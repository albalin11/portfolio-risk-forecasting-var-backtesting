"""Five-day volatility forecasts and VaR for an equal-weight ETF portfolio."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.data_preparation_checks import check_trading_dates, check_adjusted_prices
from src.feature_engineering import ASSETS, FEATURE_COLUMNS, TARGET_COLUMN, build_portfolio_features, build_future_volatility_target
from src.var_backtesting import kupiec_test


ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/raw/yahoo_ohlcv_20100101_20260916.csv'
META = RAW.with_name('yahoo_ohlcv_20100101_20260916_metadata.json')
TABLES = ROOT / 'outputs/tables'
FIGURES = ROOT / 'outputs/figures'
for folder in (RAW.parent, TABLES, FIGURES, ROOT / 'data/processed'):
    folder.mkdir(parents=True, exist_ok=True)

# Reuse the dated local snapshot so provider revisions do not silently change a run.
if RAW.exists():
    metadata = json.loads(META.read_text())
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == metadata['sha256']
    assert metadata['tickers'] == ASSETS and metadata['start'] == '2010-01-01'
    assert metadata['end_exclusive'] == '2026-09-17'
    raw = pd.read_csv(RAW, header=[0, 1], index_col=0, parse_dates=True, float_precision='round_trip')
else:
    yf.set_tz_cache_location(str(RAW.parent / 'yfinance_cache'))
    raw = yf.download(ASSETS, start='2010-01-01', end='2026-09-17',
                      auto_adjust=False, keepna=True, progress=False, threads=False)
    if raw.empty:
        raise ValueError('Yahoo download returned no prices.')

calendar = check_trading_dates(raw.index, '2010-01-01', '2026-09-16', ASSETS)
if not isinstance(raw.columns, pd.MultiIndex) or not raw.columns.is_unique:
    raise ValueError('Expected distinct Yahoo price fields by asset.')
if any(('Adj Close', asset) not in raw.columns for asset in ASSETS):
    raise ValueError('An adjusted close is missing; investigate the source data.')
prices = raw['Adj Close'][ASSETS].copy()
prices.index.name = 'Date'
check_adjusted_prices(prices, ASSETS)
if not RAW.exists():
    raw.to_csv(RAW, index_label='Date')
    META.write_text(json.dumps({
        'source': 'Yahoo Finance via yfinance', 'tickers': ASSETS,
        'start': '2010-01-01', 'end_exclusive': '2026-09-17',
        'sha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
    }, indent=2))

asset_returns = prices.pct_change(fill_method=None).iloc[1:]
assert np.isfinite(asset_returns.to_numpy()).all()
portfolio = asset_returns.mean(axis=1)  # Constant 1/7 weights, rebalanced daily.
features = build_portfolio_features(asset_returns)
target = build_future_volatility_target(portfolio)
data = features.assign(**{TARGET_COLUMN: target}).dropna()
assert data.index.equals(asset_returns.index[59:-5])

prices.to_csv(ROOT / 'data/processed/adjusted_prices.csv')
asset_returns.to_csv(ROOT / 'data/processed/daily_returns.csv')
data.to_csv(ROOT / 'data/processed/modeling_data.csv')

# The target for t uses t+1,...,t+5. Purge labels that cross each split.
return_dates = asset_returns.index
positions = return_dates.get_indexer(data.index)
target_start = pd.Series(return_dates[positions + 1], index=data.index)
target_end = pd.Series(return_dates[positions + 5], index=data.index)
first_validation = data.index[data.index >= '2018-01-01'][0]
first_test = data.index[data.index >= '2022-01-01'][0]
train = data.loc[(data.index < first_validation) & (target_end < first_validation)]
validation = data.loc[(data.index >= first_validation) & (data.index < first_test)
                      & (target_end < first_test)]
test = data.loc[data.index >= first_test]
refit = data.loc[(data.index < first_test) & (target_end < first_test)]
assert target_end.loc[train.index].max() < first_validation
assert target_end.loc[validation.index].max() < first_test
assert target_end.loc[refit.index].max() < first_test
splits = pd.DataFrame([{'split': name, 'rows': len(frame),
                        'first_date': frame.index[0].date(), 'last_date': frame.index[-1].date(),
                        'first_target_start': target_start.loc[frame.index].iloc[0].date(),
                        'last_target_end': target_end.loc[frame.index].iloc[-1].date(),
                        'purged_before_next_split': 5 if name != 'test' else 0}
                       for name, frame in [('train', train), ('validation', validation), ('test', test)]])
splits.to_csv(TABLES / 'time_splits.csv', index=False)

# EWMA starts with the first available return, including the 60-day feature warmup.
historical = features['volatility_20d']
ewma = np.sqrt(252 * portfolio.pow(2).ewm(alpha=0.06, adjust=False).mean())
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=3,
                                          min_samples_leaf=10, random_state=42, n_jobs=1),
    'XGBoost': XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=1.0,
                           objective='reg:squarederror', random_state=42, n_jobs=1),
}


def scores(actual, forecast):
    return {'MAE': mean_absolute_error(actual, forecast),
            'RMSE': np.sqrt(mean_squared_error(actual, forecast))}


validation_forecasts = {
    'Historical Volatility': historical.loc[validation.index],
    'EWMA': ewma.loc[validation.index],
}
for name, model in models.items():
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    validation_forecasts[name] = np.maximum(model.predict(validation[FEATURE_COLUMNS]), 1e-8)
validation_metrics = pd.DataFrame([
    {'model': name, **scores(validation[TARGET_COLUMN], forecast)}
    for name, forecast in validation_forecasts.items()
]).sort_values(['MAE', 'RMSE'])
validation_metrics.to_csv(TABLES / 'validation_metrics.csv', index=False)
selected = validation_metrics[validation_metrics.model.isin(models)].iloc[0]['model']

# Model choice uses validation MAE; only that ML model is refitted and scored on test.
models[selected].fit(refit[FEATURE_COLUMNS], refit[TARGET_COLUMN])
test_forecasts = pd.DataFrame({
    'actual_volatility': test[TARGET_COLUMN],
    'target_start': target_start.loc[test.index],
    'target_end': target_end.loc[test.index],
    'Historical Volatility': historical.loc[test.index],
    'EWMA': ewma.loc[test.index],
    selected: np.maximum(models[selected].predict(test[FEATURE_COLUMNS]), 1e-8),
})
test_forecasts.index.name = 'Date'
test_metrics = pd.DataFrame([
    {'model': name, **scores(test[TARGET_COLUMN], test_forecasts[name])}
    for name in ['Historical Volatility', 'EWMA', selected]
])
test_metrics.to_csv(TABLES / 'test_metrics.csv', index=False)
test_forecasts.to_csv(TABLES / 'test_predictions.csv')  # Local observation-level output.

future_returns = pd.concat([portfolio.shift(-step) for step in range(1, 6)], axis=1)
actual_5d = ((1 + future_returns).prod(axis=1) - 1).where(future_returns.notna().all(axis=1))
actual_5d = actual_5d.loc[test.index]
assert actual_5d.notna().all()
assert (test_forecasts.target_start == return_dates[return_dates.get_indexer(test.index) + 1]).all()
assert (test_forecasts.target_end == return_dates[return_dates.get_indexer(test.index) + 5]).all()

# Forecasts spaced five sessions apart have disjoint subsequent return windows.
dates = test.index[::5]
assert (target_start.loc[dates].iloc[1:].to_numpy() > target_end.loc[dates].iloc[:-1].to_numpy()).all()
risk_rows = []
for name in test_metrics.model:
    for confidence, z in [(0.95, 1.6448536269514722), (0.99, 2.3263478740408408)]:
        var = z * test_forecasts.loc[dates, name] * np.sqrt(5 / 252)
        violations = actual_5d.loc[dates] < -var
        statistic, p_value = kupiec_test(violations, 1 - confidence)
        risk_rows.append({'model': name, 'confidence': confidence, 'observations': len(dates),
                          'violations': int(violations.sum()), 'violation_rate': violations.mean(),
                          'expected_violation_rate': 1 - confidence,
                          'kupiec_statistic': statistic, 'kupiec_p_value': p_value})
backtests = pd.DataFrame(risk_rows)
backtests.to_csv(TABLES / 'kupiec_backtests.csv', index=False)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
for axis, name in zip(axes, test_metrics.model):
    axis.plot(test.index, test[TARGET_COLUMN] * 100, color='0.6', lw=0.8, label='Realised')
    axis.plot(test.index, test_forecasts[name] * 100, lw=1, label=name)
    axis.set_ylabel('Annualised vol. (%)')
    axis.legend(loc='upper right')
axes[-1].set_xlabel('Forecast date')
fig.savefig(FIGURES / 'test_volatility_forecasts.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
for row, name in enumerate(test_metrics.model):
    for col, (confidence, z) in enumerate([(0.95, 1.6448536269514722), (0.99, 2.3263478740408408)]):
        axis = axes[row, col]
        threshold = -z * test_forecasts.loc[dates, name] * np.sqrt(5 / 252)
        outcomes = actual_5d.loc[dates]
        axis.plot(dates, outcomes * 100, color='0.6', lw=0.8, label='Five-day return')
        axis.plot(dates, threshold * 100, lw=1, label='VaR threshold')
        breached = outcomes < threshold
        axis.scatter(dates[breached], outcomes[breached] * 100, color='firebrick', s=12)
        axis.set_title(f'{name}, {confidence:.0%}')
        axis.set_ylabel('Return (%)')
axes[0, 0].legend(loc='upper right')
axes[-1, 0].set_xlabel('Forecast date')
axes[-1, 1].set_xlabel('Forecast date')
fig.savefig(FIGURES / 'var_violations.png', dpi=150)
plt.close(fig)

print(f'{len(prices)} price dates ({calendar["expected_first"]} to {calendar["expected_last"]}); '
      f'{len(data)} modeling rows; selected: {selected}')
print(splits.to_string(index=False))
print(validation_metrics.to_string(index=False))
print(test_metrics.to_string(index=False))
print(backtests.to_string(index=False))
