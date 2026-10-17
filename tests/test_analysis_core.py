"""Checks the two alignments most likely to change the conclusions."""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2
from pathlib import Path

from src.feature_engineering import ASSETS, build_portfolio_features, build_future_volatility_target
from src.var_backtesting import kupiec_test


class CoreCalculations(unittest.TestCase):
    def test_features_use_only_past_and_target_uses_next_five(self):
        dates = pd.bdate_range('2020-01-01', periods=80)
        values = np.arange(80)[:, None] * np.arange(1, 8)[None, :] / 100_000
        returns = pd.DataFrame(values, index=dates, columns=ASSETS)
        full = build_portfolio_features(returns)
        prefix = build_portfolio_features(returns.iloc[:70])
        pd.testing.assert_series_equal(full.iloc[69], prefix.iloc[69])

        portfolio = returns.mean(axis=1)
        target = build_future_volatility_target(portfolio)
        expected = portfolio.iloc[70:75].std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(target.iloc[69], expected)
        self.assertTrue(target.iloc[-5:].isna().all())

    def test_nonoverlap_and_kupiec_count(self):
        dates = pd.bdate_range('2022-01-03', periods=25)
        anchors = dates[::5]
        starts = dates[[dates.get_loc(date) + 1 for date in anchors[:-1]]]
        ends = dates[[dates.get_loc(date) + 5 for date in anchors[:-1]]]
        self.assertTrue((starts[1:] > ends[:-1]).all())

        flags = np.arange(100) < 7
        statistic, p_value = kupiec_test(flags, 0.05)
        expected = 2 * (binom.logpmf(7, 100, 0.07) - binom.logpmf(7, 100, 0.05))
        self.assertAlmostEqual(statistic, expected)
        self.assertAlmostEqual(p_value, chi2.sf(expected, 1))
        zero_stat, zero_p = kupiec_test(np.zeros(100, dtype=bool), 0.05)
        self.assertTrue(np.isfinite(zero_stat) and 0 <= zero_p <= 1)

    def test_saved_backtest_from_prices_and_forecasts(self):
        root = Path(__file__).resolve().parents[1]
        returns = pd.read_csv(root / 'data/processed/daily_returns.csv', index_col='Date', parse_dates=True)
        forecasts = pd.read_csv(root / 'outputs/tables/test_predictions.csv', index_col='Date',
                                parse_dates=['Date', 'target_start', 'target_end'])
        saved = pd.read_csv(root / 'outputs/tables/kupiec_backtests.csv')
        portfolio = returns.mean(axis=1)
        anchors = forecasts.index[::5]
        for row in saved.itertuples(index=False):
            actual = []
            for date in anchors:
                position = portfolio.index.get_loc(date)
                window = portfolio.iloc[position + 1:position + 6]
                self.assertEqual(window.index[0], forecasts.loc[date, 'target_start'])
                self.assertEqual(window.index[-1], forecasts.loc[date, 'target_end'])
                actual.append(np.prod(1 + window.to_numpy()) - 1)
            z = 1.6448536269514722 if row.confidence == 0.95 else 2.3263478740408408
            threshold = -z * forecasts.loc[anchors, row.model].to_numpy() * np.sqrt(5 / 252)
            flags = np.array(actual) < threshold
            self.assertEqual(row.observations, len(anchors))
            self.assertEqual(row.violations, flags.sum())
            statistic, p_value = kupiec_test(flags, 1 - row.confidence)
            self.assertAlmostEqual(row.kupiec_statistic, statistic)
            self.assertAlmostEqual(row.kupiec_p_value, p_value)

    def test_saved_split_labels_do_not_cross_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        data = pd.read_csv(root / 'data/processed/modeling_data.csv', index_col='Date', parse_dates=True)
        returns = pd.read_csv(root / 'data/processed/daily_returns.csv', index_col='Date', parse_dates=True)
        splits = pd.read_csv(root / 'outputs/tables/time_splits.csv').set_index('split')
        positions = returns.index.get_indexer(data.index)
        ends = pd.Series(returns.index[positions + 5], index=data.index)
        validation_start = pd.Timestamp(splits.loc['validation', 'first_date'])
        test_start = pd.Timestamp(splits.loc['test', 'first_date'])
        train = data.loc[(data.index < validation_start) & (ends < validation_start)]
        validation = data.loc[(data.index >= validation_start) & (ends < test_start)]
        test = data.loc[data.index >= test_start]
        for name, frame in [('train', train), ('validation', validation), ('test', test)]:
            self.assertEqual(len(frame), splits.loc[name, 'rows'])
            self.assertEqual(frame.index[-1], pd.Timestamp(splits.loc[name, 'last_date']))
        self.assertLess(ends.loc[train.index].max(), validation_start)
        self.assertLess(ends.loc[validation.index].max(), test_start)


if __name__ == '__main__':
    unittest.main()
