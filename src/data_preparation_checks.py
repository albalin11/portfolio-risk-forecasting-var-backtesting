"""Trading-session and adjusted-price checks."""
from numbers import Real
import numpy as np
import pandas as pd
import exchange_calendars as xcals


def check_trading_dates(dates, start, end, tickers=None):
    """Require every expected US equity session, including both sample boundaries."""
    dates = pd.DatetimeIndex(dates)
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError('Expected timezone-naive daily session labels without NaT or intraday times.')
    duplicates = dates[dates.duplicated()].unique()
    if len(duplicates):
        raise ValueError(f'Duplicate dates: {duplicates.strftime("%Y-%m-%d").tolist()}')
    if not dates.is_monotonic_increasing:
        raise ValueError('Dates must be chronological; sorting would hide a raw-data issue.')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # Padding lets the requested boundary itself be a holiday or weekend.
    calendar = xcals.get_calendar('XNYS', start=start-pd.Timedelta(days=7), end=end+pd.Timedelta(days=7))
    expected = calendar.sessions_in_range(start, end).tz_localize(None)
    missing = expected.difference(dates)
    unexpected = dates.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(f'Missing trading sessions ({len(missing)}): {missing.strftime("%Y-%m-%d").tolist()}; '
                         f'unexpected dates ({len(unexpected)}): {unexpected.strftime("%Y-%m-%d").tolist()}; '
                         f'affected tickers: {tickers if tickers is not None else "all input assets"}')
    return dict(calendar='XNYS', package_version=xcals.__version__, expected_sessions=len(expected),
                expected_first=str(expected[0].date()), expected_last=str(expected[-1].date()),
                missing_sessions=len(missing), unexpected_dates=len(unexpected))


def check_adjusted_prices(prices, tickers):
    """Reject missing, nonnumeric, non-finite or non-positive adjusted prices."""
    if list(prices.columns) != tickers:
        raise ValueError(f'Expected adjusted-price columns in this order: {tickers}')
    errors = []
    for asset in tickers:
        for date, value in prices[asset].items():
            if pd.isna(value):
                reason = 'missing Adj Close'
            elif not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
                reason = 'nonnumeric Adj Close'
            elif not np.isfinite(value):
                reason = 'non-finite Adj Close'
            elif value <= 0:
                reason = 'non-positive Adj Close'
            else:
                continue
            errors.append(f'{asset}: {reason}; date={date:%Y-%m-%d}; value={value!r}')
    if errors:
        raise ValueError('Invalid adjusted prices:\n' + '\n'.join(errors))
