# Data provenance

`analysis.py` requests Yahoo Finance daily OHLCV data for SPY, QQQ, IWM, TLT, HYG, GLD and DBC from 2010-01-01 through 2026-09-16 (the API end date is exclusive). It uses the `Adj Close` field with `auto_adjust=False` and checks all expected XNYS sessions and all seven adjusted prices before calculating returns. Missing prices are neither filled nor silently dropped.

The local raw CSV and SHA-256 metadata preserve the download vintage. They and the derived observation-level files are excluded from Git. If the snapshot is absent, the script downloads a new vintage; Yahoo revisions may change the reported numbers. The published aggregate tables and figures are therefore a record of this run, while exact numerical reproduction requires the original local snapshot.

The [yfinance project](https://github.com/ranaroussi/yfinance#readme) distinguishes its software license from rights to the downloaded data. This repository does not redistribute the Yahoo price observations.
