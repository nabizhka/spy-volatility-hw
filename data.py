"""
data.py
-------
Load daily closing prices for SPY and compute daily log returns.

Primary source : Yahoo Finance (via yfinance)
Fallback       : Stooq (via pandas-datareader)
Offline option : a local CSV cached from a previous run

The functions return a pandas Series/DataFrame indexed by date.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd


CACHE = os.path.join(os.path.dirname(__file__), "..", "spy_prices.csv")


def load_spy_close(start: str, end: str, ticker: str = "SPY",
                   use_cache: bool = True) -> pd.Series:
    """Return a Series of adjusted daily closing prices between start and end.

    Dates are inclusive of `start` and exclusive of `end` in yfinance;
    we pad `end` by one day so the last requested day is included.
    """
    end_padded = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # 1) cached CSV (fast + fully reproducible once fetched)
    if use_cache and os.path.exists(CACHE):
        s = pd.read_csv(CACHE, index_col=0, parse_dates=True).iloc[:, 0]
        s = s.loc[(s.index >= start) & (s.index <= end)]
        if len(s) > 0:
            print(f"[data] loaded {len(s)} rows from cache {CACHE}")
            return s.astype(float)

    # 2) Yahoo Finance
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end_padded,
                         progress=False, auto_adjust=True)
        if len(df) > 0:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):      # multiindex guard
                close = close.iloc[:, 0]
            close.name = ticker
            close.to_csv(CACHE)
            print(f"[data] downloaded {len(close)} rows from Yahoo Finance")
            return close.astype(float)
    except Exception as exc:                          # pragma: no cover
        print(f"[data] yfinance failed: {exc}")

    # 3) Stooq
    try:
        import pandas_datareader.data as web
        df = web.DataReader(ticker, "stooq", start=start, end=end)
        df = df.sort_index()
        close = df["Close"]
        close.name = ticker
        close.to_csv(CACHE)
        print(f"[data] downloaded {len(close)} rows from Stooq")
        return close.astype(float)
    except Exception as exc:                           # pragma: no cover
        raise RuntimeError(
            "Could not load SPY prices from Yahoo or Stooq. "
            "Run once with internet access to populate spy_prices.csv, "
            "or drop a CSV (date,close) at that path."
        ) from exc


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns r_t = ln(P_t / P_{t-1})."""
    r = np.log(prices / prices.shift(1)).dropna()
    r.name = "log_return"
    return r
