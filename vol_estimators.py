"""
vol_estimators.py
-----------------
Two rolling estimators of daily variance, returned as *annualized volatility*.

Annualization: with Delta = 1/252,
    sigma_annual = sigma_daily / sqrt(Delta) = sigma_daily * sqrt(252).

Both estimators use the zero-mean squared-return convention (Hull, RiskMetrics):
the daily mean return is negligible at the daily horizon, so variance is
estimated directly from squared returns.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252
DELTA = 1.0 / TRADING_DAYS


def moving_average_vol(returns: pd.Series, n: int = 100) -> pd.Series:
    """Annualized volatility from an n-day moving average of squared returns.

        v_t = (1/n) * sum_{i=0}^{n-1} r_{t-i}^2          (window ends at t)
        sigma_annual_t = sqrt(v_t / Delta)
    """
    daily_var = returns.pow(2).rolling(window=n).mean()
    ann_vol = np.sqrt(daily_var / DELTA)
    ann_vol.name = f"MA({n})"
    return ann_vol.dropna()


def ewma_vol(returns: pd.Series, lam: float = 0.94,
             init_window: int = 100) -> pd.Series:
    """Annualized EWMA volatility.

        v_t = lambda * v_{t-1} + (1-lambda) * r_t^2
        v_init = sample variance of the first `init_window` returns.

    The recursion starts at index `init_window`; the estimate at day t
    incorporates the return of day t.
    """
    r = returns.to_numpy()
    dates = returns.index
    N = len(r)
    if N <= init_window:
        raise ValueError("series shorter than init_window")

    v = np.full(N, np.nan)
    v_prev = np.var(r[:init_window], ddof=1)          # initial variance
    for t in range(init_window, N):
        v_prev = lam * v_prev + (1.0 - lam) * r[t] ** 2
        v[t] = v_prev

    ann_vol = pd.Series(np.sqrt(v / DELTA), index=dates, name=f"EWMA(lam={lam})")
    return ann_vol.dropna()
