"""
garch_model.py
--------------
Fit a GARCH(1,1) model to daily log returns and forecast forward with a
Monte-Carlo simulation of M sample paths.

We feed returns in PERCENT (100 * log return) to the `arch` package, which is
the standard scaling that keeps the optimizer well conditioned.  The fitted
parameters therefore refer to variance in percent^2; helper functions convert
back to decimal / annualized quantities.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS = 252
DELTA = 1.0 / TRADING_DAYS


def fit_garch11(returns: pd.Series):
    """Fit GARCH(1,1) with a constant mean and Normal innovations.

    Returns the fitted `arch` results object.  `returns` are decimal log
    returns; they are scaled to percent internally.
    """
    r_pct = 100.0 * returns
    am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    return res


def garch_params_decimal(res) -> dict:
    """Extract (omega, alpha, beta) and derived long-run vol in DECIMAL units.

    arch parameters are in percent^2, so omega_dec = omega_pct / 100^2.
    """
    p = res.params
    omega_pct = p["omega"]
    alpha = p["alpha[1]"]
    beta = p["beta[1]"]
    omega_dec = omega_pct / (100.0 ** 2)
    persistence = alpha + beta
    var_long_daily = omega_dec / (1.0 - persistence)      # decimal daily var
    return {
        "omega_pct": omega_pct,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "omega_decimal": omega_dec,
        "long_run_daily_var": var_long_daily,
        "long_run_daily_vol": np.sqrt(var_long_daily),
        "long_run_annual_vol": np.sqrt(var_long_daily / DELTA),
    }


def simulate_forecast_paths(res, horizon: int, M: int = 100,
                            seed: int = 42) -> np.ndarray:
    """Simulate M forward paths of ANNUALIZED volatility over `horizon` days.

    Uses arch's simulation-based forecast, which for each simulation draws
    innovations z_t ~ N(0,1), sets X_t = sigma_t * z_t, and iterates
        sigma_{t+1}^2 = omega + alpha * X_t^2 + beta * sigma_t^2.

    Returns an array of shape (M, horizon) of annualized vol (decimal).
    """
    fc = res.forecast(horizon=horizon, method="simulation",
                      simulations=M, reindex=False)
    # residual variance per simulation: shape (1, M, horizon), percent^2
    sim_var_pct2 = fc.simulations.residual_variances[0]        # (M, horizon)
    daily_vol_dec = np.sqrt(sim_var_pct2) / 100.0             # decimal daily vol
    ann_vol = daily_vol_dec * np.sqrt(TRADING_DAYS)
    return ann_vol


def analytic_forecast(res, horizon: int) -> np.ndarray:
    """Deterministic (expected) annualized-vol forecast path, for reference.

        E[sigma_{t+k}^2] = V_L + (alpha+beta)^k (sigma_t^2 - V_L)
    """
    d = garch_params_decimal(res)
    VL = d["long_run_daily_var"]
    persistence = d["persistence"]
    # last in-sample conditional variance (percent^2 -> decimal)
    sigma2_t = (np.asarray(res.conditional_volatility)[-1] / 100.0) ** 2
    ks = np.arange(1, horizon + 1)
    Es2 = VL + persistence ** ks * (sigma2_t - VL)
    return np.sqrt(Es2 / DELTA)
