"""
main.py
-------
Runs the full assignment:

  Task 1  Annualized MA(100) and EWMA(0.94) volatility, 2021-08-01 .. 2026-07-30
  Task 2  Fit GARCH(1,1) on 2025-06-01 .. 2026-05-31
  Task 3  Monte-Carlo forecast (M=100 paths) over 2026-06-01 .. 2026-07-30,
          compared with the realized MA and EWMA estimates.

Figures are written to ../figures/.

Usage:  python -m src.main         (from the repo root)
   or:  python src/main.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:                                   # allow "python src/main.py" and "-m src.main"
    from .data import load_spy_close, log_returns
    from .vol_estimators import moving_average_vol, ewma_vol, TRADING_DAYS
    from .garch_model import (fit_garch11, garch_params_decimal,
                              simulate_forecast_paths, analytic_forecast)
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load_spy_close, log_returns
    from vol_estimators import moving_average_vol, ewma_vol, TRADING_DAYS
    from garch_model import (fit_garch11, garch_params_decimal,
                             simulate_forecast_paths, analytic_forecast)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

FULL_START, FULL_END = "2021-08-01", "2026-07-30"
FIT_START, FIT_END = "2025-06-01", "2026-05-31"
FC_START, FC_END = "2026-06-01", "2026-07-30"
LAM, N_MA, M_PATHS = 0.94, 100, 100


def task1(returns):
    ma = moving_average_vol(returns, n=N_MA)
    ew = ewma_vol(returns, lam=LAM, init_window=N_MA)

    plt.figure(figsize=(11, 5))
    plt.plot(ma.index, ma.values, label=f"MA({N_MA})", lw=1.1)
    plt.plot(ew.index, ew.values, label=f"EWMA($\\lambda$={LAM})", lw=1.1)
    plt.title("SPY annualized volatility of daily log returns (2021-08 to 2026-07)")
    plt.ylabel("annualized volatility")
    plt.xlabel("date")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "task1_ma_ewma.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[task1] saved {out}")
    return ma, ew


def task2(returns):
    fit_r = returns.loc[(returns.index >= FIT_START) & (returns.index <= FIT_END)]
    res = fit_garch11(fit_r)
    d = garch_params_decimal(res)
    print("\n[task2] GARCH(1,1) fit on", FIT_START, "..", FIT_END,
          f"({len(fit_r)} obs)")
    print(res.summary().tables[1])
    print(f"  omega (pct^2)      = {d['omega_pct']:.6g}")
    print(f"  alpha              = {d['alpha']:.6f}")
    print(f"  beta               = {d['beta']:.6f}")
    print(f"  alpha+beta         = {d['persistence']:.6f}")
    print(f"  long-run daily vol = {d['long_run_daily_vol']*100:.4f}% per day")
    print(f"  long-run ann. vol  = {d['long_run_annual_vol']*100:.2f}%")
    return res, d, fit_r


def task3(returns, res):
    fc_r = returns.loc[(returns.index >= FC_START) & (returns.index <= FC_END)]
    horizon = len(fc_r)
    print(f"\n[task3] forecast horizon = {horizon} trading days")

    sim = simulate_forecast_paths(res, horizon=horizon, M=M_PATHS)   # (M, H)
    expected = analytic_forecast(res, horizon=horizon)              # (H,)

    # realized estimators over the SAME window (need warm-up, so recompute on tail)
    tail = returns.loc[returns.index <= FC_END]
    ma = moving_average_vol(tail, n=N_MA).loc[fc_r.index]
    ew = ewma_vol(tail, lam=LAM, init_window=N_MA).loc[fc_r.index]

    x = np.arange(horizon)
    plt.figure(figsize=(11, 5))
    for m in range(sim.shape[0]):
        plt.plot(x, sim[m], color="0.75", lw=0.5, alpha=0.5,
                 label="_nolegend_" if m else "GARCH sim paths")
    plt.plot(x, sim.mean(axis=0), color="C3", lw=2, label="sim mean")
    plt.plot(x, expected, color="C1", lw=2, ls="--", label="analytic E[vol]")
    plt.plot(x, ma.values, color="C0", lw=2, label=f"realized MA({N_MA})")
    plt.plot(x, ew.values, color="C2", lw=2, label=f"realized EWMA({LAM})")
    plt.title("GARCH(1,1) forecast paths vs realized MA / EWMA (2026-06 to 2026-07)")
    plt.ylabel("annualized volatility")
    plt.xlabel("trading days into forecast window")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "task3_forecast_vs_realized.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[task3] saved {out}")
    return sim, ma, ew


def main():
    prices = load_spy_close(FULL_START, FULL_END)
    returns = log_returns(prices)
    print(f"[main] {len(returns)} daily returns "
          f"{returns.index[0].date()} .. {returns.index[-1].date()}")

    task1(returns)
    res, _, _ = task2(returns)
    task3(returns, res)
    print("\nDone. See ../figures/ for output plots.")


if __name__ == "__main__":
    main()
