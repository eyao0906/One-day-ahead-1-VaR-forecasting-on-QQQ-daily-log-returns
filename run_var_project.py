from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from var_utils import (
    christoffersen_independence_test,
    forecast_caviar_sav,
    forecast_garch,
    forecast_gjr_garch,
    forecast_historical_simulation,
    forecast_fhs_garch_t,
    forecast_fhs_gjr_t,
    forecast_bootstrap_garch_t,
    kupiec_test,
)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "log_ret", "vix_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    df = df[["Date", "log_ret", "vix_close"]].dropna().sort_values("Date").reset_index(drop=True)
    return df


def run_rolling_var(
    df: pd.DataFrame,
    alpha: float,
    initial_train: float,
    hs_window: int,
    max_steps: int | None,
) -> pd.DataFrame:
    n = len(df)
    train_end0 = int(n * initial_train)
    if train_end0 < 500:
        train_end0 = min(max(500, train_end0), n - 30)

    rows = []
    end_limit = n - 1
    if max_steps is not None:
        end_limit = min(end_limit, train_end0 + max_steps - 1)

    for t in range(train_end0, end_limit + 1):
        train = df.iloc[:t]
        test_row = df.iloc[t]

        r_train = train["log_ret"].to_numpy()

        model_funcs = [
            ("GARCH(1,1)", lambda: forecast_garch(r_train, alpha=alpha)),
            ("GJR-GARCH", lambda: forecast_gjr_garch(r_train, alpha=alpha)),
            ("CAViaR-SAV", lambda: forecast_caviar_sav(r_train, alpha=alpha)),
            ("Historical-Simulation", lambda: forecast_historical_simulation(r_train, alpha=alpha, window=hs_window)),
            ("GARCH-t-FHS", lambda: forecast_fhs_garch_t(r_train, alpha=alpha)),
            ("GJR-t-FHS", lambda: forecast_fhs_gjr_t(r_train, alpha=alpha)),
            ("GARCH-t-Bootstrap", lambda: forecast_bootstrap_garch_t(r_train, alpha=alpha)),
        ]
        realized = float(test_row["log_ret"])
        date = test_row["Date"]
        for name, func in model_funcs:
            try:
                fc = func()
                var = float(fc.var)
                lower_bound = float(fc.meta.get("lower_bound", var)) if isinstance(fc.meta, dict) else var
                upper_bound = float(fc.meta.get("upper_bound", var)) if isinstance(fc.meta, dict) else var
                hit = int(realized < var)

                row = {
                    "Date": date,
                    "Model": name,
                    "alpha": alpha,
                    "VaR": var,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "Return": realized,
                    "Violation": hit,
                }

                if isinstance(fc.meta, dict):
                    for k, v in fc.meta.items():
                        if k in {"lower_bound", "upper_bound"}:
                            continue
                        if np.isscalar(v):
                            row[f"meta_{k}"] = float(v) if v == v else np.nan

                rows.append(row)

            except Exception as e:
                print(f"[warn] step {t}, model {name} failed: {e}")

    return pd.DataFrame(rows)


def summarize_backtests(forecasts: pd.DataFrame, alpha: float) -> pd.DataFrame:
    records = []
    for model, g in forecasts.groupby("Model"):
        v = g["Violation"].to_numpy()
        lr_uc, p_uc = kupiec_test(v, alpha)
        lr_ind, p_ind = christoffersen_independence_test(v)
        lr_cc = lr_uc + lr_ind
        p_cc = float(1 - __import__("scipy").stats.chi2.cdf(lr_cc, df=2))
        records.append(
            {
                "Model": model,
                "N": len(g),
                "Violations": int(v.sum()),
                "ExpectedViolations": float(alpha * len(g)),
                "ViolationRate": float(v.mean()),
                "Kupiec_LRuc": lr_uc,
                "Kupiec_pvalue": p_uc,
                "Christoffersen_LRind": lr_ind,
                "Christoffersen_pvalue": p_ind,
                "ConditionalCoverage_LRcc": lr_cc,
                "ConditionalCoverage_pvalue": p_cc,
            }
        )

    out = pd.DataFrame(records).sort_values("ConditionalCoverage_pvalue", ascending=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="QQQ one-day-ahead VaR backtesting project")
    parser.add_argument("--data", default="data/model_data.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--initial-train", type=float, default=0.7)
    parser.add_argument("--hs-window", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    forecasts = run_rolling_var(
        df,
        alpha=args.alpha,
        initial_train=args.initial_train,
        hs_window=args.hs_window,
        max_steps=args.max_steps,
    )
    if forecasts.empty:
        raise RuntimeError("No forecasts produced. Try lowering initial_train or checking data.")

    summary = summarize_backtests(forecasts, alpha=args.alpha)

    forecasts.to_csv(outdir / "var_forecasts.csv", index=False)
    summary.to_csv(outdir / "var_backtests.csv", index=False)

    print(f"Saved: {outdir / 'var_forecasts.csv'}")
    print(f"Saved: {outdir / 'var_backtests.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
