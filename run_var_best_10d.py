from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from var_utils_best_10d import (
    christoffersen_independence_test,
    forecast_fhs_garch_10d,
    forecast_fhs_gjr_10d,
    forecast_historical_simulation_10d,
    kupiec_test,
)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "log_ret"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    df = df[["Date", "log_ret"]].dropna().sort_values("Date").reset_index(drop=True)
    return df


def run_rolling_var(
    df: pd.DataFrame,
    alpha: float,
    initial_train: float,
    hs_window: int,
    horizon: int,
    n_sims: int,
    max_steps: int | None,
) -> pd.DataFrame:
    n = len(df)
    h = max(int(horizon), 1)
    train_end0 = int(n * initial_train)
    if train_end0 < 500:
        train_end0 = min(max(500, train_end0), n - h)

    rows: list[dict] = []
    end_limit = n - h
    if max_steps is not None:
        end_limit = min(end_limit, train_end0 + max_steps - 1)

    for t in range(train_end0, end_limit + 1):
        train = df.iloc[:t]
        future = df.iloc[t : t + h]
        r_train = train["log_ret"].to_numpy()
        realized = float(future["log_ret"].sum())
        origin_date = train.iloc[-1]["Date"]
        end_date = future.iloc[-1]["Date"]

        model_funcs = [
            (
                "Historical-Simulation-10D",
                lambda: forecast_historical_simulation_10d(
                    r_train,
                    alpha=alpha,
                    horizon=h,
                    window=hs_window,
                ),
            ),
            (
                "GARCH-FHS-10D",
                lambda: forecast_fhs_garch_10d(
                    r_train,
                    alpha=alpha,
                    horizon=h,
                    n_sims=n_sims,
                    seed=1000003 * t + 11,
                ),
            ),
            (
                "GJR-FHS-10D",
                lambda: forecast_fhs_gjr_10d(
                    r_train,
                    alpha=alpha,
                    horizon=h,
                    n_sims=n_sims,
                    seed=1000003 * t + 29,
                ),
            ),
        ]

        for name, func in model_funcs:
            try:
                fc = func()
                var = float(fc.var)
                hit = int(realized < var)
                row = {
                    "Date": end_date,
                    "OriginDate": origin_date,
                    "Model": name,
                    "alpha": alpha,
                    "HorizonDays": h,
                    "VaR": var,
                    "Return": realized,
                    "Violation": hit,
                }
                if isinstance(fc.meta, dict):
                    for k, v in fc.meta.items():
                        if np.isscalar(v):
                            row[f"meta_{k}"] = float(v) if v == v else np.nan
                rows.append(row)
            except Exception as exc:
                print(f"[warn] step {t}, model {name} failed: {exc}")

        if (t - train_end0 + 1) % 50 == 0:
            print(f"Processed {t - train_end0 + 1} forecast origins...")

    return pd.DataFrame(rows)


def summarize_backtests(forecasts: pd.DataFrame, alpha: float) -> pd.DataFrame:
    records = []
    for model, g in forecasts.groupby("Model"):
        v = g["Violation"].to_numpy()
        lr_uc, p_uc = kupiec_test(v, alpha)
        lr_ind, p_ind = christoffersen_independence_test(v)
        lr_cc = lr_uc + lr_ind
        p_cc = float(1 - chi2.cdf(lr_cc, df=2))
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
                "Note": "10-day horizons overlap; independence p-values should be interpreted cautiously.",
            }
        )

    return pd.DataFrame(records).sort_values("ConditionalCoverage_pvalue", ascending=False)


def plot_forecasts(forecasts: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for model, g in forecasts.groupby("Model"):
        g = g.sort_values("Date")
        plt.figure(figsize=(12, 4.5))
        plt.plot(g["Date"], g["Return"], label="Realized 10-day Return", linewidth=1)
        plt.plot(g["Date"], g["VaR"], label="Forecast 10-day VaR", linewidth=1)
        viol = g[g["Violation"] == 1]
        if not viol.empty:
            plt.scatter(viol["Date"], viol["Return"], s=14, label="Violations")
        plt.title(f"10-Day VaR Backtest: {model}")
        plt.legend()
        plt.tight_layout()
        safe = model.replace("/", "-").replace("(", "").replace(")", "").replace(" ", "_")
        plt.savefig(outdir / f"var_{safe}.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="QQQ 10-day-ahead VaR workflow for best-performing models")
    parser.add_argument("--data", default="data/model_data.csv")
    parser.add_argument("--outdir", default="outputs_10d")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--initial-train", type=float, default=0.7)
    parser.add_argument("--hs-window", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    forecasts = run_rolling_var(
        df=df,
        alpha=args.alpha,
        initial_train=args.initial_train,
        hs_window=args.hs_window,
        horizon=args.horizon,
        n_sims=args.n_sims,
        max_steps=args.max_steps,
    )
    if forecasts.empty:
        raise RuntimeError("No forecasts produced. Check the input data or reduce initial_train.")

    summary = summarize_backtests(forecasts, alpha=args.alpha)

    forecasts.to_csv(outdir / "var_forecasts_best_10d.csv", index=False)
    summary.to_csv(outdir / "var_backtests_best_10d.csv", index=False)
    plot_forecasts(forecasts, outdir / "plots")

    print(f"Saved: {outdir / 'var_forecasts_best_10d.csv'}")
    print(f"Saved: {outdir / 'var_backtests_best_10d.csv'}")
    print(f"Saved plots to: {outdir / 'plots'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
