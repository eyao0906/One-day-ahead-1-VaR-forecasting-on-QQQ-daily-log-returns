from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VaR extensions vs realized returns")
    parser.add_argument("--forecasts", default="outputs/var_forecasts_ext.csv")
    parser.add_argument("--outdir", default="outputs/plots_ext")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.forecasts, parse_dates=["Date"])

    for model, g in df.groupby("Model"):
        g = g.sort_values("Date")
        print(model, g["VaR"].describe())
        plt.figure(figsize=(12, 4))
        plt.plot(g["Date"], g["Return"], label="Realized Return", linewidth=1, color="black", alpha=0.7)
        plt.plot(g["Date"], g["VaR"], label=f"Forecast VaR ({model})", linewidth=1.5, color="red")
        if {"lower_bound", "upper_bound"}.issubset(g.columns):
            plt.fill_between(g["Date"], g["lower_bound"], g["upper_bound"], alpha=0.15, color="red", label="95% CI")
        viol = g[g["Violation"] == 1]
        if not viol.empty:
            plt.scatter(viol["Date"], viol["Return"], s=20, color="blue", label="Violations", zorder=5)
        plt.title(f"VaR Backtest: {model}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = model.replace("/", "-").replace("(", "").replace(")", "").replace(" ", "_")
        plt.savefig(outdir / f"var_{safe}.png", dpi=150)
        plt.close()

    print(f"Saved extension plots to: {outdir}")


if __name__ == "__main__":
    main()
