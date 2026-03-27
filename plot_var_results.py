from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VaR forecasts vs realized returns")
    parser.add_argument("--forecasts", default="outputs/var_forecasts.csv")
    parser.add_argument("--outdir", default="outputs/plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.forecasts, parse_dates=["Date"])

    for model, g in df.groupby("Model"):
        g = g.sort_values("Date")
        print(model, g["VaR"].describe())
        plt.figure(figsize=(12, 4))
        plt.plot(g["Date"], g["Return"], label="Realized Return", linewidth=1)
        plt.plot(g["Date"], g["VaR"], label="Forecast VaR", linewidth=1)
        viol = g[g["Violation"] == 1]
        if not viol.empty:
            plt.scatter(viol["Date"], viol["Return"], s=12, label="Violations")
        plt.title(f"VaR Backtest: {model}")
        plt.legend()
        plt.tight_layout()
        safe = model.replace("/", "-").replace("(", "").replace(")", "").replace(" ", "_")
        plt.savefig(outdir / f"var_{safe}.png", dpi=150)
        plt.close()

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
