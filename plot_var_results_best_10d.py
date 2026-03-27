from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 10-day VaR forecasts vs realized returns")
    parser.add_argument("--forecasts", default="outputs_10d/var_forecasts_best_10d.csv")
    parser.add_argument("--outdir", default="outputs_10d/plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.forecasts, parse_dates=["Date", "OriginDate"])

    for model, g in df.groupby("Model"):
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

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
