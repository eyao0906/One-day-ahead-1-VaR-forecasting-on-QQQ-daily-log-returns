from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def quantile_check_loss(y: pd.Series, q: pd.Series, alpha: float) -> pd.Series:
    """Quantile check loss (pinball loss) for an alpha-quantile forecast."""
    u = y - q
    return u * (alpha - (u < 0).astype(float))

PROJECT_ROOT = Path().resolve()
DATA_DIR = PROJECT_ROOT / "outputs/Integrated_Forecast_Results.csv"
OPT_DIR = PROJECT_ROOT / "outputs"
def summarize_check_loss(df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"Model", "VaR", "Return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["error"] = out["Return"] - out["VaR"]
    out["violation_flag"] = (out["Return"] < out["VaR"]).astype(int)
    out["check_loss"] = quantile_check_loss(out["Return"], out["VaR"], alpha)
    out["violation_loss"] = np.where(out["violation_flag"] == 1, out["check_loss"], 0.0)
    out["non_violation_loss"] = np.where(out["violation_flag"] == 0, out["check_loss"], 0.0)

    summary = (
        out.groupby("Model", dropna=False)
        .agg(
            N=("check_loss", "size"),
            Violations=("violation_flag", "sum"),
            ViolationRate=("violation_flag", "mean"),
            TotalCheckLoss=("check_loss", "sum"),
            MeanCheckLoss=("check_loss", "mean"),
            MedianCheckLoss=("check_loss", "median"),
            StdCheckLoss=("check_loss", "std"),
            TotalViolationLoss=("violation_loss", "sum"),
            TotalNonViolationLoss=("non_violation_loss", "sum"),
            MeanError=("error", "mean"),
            MeanVaR=("VaR", "mean"),
            MeanReturn=("Return", "mean"),
        )
        .reset_index()
        .sort_values(["MeanCheckLoss", "TotalCheckLoss", "Model"], ascending=[True, True, True])
    )

    summary["ViolationLossShare"] = np.where(
        summary["TotalCheckLoss"] > 0,
        summary["TotalViolationLoss"] / summary["TotalCheckLoss"],
        np.nan,
    )
    summary.insert(0, "Rank", np.arange(1, len(summary) + 1))
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute quantile check loss for VaR/quantile forecasts by model."
    )
    parser.add_argument(
        "--forecasts",
        default="Integrated_Forecast_Results.csv",
        help="Path to forecast CSV containing at least Model, VaR, Return columns.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Target quantile level. Default: 0.01 for 1%% VaR.",
    )
    parser.add_argument(
        "--outdir",
        default="check_loss_outputs",
        help="Directory for row-level and summary CSV outputs.",
    )
    args = parser.parse_args()

    alpha = float(args.alpha)
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0, 1).")

    forecasts_path = DATA_DIR
    header = pd.read_csv(forecasts_path, nrows=0)
    parse_dates = ["Date"] if "Date" in header.columns else None
    df = pd.read_csv(forecasts_path, parse_dates=parse_dates)

    outdir = OPT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    row_level, summary = summarize_check_loss(df, alpha=alpha)

    row_path = outdir / "forecast_check_loss_rows.csv"
    summary_path = outdir / "model_check_loss_summary.csv"

    row_level.to_csv(row_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved row-level output to: {row_path}")
    print(f"Saved model summary to:   {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
