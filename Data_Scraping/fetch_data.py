from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for {symbol}.")

    # Normalize column names for multi-index/single-index output
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index.name = "Date"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch QQQ and VIX from Yahoo Finance")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    qqq = fetch_prices("QQQ", args.start, args.end)
    vix = fetch_prices("^VIX", args.start, args.end)

    qqq.to_csv(outdir / "qqq.csv")
    vix.to_csv(outdir / "vix.csv")

    # Merge core modeling frame
    merged = pd.DataFrame(index=qqq.index)
    merged["qqq_close"] = qqq["Close"]
    merged["vix_close"] = vix["Close"].reindex(qqq.index).ffill()
    gross_ret = merged["qqq_close"].pct_change() + 1.0
    gross_ret = gross_ret.where(gross_ret > 0)
    merged["log_ret"] = np.log(gross_ret)
    merged = merged.dropna()
    merged.to_csv(outdir / "model_data.csv")

    print(f"Saved: {outdir / 'qqq.csv'}")
    print(f"Saved: {outdir / 'vix.csv'}")
    print(f"Saved: {outdir / 'model_data.csv'}")


if __name__ == "__main__":
    main()
