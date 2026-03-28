from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


# =====================================================================
# 1. RiskMetrics Forecasting Function
# =====================================================================

def forecast_riskmetrics(returns: np.ndarray, alpha: float = 0.01, lambd: float = 0.94) -> float:
    """
    RiskMetrics EWMA Variance Forecast (Longerstaey, 1996).
    
    The recursion is: sigma^2_{t} = lambda * sigma^2_{t-1} + (1 - lambda) * r_{t-1}^2
    Initialized using the sample variance of the first 250 observations.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    
    # Initialize with the variance estimate from the first 250 observations
    init_window = min(n, 250)
    
    # Standard RiskMetrics assumes a mean of 0 for daily returns
    sigma2 = np.var(r[:init_window])
    
    # Apply the RiskMetrics recursion through the rest of the training set
    for i in range(init_window, n):
        sigma2 = lambd * sigma2 + (1.0 - lambd) * (r[i] ** 2)
        
    # The loop finishes having incorporated r[n-1], so sigma2 is the forecast for t+1
    sigma_next = np.sqrt(max(sigma2, 1e-12))
    
    # Calculate 1-day ahead VaR using the Normal Distribution
    q = float(norm.ppf(np.clip(alpha, 1e-4, 0.2)))
    
    # While RiskMetrics assumes mu=0 for variance, we apply the empirical mean 
    # to the final VaR threshold for consistency with our other models.
    mu = float(np.mean(r)) 
    
    return mu + sigma_next * q


# =====================================================================
# 2. Backtesting Functions
# =====================================================================

def kupiec_test(violations: np.ndarray, alpha: float) -> Tuple[float, float]:
    v = np.asarray(violations).astype(int)
    n = len(v)
    x = int(v.sum())
    pi_hat = np.clip(x / max(n, 1), 1e-8, 1 - 1e-8)
    a = np.clip(alpha, 1e-8, 1 - 1e-8)

    lr = -2.0 * (
        (n - x) * np.log(1 - a)
        + x * np.log(a)
        - (n - x) * np.log(1 - pi_hat)
        - x * np.log(pi_hat)
    )
    pval = float(1 - chi2.cdf(lr, df=1))
    return float(lr), pval


def christoffersen_independence_test(violations: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(violations).astype(int)
    if len(v) < 2:
        return float("nan"), float("nan")

    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(v)):
        prev, cur = v[i - 1], v[i]
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1

    p01 = np.clip(n01 / max(n00 + n01, 1), 1e-8, 1 - 1e-8)
    p11 = np.clip(n11 / max(n10 + n11, 1), 1e-8, 1 - 1e-8)
    p = np.clip((n01 + n11) / max(n00 + n01 + n10 + n11, 1), 1e-8, 1 - 1e-8)

    ll_ind = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    ll_dep = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)

    lr = -2.0 * (ll_ind - ll_dep)
    pval = float(1 - chi2.cdf(lr, df=1))
    return float(lr), pval


# =====================================================================
# 3. Main Workflow & Execution
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RiskMetrics 1-Day VaR Forecasting")
    parser.add_argument("--data", default="data/model_data.csv")
    parser.add_argument("--outdir", default="outputs/riskmetrics")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--initial-train", type=float, default=0.7)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load Data
    df = pd.read_csv(args.data, parse_dates=["Date"])
    df = df[["Date", "log_ret"]].dropna().sort_values("Date").reset_index(drop=True)
    
    n = len(df)
    train_end0 = int(n * args.initial_train)
    if train_end0 < 500:
        train_end0 = min(max(500, train_end0), n - 30)

    rows = []
    
    print(f"Starting RiskMetrics rolling forecast (Total evaluated days: {n - train_end0})...")
    
    # Rolling Walk-Forward
    for t in range(train_end0, n):
        train = df.iloc[:t]
        test_row = df.iloc[t]

        r_train = train["log_ret"].to_numpy()
        realized = float(test_row["log_ret"])
        target_date = test_row["Date"]

        try:
            # Generate the RiskMetrics forecast
            var = forecast_riskmetrics(r_train, alpha=args.alpha, lambd=0.94)
            hit = int(realized < var)

            rows.append({
                "Date": target_date,
                "Model": "RiskMetrics (lambda=0.94)",
                "alpha": args.alpha,
                "VaR": var,
                "Return": realized,
                "Violation": hit,
            })
            
        except Exception as e:
            print(f"[warn] step {t}, model RiskMetrics failed: {e}")
            
        if t % 100 == 0:
            print(f"Processed step {t} / {n}")

    forecasts = pd.DataFrame(rows)
    
    if forecasts.empty:
        raise RuntimeError("No forecasts produced. Check data format.")
        
    forecasts.to_csv(outdir / "var_riskmetrics_forecasts.csv", index=False)

    # Summarize Backtests
    v = forecasts["Violation"].to_numpy()
    lr_uc, p_uc = kupiec_test(v, args.alpha)
    lr_ind, p_ind = christoffersen_independence_test(v)
    lr_cc = lr_uc + lr_ind
    p_cc = float(1 - chi2.cdf(lr_cc, df=2))
    
    summary = pd.DataFrame([{
        "Model": "RiskMetrics (lambda=0.94)",
        "N": len(forecasts),
        "Violations": int(v.sum()),
        "ExpectedViolations": float(args.alpha * len(forecasts)),
        "ViolationRate": float(v.mean()),
        "Kupiec_LRuc": lr_uc,
        "Kupiec_pvalue": p_uc,
        "Christoffersen_LRind": lr_ind,
        "Christoffersen_pvalue": p_ind,
        "ConditionalCoverage_LRcc": lr_cc,
        "ConditionalCoverage_pvalue": p_cc,
    }])
    
    summary.to_csv(outdir / "var_riskmetrics_backtests.csv", index=False)
    
    print("\n--- RISKMETRICS BACKTEST SUMMARY ---")
    print(summary.to_string(index=False))

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(forecasts["Date"], forecasts["Return"], label="Realized Return", linewidth=1, color="black", alpha=0.7)
    plt.plot(forecasts["Date"], forecasts["VaR"], label="1-Day VaR (RiskMetrics)", linewidth=1.5, color="red")
    
    viol = forecasts[forecasts["Violation"] == 1]
    if not viol.empty:
        plt.scatter(viol["Date"], viol["Return"], s=20, color="blue", label="Violations", zorder=5)
        
    plt.title("VaR Backtest: RiskMetrics ($\lambda=0.94$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "plot_var_riskmetrics.png", dpi=150)
    plt.close()

    print(f"\nAll RiskMetrics forecasts and plots saved to: {outdir}")

if __name__ == "__main__":
    main()