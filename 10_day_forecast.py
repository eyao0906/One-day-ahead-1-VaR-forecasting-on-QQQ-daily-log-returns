from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2


# =====================================================================
# 1. 10-Day Forecasting Functions (Pure Normal vs Empirical Bootstrap)
# =====================================================================

def forecast_hs_10d(returns: np.ndarray, alpha: float = 0.01, window: int = 250) -> float:
    """Historical Simulation for 10-day VaR."""
    r = returns[-window:]
    if len(r) < 10:
        return np.nan
    
    roll_10d = np.convolve(r, np.ones(10), 'valid')
    return float(np.quantile(roll_10d, alpha))


# --- PARAMETRIC SIMULATION (Pure Normal Math) ---

def forecast_parametric_garch_10d(returns: np.ndarray, alpha: float = 0.01, n_paths: int = 10000) -> float:
    """Parametric Monte Carlo: Draws shocks from a perfect Standard Normal Distribution."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0))
    omega = float(res.params["omega"])
    alpha_coef = float(res.params["alpha[1]"])
    beta_coef = float(res.params["beta[1]"])

    sigma2_next = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])

    # THE DIFFERENCE: Draw from a purely theoretical Normal distribution
    np.random.seed(42)
    Z = np.random.standard_normal(size=(10, n_paths))
    
    sigma2 = np.full(n_paths, sigma2_next)
    path_returns = np.zeros((10, n_paths))

    for step in range(10):
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        step_r = mu + sigma * Z[step]
        path_returns[step] = step_r
        eps = step_r - mu
        if step < 9:
            sigma2 = omega + alpha_coef * (eps**2) + beta_coef * sigma2

    cum_returns = np.sum(path_returns, axis=0)
    return float(np.quantile(cum_returns, alpha)) / 100.0


def forecast_parametric_gjr_10d(returns: np.ndarray, alpha: float = 0.01, n_paths: int = 10000) -> float:
    """Parametric Monte Carlo: GJR-GARCH with perfect Standard Normal shocks."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0))
    omega = float(res.params["omega"])
    alpha_coef = float(res.params["alpha[1]"])
    gamma_coef = float(res.params["gamma[1]"])
    beta_coef = float(res.params["beta[1]"])

    sigma2_next = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])

    # THE DIFFERENCE: Draw from a purely theoretical Normal distribution
    np.random.seed(42)
    Z = np.random.standard_normal(size=(10, n_paths))
    
    sigma2 = np.full(n_paths, sigma2_next)
    path_returns = np.zeros((10, n_paths))

    for step in range(10):
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        step_r = mu + sigma * Z[step]
        path_returns[step] = step_r
        eps = step_r - mu
        if step < 9:
            asym = gamma_coef * (eps**2) * (eps < 0)
            sigma2 = omega + alpha_coef * (eps**2) + asym + beta_coef * sigma2

    cum_returns = np.sum(path_returns, axis=0)
    return float(np.quantile(cum_returns, alpha)) / 100.0


# --- EMPIRICAL BOOTSTRAP (FHS with Normal Engine) ---
def forecast_bootstrap_garch_10d(returns: np.ndarray, alpha: float = 0.01, n_paths: int = 10000) -> float:
    """Empirical Bootstrap: Fits Normal GARCH, but draws shocks from the actual data."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0))
    omega = float(res.params["omega"])
    alpha_coef = float(res.params["alpha[1]"])
    beta_coef = float(res.params["beta[1]"])

    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]
    sigma2_next = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])

    np.random.seed(42)
    Z = np.random.choice(std_resid, size=(10, n_paths), replace=True)
    
    sigma2 = np.full(n_paths, sigma2_next)
    path_returns = np.zeros((10, n_paths))

    for step in range(10):
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        step_r = mu + sigma * Z[step]
        path_returns[step] = step_r
        eps = step_r - mu
        if step < 9:
            sigma2 = omega + alpha_coef * (eps**2) + beta_coef * sigma2

    cum_returns = np.sum(path_returns, axis=0)
    return float(np.quantile(cum_returns, alpha)) / 100.0


def forecast_bootstrap_gjr_10d(returns: np.ndarray, alpha: float = 0.01, n_paths: int = 10000) -> float:
    """Empirical Bootstrap: Fits Normal GJR-GARCH, but draws shocks from the actual data."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0))
    omega = float(res.params["omega"])
    alpha_coef = float(res.params["alpha[1]"])
    gamma_coef = float(res.params["gamma[1]"])
    beta_coef = float(res.params["beta[1]"])

    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]
    sigma2_next = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])

    np.random.seed(42)
    Z = np.random.choice(std_resid, size=(10, n_paths), replace=True)
    
    sigma2 = np.full(n_paths, sigma2_next)
    path_returns = np.zeros((10, n_paths))

    for step in range(10):
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        step_r = mu + sigma * Z[step]
        path_returns[step] = step_r
        eps = step_r - mu
        if step < 9:
            asym = gamma_coef * (eps**2) * (eps < 0)
            sigma2 = omega + alpha_coef * (eps**2) + asym + beta_coef * sigma2

    cum_returns = np.sum(path_returns, axis=0)
    # The line below must be included for the VaR threshold to calculate correctly!
    return float(np.quantile(cum_returns, alpha)) / 100.0

# =====================================================================
# 2. Backtesting Stats
# =====================================================================

def kupiec_test(violations: np.ndarray, alpha: float) -> Tuple[float, float]:
    v = np.asarray(violations).astype(int)
    n = len(v)
    x = int(v.sum())
    pi_hat = np.clip(x / max(n, 1), 1e-8, 1 - 1e-8)
    a = np.clip(alpha, 1e-8, 1 - 1e-8)
    lr = -2.0 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - pi_hat) - x * np.log(pi_hat))
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def christoffersen_independence_test(violations: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(violations).astype(int)
    if len(v) < 2: return float("nan"), float("nan")
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
    return float(lr), float(1 - chi2.cdf(lr, df=1))


# =====================================================================
# 3. Main Workflow & Execution
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="10-Day VaR Forecasting and Plotting")
    parser.add_argument("--data", default="data/model_data.csv")
    parser.add_argument("--outdir", default="outputs/10_day_var_normal")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--initial-train", type=float, default=0.7)
    parser.add_argument("--hs-window", type=int, default=250)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["Date"]).dropna().sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end0 = int(n * args.initial_train)
    
    rows = []
    print(f"Starting 10-day VaR rolling forecast (Total evaluated days: {n - 9 - train_end0})...")
    
    for t in range(train_end0, n - 9):
        train = df.iloc[:t]
        r_train = train["log_ret"].to_numpy()
        realized_10d = df["log_ret"].iloc[t : t+10].sum()
        target_date = df["Date"].iloc[t]

        models = {
            "Historical-Sim (10d)": lambda: forecast_hs_10d(r_train, alpha=args.alpha, window=args.hs_window),
            "GARCH-Normal-Parametric (10d)": lambda: forecast_parametric_garch_10d(r_train, alpha=args.alpha),
            "GJR-Normal-Parametric (10d)": lambda: forecast_parametric_gjr_10d(r_train, alpha=args.alpha),
            "GARCH-Normal-Bootstrap (10d)": lambda: forecast_bootstrap_garch_10d(r_train, alpha=args.alpha),
            "GJR-Normal-Bootstrap (10d)": lambda: forecast_bootstrap_gjr_10d(r_train, alpha=args.alpha)
        }

        for name, func in models.items():
            try:
                var_10d = func()
                hit = int(realized_10d < var_10d)
                rows.append({
                    "Date": target_date,
                    "Model": name,
                    "VaR_10d": var_10d,
                    "Realized_10d": realized_10d,
                    "Violation": hit
                })
            except Exception as e:
                print(f"[warn] step {t}, model {name} failed: {e}")
                
        if t % 50 == 0:
            print(f"Processed step {t} / {n-9}")

    forecasts = pd.DataFrame(rows)
    forecasts.to_csv(outdir / "var_10d_forecasts.csv", index=False)

    records = []
    for model, g in forecasts.groupby("Model"):
        v = g["Violation"].to_numpy()
        lr_uc, p_uc = kupiec_test(v, args.alpha)
        lr_ind, p_ind = christoffersen_independence_test(v)
        records.append({
            "Model": model, "Violations": int(v.sum()), "ViolationRate": float(v.mean()),
            "Kupiec_pvalue": p_uc, "Christoffersen_pvalue": p_ind
        })
    
    summary = pd.DataFrame(records)
    summary.to_csv(outdir / "var_10d_backtests.csv", index=False)
    print("\n--- COMBINED 10-DAY BACKTEST SUMMARY ---")
    print(summary.to_string(index=False))

    for model, g in forecasts.groupby("Model"):
        g = g.sort_values("Date")
        plt.figure(figsize=(12, 4))
        plt.plot(g["Date"], g["Realized_10d"], label="Realized 10-Day Return", linewidth=1, color="black", alpha=0.7)
        plt.plot(g["Date"], g["VaR_10d"], label=f"10-Day VaR ({model})", linewidth=1.5, color="red")
        
        viol = g[g["Violation"] == 1]
        if not viol.empty:
            plt.scatter(viol["Date"], viol["Realized_10d"], s=20, color="blue", label="Violations", zorder=5)
            
        plt.title(f"10-Day VaR Forecast vs Realized: {model}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = model.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(outdir / f"plot_10d_{safe_name}.png", dpi=150)
        plt.close()

    print(f"\nAll 10-day forecasts and plots saved to: {outdir}")

if __name__ == "__main__":
    main()