from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from arch import arch_model
from scipy.stats import chi2, norm


@dataclass
class ForecastResult:
    var: float
    meta: Dict[str, float]


def _safe_quantile(alpha: float) -> float:
    return float(np.clip(alpha, 1e-4, 0.2))


def forecast_garch_gaussian(returns: np.ndarray, alpha: float = 0.01) -> ForecastResult:
    """Standard GARCH(1,1) with Normal distribution."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma = float(np.sqrt(max(sigma2, 1e-12)))

    q = float(norm.ppf(_safe_quantile(alpha)))
    var = mu + sigma * q
    return ForecastResult(var=var, meta={"mu": mu, "sigma": sigma})


def forecast_fhs_garch(returns: np.ndarray, alpha: float = 0.01) -> ForecastResult:
    """Filtered Historical Simulation using GARCH(1,1)."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma_next = float(np.sqrt(max(sigma2, 1e-12)))

    # FIX: Use numpy isnan mask instead of pandas dropna
    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]
    
    # Empirical quantile
    q_emp = float(np.quantile(std_resid, _safe_quantile(alpha)))

    var = mu + sigma_next * q_emp
    return ForecastResult(var=var, meta={"mu": mu, "sigma": sigma_next, "q_emp": q_emp})


def forecast_fhs_gjr(returns: np.ndarray, alpha: float = 0.01) -> ForecastResult:
    """Filtered Historical Simulation using GJR-GARCH(1,1,1)."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma_next = float(np.sqrt(max(sigma2, 1e-12)))

    # FIX: Use numpy isnan mask instead of pandas dropna
    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]
    
    # Empirical quantile
    q_emp = float(np.quantile(std_resid, _safe_quantile(alpha)))

    var = mu + sigma_next * q_emp
    return ForecastResult(var=var, meta={"mu": mu, "sigma": sigma_next, "q_emp": q_emp})


def forecast_bootstrap_garch(returns: np.ndarray, alpha: float = 0.01, n_boot: int = 10000) -> ForecastResult:
    """Residual Bootstrap GARCH for 1-step ahead VaR."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma_next = float(np.sqrt(max(sigma2, 1e-12)))

    # FIX: Use numpy isnan mask instead of pandas dropna
    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]
    
    # Resample standardized residuals
    np.random.seed(42) # Optional: ensures reproducible bootstrap runs
    boot_z = np.random.choice(std_resid, size=n_boot, replace=True)
    
    # Simulate 1-step returns
    boot_returns = mu + sigma_next * boot_z
    
    # VaR is the empirical quantile of the simulated returns
    var = float(np.quantile(boot_returns, _safe_quantile(alpha)))
    return ForecastResult(var=var, meta={"mu": mu, "sigma": sigma_next})


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
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    p01 = np.clip(p01, 1e-8, 1 - 1e-8)
    p11 = np.clip(p11, 1e-8, 1 - 1e-8)
    p = np.clip(p, 1e-8, 1 - 1e-8)

    ll_ind = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    ll_dep = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)

    lr = -2.0 * (ll_ind - ll_dep)
    pval = float(1 - chi2.cdf(lr, df=1))
    return float(lr), pval