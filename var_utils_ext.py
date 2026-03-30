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


def _z_critical(confidence: float = 0.95) -> float:
    return float(norm.ppf(0.5 + confidence / 2.0))


def _quantile_density_from_sample(sample: np.ndarray, alpha: float) -> float:
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float("nan")

    a = _safe_quantile(alpha)
    eps = max(2.0 / n, 0.0025)
    lo = max(1e-4, a - eps)
    hi = min(0.9999, a + eps)
    if hi <= lo:
        hi = min(0.9999, lo + max(1.0 / n, 1e-4))

    q_lo = float(np.quantile(x, lo))
    q_hi = float(np.quantile(x, hi))
    width = max(q_hi - q_lo, 1e-8)
    return float((hi - lo) / width)


def _sample_quantile_ci(sample: np.ndarray, alpha: float, point_quantile: float | None = None, confidence: float = 0.95) -> Tuple[float, float]:
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("Cannot build a quantile CI from an empty sample.")

    a = _safe_quantile(alpha)
    q_hat = float(np.quantile(x, a)) if point_quantile is None else float(point_quantile)
    f_hat = _quantile_density_from_sample(x, a)
    if not np.isfinite(f_hat) or f_hat <= 0.0:
        return q_hat, q_hat

    se = np.sqrt(a * (1.0 - a) / (len(x) * (f_hat ** 2)))
    z = _z_critical(confidence)
    return float(q_hat - z * se), float(q_hat + z * se)


def _parametric_quantile_ci(mu: float, sigma: float, alpha: float, n_eff: int, confidence: float = 0.95) -> Tuple[float, float]:
    a = _safe_quantile(alpha)
    if n_eff <= 0:
        return mu, mu

    q = float(norm.ppf(a))
    f_q = max(float(norm.pdf(q)), 1e-8)
    se_q = np.sqrt(a * (1.0 - a) / (n_eff * (f_q ** 2)))
    z = _z_critical(confidence)
    return float(mu + sigma * (q - z * se_q)), float(mu + sigma * (q + z * se_q))


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
    lower_bound, upper_bound = _parametric_quantile_ci(mu, sigma, alpha, n_eff=len(y))
    return ForecastResult(
        var=var,
        meta={"mu": mu, "sigma": sigma, "lower_bound": lower_bound, "upper_bound": upper_bound},
    )


def forecast_fhs_garch(returns: np.ndarray, alpha: float = 0.01) -> ForecastResult:
    """Filtered Historical Simulation using GARCH(1,1)."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma_next = float(np.sqrt(max(sigma2, 1e-12)))

    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]

    q_emp = float(np.quantile(std_resid, _safe_quantile(alpha)))
    q_lo, q_hi = _sample_quantile_ci(std_resid, alpha, point_quantile=q_emp)

    var = mu + sigma_next * q_emp
    return ForecastResult(
        var=var,
        meta={
            "mu": mu,
            "sigma": sigma_next,
            "q_emp": q_emp,
            "lower_bound": float(mu + sigma_next * q_lo),
            "upper_bound": float(mu + sigma_next * q_hi),
        },
    )


def forecast_fhs_gjr(returns: np.ndarray, alpha: float = 0.01) -> ForecastResult:
    """Filtered Historical Simulation using GJR-GARCH(1,1,1)."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma_next = float(np.sqrt(max(sigma2, 1e-12)))

    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]

    q_emp = float(np.quantile(std_resid, _safe_quantile(alpha)))
    q_lo, q_hi = _sample_quantile_ci(std_resid, alpha, point_quantile=q_emp)

    var = mu + sigma_next * q_emp
    return ForecastResult(
        var=var,
        meta={
            "mu": mu,
            "sigma": sigma_next,
            "q_emp": q_emp,
            "lower_bound": float(mu + sigma_next * q_lo),
            "upper_bound": float(mu + sigma_next * q_hi),
        },
    )


def forecast_bootstrap_garch(returns: np.ndarray, alpha: float = 0.01, n_boot: int = 10000) -> ForecastResult:
    """Residual Bootstrap GARCH for 1-step ahead VaR."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma_next = float(np.sqrt(max(sigma2, 1e-12)))

    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]

    np.random.seed(42)  # Optional: ensures reproducible bootstrap runs
    boot_z = np.random.choice(std_resid, size=n_boot, replace=True)

    boot_returns = mu + sigma_next * boot_z

    var = float(np.quantile(boot_returns, _safe_quantile(alpha)))
    lower_bound, upper_bound = _sample_quantile_ci(boot_returns, alpha, point_quantile=var)
    return ForecastResult(
        var=var,
        meta={"mu": mu, "sigma": sigma_next, "lower_bound": lower_bound, "upper_bound": upper_bound},
    )


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
