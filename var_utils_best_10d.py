from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from arch import arch_model
from scipy.stats import chi2


@dataclass
class ForecastResult:
    var: float
    meta: Dict[str, float]


def _safe_quantile(alpha: float) -> float:
    return float(np.clip(alpha, 1e-4, 0.2))


def _draw_empirical_standardized_residuals(
    std_resid: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    z = np.asarray(std_resid, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        raise ValueError("No valid standardized residuals available for resampling.")
    idx = rng.integers(0, len(z), size=size)
    return z[idx]


def _extract_garch_params(res) -> tuple[float, float, float, float]:
    params = res.params
    mu = float(params.get("mu", 0.0)) / 100.0
    omega = float(params.get("omega", np.nan)) / (100.0**2)
    alpha1 = float(params.get("alpha[1]", np.nan))
    beta1 = float(params.get("beta[1]", np.nan))
    return mu, omega, alpha1, beta1


def _extract_gjr_params(res) -> tuple[float, float, float, float, float]:
    params = res.params
    mu = float(params.get("mu", 0.0)) / 100.0
    omega = float(params.get("omega", np.nan)) / (100.0**2)
    alpha1 = float(params.get("alpha[1]", np.nan))
    gamma1 = float(params.get("gamma[1]", np.nan))
    beta1 = float(params.get("beta[1]", np.nan))
    return mu, omega, alpha1, gamma1, beta1


def _simulate_garch_empirical_paths(
    mu: float,
    omega: float,
    alpha1: float,
    beta1: float,
    sigma2_last: float,
    eps_last: float,
    std_resid: np.ndarray,
    horizon: int,
    n_sims: int,
    seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        sigma2_prev = float(max(sigma2_last, 1e-12))
        eps_prev = float(eps_last)
        total = 0.0
        z_path = _draw_empirical_standardized_residuals(std_resid, horizon, rng)

        for h in range(horizon):
            sigma2_next = omega + alpha1 * (eps_prev**2) + beta1 * sigma2_prev
            sigma2_next = float(max(sigma2_next, 1e-12))
            eps_next = float(np.sqrt(sigma2_next) * z_path[h])
            total += mu + eps_next
            sigma2_prev = sigma2_next
            eps_prev = eps_next

        out[i] = total

    return out


def _simulate_gjr_empirical_paths(
    mu: float,
    omega: float,
    alpha1: float,
    gamma1: float,
    beta1: float,
    sigma2_last: float,
    eps_last: float,
    std_resid: np.ndarray,
    horizon: int,
    n_sims: int,
    seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        sigma2_prev = float(max(sigma2_last, 1e-12))
        eps_prev = float(eps_last)
        total = 0.0
        z_path = _draw_empirical_standardized_residuals(std_resid, horizon, rng)

        for h in range(horizon):
            indicator = 1.0 if eps_prev < 0.0 else 0.0
            sigma2_next = (
                omega
                + alpha1 * (eps_prev**2)
                + gamma1 * (eps_prev**2) * indicator
                + beta1 * sigma2_prev
            )
            sigma2_next = float(max(sigma2_next, 1e-12))
            eps_next = float(np.sqrt(sigma2_next) * z_path[h])
            total += mu + eps_next
            sigma2_prev = sigma2_next
            eps_prev = eps_next

        out[i] = total

    return out


def forecast_historical_simulation_10d(
    returns: np.ndarray,
    alpha: float = 0.01,
    horizon: int = 10,
    window: int = 250,
) -> ForecastResult:
    """Historical Simulation for h-step cumulative log returns.

    Uses overlapping h-day sums formed from the most recent `window` daily returns.
    Since the pipeline works with log returns, cumulative h-day returns are additive.
    """
    r = np.asarray(returns, dtype=float)
    h = max(int(horizon), 1)
    if len(r) < h:
        raise ValueError(f"Need at least {h} returns for {h}-day HS VaR.")

    if len(r) < window:
        window = len(r)
    tail = r[-window:]
    if len(tail) < h:
        raise ValueError("Historical window is too short for the requested horizon.")

    cumulative = np.convolve(tail, np.ones(h, dtype=float), mode="valid")
    var = float(np.quantile(cumulative, _safe_quantile(alpha)))
    return ForecastResult(
        var=var,
        meta={
            "window": float(window),
            "horizon": float(h),
            "effective_paths": float(len(cumulative)),
        },
    )


def forecast_fhs_garch_10d(
    returns: np.ndarray,
    alpha: float = 0.01,
    horizon: int = 10,
    n_sims: int = 10000,
    seed: int | None = 42,
) -> ForecastResult:
    """Filtered Historical Simulation using GARCH(1,1) and empirical innovations.

    For horizon>1, this simulates the GARCH recursion forward and takes the empirical
    quantile of cumulative h-day returns.
    """
    y = np.asarray(returns, dtype=float) * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu, omega, alpha1, beta1 = _extract_garch_params(res)
    sigma2_path = (np.asarray(res.conditional_volatility, dtype=float) / 100.0) ** 2
    eps_path = np.asarray(res.resid, dtype=float) / 100.0
    sigma2_last = float(max(sigma2_path[-1], 1e-12))
    eps_last = float(eps_path[-1])

    std_resid = np.asarray(res.resid / res.conditional_volatility, dtype=float)
    std_resid = std_resid[np.isfinite(std_resid)]

    sim_returns = _simulate_garch_empirical_paths(
        mu=mu,
        omega=omega,
        alpha1=alpha1,
        beta1=beta1,
        sigma2_last=sigma2_last,
        eps_last=eps_last,
        std_resid=std_resid,
        horizon=max(int(horizon), 1),
        n_sims=n_sims,
        seed=seed,
    )
    var = float(np.quantile(sim_returns, _safe_quantile(alpha)))
    return ForecastResult(
        var=var,
        meta={
            "mu": mu,
            "horizon": float(max(int(horizon), 1)),
            "n_sims": float(n_sims),
            "omega": omega,
            "alpha1": alpha1,
            "beta1": beta1,
        },
    )


def forecast_fhs_gjr_10d(
    returns: np.ndarray,
    alpha: float = 0.01,
    horizon: int = 10,
    n_sims: int = 10000,
    seed: int | None = 42,
) -> ForecastResult:
    """Filtered Historical Simulation using GJR-GARCH(1,1,1) and empirical innovations."""
    y = np.asarray(returns, dtype=float) * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
    res = model.fit(disp="off")

    mu, omega, alpha1, gamma1, beta1 = _extract_gjr_params(res)
    sigma2_path = (np.asarray(res.conditional_volatility, dtype=float) / 100.0) ** 2
    eps_path = np.asarray(res.resid, dtype=float) / 100.0
    sigma2_last = float(max(sigma2_path[-1], 1e-12))
    eps_last = float(eps_path[-1])

    std_resid = np.asarray(res.resid / res.conditional_volatility, dtype=float)
    std_resid = std_resid[np.isfinite(std_resid)]

    sim_returns = _simulate_gjr_empirical_paths(
        mu=mu,
        omega=omega,
        alpha1=alpha1,
        gamma1=gamma1,
        beta1=beta1,
        sigma2_last=sigma2_last,
        eps_last=eps_last,
        std_resid=std_resid,
        horizon=max(int(horizon), 1),
        n_sims=n_sims,
        seed=seed,
    )
    var = float(np.quantile(sim_returns, _safe_quantile(alpha)))
    return ForecastResult(
        var=var,
        meta={
            "mu": mu,
            "horizon": float(max(int(horizon), 1)),
            "n_sims": float(n_sims),
            "omega": omega,
            "alpha1": alpha1,
            "gamma1": gamma1,
            "beta1": beta1,
        },
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
