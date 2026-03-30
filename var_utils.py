from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import chi2, norm, t

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


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


def _standardized_t_pdf(z: float, nu: float) -> float:
    scale = np.sqrt((nu - 2.0) / nu)
    return float(t.pdf(z / scale, df=nu) / scale)


def _parametric_quantile_ci(mu: float, sigma: float, alpha: float, n_eff: int, dist: str = "normal", nu: float | None = None, confidence: float = 0.95) -> Tuple[float, float]:
    a = _safe_quantile(alpha)
    if n_eff <= 0:
        return mu, mu

    if dist == "t":
        nu = float(max(nu if nu is not None and np.isfinite(nu) else 8.0, 2.2))
        q = float(t.ppf(a, df=nu)) * np.sqrt((nu - 2.0) / nu)
        f_q = _standardized_t_pdf(q, nu)
    else:
        q = float(norm.ppf(a))
        f_q = float(norm.pdf(q))

    f_q = max(f_q, 1e-8)
    se_q = np.sqrt(a * (1.0 - a) / (n_eff * (f_q ** 2)))
    z = _z_critical(confidence)
    return float(mu + sigma * (q - z * se_q)), float(mu + sigma * (q + z * se_q))


def _centered_sample_ci(target_quantile: float, sample: np.ndarray, alpha: float, confidence: float = 0.95) -> Tuple[float, float]:
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return target_quantile, target_quantile

    a = _safe_quantile(alpha)
    base_q = float(np.quantile(x, a))
    shifted = x + (target_quantile - base_q)
    return _sample_quantile_ci(shifted, alpha=a, point_quantile=target_quantile, confidence=confidence)


def forecast_garch(returns: np.ndarray, alpha: float = 0.01, dist: str = "t") -> ForecastResult:
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist=("StudentsT" if dist == "t" else "normal"))
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma = float(np.sqrt(max(sigma2, 1e-12)))

    if dist == "t" and "nu" in res.params.index:
        nu = float(res.params["nu"])
        q = float(t.ppf(_safe_quantile(alpha), df=max(nu, 2.2)))
        # standardize t to variance 1
        q *= np.sqrt((nu - 2.0) / nu)
        lower_bound, upper_bound = _parametric_quantile_ci(mu, sigma, alpha, n_eff=len(y), dist="t", nu=nu)
    else:
        nu = np.nan
        q = float(norm.ppf(_safe_quantile(alpha)))
        lower_bound, upper_bound = _parametric_quantile_ci(mu, sigma, alpha, n_eff=len(y), dist="normal")

    var = mu + sigma * q
    return ForecastResult(
        var=var,
        meta={"mu": mu, "sigma": sigma, "nu": nu, "lower_bound": lower_bound, "upper_bound": upper_bound},
    )


def forecast_fhs_garch_t(returns: np.ndarray, alpha: float = 0.01) -> ForecastResult:
    """Filtered Historical Simulation using Student-t GARCH(1,1)."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="StudentsT")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma_next = float(np.sqrt(max(sigma2, 1e-12)))

    # Compute standardized residuals and remove initial NaNs
    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]

    # Empirical quantile of the t-standardized residuals
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


def forecast_bootstrap_garch_t(returns: np.ndarray, alpha: float = 0.01, n_boot: int = 10000) -> ForecastResult:
    """Residual Bootstrap using Student-t GARCH(1,1) for 1-step ahead VaR."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="StudentsT")
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma_next = float(np.sqrt(max(sigma2, 1e-12)))

    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]

    # Resample standardized residuals
    np.random.seed(42)  # Ensures reproducibility
    boot_z = np.random.choice(std_resid, size=n_boot, replace=True)

    # Simulate 1-step returns
    boot_returns = mu + sigma_next * boot_z

    var = float(np.quantile(boot_returns, _safe_quantile(alpha)))
    lower_bound, upper_bound = _sample_quantile_ci(boot_returns, alpha, point_quantile=var)
    return ForecastResult(
        var=var,
        meta={"mu": mu, "sigma": sigma_next, "lower_bound": lower_bound, "upper_bound": upper_bound},
    )


def forecast_gjr_garch(returns: np.ndarray, alpha: float = 0.01, dist: str = "t") -> ForecastResult:
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=("StudentsT" if dist == "t" else "normal"))
    res = model.fit(disp="off")

    mu = float(res.params.get("mu", 0.0)) / 100.0
    sigma2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / (100.0**2)
    sigma = float(np.sqrt(max(sigma2, 1e-12)))

    if dist == "t" and "nu" in res.params.index:
        nu = float(res.params["nu"])
        q = float(t.ppf(_safe_quantile(alpha), df=max(nu, 2.2)))
        q *= np.sqrt((nu - 2.0) / nu)
        lower_bound, upper_bound = _parametric_quantile_ci(mu, sigma, alpha, n_eff=len(y), dist="t", nu=nu)
    else:
        nu = np.nan
        q = float(norm.ppf(_safe_quantile(alpha)))
        lower_bound, upper_bound = _parametric_quantile_ci(mu, sigma, alpha, n_eff=len(y), dist="normal")

    var = mu + sigma * q
    return ForecastResult(
        var=var,
        meta={"mu": mu, "sigma": sigma, "nu": nu, "lower_bound": lower_bound, "upper_bound": upper_bound},
    )


def forecast_fhs_gjr_t(returns: np.ndarray, alpha: float = 0.01) -> ForecastResult:
    """Filtered Historical Simulation using Student-t GJR-GARCH(1,1,1)."""
    y = returns * 100.0
    model = arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="StudentsT")
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


def forecast_historical_simulation(returns: np.ndarray, alpha: float = 0.01, window: int = 250) -> ForecastResult:
    r = np.asarray(returns)
    if len(r) < window:
        window = len(r)
    tail_sample = np.asarray(r[-window:], dtype=float)
    var = float(np.quantile(tail_sample, _safe_quantile(alpha)))
    lower_bound, upper_bound = _sample_quantile_ci(tail_sample, alpha, point_quantile=var)
    return ForecastResult(
        var=var,
        meta={"window": float(window), "lower_bound": lower_bound, "upper_bound": upper_bound},
    )


@njit(cache=True)
def _caviar_sav_loss(params: np.ndarray, r_scaled: np.ndarray, alpha: float) -> float:
    """
    CAViaR-SAV estimated on the positive VaR magnitude v_t, then mapped back
    to the return quantile q_t = -v_t. This aligns the recursion with the
    original Engle-Manganelli SAV specification, where VaR is updated as a
    positive risk measure.
    """
    b0, b1, b2 = params[0], params[1], params[2]

    if b0 < 0.0 or b1 < 0.0 or b1 >= 0.999 or b2 < 0.0:
        return 1e12

    n = len(r_scaled)
    v_prev = max(1e-6, -np.quantile(r_scaled, alpha))
    loss = 0.0

    q = -v_prev
    u = r_scaled[0] - q
    loss += u * (alpha - (1.0 if u < 0.0 else 0.0))

    for t in range(1, n):
        v_t = b0 + b1 * v_prev + b2 * abs(r_scaled[t - 1])
        if (not np.isfinite(v_t)) or v_t <= 0.0:
            return 1e12

        q = -v_t
        u = r_scaled[t] - q
        loss += u * (alpha - (1.0 if u < 0.0 else 0.0))
        v_prev = v_t

    return loss


def _caviar_loss_objective(params: np.ndarray, r_scaled: np.ndarray, alpha: float) -> float:
    return float(_caviar_sav_loss(np.asarray(params, dtype=float), r_scaled, float(_safe_quantile(alpha))))


def _fit_caviar_sav(r_scaled: np.ndarray, alpha: float) -> np.ndarray:
    emp_var = max(0.1, -float(np.quantile(r_scaled, _safe_quantile(alpha))))
    starts = [
        np.array([0.05 * emp_var, 0.90, 0.10]),
        np.array([0.10 * emp_var, 0.95, 0.05]),
        np.array([0.20 * emp_var, 0.85, 0.15]),
        np.array([0.01, 0.99, 0.01]),
    ]

    best = None
    for s in starts:
        res = minimize(
            _caviar_loss_objective,
            s,
            args=(r_scaled, alpha),
            method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4, "disp": False},
        )
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("CAViaR-SAV optimization failed to produce a candidate.")

    return np.asarray(best.x, dtype=float)


def _caviar_sav_next_quantile(params: np.ndarray, r_scaled: np.ndarray, alpha: float) -> float:
    b0, b1, b2 = params
    v_prev = max(1e-6, -float(np.quantile(r_scaled, _safe_quantile(alpha))))

    for t in range(1, len(r_scaled)):
        v_prev = b0 + b1 * v_prev + b2 * abs(r_scaled[t - 1])

    v_next = b0 + b1 * v_prev + b2 * abs(r_scaled[-1])
    return -float(v_next)


def forecast_caviar_sav(returns: np.ndarray, alpha: float = 0.01) -> ForecastResult:
    r_scaled = np.asarray(returns, dtype=float) * 100.0
    if len(r_scaled) < 30:
        raise ValueError("CAViaR-SAV needs at least 30 observations.")

    params = _fit_caviar_sav(r_scaled, alpha)
    q_next_scaled = _caviar_sav_next_quantile(params, r_scaled, alpha)

    proxy_window = min(250, len(r_scaled))
    lower_scaled, upper_scaled = _centered_sample_ci(q_next_scaled, r_scaled[-proxy_window:], alpha)

    return ForecastResult(
        var=q_next_scaled / 100.0,
        meta={
            "b0": float(params[0]),
            "b1": float(params[1]),
            "b2": float(params[2]),
            "var_level": -q_next_scaled / 100.0,
            "lower_bound": float(lower_scaled / 100.0),
            "upper_bound": float(upper_scaled / 100.0),
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
