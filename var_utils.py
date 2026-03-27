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
    else:
        nu = np.nan
        q = float(norm.ppf(_safe_quantile(alpha)))

    var = mu + sigma * q
    return ForecastResult(var=var, meta={"mu": mu, "sigma": sigma, "nu": nu})

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

    var = mu + sigma_next * q_emp
    return ForecastResult(var=var, meta={"mu": mu, "sigma": sigma_next, "q_emp": q_emp})

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
    np.random.seed(42) # Ensures reproducibility
    boot_z = np.random.choice(std_resid, size=n_boot, replace=True)
    
    # Simulate 1-step returns
    boot_returns = mu + sigma_next * boot_z
    
    var = float(np.quantile(boot_returns, _safe_quantile(alpha)))
    return ForecastResult(var=var, meta={"mu": mu, "sigma": sigma_next})

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
    else:
        nu = np.nan
        q = float(norm.ppf(_safe_quantile(alpha)))

    var = mu + sigma * q
    return ForecastResult(var=var, meta={"mu": mu, "sigma": sigma, "nu": nu})

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

    var = mu + sigma_next * q_emp
    return ForecastResult(var=var, meta={"mu": mu, "sigma": sigma_next, "q_emp": q_emp})

def _prepare_garchx_regressor(x: np.ndarray) -> np.ndarray:
    """Prepare a strictly positive VIX-based regressor for the variance equation.

    To keep the mechanics aligned with the GARCHX note, the exogenous term enters
    the variance recursion through a squared regressor. We standardize VIX first so
    the squared term is numerically stable, then use x_t^2 in the recursion.
    """
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)):
        raise ValueError("Exogenous regressor contains non-finite values.")
    x = np.clip(x, 1e-8, None)
    z = (x - np.mean(x)) / max(np.std(x), 1e-8)
    x_pos = 1.0 + z
    x_pos = np.clip(x_pos, 1e-4, None)
    return x_pos


def _garchx_negloglik(theta: np.ndarray, r: np.ndarray, x_sq: np.ndarray) -> float:
    """Gaussian quasi-log-likelihood for GARCHX(1,1) with x_{t-1}^2 in variance.

    sigma_t^2 = omega + a * eps_{t-1}^2 + b * sigma_{t-1}^2 + g * x_{t-1}^2
    """
    omega, a, b, g, mu = theta

    if (
        not np.all(np.isfinite(theta))
        or omega <= 0.0
        or a < 0.0
        or b < 0.0
        or g < 0.0
        or (a + b) >= 0.999
    ):
        return 1e12

    eps = np.asarray(r, dtype=float) - mu
    x_sq = np.asarray(x_sq, dtype=float)
    n = len(eps)
    sigma2 = np.empty(n, dtype=float)
    unc = omega / max(1.0 - a - b, 1e-6)
    sigma2[0] = max(np.var(eps), unc, 1e-8)

    for i in range(1, n):
        sigma2[i] = omega + a * eps[i - 1] ** 2 + b * sigma2[i - 1] + g * x_sq[i - 1]
        if sigma2[i] <= 0.0 or not np.isfinite(sigma2[i]):
            return 1e12

    ll = -0.5 * np.sum(np.log(sigma2) + (eps**2) / sigma2)
    return 1e12 if not np.isfinite(ll) else -float(ll)


def _compute_garchx_path(theta: np.ndarray, r: np.ndarray, x_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    omega, a, b, g, mu = theta
    eps = np.asarray(r, dtype=float) - mu
    x_sq = np.asarray(x_sq, dtype=float)

    sigma2 = np.empty(len(r), dtype=float)
    unc = omega / max(1.0 - a - b, 1e-6)
    sigma2[0] = max(np.var(eps), unc, 1e-8)

    for i in range(1, len(r)):
        sigma2[i] = omega + a * eps[i - 1] ** 2 + b * sigma2[i - 1] + g * x_sq[i - 1]
        sigma2[i] = max(sigma2[i], 1e-10)

    return eps, sigma2


def forecast_garchx(
    returns: np.ndarray,
    x: np.ndarray,
    alpha: float = 0.01,
    fallback_to_garch: bool = True,
) -> ForecastResult:
    r_raw = np.asarray(returns, dtype=float)
    r = r_raw * 100.0
    x_prepared = _prepare_garchx_regressor(x)
    x_sq = x_prepared**2

    if len(r) != len(x_sq):
        raise ValueError("returns and x must have the same length")

    var_r = max(float(np.var(r)), 1e-4)
    x0 = np.array([
        0.02 * var_r,
        0.05,
        0.90,
        0.02 * var_r / max(float(np.mean(x_sq)), 1e-6),
        float(np.mean(r)),
    ])

    bounds = [
        (1e-8, 10.0 * var_r),
        (1e-8, 0.5),
        (1e-8, 0.999),
        (0.0, 10.0 * var_r),
        (-5.0, 5.0),
    ]

    opt = minimize(
        _garchx_negloglik,
        x0,
        args=(r, x_sq),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not opt.success:
        if fallback_to_garch:
            fc = forecast_garch(r_raw, alpha=alpha, dist="t")
            fc.meta["fallback"] = 1.0
            return fc
        raise RuntimeError(f"GARCHX optimization failed: {opt.message}")

    theta = np.asarray(opt.x, dtype=float)
    omega, a, b, g, mu = theta
    eps, sigma2 = _compute_garchx_path(theta, r, x_sq)
    sigma2_next = omega + a * eps[-1] ** 2 + b * sigma2[-1] + g * x_sq[-1]
    sigma2_next = max(float(sigma2_next), 1e-12)
    sigma_next = float(np.sqrt(sigma2_next))

    q = float(norm.ppf(_safe_quantile(alpha)))
    var_scaled = float(mu + sigma_next * q)
    var_unscaled = var_scaled / 100.0

    return ForecastResult(
        var=var_unscaled,
        meta={
            "mu": float(mu) / 100.0,
            "sigma": sigma_next / 100.0,
            "omega": float(omega),
            "a": float(a),
            "b": float(b),
            "g": float(g),
            "x_term_is_squared": 1.0,
            "fallback": 0.0,
        },
    )

def forecast_historical_simulation(returns: np.ndarray, alpha: float = 0.01, window: int = 250) -> ForecastResult:
    r = np.asarray(returns)
    if len(r) < window:
        window = len(r)
    var = float(np.quantile(r[-window:], _safe_quantile(alpha)))
    return ForecastResult(var=var, meta={"window": float(window)})


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

    return ForecastResult(
        var=q_next_scaled / 100.0,
        meta={
            "b0": float(params[0]),
            "b1": float(params[1]),
            "b2": float(params[2]),
            "var_level": -q_next_scaled / 100.0,
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
