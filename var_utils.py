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
