# QQQ One-Day-Ahead VaR Forecasting Project

This project implements your STAT 929 proposal:

- GARCH(1,1)
- GARCHX with VIX exogenous variance input
- GJR-GARCH(1,1)
- CAViaR (SAV specification)
- Historical Simulation benchmark

with rolling one-step-ahead VaR forecasts and backtests.

## Data

Uses Yahoo Finance data:
- `QQQ` adjusted close (for log returns)
- `^VIX` close (exogenous volatility proxy)

## Outputs

- `outputs/var_forecasts.csv` (all rolling forecasts + violations)
- `outputs/var_backtests.csv` (Kupiec, Christoffersen, conditional coverage)
- `outputs/plots/var_*.png` (per-model VaR chart)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/fetch_data.py --start 2015-01-01 --outdir data
python run_var_project.py --data data/model_data.csv --outdir outputs --alpha 0.01 --initial-train 0.7
python plot_var_results.py --forecasts outputs/var_forecasts.csv --outdir outputs/plots
```

Optional speed-up while iterating:

```bash
python run_var_project.py --data data/model_data.csv --outdir outputs --alpha 0.01 --max-steps 200
```

## Notes

- Forecast framework is strictly rolling and out-of-sample.
- VaR violation indicator is `1(Return < VaR)`.
- GARCH/GJR use `arch` package (Student-t innovations).
- GARCHX is estimated via custom Gaussian QMLE recursion with VIX term in conditional variance.
- CAViaR uses quantile-regression style objective with SAV dynamics.
