# QQQ One-Day-Ahead VaR Forecasting Project

This project implements:

- Risk Metrics
- GARCH(1,1)
- GARCHX with VIX exogenous variance input
- GJR-GARCH(1,1)
- CAViaR (SAV specification)
- Historical Simulation benchmark

with rolling one-step-ahead VaR forecasts and backtests.

## Data

Uses Yahoo Finance data running from 2015-01-01 to 2026-03-27:
- `QQQ` adjusted close (for log returns)
- `^VIX` close (exogenous volatility proxy)

## Outputs

- `outputs/Integrated_Forecast_Results.csv` (all rolling forecasts + violations)
- `outputs/Integrated_Backtest_Results.csv` (Kupiec, Christoffersen, conditional coverage, Check Loss)
- `outputs/plots/var_*.png` (per-model VaR chart)
- `outputs/plots_ext/var_*.png`
- `outputs/riskmetrics/plot_var_riskmetrics.png`
- `outputs/garchx_only/plot_garchx_vix.png`

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/fetch_data.py --start 2015-01-01 --outdir data
python run_var_project.py --data data/model_data.csv --outdir outputs --alpha 0.01 --initial-train 0.7
python plot_var_results.py --forecasts outputs/var_forecasts.csv --outdir outputs/plots
python run_var_ext.py --data data/model_data.csv --outdir outputs --alpha 0.01 --initial-train 0.7
python plot_var_ext.py --forecasts outputs/var_forecasts.csv --outdir outputs/plots
python run_var_riskmetrics.py --data data/model_data.csv --outdir outputs --alpha 0.01 --initial-train 0.7
```
## Notes

- Forecast framework is strictly rolling and out-of-sample.
- VaR violation indicator is `1(Return < VaR)`.
- GARCH/GJR use `arch` package (Student-t innovations).
- GARCHX uses R CRAN package
- CAViaR uses quantile-regression style objective with SAV dynamics.
