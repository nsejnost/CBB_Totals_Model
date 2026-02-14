# NCAAB Totals Predictor

## Overview
An advanced college basketball game-total projection model built with Streamlit. It uses machine learning (XGBoost + LightGBM + Ridge ensemble) to predict NCAAB game totals, improving on KenPom-style approaches with non-linear pace interactions, recency-weighted stats, and walk-forward backtesting.

## Project Architecture
- **app.py** - Main Streamlit dashboard with tabs for overview, backtest results, game explorer, feature importance, and methodology
- **data_loader.py** - Data loading and processing (Vegas data, Torvik stats, name mapping, rolling stats)
- **features.py** - Feature engineering (builds feature matrix from merged datasets)
- **model.py** - ML model definitions (XGBoost, LightGBM, Ridge ensemble), walk-forward backtest, metrics

## Data Files
- `2025_super_sked.csv` / `2026_super_sked.csv` - Season schedule data
- `ncaab_proxy_close_spreads_totals_*.csv` - Vegas closing lines data
- `team_name_mapping.csv` - Team name normalization

## Tech Stack
- Python 3.12
- Streamlit (frontend)
- pandas, numpy, scikit-learn, xgboost, lightgbm (ML/data)
- plotly (visualization)

## Configuration
- Streamlit config in `.streamlit/config.toml` - runs on port 5000, bound to 0.0.0.0
- System dependency: libgcc (provides libgomp for LightGBM)

## Running
```
streamlit run app.py
```
