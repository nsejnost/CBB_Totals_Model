"""
NCAAB Totals Predictor – Streamlit Dashboard

An advanced college basketball game-total projection model that improves
on KenPom-style approaches via:
  • Machine learning (XGBoost + LightGBM + Ridge ensemble)
  • Non-linear pace × efficiency interaction features
  • Recency-weighted rolling stats (EMA) instead of season-long averages
  • Matchup-specific tempo modelling
  • Scoring volatility capture
  • Walk-forward backtesting vs Vegas closing lines

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_loader import (
    load_vegas_data, load_all_torvik, build_name_map,
    build_rolling_stats, merge_datasets,
)
from features import build_feature_matrix
from model import (
    DEFAULT_PARAMS, TotalsModel, walk_forward_backtest,
    compute_backtest_metrics,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NCAAB Totals Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: calibration parameters ─────────────────────────────────────────

st.sidebar.title("Model Calibration")

st.sidebar.markdown("---")
st.sidebar.subheader("Feature Weights")

recency_weight = st.sidebar.slider(
    "Recency weight (EMA vs season avg)",
    min_value=0.0, max_value=1.0,
    value=DEFAULT_PARAMS["recency_weight"], step=0.05,
    help="Higher = more weight on recent games (EMA). Lower = more on season average.",
)
pace_interaction_weight = st.sidebar.slider(
    "Pace-interaction emphasis",
    min_value=0.0, max_value=3.0,
    value=DEFAULT_PARAMS["pace_interaction_weight"], step=0.1,
    help="Multiplier on tempo/pace interaction features. >1 increases emphasis.",
)
volatility_weight = st.sidebar.slider(
    "Volatility emphasis",
    min_value=0.0, max_value=3.0,
    value=DEFAULT_PARAMS["volatility_weight"], step=0.1,
    help="Multiplier on scoring volatility features.",
)
vegas_anchor_weight = st.sidebar.slider(
    "Vegas anchor weight",
    min_value=0.0, max_value=1.0,
    value=DEFAULT_PARAMS["vegas_anchor_weight"], step=0.05,
    help="Final blend: weight on Vegas line. 1.0 = pure Vegas, 0.0 = pure model.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Ensemble Weights")
w_xgb = st.sidebar.slider("XGBoost weight", 0.0, 1.0, DEFAULT_PARAMS["w_xgb"], 0.05)
w_lgb = st.sidebar.slider("LightGBM weight", 0.0, 1.0, DEFAULT_PARAMS["w_lgb"], 0.05)
w_ridge = st.sidebar.slider("Ridge weight", 0.0, 1.0, DEFAULT_PARAMS["w_ridge"], 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("XGBoost Hyperparameters")
xgb_max_depth = st.sidebar.slider("XGB max depth", 2, 10, int(DEFAULT_PARAMS["xgb_max_depth"]))
xgb_learning_rate = st.sidebar.slider("XGB learning rate", 0.01, 0.3, DEFAULT_PARAMS["xgb_learning_rate"], 0.01)
xgb_n_estimators = st.sidebar.slider("XGB # estimators", 50, 800, int(DEFAULT_PARAMS["xgb_n_estimators"]), 50)

st.sidebar.markdown("---")
st.sidebar.subheader("LightGBM Hyperparameters")
lgb_max_depth = st.sidebar.slider("LGB max depth", 2, 10, int(DEFAULT_PARAMS["lgb_max_depth"]))
lgb_learning_rate = st.sidebar.slider("LGB learning rate", 0.01, 0.3, DEFAULT_PARAMS["lgb_learning_rate"], 0.01)
lgb_n_estimators = st.sidebar.slider("LGB # estimators", 50, 800, int(DEFAULT_PARAMS["lgb_n_estimators"]), 50)

st.sidebar.markdown("---")
st.sidebar.subheader("Ridge Regression")
ridge_alpha = st.sidebar.slider("Ridge alpha", 0.1, 100.0, DEFAULT_PARAMS["ridge_alpha"], 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Settings")
min_training_games = st.sidebar.slider(
    "Min training games", 100, 2000,
    int(DEFAULT_PARAMS["min_training_games"]), 100,
    help="How many games to use before making first prediction.",
)
retrain_every = st.sidebar.slider(
    "Retrain every N games", 50, 500,
    int(DEFAULT_PARAMS["retrain_every"]), 50,
    help="How often (in games) to retrain the model during walk-forward.",
)

# Collect all params
user_params = {
    "recency_weight": recency_weight,
    "pace_interaction_weight": pace_interaction_weight,
    "volatility_weight": volatility_weight,
    "vegas_anchor_weight": vegas_anchor_weight,
    "w_xgb": w_xgb,
    "w_lgb": w_lgb,
    "w_ridge": w_ridge,
    "xgb_max_depth": xgb_max_depth,
    "xgb_learning_rate": xgb_learning_rate,
    "xgb_n_estimators": xgb_n_estimators,
    "xgb_subsample": DEFAULT_PARAMS["xgb_subsample"],
    "xgb_colsample_bytree": DEFAULT_PARAMS["xgb_colsample_bytree"],
    "xgb_reg_alpha": DEFAULT_PARAMS["xgb_reg_alpha"],
    "xgb_reg_lambda": DEFAULT_PARAMS["xgb_reg_lambda"],
    "lgb_max_depth": lgb_max_depth,
    "lgb_learning_rate": lgb_learning_rate,
    "lgb_n_estimators": lgb_n_estimators,
    "lgb_subsample": DEFAULT_PARAMS["lgb_subsample"],
    "lgb_colsample_bytree": DEFAULT_PARAMS["lgb_colsample_bytree"],
    "lgb_reg_alpha": DEFAULT_PARAMS["lgb_reg_alpha"],
    "lgb_reg_lambda": DEFAULT_PARAMS["lgb_reg_lambda"],
    "ridge_alpha": ridge_alpha,
    "min_training_games": min_training_games,
    "retrain_every": retrain_every,
}

# ── Main content ─────────────────────────────────────────────────────────────

st.title("NCAAB Totals Predictor")
st.caption(
    "Advanced ML model for projecting college basketball game totals. "
    "Overcomes KenPom-style shortcomings with non-linear pace interactions, "
    "recency-weighted stats, and ensemble ML."
)

# Tabs
tab_overview, tab_backtest, tab_explore, tab_features, tab_methodology = st.tabs([
    "Overview & Predictions",
    "Backtest Results",
    "Game Explorer",
    "Feature Importance",
    "Methodology",
])

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading Vegas data…")
def _load_vegas():
    return load_vegas_data()

@st.cache_data(show_spinner="Loading Barttorvik data…")
def _load_torvik():
    return load_all_torvik()

@st.cache_data(show_spinner="Building rolling team stats…")
def _build_rolling(_torvik_hash):
    """Build rolling stats. _torvik_hash is used for cache invalidation only."""
    torvik_df = _load_torvik()
    return build_rolling_stats(torvik_df)

@st.cache_data(show_spinner="Building name map…")
def _build_name_map(_vegas_hash, _torvik_hash):
    vegas_df = _load_vegas()
    torvik_df = _load_torvik()
    vegas_home = vegas_df["home_team"].unique().tolist()
    vegas_away = vegas_df["away_team"].unique().tolist()
    vegas_names = list(set(vegas_home + vegas_away))
    torvik_names = list(set(
        torvik_df["team1"].dropna().unique().tolist()
        + torvik_df["team2"].dropna().unique().tolist()
    ))
    return build_name_map(vegas_names, torvik_names)

@st.cache_data(show_spinner="Merging datasets & building features…")
def _build_features(_vegas_hash, _torvik_hash):
    vegas_df = _load_vegas()
    rolling_df = _build_rolling(_torvik_hash)
    name_map = _build_name_map(_vegas_hash, _torvik_hash)
    merged_df = merge_datasets(vegas_df, rolling_df, name_map)
    feature_df, feat_cols = build_feature_matrix(merged_df)
    return feature_df, feat_cols, name_map, len(merged_df)


with st.spinner("Loading data pipeline…"):
    vegas_df = _load_vegas()
    torvik_df = _load_torvik()

    if torvik_df.empty or len(torvik_df) < 100:
        st.error(
            "Could not load Barttorvik data. Make sure 2025_super_sked.csv "
            "and/or 2026_super_sked.csv are in the project directory."
        )
        st.stop()

    # Use len as a simple hash for caching
    vegas_hash = len(vegas_df)
    torvik_hash = len(torvik_df)

    feature_df, feat_cols, name_map, n_merged = _build_features(vegas_hash, torvik_hash)

    if n_merged < 200:
        st.warning(
            f"Only matched {n_merged} games between Vegas and Torvik data. "
            "Team-name matching may need improvement. Proceeding with available data."
        )

    st.sidebar.markdown("---")
    st.sidebar.metric("Matched games", f"{len(feature_df):,}")
    st.sidebar.metric("Features", f"{len(feat_cols)}")


# ── Run backtest ─────────────────────────────────────────────────────────────

run_backtest = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
    st.session_state.backtest_params = None
    st.session_state.backtest_metrics = None

if run_backtest:
    with st.spinner("Running walk-forward backtest… this may take a minute."):
        bt = walk_forward_backtest(feature_df, feat_cols, user_params)
        metrics = compute_backtest_metrics(bt)
        st.session_state.backtest_results = bt
        st.session_state.backtest_params = user_params.copy()
        st.session_state.backtest_metrics = metrics

bt = st.session_state.backtest_results
metrics = st.session_state.backtest_metrics


# ── Tab: Overview & Predictions ──────────────────────────────────────────────

with tab_overview:
    st.header("Model Overview")

    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Model MAE",
            f"{metrics['model_mae']:.2f}",
            delta=f"{metrics['mae_improvement']:.2f} vs Vegas",
            delta_color="normal",
        )
        col2.metric(
            "Vegas MAE",
            f"{metrics['vegas_mae']:.2f}",
        )
        col3.metric(
            "Directional Accuracy",
            f"{metrics.get('directional_accuracy', 0):.1%}",
            help="When model disagrees with Vegas, how often is the model right?",
        )
        col4.metric(
            "Games Backtested",
            f"{metrics['n_games']:,}",
        )

        st.markdown("---")

        # Betting summary
        if "n_bets" in metrics:
            st.subheader("Simulated Betting Results (flat $100, -110 juice)")
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Total Bets (1+ pt edge)", metrics["n_bets"])
            bc2.metric("Win Rate", f"{metrics['bet_win_pct']:.1%}")
            bc3.metric("Profit/Loss", f"${metrics['bet_profit']:,.0f}")
            bc4.metric("ROI", f"{metrics['bet_roi']:.1%}")

            # Edge-stratified results
            st.subheader("Results by Edge Size")
            edge_data = []
            for edge_min in [1.5, 2.0, 3.0, 4.0, 5.0]:
                tag = str(edge_min).replace(".", "p")
                if f"edge{tag}_n" in metrics:
                    edge_data.append({
                        "Min Edge (pts)": edge_min,
                        "# Bets": metrics[f"edge{tag}_n"],
                        "Win %": f"{metrics[f'edge{tag}_winpct']:.1%}",
                        "Profit": f"${metrics[f'edge{tag}_profit']:,.0f}",
                        "ROI": f"{metrics[f'edge{tag}_roi']:.1%}",
                    })
            if edge_data:
                st.dataframe(pd.DataFrame(edge_data), use_container_width=True, hide_index=True)

    else:
        st.info(
            "Click **Run Backtest** in the sidebar to train the model and "
            "see results. Adjust calibration parameters first if desired."
        )

    # KenPom comparison explanation
    with st.expander("How this model improves on KenPom-style approaches"):
        st.markdown("""
**KenPom-style shortcomings for totals prediction:**

| Limitation | KenPom Approach | This Model's Solution |
|---|---|---|
| **Linear efficiency model** | Total ≈ Tempo × (AdjO_A + AdjO_B) / 100 | Non-linear ML captures complex interactions |
| **Single tempo estimate** | One AdjT per team, season-long | Matchup-specific pace features (tempo product, min/max, interaction terms) |
| **No recency weighting** | Treats November and February games equally | Exponential moving averages + rolling windows with tunable recency weight |
| **Ignores scoring volatility** | Point estimate only | Rolling standard deviations capture team consistency |
| **No market context** | Doesn't reference Vegas lines | Model learns systematic market biases, anchored to Vegas |
| **Static ratings** | Updated daily but still season-long | Walk-forward retraining captures regime changes |
| **No opponent-quality adjustment on trends** | Season-wide SOS | Rolling opponent quality EMA tracks recent schedule difficulty |
        """)


# ── Tab: Backtest Results ────────────────────────────────────────────────────

with tab_backtest:
    st.header("Walk-Forward Backtest")

    if bt is not None:
        valid = bt.dropna(subset=["model_total"])

        # Cumulative error comparison
        st.subheader("Cumulative Absolute Error: Model vs Vegas")
        valid = valid.sort_values("game_date").reset_index(drop=True)
        valid["cum_vegas_ae"] = valid["vegas_error"].abs().cumsum()
        valid["cum_model_ae"] = valid["model_error"].abs().cumsum()
        valid["game_num"] = range(1, len(valid) + 1)

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=valid["game_num"], y=valid["cum_vegas_ae"],
            name="Vegas cumulative AE", line=dict(color="red", width=2),
        ))
        fig_cum.add_trace(go.Scatter(
            x=valid["game_num"], y=valid["cum_model_ae"],
            name="Model cumulative AE", line=dict(color="blue", width=2),
        ))
        fig_cum.update_layout(
            xaxis_title="Game #", yaxis_title="Cumulative Absolute Error",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # Rolling MAE
        st.subheader("Rolling MAE (100-game window)")
        valid["rolling_vegas_mae"] = valid["vegas_error"].abs().rolling(100, min_periods=20).mean()
        valid["rolling_model_mae"] = valid["model_error"].abs().rolling(100, min_periods=20).mean()

        fig_rmae = go.Figure()
        fig_rmae.add_trace(go.Scatter(
            x=valid["game_date"], y=valid["rolling_vegas_mae"],
            name="Vegas rolling MAE", line=dict(color="red", width=2),
        ))
        fig_rmae.add_trace(go.Scatter(
            x=valid["game_date"], y=valid["rolling_model_mae"],
            name="Model rolling MAE", line=dict(color="blue", width=2),
        ))
        fig_rmae.update_layout(
            xaxis_title="Date", yaxis_title="MAE (100-game rolling)",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_rmae, use_container_width=True)

        # Residual distribution
        st.subheader("Prediction Error Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig_vhist = px.histogram(
                valid, x="vegas_error", nbins=60,
                title="Vegas Error Distribution",
                labels={"vegas_error": "Vegas Total - Actual Total"},
                color_discrete_sequence=["red"],
            )
            fig_vhist.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_vhist, use_container_width=True)

        with col2:
            fig_mhist = px.histogram(
                valid, x="model_error", nbins=60,
                title="Model Error Distribution",
                labels={"model_error": "Model Total - Actual Total"},
                color_discrete_sequence=["blue"],
            )
            fig_mhist.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_mhist, use_container_width=True)

        # Scatter: predicted vs actual
        st.subheader("Predicted vs Actual Total")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=valid["actual_total"], y=valid["vegas_total"],
            mode="markers", name="Vegas",
            marker=dict(color="red", opacity=0.3, size=4),
        ))
        fig_scatter.add_trace(go.Scatter(
            x=valid["actual_total"], y=valid["model_total"],
            mode="markers", name="Model",
            marker=dict(color="blue", opacity=0.3, size=4),
        ))
        # Perfect line
        mn = valid["actual_total"].min()
        mx = valid["actual_total"].max()
        fig_scatter.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", name="Perfect",
            line=dict(color="green", dash="dash"),
        ))
        fig_scatter.update_layout(
            xaxis_title="Actual Total", yaxis_title="Predicted Total",
            height=500, template="plotly_white",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Betting P&L curve
        if metrics and "n_bets" in metrics:
            st.subheader("Cumulative Betting P&L (1+ pt edge, -110 juice)")
            bets = valid[
                (valid["model_total"] - valid["vegas_total"]).abs() >= 1.0
            ].copy()
            bets["bet_over"] = bets["model_total"] > bets["vegas_total"]
            bets["won"] = (
                (bets["bet_over"] & (bets["actual_total"] > bets["vegas_total"])) |
                (~bets["bet_over"] & (bets["actual_total"] < bets["vegas_total"]))
            )
            bets["push"] = bets["actual_total"] == bets["vegas_total"]
            bets["pnl"] = np.where(
                bets["push"], 0,
                np.where(bets["won"], 100, -110),
            )
            bets["cum_pnl"] = bets["pnl"].cumsum()

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=bets["game_date"], y=bets["cum_pnl"],
                fill="tozeroy", line=dict(color="green", width=2),
            ))
            fig_pnl.update_layout(
                xaxis_title="Date", yaxis_title="Cumulative P&L ($)",
                height=400, template="plotly_white",
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        # Monthly breakdown
        st.subheader("Monthly Backtest Summary")
        valid["month"] = valid["game_date"].dt.to_period("M").astype(str)
        monthly = valid.groupby("month").agg(
            games=("actual_total", "count"),
            vegas_mae=("vegas_error", lambda x: np.mean(np.abs(x))),
            model_mae=("model_error", lambda x: np.mean(np.abs(x))),
            vegas_bias=("vegas_error", "mean"),
            model_bias=("model_error", "mean"),
        ).reset_index()
        monthly["mae_diff"] = monthly["vegas_mae"] - monthly["model_mae"]
        monthly = monthly.rename(columns={
            "month": "Month", "games": "Games",
            "vegas_mae": "Vegas MAE", "model_mae": "Model MAE",
            "vegas_bias": "Vegas Bias", "model_bias": "Model Bias",
            "mae_diff": "MAE Improvement",
        })
        st.dataframe(
            monthly.style.format({
                "Vegas MAE": "{:.2f}", "Model MAE": "{:.2f}",
                "Vegas Bias": "{:.2f}", "Model Bias": "{:.2f}",
                "MAE Improvement": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

    else:
        st.info("Run a backtest from the sidebar to see results here.")


# ── Tab: Game Explorer ───────────────────────────────────────────────────────

with tab_explore:
    st.header("Game Explorer")

    if bt is not None:
        valid = bt.dropna(subset=["model_total"]).copy()

        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Date range",
                value=(valid["game_date"].min(), valid["game_date"].max()),
                min_value=valid["game_date"].min(),
                max_value=valid["game_date"].max(),
            )
        with col2:
            min_edge = st.slider("Minimum model edge (pts)", 0.0, 10.0, 0.0, 0.5)

        if isinstance(date_range, tuple) and len(date_range) == 2:
            mask = (
                (valid["game_date"].dt.date >= date_range[0])
                & (valid["game_date"].dt.date <= date_range[1])
                & ((valid["model_total"] - valid["vegas_total"]).abs() >= min_edge)
            )
            filtered = valid[mask].sort_values("game_date", ascending=False)

            display_cols = [
                "game_date", "home_team", "away_team",
                "vegas_total", "model_total", "actual_total",
                "model_error", "vegas_error",
            ]
            display = filtered[
                [c for c in display_cols if c in filtered.columns]
            ].copy()
            display["model_edge"] = display["model_total"] - display["vegas_total"]
            display = display.rename(columns={
                "game_date": "Date", "home_team": "Home", "away_team": "Away",
                "vegas_total": "Vegas", "model_total": "Model",
                "actual_total": "Actual", "model_error": "Model Err",
                "vegas_error": "Vegas Err", "model_edge": "Edge",
            })
            st.dataframe(
                display.style.format({
                    "Vegas": "{:.1f}", "Model": "{:.1f}", "Actual": "{:.1f}",
                    "Model Err": "{:.1f}", "Vegas Err": "{:.1f}", "Edge": "{:.1f}",
                }),
                use_container_width=True, hide_index=True,
                height=500,
            )
            st.caption(f"Showing {len(display):,} games")
    else:
        st.info("Run a backtest to explore individual games.")


# ── Tab: Feature Importance ──────────────────────────────────────────────────

with tab_features:
    st.header("Feature Importance")

    if bt is not None and st.session_state.backtest_params is not None:
        # Train final model on all data for feature importance
        with st.spinner("Computing feature importance…"):
            final_model = TotalsModel(st.session_state.backtest_params)
            target = feature_df["actual_total"] - feature_df["vegas_total"]
            mask = target.notna() & feature_df[feat_cols].notna().all(axis=1)
            final_model.fit(feature_df[mask], target[mask], feat_cols)
            imp_df = final_model.feature_importance()

        if not imp_df.empty:
            # Top 30 features by model
            for model_name in imp_df["model"].unique():
                sub = imp_df[imp_df["model"] == model_name].nlargest(30, "importance")
                fig = px.bar(
                    sub, x="importance", y="feature", orientation="h",
                    title=f"Top 30 Features – {model_name}",
                    color_discrete_sequence=["steelblue"],
                )
                fig.update_layout(
                    height=600, template="plotly_white",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No feature importance available (tree models may not be installed).")
    else:
        st.info("Run a backtest first to see feature importance.")


# ── Tab: Methodology ─────────────────────────────────────────────────────────

with tab_methodology:
    st.header("Methodology")

    st.markdown("""
## Why Not Just KenPom?

KenPom's adjusted efficiency model is elegant and powerful for **margin-of-victory**
estimation, but has systematic shortcomings for **totals** prediction:

### 1. Linear Efficiency Assumption
KenPom estimates game totals as:
```
Total ≈ AdjTempo × (Team1_AdjO + Team2_AdjO) / 100
```
This is a **linear** model that cannot capture how a fast team's offense interacts
**non-linearly** with a slow team's defensive style. In reality, the pace-controlling
team's influence is asymmetric and context-dependent.

### 2. Season-Long Averages Miss Streaks & Trends
KenPom updates daily, but his ratings are season-long adjusted averages. A team
that has dramatically changed its style (new rotation, injury returns, strategic shift)
mid-season is poorly captured. This model uses **exponential moving averages** and
**rolling windows** (5/10/20 games) to capture recent form.

### 3. No Matchup-Specific Tempo Modeling
KenPom uses one AdjTempo per team. But the actual game pace depends on the
**interaction** of both teams' preferred tempos. This model includes:
- Tempo product (both fast = very high scoring)
- Tempo minimum (slower team controls pace)
- Tempo differential
- Non-linear tempo × efficiency cross terms

### 4. No Volatility Capture
Two teams can have the same average but very different variances. A volatile
team creates more over/under opportunities. This model tracks **rolling standard
deviations** of scoring, pace, and efficiency.

### 5. No Market Context
KenPom operates in a vacuum. Vegas lines incorporate information from sharps,
injury news, and other signals. This model is **anchored to Vegas** and learns
to predict the **residual** (actual – Vegas), capturing systematic biases the
market misses.

---

## Model Architecture

### Feature Engineering (120+ features)
- Rolling stats: 5/10/20-game windows + EMA + season average
- Pace interactions: product, min, max, differential
- Efficiency interactions: offense vs opposing defense gaps
- Scoring volatility: rolling standard deviations
- Market features: Vegas total, implied team totals, spread
- Momentum: recent trends vs season baselines
- Opponent quality: rolling strength-of-schedule

### Ensemble Model
Three sub-models, each with distinct strengths:

| Model | Strength | Role |
|-------|----------|------|
| **XGBoost** | Complex non-linear interactions | Primary predictor |
| **LightGBM** | Fast, handles sparse features well | Secondary predictor |
| **Ridge Regression** | Stable linear baseline | Regularization anchor |

The ensemble blends these with user-configurable weights, then anchors the
final prediction to the Vegas line with a tunable anchor weight.

### Walk-Forward Backtesting
- **No lookahead bias**: model only sees past games at each prediction point
- **Periodic retraining**: model retrains every N games (configurable)
- **Realistic evaluation**: all metrics computed out-of-sample

### Calibration Philosophy
Every key parameter is exposed in the sidebar so you can see how changes
affect backtest results:
- **Recency weight**: Balance recent form vs season-long track record
- **Pace emphasis**: How much to weight matchup-specific pace dynamics
- **Volatility emphasis**: How much scoring variance matters
- **Vegas anchor**: Trust the market vs trust the model
- **Ensemble weights**: Blend between ML models and linear baseline
- **Tree hyperparameters**: Control model complexity and overfitting
    """)

    st.markdown("---")
    st.caption(
        "Data sources: Barttorvik (game-level advanced stats), "
        "DraftKings/FanDuel (closing lines via user-provided dataset)"
    )
