"""
NCAAB Totals Predictor – Streamlit Dashboard

An advanced college basketball game-total projection model that improves
on KenPom-style approaches via:
  * Machine learning (XGBoost + LightGBM + Ridge ensemble with stacking)
  * Non-linear pace x efficiency interaction features
  * Recency-weighted rolling stats (EMA) instead of season-long averages
  * Matchup-specific tempo modelling
  * Scoring volatility capture
  * Walk-forward backtesting vs Vegas closing lines
  * SHAP-based feature explanations
  * Cross-validation confidence intervals

Run:  streamlit run app.py
"""

import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_loader import (
    load_vegas_data, load_all_torvik, build_name_map,
    build_rolling_stats, merge_datasets, build_upcoming_rows,
)
from features import build_feature_matrix
from model import (
    DEFAULT_PARAMS, PARAM_PRESETS, TotalsModel,
    walk_forward_backtest, compute_backtest_metrics,
    train_full_model, predict_upcoming, cross_validate_model,
    save_model, list_saved_models, load_saved_model, delete_saved_model,
    HAS_SHAP, HAS_JOBLIB,
)
from odds_api import fetch_upcoming_odds, pick_consensus_line

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NCAAB Totals Predictor",
    page_icon="\U0001f3c0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: calibration parameters ─────────────────────────────────────────

st.sidebar.title("Model Calibration")

# --- Parameter presets ---
st.sidebar.markdown("---")
preset_names = list(PARAM_PRESETS.keys())
selected_preset = st.sidebar.selectbox(
    "Parameter Preset",
    preset_names,
    index=0,
    help="Load a named parameter configuration. 'Default' is the baseline. "
         "'Conservative' trusts Vegas more. 'Aggressive' trusts the model more.",
)
preset_params = PARAM_PRESETS[selected_preset]

if st.sidebar.button("Apply Preset", use_container_width=True):
    for key, val in preset_params.items():
        st.session_state[f"param_{key}"] = val
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Feature Weights")


def _get_param(key, default_val):
    """Get parameter value from session state or preset."""
    return st.session_state.get(f"param_{key}", default_val)


recency_weight = st.sidebar.slider(
    "Recency weight (EMA vs season avg)",
    min_value=0.0, max_value=1.0,
    value=_get_param("recency_weight", DEFAULT_PARAMS["recency_weight"]),
    step=0.05,
    help="Higher = more weight on recent games (EMA). Lower = more on season average.",
)
pace_interaction_weight = st.sidebar.slider(
    "Pace-interaction emphasis",
    min_value=0.0, max_value=3.0,
    value=_get_param("pace_interaction_weight", DEFAULT_PARAMS["pace_interaction_weight"]),
    step=0.1,
    help="Multiplier on tempo/pace interaction features. >1 increases emphasis.",
)
volatility_weight = st.sidebar.slider(
    "Volatility emphasis",
    min_value=0.0, max_value=3.0,
    value=_get_param("volatility_weight", DEFAULT_PARAMS["volatility_weight"]),
    step=0.1,
    help="Multiplier on scoring volatility features.",
)
vegas_anchor_weight = st.sidebar.slider(
    "Vegas anchor weight",
    min_value=0.0, max_value=1.0,
    value=_get_param("vegas_anchor_weight", DEFAULT_PARAMS["vegas_anchor_weight"]),
    step=0.05,
    help="Final blend: weight on Vegas line. 1.0 = pure Vegas, 0.0 = pure model.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Ensemble & Stacking")
use_stacking = st.sidebar.checkbox(
    "Use stacking meta-learner",
    value=True,
    help="Train a Ridge meta-learner on out-of-fold base model predictions. "
         "If disabled, falls back to manual ensemble weights below.",
)
w_xgb = st.sidebar.slider("XGBoost weight (fallback)", 0.0, 1.0,
                           _get_param("w_xgb", DEFAULT_PARAMS["w_xgb"]), 0.05)
w_lgb = st.sidebar.slider("LightGBM weight (fallback)", 0.0, 1.0,
                           _get_param("w_lgb", DEFAULT_PARAMS["w_lgb"]), 0.05)
w_ridge = st.sidebar.slider("Ridge weight (fallback)", 0.0, 1.0,
                             _get_param("w_ridge", DEFAULT_PARAMS["w_ridge"]), 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Season & Feature Selection")
season_decay = st.sidebar.slider(
    "Season decay",
    min_value=0.5, max_value=1.0,
    value=_get_param("season_decay", DEFAULT_PARAMS["season_decay"]),
    step=0.05,
    help="Weight multiplier per season back. 1.0 = no decay, 0.5 = strong recency preference.",
)
corr_threshold = st.sidebar.slider(
    "Correlation threshold",
    min_value=0.80, max_value=1.0,
    value=_get_param("correlation_threshold", DEFAULT_PARAMS["correlation_threshold"]),
    step=0.01,
    help="Drop features with pairwise correlation above this. Lower = fewer features.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("XGBoost Hyperparameters")
xgb_max_depth = st.sidebar.slider("XGB max depth", 2, 10,
                                   int(_get_param("xgb_max_depth", DEFAULT_PARAMS["xgb_max_depth"])))
xgb_learning_rate = st.sidebar.slider("XGB learning rate", 0.01, 0.3,
                                       _get_param("xgb_learning_rate", DEFAULT_PARAMS["xgb_learning_rate"]), 0.01)
xgb_n_estimators = st.sidebar.slider("XGB # estimators", 50, 800,
                                      int(_get_param("xgb_n_estimators", DEFAULT_PARAMS["xgb_n_estimators"])), 50)

st.sidebar.markdown("---")
st.sidebar.subheader("LightGBM Hyperparameters")
lgb_max_depth = st.sidebar.slider("LGB max depth", 2, 10,
                                   int(_get_param("lgb_max_depth", DEFAULT_PARAMS["lgb_max_depth"])))
lgb_learning_rate = st.sidebar.slider("LGB learning rate", 0.01, 0.3,
                                       _get_param("lgb_learning_rate", DEFAULT_PARAMS["lgb_learning_rate"]), 0.01)
lgb_n_estimators = st.sidebar.slider("LGB # estimators", 50, 800,
                                      int(_get_param("lgb_n_estimators", DEFAULT_PARAMS["lgb_n_estimators"])), 50)

st.sidebar.markdown("---")
st.sidebar.subheader("Ridge Regression")
ridge_alpha = st.sidebar.slider("Ridge alpha", 0.1, 100.0,
                                 _get_param("ridge_alpha", DEFAULT_PARAMS["ridge_alpha"]), 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Settings")
min_training_games = st.sidebar.slider(
    "Min training games", 100, 2000,
    int(_get_param("min_training_games", DEFAULT_PARAMS["min_training_games"])), 100,
    help="How many games to use before making first prediction.",
)
retrain_every = st.sidebar.slider(
    "Retrain every N games", 50, 500,
    int(_get_param("retrain_every", DEFAULT_PARAMS["retrain_every"])), 50,
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
    "use_stacking": use_stacking,
    "season_decay": season_decay,
    "correlation_threshold": corr_threshold,
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
    "recency-weighted stats, and ensemble ML with stacking."
)

# Tabs
tab_overview, tab_value, tab_backtest, tab_explore, tab_features, tab_cv, tab_models, tab_diagnostics, tab_methodology = st.tabs([
    "Overview & Predictions",
    "Today's Value Bets",
    "Backtest Results",
    "Game Explorer",
    "Feature Importance",
    "Cross-Validation",
    "Saved Models",
    "Diagnostics",
    "Methodology",
])

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading Vegas data\u2026")
def _load_vegas():
    return load_vegas_data()

@st.cache_data(show_spinner="Loading Barttorvik data\u2026")
def _load_torvik():
    return load_all_torvik()

@st.cache_data(show_spinner="Building rolling team stats\u2026")
def _build_rolling(_torvik_hash):
    """Build rolling stats. _torvik_hash is used for cache invalidation only."""
    torvik_df = _load_torvik()
    return build_rolling_stats(torvik_df)

@st.cache_data(show_spinner="Building name map\u2026")
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
    name_map, unmatched = build_name_map(vegas_names, torvik_names)
    return name_map, unmatched

@st.cache_data(show_spinner="Merging datasets & building features\u2026")
def _build_features(_vegas_hash, _torvik_hash):
    vegas_df = _load_vegas()
    rolling_df = _build_rolling(_torvik_hash)
    name_map, unmatched = _build_name_map(_vegas_hash, _torvik_hash)
    merged_df = merge_datasets(vegas_df, rolling_df, name_map)
    feature_df, feat_cols = build_feature_matrix(merged_df)
    return feature_df, feat_cols, name_map, unmatched, len(merged_df)


with st.spinner("Loading data pipeline\u2026"):
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

    feature_df, feat_cols, name_map, unmatched_teams, n_merged = _build_features(vegas_hash, torvik_hash)

    if n_merged < 200:
        st.warning(
            f"Only matched {n_merged} games between Vegas and Torvik data. "
            "Team-name matching may need improvement. Check the Diagnostics tab."
        )

    st.sidebar.markdown("---")
    st.sidebar.metric("Matched games", f"{len(feature_df):,}")
    st.sidebar.metric("Features", f"{len(feat_cols)}")


# ── Run backtest ─────────────────────────────────────────────────────────────

run_backtest = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

st.sidebar.markdown("---")
odds_api_key = os.environ.get("ODDS_API_KEY", "")
min_edge_filter = st.sidebar.slider(
    "Min edge to display (pts)", 0.0, 5.0, 1.0, 0.5,
    help="Only show games where model edge exceeds this threshold.",
)

if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
    st.session_state.backtest_params = None
    st.session_state.backtest_metrics = None

if run_backtest:
    progress_bar = st.progress(0, text="Running walk-forward backtest\u2026")
    bt = walk_forward_backtest(
        feature_df, feat_cols, user_params,
        progress_callback=lambda pct: progress_bar.progress(
            min(pct, 1.0),
            text=f"Backtesting\u2026 {pct:.0%} complete",
        ),
    )
    progress_bar.progress(1.0, text="Backtest complete!")
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
| **Linear efficiency model** | Total ~ Tempo x (AdjO_A + AdjO_B) / 100 | Non-linear ML captures complex interactions |
| **Single tempo estimate** | One AdjT per team, season-long | Matchup-specific pace features (tempo product, min/max, interaction terms) |
| **No recency weighting** | Treats November and February games equally | Exponential moving averages + rolling windows with tunable recency weight |
| **Ignores scoring volatility** | Point estimate only | Rolling standard deviations capture team consistency |
| **No market context** | Doesn't reference Vegas lines | Model learns systematic market biases, anchored to Vegas |
| **Static ratings** | Updated daily but still season-long | Walk-forward retraining captures regime changes |
| **No opponent-quality adjustment on trends** | Season-wide SOS | Rolling opponent quality EMA tracks recent schedule difficulty |
| **No home-court / conference context** | Single home-court factor | Venue, conference matchup, and power-conference indicators |
        """)


# ── Tab: Today's Value Bets ─────────────────────────────────────────────────

with tab_value:
    st.header("Today's Value Bets")

    if not odds_api_key:
        st.info(
            "Set the **ODDS_API_KEY** secret in Replit to fetch live odds "
            "for upcoming games. Get a free key at https://the-odds-api.com"
        )
    else:
        fetch_odds_btn = st.button("Fetch Live Odds & Run Model", type="primary")

        if fetch_odds_btn:
            # 1. Fetch odds
            with st.spinner("Fetching upcoming odds from The Odds API..."):
                try:
                    raw_odds, api_info = fetch_upcoming_odds(odds_api_key)
                except Exception as e:
                    st.error(f"Failed to fetch odds: {e}")
                    raw_odds = pd.DataFrame()
                    api_info = {}

            if raw_odds.empty:
                st.warning("No upcoming games with totals found from the API.")
            else:
                st.caption(
                    f"API requests remaining: {api_info.get('requests_remaining', '?')} | "
                    f"Used: {api_info.get('requests_used', '?')}"
                )

                # 2. Pick consensus line across bookmakers
                consensus = pick_consensus_line(raw_odds)
                st.caption(f"Found {len(consensus)} upcoming games with odds.")

                # 3. Build upcoming game features
                with st.spinner("Building features for upcoming games..."):
                    rolling_df = _build_rolling(torvik_hash)
                    name_map_all, _ = _build_name_map(vegas_hash, torvik_hash)

                    # Extend the name map with upcoming game team names
                    new_vegas_names = list(set(
                        consensus["home_team"].tolist()
                        + consensus["away_team"].tolist()
                    ))
                    torvik_names_all = list(set(
                        torvik_df["team1"].dropna().unique().tolist()
                        + torvik_df["team2"].dropna().unique().tolist()
                    ))
                    live_name_map, _ = build_name_map(new_vegas_names, torvik_names_all)
                    # Merge: prefer existing map, fill in from live map
                    full_name_map = {**live_name_map, **name_map_all}

                    upcoming_rows = build_upcoming_rows(
                        consensus, rolling_df, full_name_map,
                    )

                if upcoming_rows.empty:
                    st.warning(
                        "Could not match any upcoming game teams to Torvik data. "
                        "Team name mapping may need updating. Check the Diagnostics tab."
                    )
                else:
                    # 4. Build feature matrix
                    with st.spinner("Engineering features..."):
                        upcoming_feat, uf_cols = build_feature_matrix(upcoming_rows)

                    # 5. Train model on full historical data and predict
                    with st.spinner("Training model on full history & predicting..."):
                        try:
                            model = train_full_model(
                                feature_df, feat_cols, user_params,
                            )
                            results = predict_upcoming(model, upcoming_feat)
                        except Exception as e:
                            st.error(f"Model error: {e}")
                            results = pd.DataFrame()

                    if not results.empty:
                        # Store in session state for persistence (with timestamp)
                        st.session_state["value_bets"] = results
                        st.session_state["odds_fetched_at"] = datetime.now().strftime(
                            "%Y-%m-%d %I:%M %p ET"
                        )

        # Display results (persisted across reruns)
        if "value_bets" in st.session_state and not st.session_state["value_bets"].empty:
            results = st.session_state["value_bets"]

            # Show when odds were fetched
            if "odds_fetched_at" in st.session_state:
                st.caption(f"Odds fetched at: {st.session_state['odds_fetched_at']}")

            # Filter by minimum edge
            display = results[results["abs_edge"] >= min_edge_filter].copy()
            display = display.sort_values("abs_edge", ascending=False)

            if display.empty:
                st.info(
                    f"No games found with edge >= {min_edge_filter} pts. "
                    "Try lowering the minimum edge filter in the sidebar."
                )
            else:
                # Summary metrics
                n_value = len(display)
                avg_edge = display["abs_edge"].mean()
                n_over = (display["bet_side"] == "OVER").sum()
                n_under = (display["bet_side"] == "UNDER").sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Value Bets Found", n_value)
                c2.metric("Avg Edge", f"{avg_edge:.1f} pts")
                c3.metric("Overs", n_over)
                c4.metric("Unders", n_under)

                st.markdown("---")

                # Detailed table
                table_cols = {
                    "commence_time": "Game Time",
                    "home_team": "Home",
                    "away_team": "Away",
                    "vegas_total": "Vegas Total",
                    "model_total": "Model Total",
                    "model_edge": "Edge",
                    "bet_side": "Side",
                    "over_price": "Over Price",
                    "under_price": "Under Price",
                }
                avail = [c for c in table_cols if c in display.columns]
                show = display[avail].rename(columns=table_cols)

                # Format game time (convert UTC -> Eastern)
                if "Game Time" in show.columns:
                    show["Game Time"] = (
                        pd.to_datetime(show["Game Time"], utc=True)
                        .dt.tz_convert("US/Eastern")
                        .dt.strftime("%b %d  %I:%M %p ET")
                    )

                fmt = {}
                if "Vegas Total" in show.columns:
                    fmt["Vegas Total"] = "{:.1f}"
                if "Model Total" in show.columns:
                    fmt["Model Total"] = "{:.1f}"
                if "Edge" in show.columns:
                    fmt["Edge"] = "{:+.1f}"
                if "Over Price" in show.columns:
                    fmt["Over Price"] = "{:+.0f}"
                if "Under Price" in show.columns:
                    fmt["Under Price"] = "{:+.0f}"

                st.dataframe(
                    show.style.format(fmt),
                    use_container_width=True,
                    hide_index=True,
                    height=min(600, 50 + 35 * len(show)),
                )

                # Individual game cards for top edges
                st.subheader("Top Value Plays")
                for _, row in display.head(5).iterrows():
                    edge = row["model_edge"]
                    side = row["bet_side"]
                    price_col = "over_price" if side == "OVER" else "under_price"
                    price = row.get(price_col, None)
                    price_str = f" ({price:+.0f})" if pd.notna(price) else ""

                    with st.container(border=True):
                        gc1, gc2, gc3 = st.columns([3, 2, 2])
                        gc1.markdown(
                            f"**{row['away_team']}** @ **{row['home_team']}**"
                        )
                        gc2.metric(
                            "Model Total",
                            f"{row['model_total']:.1f}",
                            delta=f"{edge:+.1f} vs Vegas {row['vegas_total']:.1f}",
                        )
                        gc3.markdown(
                            f"### {side}{price_str}"
                        )

                st.caption(
                    "Edge = Model Total - Vegas Total. "
                    "Positive edge = model expects higher scoring (lean OVER). "
                    "Negative edge = model expects lower scoring (lean UNDER)."
                )


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


# ── Tab: Feature Importance (with SHAP) ─────────────────────────────────────

with tab_features:
    st.header("Feature Importance")

    if bt is not None and st.session_state.backtest_params is not None:
        # Train final model on all data for feature importance
        with st.spinner("Computing feature importance\u2026"):
            final_model = TotalsModel(st.session_state.backtest_params)
            target = feature_df["actual_total"] - feature_df["vegas_total"]
            mask = target.notna() & feature_df[feat_cols].notna().all(axis=1)
            final_model.fit(feature_df[mask], target[mask], feat_cols)
            imp_df = final_model.feature_importance()

        if not imp_df.empty:
            # Show how many features were selected vs total
            n_orig = len(feat_cols)
            n_selected = len(final_model.selected_cols)
            if n_selected < n_orig:
                st.info(
                    f"Feature selection reduced {n_orig} features to {n_selected} "
                    f"(dropped {n_orig - n_selected} highly correlated features)."
                )

            # Gain-based importance
            st.subheader("Gain-Based Feature Importance")
            for model_name in imp_df["model"].unique():
                sub = imp_df[imp_df["model"] == model_name].nlargest(30, "importance")
                fig = px.bar(
                    sub, x="importance", y="feature", orientation="h",
                    title=f"Top 30 Features - {model_name}",
                    color_discrete_sequence=["steelblue"],
                )
                fig.update_layout(
                    height=600, template="plotly_white",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # SHAP values
            if HAS_SHAP:
                st.subheader("SHAP Feature Impact")
                with st.spinner("Computing SHAP values (this may take a moment)\u2026"):
                    shap_result = final_model.compute_shap_values(feature_df[mask])

                if shap_result:
                    for model_name, shap_df in shap_result.items():
                        label = "XGBoost" if model_name == "xgb" else "LightGBM"
                        st.markdown(f"**{label} - Mean |SHAP| Impact**")

                        mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(30)
                        fig_shap = px.bar(
                            x=mean_abs.values, y=mean_abs.index,
                            orientation="h",
                            labels={"x": "Mean |SHAP value|", "y": "Feature"},
                            title=f"Top 30 SHAP Features - {label}",
                            color_discrete_sequence=["coral"],
                        )
                        fig_shap.update_layout(
                            height=600, template="plotly_white",
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.warning("SHAP computation returned no results.")
            else:
                st.info("Install `shap` package for SHAP-based feature explanations.")
        else:
            st.warning("No feature importance available (tree models may not be installed).")
    else:
        st.info("Run a backtest first to see feature importance.")


# ── Tab: Cross-Validation ───────────────────────────────────────────────────

with tab_cv:
    st.header("Cross-Validation")
    st.caption(
        "Time-series cross-validation provides confidence intervals on model "
        "performance, helping distinguish real improvement from noise."
    )

    run_cv = st.button("Run Cross-Validation (5-fold)", type="primary")

    if run_cv:
        with st.spinner("Running time-series cross-validation\u2026"):
            cv_results = cross_validate_model(feature_df, feat_cols, user_params, n_splits=5)

        if cv_results:
            st.session_state["cv_results"] = cv_results

    if "cv_results" in st.session_state and st.session_state["cv_results"]:
        cv = st.session_state["cv_results"]

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Mean Model MAE",
            f"{cv['mean_model_mae']:.2f}",
            delta=f"\u00b1{cv['std_model_mae']:.2f}",
        )
        c2.metric(
            "Mean MAE Improvement",
            f"{cv['mean_mae_improvement']:.2f}",
            delta=f"\u00b1{cv['std_mae_improvement']:.2f}",
        )
        c3.metric(
            "95% CI (Improvement)",
            f"[{cv['ci_lower_improvement']:.2f}, {cv['ci_upper_improvement']:.2f}]",
            help="If the lower bound is positive, the improvement is statistically significant.",
        )

        st.markdown("---")
        st.subheader("Per-Fold Results")
        folds = cv["folds"]
        st.dataframe(
            folds.style.format({
                "model_mae": "{:.2f}",
                "vegas_mae": "{:.2f}",
                "mae_improvement": "{:.2f}",
                "model_rmse": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

        # Visualization
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=folds["fold"], y=folds["vegas_mae"],
            name="Vegas MAE", marker_color="red",
        ))
        fig_cv.add_trace(go.Bar(
            x=folds["fold"], y=folds["model_mae"],
            name="Model MAE", marker_color="blue",
        ))
        fig_cv.update_layout(
            barmode="group",
            xaxis_title="Fold", yaxis_title="MAE",
            height=400, template="plotly_white",
            title="MAE by Cross-Validation Fold",
        )
        st.plotly_chart(fig_cv, use_container_width=True)


# ── Tab: Saved Models ───────────────────────────────────────────────────────

with tab_models:
    st.header("Saved Models")

    if not HAS_JOBLIB:
        st.warning("Install `joblib` to enable model persistence.")
    else:
        # Save current model
        if metrics and bt is not None:
            st.subheader("Save Current Model")
            model_tag = st.text_input(
                "Model tag (optional label)",
                placeholder="e.g., baseline_v1",
            )
            if st.button("Save Model & Backtest Results", type="primary"):
                with st.spinner("Training and saving model\u2026"):
                    try:
                        full_model = train_full_model(
                            feature_df, feat_cols, user_params,
                        )
                        path = save_model(full_model, metrics, user_params, tag=model_tag)
                        st.success(f"Model saved successfully.")
                    except Exception as e:
                        st.error(f"Failed to save model: {e}")

        st.markdown("---")

        # List saved models
        st.subheader("Previously Saved Models")
        saved = list_saved_models()
        if not saved:
            st.info("No saved models found. Run a backtest and save a model above.")
        else:
            for i, meta in enumerate(reversed(saved)):
                ts = meta.get("timestamp", "?")
                tag = meta.get("tag", "")
                n_feat = meta.get("n_features", "?")
                m = meta.get("metrics", {})
                mae = m.get("model_mae", "?")
                mae_str = f"{mae:.2f}" if isinstance(mae, (int, float)) else str(mae)

                label = f"{ts}"
                if tag:
                    label += f" [{tag}]"
                label += f" | MAE: {mae_str} | {n_feat} features"

                with st.expander(label):
                    # Show params
                    params_display = meta.get("params", {})
                    key_params = {
                        "vegas_anchor_weight": params_display.get("vegas_anchor_weight"),
                        "recency_weight": params_display.get("recency_weight"),
                        "season_decay": params_display.get("season_decay"),
                        "use_stacking": params_display.get("use_stacking"),
                        "correlation_threshold": params_display.get("correlation_threshold"),
                    }
                    st.json(key_params)

                    # Show metrics
                    st.markdown("**Backtest Metrics:**")
                    metrics_display = {
                        k: f"{v:.3f}" if isinstance(v, float) else v
                        for k, v in m.items()
                        if k not in ("folds",) and isinstance(v, (int, float, str))
                    }
                    st.json(metrics_display)

                    # Delete button
                    if st.button(f"Delete", key=f"del_{i}"):
                        delete_saved_model(meta["model_path"])
                        st.rerun()


# ── Tab: Diagnostics ────────────────────────────────────────────────────────

with tab_diagnostics:
    st.header("Diagnostics")

    # --- Team Name Matching ---
    st.subheader("Team Name Matching")

    n_matched = len(name_map)
    n_unmatched = len(unmatched_teams)
    total_teams = n_matched + n_unmatched

    c1, c2, c3 = st.columns(3)
    c1.metric("Matched Teams", n_matched)
    c2.metric("Unmatched Teams", n_unmatched)
    c3.metric("Match Rate", f"{n_matched / max(1, total_teams):.0%}")

    if unmatched_teams:
        st.warning(
            f"{n_unmatched} Vegas team names could not be matched to Torvik data. "
            "These games are excluded from the model."
        )
        with st.expander(f"Show {n_unmatched} unmatched team names"):
            for name in sorted(unmatched_teams):
                st.text(name)
    else:
        st.success("All Vegas team names were matched to Torvik data.")

    st.markdown("---")

    # --- Name map explorer ---
    st.subheader("Name Map Explorer")
    with st.expander("Show full team name mapping"):
        map_df = pd.DataFrame([
            {"Vegas Name": k, "Torvik Name": v}
            for k, v in sorted(name_map.items())
        ])
        st.dataframe(map_df, use_container_width=True, hide_index=True, height=400)

    st.markdown("---")

    # --- Data quality ---
    st.subheader("Data Quality")
    dq1, dq2, dq3 = st.columns(3)
    dq1.metric("Torvik Games Loaded", f"{len(torvik_df):,}")
    dq2.metric("Vegas Lines Loaded", f"{len(vegas_df):,}")
    dq3.metric("Merged & Feature-Ready", f"{len(feature_df):,}")

    # Feature availability
    with st.expander("Feature availability"):
        avail = {}
        for c in feat_cols:
            if c in feature_df.columns:
                avail[c] = f"{feature_df[c].notna().mean():.0%}"
            else:
                avail[c] = "MISSING"
        avail_df = pd.DataFrame([
            {"Feature": k, "Available %": v} for k, v in avail.items()
        ])
        st.dataframe(avail_df, use_container_width=True, hide_index=True, height=400)


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
Total ~ AdjTempo x (Team1_AdjO + Team2_AdjO) / 100
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
- Non-linear tempo x efficiency cross terms

### 4. No Volatility Capture
Two teams can have the same average but very different variances. A volatile
team creates more over/under opportunities. This model tracks **rolling standard
deviations** of scoring, pace, and efficiency.

### 5. No Market Context
KenPom operates in a vacuum. Vegas lines incorporate information from sharps,
injury news, and other signals. This model is **anchored to Vegas** and learns
to predict the **residual** (actual - Vegas), capturing systematic biases the
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
- Opponent quality: rolling strength-of-schedule (with lookahead-free opponent stats)
- Home-court advantage: venue indicators
- Conference context: conference game, major conference, power matchup indicators

### Feature Selection
Highly correlated features (above configurable threshold) are automatically removed
to reduce multicollinearity and improve generalization.

### Ensemble Model with Stacking
Three base models, each with distinct strengths:

| Model | Strength | Role |
|-------|----------|------|
| **XGBoost** | Complex non-linear interactions | Primary predictor |
| **LightGBM** | Fast, handles sparse features well | Secondary predictor |
| **Ridge Regression** | Stable linear baseline | Regularization anchor |

A **stacking meta-learner** (Ridge regression on out-of-fold predictions) learns
the optimal blend of base models, replacing the static weighted average. When
stacking is disabled, falls back to user-configurable ensemble weights.

### Season Weighting
Training samples are weighted by recency: recent seasons receive full weight,
older seasons are exponentially decayed. This helps the model adapt to evolving
playstyles and rule changes.

### Walk-Forward Backtesting
- **No lookahead bias**: model only sees past games at each prediction point
- **Periodic retraining**: model retrains every N games (configurable)
- **Realistic evaluation**: all metrics computed out-of-sample
- **Cross-validation**: 5-fold time-series CV provides confidence intervals

### SHAP Feature Explanations
When available, SHAP (SHapley Additive exPlanations) values show the impact
of each feature on individual predictions, providing more reliable importance
rankings than gain-based metrics.

### Calibration Philosophy
Every key parameter is exposed in the sidebar so you can see how changes
affect backtest results:
- **Presets**: Quick-load Conservative, Default, or Aggressive configurations
- **Recency weight**: Balance recent form vs season-long track record
- **Pace emphasis**: How much to weight matchup-specific pace dynamics
- **Volatility emphasis**: How much scoring variance matters
- **Vegas anchor**: Trust the market vs trust the model
- **Season decay**: How much to discount older seasons
- **Correlation threshold**: Control feature selection aggressiveness
- **Stacking**: Enable/disable the meta-learner
- **Tree hyperparameters**: Control model complexity and overfitting
    """)

    st.markdown("---")
    st.caption(
        "Data sources: Barttorvik (game-level advanced stats), "
        "DraftKings/FanDuel (closing lines via user-provided dataset)"
    )
