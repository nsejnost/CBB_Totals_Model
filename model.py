"""
ML model training, prediction, and backtesting for NCAAB totals.

Model stack:
  - Gradient-boosted trees (XGBoost + LightGBM)
  - Ridge regression as a linear baseline
  - Stacked ensemble that blends all three

The target is *actual game total – Vegas total* (the residual).
By modelling the residual we let Vegas set the baseline and our
model learns systematic biases the market misses.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Optional: XGBoost / LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ---------------------------------------------------------------------------
# Calibration parameters (user-tunable via the Streamlit UI)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # --- Feature weighting ---
    "recency_weight": 0.70,        # How much to weight recent (EMA) vs season-long
    "pace_interaction_weight": 1.0, # Multiplier on pace-interaction features
    "volatility_weight": 1.0,      # Multiplier on volatility features
    "vegas_anchor_weight": 0.60,   # Final blend: weight on Vegas line vs model

    # --- Model hyperparameters ---
    "xgb_max_depth": 5,
    "xgb_learning_rate": 0.05,
    "xgb_n_estimators": 300,
    "xgb_subsample": 0.8,
    "xgb_colsample_bytree": 0.8,
    "xgb_reg_alpha": 0.1,
    "xgb_reg_lambda": 1.0,

    "lgb_max_depth": 5,
    "lgb_learning_rate": 0.05,
    "lgb_n_estimators": 300,
    "lgb_subsample": 0.8,
    "lgb_colsample_bytree": 0.8,
    "lgb_reg_alpha": 0.1,
    "lgb_reg_lambda": 1.0,

    "ridge_alpha": 10.0,

    # --- Ensemble weights ---
    "w_xgb": 0.45,
    "w_lgb": 0.35,
    "w_ridge": 0.20,

    # --- Backtesting ---
    "min_training_games": 500,  # Minimum games before first prediction
    "retrain_every": 200,       # Retrain every N games
}


def _apply_feature_weights(
    X: pd.DataFrame, feat_cols: list[str], params: dict
) -> pd.DataFrame:
    """Apply calibration weights to feature groups."""
    X = X.copy()
    recency_w = params.get("recency_weight", 0.7)
    pace_w = params.get("pace_interaction_weight", 1.0)
    vol_w = params.get("volatility_weight", 1.0)

    for c in feat_cols:
        if c in X.columns:
            # Down-weight season features, up-weight EMA features
            if "_season" in c:
                X[c] *= (1.0 - recency_w)
            elif "_ema" in c:
                X[c] *= recency_w

            # Pace interaction weight
            if "tempo_" in c or "pace_" in c:
                X[c] *= pace_w

            # Volatility weight
            if "_std" in c or "vol" in c:
                X[c] *= vol_w

    return X


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class TotalsModel:
    """Ensemble model that predicts game total residual vs Vegas."""

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.scaler = StandardScaler()
        self.ridge = None
        self.xgb_model = None
        self.lgb_model = None
        self.feat_cols: list[str] = []
        self.is_fitted = False
        self._medians: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, feat_cols: list[str]):
        """Train all sub-models."""
        self.feat_cols = feat_cols
        Xf = X[feat_cols].copy()

        # Handle NaN/inf – save medians for prediction-time imputation
        Xf = Xf.replace([np.inf, -np.inf], np.nan)
        self._medians = Xf.median()
        Xf = Xf.fillna(self._medians)

        Xw = _apply_feature_weights(Xf, feat_cols, self.params)

        # Ridge
        Xs = self.scaler.fit_transform(Xw)
        self.ridge = Ridge(alpha=self.params["ridge_alpha"])
        self.ridge.fit(Xs, y)

        # XGBoost
        if HAS_XGB:
            self.xgb_model = xgb.XGBRegressor(
                max_depth=int(self.params["xgb_max_depth"]),
                learning_rate=self.params["xgb_learning_rate"],
                n_estimators=int(self.params["xgb_n_estimators"]),
                subsample=self.params["xgb_subsample"],
                colsample_bytree=self.params["xgb_colsample_bytree"],
                reg_alpha=self.params["xgb_reg_alpha"],
                reg_lambda=self.params["xgb_reg_lambda"],
                random_state=42,
                verbosity=0,
            )
            self.xgb_model.fit(Xw, y)

        # LightGBM
        if HAS_LGB:
            self.lgb_model = lgb.LGBMRegressor(
                max_depth=int(self.params["lgb_max_depth"]),
                learning_rate=self.params["lgb_learning_rate"],
                n_estimators=int(self.params["lgb_n_estimators"]),
                subsample=self.params["lgb_subsample"],
                colsample_bytree=self.params["lgb_colsample_bytree"],
                reg_alpha=self.params["lgb_reg_alpha"],
                reg_lambda=self.params["lgb_reg_lambda"],
                random_state=42,
                verbose=-1,
            )
            self.lgb_model.fit(Xw, y)

        self.is_fitted = True

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the residual (actual - vegas) for each game."""
        Xf = X[self.feat_cols].copy()
        Xf = Xf.replace([np.inf, -np.inf], np.nan)
        # Use training medians for imputation (consistent with fit)
        if self._medians is not None:
            Xf = Xf.fillna(self._medians)
        Xf = Xf.fillna(0)
        Xw = _apply_feature_weights(Xf, self.feat_cols, self.params)

        preds = np.zeros(len(Xw))
        total_weight = 0.0

        # Ridge
        Xs = self.scaler.transform(Xw)
        ridge_pred = self.ridge.predict(Xs)
        w_ridge = self.params["w_ridge"]
        preds += w_ridge * ridge_pred
        total_weight += w_ridge

        # XGBoost
        if HAS_XGB and self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(Xw)
            w_xgb = self.params["w_xgb"]
            preds += w_xgb * xgb_pred
            total_weight += w_xgb

        # LightGBM
        if HAS_LGB and self.lgb_model is not None:
            lgb_pred = self.lgb_model.predict(Xw)
            w_lgb = self.params["w_lgb"]
            preds += w_lgb * lgb_pred
            total_weight += w_lgb

        if total_weight > 0:
            preds /= total_weight

        return preds

    def predict_total(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the game total, blending model prediction with Vegas line.

        Final = vegas_anchor_weight * Vegas + (1 - anchor) * model_raw_total
        """
        residual = self.predict_residual(X)
        model_total = X["vegas_total"].values + residual
        anchor = self.params["vegas_anchor_weight"]
        return anchor * X["vegas_total"].values + (1 - anchor) * model_total

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances from tree models."""
        records = []
        if HAS_XGB and self.xgb_model is not None:
            imp = self.xgb_model.feature_importances_
            for i, c in enumerate(self.feat_cols):
                records.append({"feature": c, "importance": imp[i], "model": "XGBoost"})
        if HAS_LGB and self.lgb_model is not None:
            imp = self.lgb_model.feature_importances_
            total = imp.sum() if imp.sum() > 0 else 1
            for i, c in enumerate(self.feat_cols):
                records.append({"feature": c, "importance": imp[i] / total, "model": "LightGBM"})
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Walk-forward backtesting
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    df: pd.DataFrame,
    feat_cols: list[str],
    params: dict | None = None,
) -> pd.DataFrame:
    """
    Time-ordered walk-forward backtest.

    Trains on games [0..i], predicts game i+1, slides forward.
    Retrains every `retrain_every` games to keep it tractable.

    Returns a DataFrame of predictions aligned with game rows.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    min_train = int(params["min_training_games"])
    retrain_every = int(params["retrain_every"])

    # Sort by date
    df = df.sort_values("game_date").reset_index(drop=True)
    target = df["actual_total"] - df["vegas_total"]  # residual

    n = len(df)
    preds = np.full(n, np.nan)
    model = None
    last_train_idx = -1

    for i in range(min_train, n):
        # Retrain periodically
        if model is None or (i - last_train_idx) >= retrain_every:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = TotalsModel(params)
                train_X = df.iloc[:i]
                train_y = target.iloc[:i]
                # Drop rows with NaN target; allow some NaN features (model imputes)
                mask = train_y.notna()
                if mask.sum() < 50:
                    continue
                model.fit(train_X[mask], train_y[mask], feat_cols)
                last_train_idx = i

        # Predict (model handles NaN via imputation)
        if model is not None and model.is_fitted:
            row = df.iloc[[i]]
            preds[i] = model.predict_total(row)[0]

    df = df.copy()
    df["model_total"] = preds
    df["model_residual"] = preds - df["vegas_total"]
    df["vegas_error"] = df["vegas_total"] - df["actual_total"]
    df["model_error"] = df["model_total"] - df["actual_total"]

    return df


# ---------------------------------------------------------------------------
# Backtest metrics
# ---------------------------------------------------------------------------

def train_full_model(
    df: pd.DataFrame,
    feat_cols: list[str],
    params: dict | None = None,
) -> TotalsModel:
    """
    Train a TotalsModel on the *entire* historical dataset.

    Use this for making forward-looking predictions on upcoming games
    (as opposed to walk-forward backtesting which is for evaluation).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    target = df["actual_total"] - df["vegas_total"]
    mask = target.notna()
    if mask.sum() < 50:
        raise ValueError("Not enough training data to fit model")

    model = TotalsModel(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df[mask], target[mask], feat_cols)
    return model


def predict_upcoming(
    model: TotalsModel,
    upcoming_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply a fitted model to upcoming games and compute edges vs Vegas.

    Returns the upcoming_features DataFrame with added columns:
        model_total, model_edge, bet_side
    """
    df = upcoming_features.copy()
    df["model_total"] = model.predict_total(df)
    df["model_edge"] = df["model_total"] - df["vegas_total"]
    df["bet_side"] = np.where(df["model_edge"] > 0, "OVER", "UNDER")
    df["abs_edge"] = df["model_edge"].abs()
    return df


def compute_backtest_metrics(bt: pd.DataFrame) -> dict:
    """Compute summary metrics from backtest results."""
    valid = bt.dropna(subset=["model_total", "actual_total"])
    if len(valid) == 0:
        return {}

    vegas_err = valid["vegas_total"] - valid["actual_total"]
    model_err = valid["model_total"] - valid["actual_total"]

    metrics = {
        "n_games": len(valid),
        "vegas_mae": np.mean(np.abs(vegas_err)),
        "model_mae": np.mean(np.abs(model_err)),
        "vegas_rmse": np.sqrt(np.mean(vegas_err ** 2)),
        "model_rmse": np.sqrt(np.mean(model_err ** 2)),
        "vegas_bias": np.mean(vegas_err),
        "model_bias": np.mean(model_err),
        "mae_improvement": np.mean(np.abs(vegas_err)) - np.mean(np.abs(model_err)),
        "rmse_improvement": (
            np.sqrt(np.mean(vegas_err ** 2)) - np.sqrt(np.mean(model_err ** 2))
        ),
    }

    # Directional accuracy: when model says over/under vs Vegas,
    # how often is it right?
    model_says_over = valid["model_total"] > valid["vegas_total"]
    actual_over = valid["actual_total"] > valid["vegas_total"]
    agree = model_says_over == actual_over
    disagree_mask = valid["model_total"] != valid["vegas_total"]
    if disagree_mask.sum() > 0:
        metrics["directional_accuracy"] = agree[disagree_mask].mean()
        metrics["n_disagree"] = int(disagree_mask.sum())
    else:
        metrics["directional_accuracy"] = 0.5
        metrics["n_disagree"] = 0

    # Betting simulation: flat-bet $100 on model's side vs Vegas line
    # when model disagrees by >= 1 point
    edge_mask = (valid["model_total"] - valid["vegas_total"]).abs() >= 1.0
    if edge_mask.sum() > 0:
        bets = valid[edge_mask].copy()
        bets["bet_over"] = bets["model_total"] > bets["vegas_total"]
        bets["won"] = (
            (bets["bet_over"] & (bets["actual_total"] > bets["vegas_total"])) |
            (~bets["bet_over"] & (bets["actual_total"] < bets["vegas_total"]))
        )
        bets["push"] = bets["actual_total"] == bets["vegas_total"]
        n_bets = len(bets)
        n_wins = bets["won"].sum()
        n_push = bets["push"].sum()
        # Standard -110 juice
        profit = n_wins * 100 - (n_bets - n_wins - n_push) * 110
        metrics["n_bets"] = n_bets
        metrics["bet_wins"] = int(n_wins)
        metrics["bet_pushes"] = int(n_push)
        metrics["bet_win_pct"] = n_wins / max(1, n_bets - n_push)
        metrics["bet_profit"] = profit
        metrics["bet_roi"] = profit / (n_bets * 110) if n_bets > 0 else 0

    # Edge-stratified: results by model edge size
    for edge_min in [1.5, 2.0, 3.0, 4.0, 5.0]:
        em = (valid["model_total"] - valid["vegas_total"]).abs() >= edge_min
        if em.sum() >= 10:
            sub = valid[em].copy()
            sub["bet_over"] = sub["model_total"] > sub["vegas_total"]
            sub["won"] = (
                (sub["bet_over"] & (sub["actual_total"] > sub["vegas_total"])) |
                (~sub["bet_over"] & (sub["actual_total"] < sub["vegas_total"]))
            )
            sub["push"] = sub["actual_total"] == sub["vegas_total"]
            nw = sub["won"].sum()
            nb = len(sub)
            np_ = sub["push"].sum()
            profit = nw * 100 - (nb - nw - np_) * 110
            tag = str(edge_min).replace(".", "p")
            metrics[f"edge{tag}_n"] = nb
            metrics[f"edge{tag}_winpct"] = nw / max(1, nb - np_)
            metrics[f"edge{tag}_profit"] = profit
            metrics[f"edge{tag}_roi"] = profit / (nb * 110) if nb > 0 else 0

    return metrics
