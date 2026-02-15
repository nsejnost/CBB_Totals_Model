"""
ML model training, prediction, and backtesting for NCAAB totals.

Model stack:
  - Gradient-boosted trees (XGBoost + LightGBM)
  - Ridge regression as a linear baseline
  - Stacking meta-learner (Ridge on out-of-fold predictions)

The target is *actual game total - Vegas total* (the residual).
By modelling the residual we let Vegas set the baseline and our
model learns systematic biases the market misses.
"""

import os
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Optional: XGBoost / LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except (ImportError, OSError):
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    HAS_LGB = False

# Optional: SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


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

    # --- Ensemble weights (now used as fallback if stacking disabled) ---
    "w_xgb": 0.45,
    "w_lgb": 0.35,
    "w_ridge": 0.20,

    # --- Stacking ---
    "use_stacking": True,

    # --- Season weighting ---
    "season_decay": 0.85,  # Weight multiplier per season back (1.0 = no decay)

    # --- Feature selection ---
    "correlation_threshold": 0.95,  # Drop features with pairwise corr above this

    # --- Backtesting ---
    "min_training_games": 500,  # Minimum games before first prediction
    "retrain_every": 200,       # Retrain every N games
}

# Named parameter presets
PARAM_PRESETS = {
    "Default": {**DEFAULT_PARAMS},
    "Conservative": {
        **DEFAULT_PARAMS,
        "vegas_anchor_weight": 0.75,
        "recency_weight": 0.50,
        "xgb_max_depth": 4,
        "lgb_max_depth": 4,
        "xgb_n_estimators": 200,
        "lgb_n_estimators": 200,
        "correlation_threshold": 0.90,
        "season_decay": 0.90,
    },
    "Aggressive": {
        **DEFAULT_PARAMS,
        "vegas_anchor_weight": 0.35,
        "recency_weight": 0.85,
        "pace_interaction_weight": 1.5,
        "volatility_weight": 1.5,
        "xgb_max_depth": 6,
        "lgb_max_depth": 6,
        "xgb_n_estimators": 500,
        "lgb_n_estimators": 500,
        "season_decay": 0.75,
    },
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
# Feature selection
# ---------------------------------------------------------------------------

def select_features(
    X: pd.DataFrame,
    feat_cols: list[str],
    corr_threshold: float = 0.95,
) -> list[str]:
    """
    Remove highly correlated features to reduce multicollinearity.

    For each pair with |corr| > threshold, drop the one that appears later
    in the feature list (preserving the earlier, presumably more important one).
    """
    Xf = X[feat_cols].copy()
    Xf = Xf.replace([np.inf, -np.inf], np.nan)
    # Only compute correlation on columns with enough non-null data
    valid_cols = [c for c in feat_cols if Xf[c].notna().mean() > 0.3]
    if len(valid_cols) < 2:
        return feat_cols

    corr = Xf[valid_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high = upper.index[upper[col] > corr_threshold].tolist()
        to_drop.update(high)

    selected = [c for c in feat_cols if c not in to_drop]
    return selected


# ---------------------------------------------------------------------------
# Season weighting
# ---------------------------------------------------------------------------

def compute_sample_weights(
    game_dates: pd.Series, decay: float = 0.85
) -> np.ndarray:
    """
    Compute per-sample weights that decay exponentially by season age.

    Recent seasons get weight 1.0, each prior season is multiplied by `decay`.
    """
    if decay >= 1.0:
        return np.ones(len(game_dates))

    dates = pd.to_datetime(game_dates)
    # Approximate season by academic year (Nov-Apr)
    # Use year of the spring semester as the season identifier
    season_id = dates.dt.year + (dates.dt.month >= 8).astype(int)
    max_season = season_id.max()
    seasons_back = max_season - season_id
    weights = decay ** seasons_back.values.astype(float)
    return weights


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
        self.meta_model = None  # Stacking meta-learner
        self.feat_cols: list[str] = []
        self.selected_cols: list[str] = []  # After feature selection
        self.is_fitted = False
        self._medians: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, feat_cols: list[str]):
        """Train all sub-models with optional stacking meta-learner."""
        self.feat_cols = feat_cols

        # Feature selection
        corr_thresh = self.params.get("correlation_threshold", 0.95)
        if corr_thresh < 1.0:
            self.selected_cols = select_features(X, feat_cols, corr_thresh)
        else:
            self.selected_cols = list(feat_cols)

        Xf = X[self.selected_cols].copy()

        # Handle NaN/inf – save medians for prediction-time imputation
        Xf = Xf.replace([np.inf, -np.inf], np.nan)
        self._medians = Xf.median()
        Xf = Xf.fillna(self._medians)

        Xw = _apply_feature_weights(Xf, self.selected_cols, self.params)

        # Season weighting
        sample_weights = None
        decay = self.params.get("season_decay", 1.0)
        if decay < 1.0 and "game_date" in X.columns:
            sample_weights = compute_sample_weights(X["game_date"], decay)

        # Ridge
        Xs = self.scaler.fit_transform(Xw)
        self.ridge = Ridge(alpha=self.params["ridge_alpha"])
        self.ridge.fit(Xs, y, sample_weight=sample_weights)

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
            self.xgb_model.fit(Xw, y, sample_weight=sample_weights)

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
            self.lgb_model.fit(Xw, y, sample_weight=sample_weights)

        # --- Stacking meta-learner ---
        if self.params.get("use_stacking", True) and HAS_XGB and HAS_LGB:
            self._fit_meta_learner(Xw, Xs, y, sample_weights)

        self.is_fitted = True

    def _fit_meta_learner(
        self, Xw: pd.DataFrame, Xs: np.ndarray, y: pd.Series,
        sample_weights: np.ndarray | None,
    ):
        """Train a Ridge meta-learner on out-of-fold base model predictions."""
        n_splits = min(5, max(2, len(y) // 100))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        oof_ridge = np.full(len(y), np.nan)
        oof_xgb = np.full(len(y), np.nan)
        oof_lgb = np.full(len(y), np.nan)

        for train_idx, val_idx in tscv.split(Xw):
            Xt, Xv = Xw.iloc[train_idx], Xw.iloc[val_idx]
            yt = y.iloc[train_idx]
            sw_t = sample_weights[train_idx] if sample_weights is not None else None

            # Ridge OOF
            scaler_fold = StandardScaler()
            Xts = scaler_fold.fit_transform(Xt)
            Xvs = scaler_fold.transform(Xv)
            r = Ridge(alpha=self.params["ridge_alpha"])
            r.fit(Xts, yt, sample_weight=sw_t)
            oof_ridge[val_idx] = r.predict(Xvs)

            # XGBoost OOF
            xm = xgb.XGBRegressor(
                max_depth=int(self.params["xgb_max_depth"]),
                learning_rate=self.params["xgb_learning_rate"],
                n_estimators=int(self.params["xgb_n_estimators"]),
                subsample=self.params["xgb_subsample"],
                colsample_bytree=self.params["xgb_colsample_bytree"],
                reg_alpha=self.params["xgb_reg_alpha"],
                reg_lambda=self.params["xgb_reg_lambda"],
                random_state=42, verbosity=0,
            )
            xm.fit(Xt, yt, sample_weight=sw_t)
            oof_xgb[val_idx] = xm.predict(Xv)

            # LightGBM OOF
            lm = lgb.LGBMRegressor(
                max_depth=int(self.params["lgb_max_depth"]),
                learning_rate=self.params["lgb_learning_rate"],
                n_estimators=int(self.params["lgb_n_estimators"]),
                subsample=self.params["lgb_subsample"],
                colsample_bytree=self.params["lgb_colsample_bytree"],
                reg_alpha=self.params["lgb_reg_alpha"],
                reg_lambda=self.params["lgb_reg_lambda"],
                random_state=42, verbose=-1,
            )
            lm.fit(Xt, yt, sample_weight=sw_t)
            oof_lgb[val_idx] = lm.predict(Xv)

        # Fit meta-learner on valid (non-NaN) OOF predictions
        valid = ~(np.isnan(oof_ridge) | np.isnan(oof_xgb) | np.isnan(oof_lgb))
        if valid.sum() > 20:
            meta_X = np.column_stack([oof_ridge[valid], oof_xgb[valid], oof_lgb[valid]])
            self.meta_model = Ridge(alpha=1.0)
            meta_sw = sample_weights[valid] if sample_weights is not None else None
            self.meta_model.fit(meta_X, y.values[valid], sample_weight=meta_sw)

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the residual (actual - vegas) for each game."""
        Xf = X[self.selected_cols].copy()
        Xf = Xf.replace([np.inf, -np.inf], np.nan)
        # Use training medians for imputation (consistent with fit)
        if self._medians is not None:
            Xf = Xf.fillna(self._medians)
        Xf = Xf.fillna(0)
        Xw = _apply_feature_weights(Xf, self.selected_cols, self.params)

        # Get base model predictions
        Xs = self.scaler.transform(Xw)
        ridge_pred = self.ridge.predict(Xs)

        xgb_pred = ridge_pred.copy()  # fallback
        lgb_pred = ridge_pred.copy()  # fallback

        if HAS_XGB and self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(Xw)

        if HAS_LGB and self.lgb_model is not None:
            lgb_pred = self.lgb_model.predict(Xw)

        # Use stacking meta-learner if available
        if self.meta_model is not None:
            meta_X = np.column_stack([ridge_pred, xgb_pred, lgb_pred])
            preds = self.meta_model.predict(meta_X)
        else:
            # Fallback to weighted average
            preds = np.zeros(len(Xw))
            total_weight = 0.0

            w_ridge = self.params["w_ridge"]
            preds += w_ridge * ridge_pred
            total_weight += w_ridge

            if HAS_XGB and self.xgb_model is not None:
                w_xgb = self.params["w_xgb"]
                preds += w_xgb * xgb_pred
                total_weight += w_xgb

            if HAS_LGB and self.lgb_model is not None:
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
            for i, c in enumerate(self.selected_cols):
                records.append({"feature": c, "importance": imp[i], "model": "XGBoost"})
        if HAS_LGB and self.lgb_model is not None:
            imp = self.lgb_model.feature_importances_
            total = imp.sum() if imp.sum() > 0 else 1
            for i, c in enumerate(self.selected_cols):
                records.append({"feature": c, "importance": imp[i] / total, "model": "LightGBM"})
        return pd.DataFrame(records)

    def compute_shap_values(self, X: pd.DataFrame) -> dict | None:
        """Compute SHAP values for the tree models.

        Returns dict with keys 'xgb' and/or 'lgb', each containing
        a DataFrame of SHAP values with feature names as columns.
        Returns None if SHAP is not available.
        """
        if not HAS_SHAP:
            return None

        Xf = X[self.selected_cols].copy()
        Xf = Xf.replace([np.inf, -np.inf], np.nan)
        if self._medians is not None:
            Xf = Xf.fillna(self._medians)
        Xf = Xf.fillna(0)
        Xw = _apply_feature_weights(Xf, self.selected_cols, self.params)

        result = {}
        # Limit to 500 samples for speed
        sample = Xw.head(500) if len(Xw) > 500 else Xw

        if HAS_XGB and self.xgb_model is not None:
            explainer = shap.TreeExplainer(self.xgb_model)
            sv = explainer.shap_values(sample)
            result["xgb"] = pd.DataFrame(sv, columns=self.selected_cols)

        if HAS_LGB and self.lgb_model is not None:
            explainer = shap.TreeExplainer(self.lgb_model)
            sv = explainer.shap_values(sample)
            result["lgb"] = pd.DataFrame(sv, columns=self.selected_cols)

        return result if result else None


# ---------------------------------------------------------------------------
# Cross-validation for uncertainty estimates
# ---------------------------------------------------------------------------

def cross_validate_model(
    df: pd.DataFrame,
    feat_cols: list[str],
    params: dict | None = None,
    n_splits: int = 5,
) -> dict:
    """
    Time-series cross-validation to estimate model uncertainty.

    Returns dict with per-fold and aggregate metrics.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    df = df.sort_values("game_date").reset_index(drop=True)
    target = df["actual_total"] - df["vegas_total"]

    valid_mask = target.notna()
    df_valid = df[valid_mask].reset_index(drop=True)
    y_valid = target[valid_mask].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(df_valid)):
        if len(train_idx) < 50 or len(val_idx) < 10:
            continue

        model = TotalsModel(params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df_valid.iloc[train_idx], y_valid.iloc[train_idx], feat_cols)

        preds = model.predict_total(df_valid.iloc[val_idx])
        actuals = df_valid.iloc[val_idx]["actual_total"].values
        vegas = df_valid.iloc[val_idx]["vegas_total"].values

        model_mae = np.mean(np.abs(preds - actuals))
        vegas_mae = np.mean(np.abs(vegas - actuals))
        model_rmse = np.sqrt(np.mean((preds - actuals) ** 2))

        fold_results.append({
            "fold": fold_i + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "model_mae": model_mae,
            "vegas_mae": vegas_mae,
            "mae_improvement": vegas_mae - model_mae,
            "model_rmse": model_rmse,
        })

    if not fold_results:
        return {}

    folds_df = pd.DataFrame(fold_results)
    return {
        "folds": folds_df,
        "mean_model_mae": folds_df["model_mae"].mean(),
        "std_model_mae": folds_df["model_mae"].std(),
        "mean_vegas_mae": folds_df["vegas_mae"].mean(),
        "mean_mae_improvement": folds_df["mae_improvement"].mean(),
        "std_mae_improvement": folds_df["mae_improvement"].std(),
        "ci_lower_improvement": folds_df["mae_improvement"].mean() - 1.96 * folds_df["mae_improvement"].std(),
        "ci_upper_improvement": folds_df["mae_improvement"].mean() + 1.96 * folds_df["mae_improvement"].std(),
    }


# ---------------------------------------------------------------------------
# Walk-forward backtesting
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    df: pd.DataFrame,
    feat_cols: list[str],
    params: dict | None = None,
    progress_callback=None,
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

    total_steps = n - min_train
    for i in range(min_train, n):
        # Report progress
        if progress_callback is not None and total_steps > 0:
            progress_callback((i - min_train) / total_steps)

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
# Model persistence
# ---------------------------------------------------------------------------

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def _params_hash(params: dict) -> str:
    """Create a short hash of model parameters for identification."""
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def save_model(model: TotalsModel, metrics: dict, params: dict, tag: str = "") -> str:
    """
    Save a trained model and its metadata to disk.

    Returns the save path.
    """
    if not HAS_JOBLIB:
        raise RuntimeError("joblib is required for model persistence")

    os.makedirs(_MODELS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    phash = _params_hash(params)
    name = f"model_{timestamp}_{phash}"
    if tag:
        name += f"_{tag}"

    model_path = os.path.join(_MODELS_DIR, f"{name}.joblib")
    meta_path = os.path.join(_MODELS_DIR, f"{name}_meta.json")

    joblib.dump(model, model_path)

    # Save metadata (params + metrics)
    meta = {
        "timestamp": timestamp,
        "params": {k: (v if not isinstance(v, np.generic) else v.item())
                   for k, v in params.items()},
        "metrics": {k: (v if not isinstance(v, (np.generic, pd.DataFrame)) else
                        v.item() if isinstance(v, np.generic) else "DataFrame")
                    for k, v in metrics.items()},
        "n_features": len(model.selected_cols),
        "tag": tag,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return model_path


def load_saved_model(path: str) -> TotalsModel:
    """Load a previously saved model."""
    if not HAS_JOBLIB:
        raise RuntimeError("joblib is required for model persistence")
    return joblib.load(path)


def list_saved_models() -> list[dict]:
    """List all saved models with their metadata."""
    if not os.path.isdir(_MODELS_DIR):
        return []

    models = []
    for f in sorted(os.listdir(_MODELS_DIR)):
        if f.endswith("_meta.json"):
            meta_path = os.path.join(_MODELS_DIR, f)
            model_path = meta_path.replace("_meta.json", ".joblib")
            if os.path.exists(model_path):
                with open(meta_path) as mf:
                    meta = json.load(mf)
                meta["model_path"] = model_path
                models.append(meta)

    return models


def delete_saved_model(model_path: str):
    """Delete a saved model and its metadata."""
    if os.path.exists(model_path):
        os.remove(model_path)
    meta_path = model_path.replace(".joblib", "_meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)


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
