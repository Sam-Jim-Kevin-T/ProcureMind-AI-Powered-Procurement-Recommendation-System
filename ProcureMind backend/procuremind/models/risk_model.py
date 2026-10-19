"""AI models for vendor risk.

Three models, all trained on the vendor profile features:

1. RISK CLASSIFIER  (ExtraTreesClassifier)  -> Low / Moderate / High Risk
   The classifier specified in the project summary (section 8).

2. RISK REGRESSOR   (GradientBoostingRegressor) -> continuous risk score 0-1
   The regression-based model requested for the backend. A continuous risk
   score is more useful for ranking than a 3-class label because it breaks
   ties inside the same risk band.

3. DELAY REGRESSOR  (RandomForestRegressor) -> expected delivery deviation
   (days). Feeds the Delivery Reliability component with a model-predicted
   delay instead of only historical averages.

IMPORTANT PROTOTYPE LIMITATION (must be kept in the report):
The training table has only ~100 vendor profiles and the labels were
themselves derived from structured/synthetic data. Cross-validated metrics
here are prototype metrics, not production accuracy claims.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from .. import config


# ---------------------------------------------------------------- utils
def _feature_frame(profiles: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in config.RISK_FEATURES if c not in profiles.columns]
    if missing:
        raise ValueError(f"Vendor profiles missing feature columns: {missing}")
    return profiles[config.RISK_FEATURES].apply(pd.to_numeric, errors="coerce")


def _pipe(estimator) -> Pipeline:
    # tree ensembles need no scaling, but do need NaN handling
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", estimator)])


# ------------------------------------------------------------ training
def train_risk_classifier(profiles: pd.DataFrame, save: bool = True):
    """ExtraTrees classifier for Low/Moderate/High risk with CV evaluation."""
    X = _feature_frame(profiles)
    y = profiles[config.RISK_LABEL].astype(str)

    pipe = _pipe(ExtraTreesClassifier(
        n_estimators=400, random_state=42, class_weight="balanced"))

    # honest evaluation on a small dataset: stratified 5-fold CV predictions
    n_splits = min(5, y.value_counts().min())
    metrics = {}
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_hat = cross_val_predict(pipe, X, y, cv=cv)
        metrics = {
            "cv_accuracy": accuracy_score(y, y_hat),
            "cv_macro_f1": f1_score(y, y_hat, average="macro"),
            "report": classification_report(y, y_hat, zero_division=0),
        }

    pipe.fit(X, y)
    if save:
        joblib.dump(pipe, config.RISK_CLASSIFIER_PATH)
    return pipe, metrics


def train_risk_regressor(profiles: pd.DataFrame, save: bool = True):
    """Gradient-boosted regressor for the continuous risk score (0-1)."""
    X = _feature_frame(profiles)
    y = pd.to_numeric(profiles[config.RISK_SCORE], errors="coerce")
    mask = y.notna()
    X, y = X[mask], y[mask]

    pipe = _pipe(GradientBoostingRegressor(random_state=42))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_hat = cross_val_predict(pipe, X, y, cv=cv)
    metrics = {
        "cv_mae": mean_absolute_error(y, y_hat),
        "cv_r2": r2_score(y, y_hat),
    }

    pipe.fit(X, y)
    if save:
        joblib.dump(pipe, config.RISK_REGRESSOR_PATH)
    return pipe, metrics


def train_delay_regressor(profiles: pd.DataFrame, save: bool = True):
    """RandomForest regressor for expected delivery deviation (days)."""
    feats = [f for f in config.RISK_FEATURES if f != config.DELAY_TARGET]
    X = profiles[feats].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(profiles[config.DELAY_TARGET], errors="coerce")
    mask = y.notna()
    X, y = X[mask], y[mask]

    pipe = _pipe(RandomForestRegressor(n_estimators=400, random_state=42))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_hat = cross_val_predict(pipe, X, y, cv=cv)
    metrics = {
        "cv_mae_days": mean_absolute_error(y, y_hat),
        "cv_r2": r2_score(y, y_hat),
    }

    pipe.fit(X, y)
    if save:
        joblib.dump(pipe, config.DELAY_REGRESSOR_PATH)
    return pipe, metrics


def train_all(profiles: pd.DataFrame) -> dict:
    """Train all three models; return the CV metric summary."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _, m_clf = train_risk_classifier(profiles)
    _, m_reg = train_risk_regressor(profiles)
    _, m_dly = train_delay_regressor(profiles)
    return {"risk_classifier": m_clf,
            "risk_regressor": m_reg,
            "delay_regressor": m_dly}


# ----------------------------------------------------------- inference
def load_models() -> dict:
    return {
        "risk_classifier": joblib.load(config.RISK_CLASSIFIER_PATH),
        "risk_regressor": joblib.load(config.RISK_REGRESSOR_PATH),
        "delay_regressor": joblib.load(config.DELAY_REGRESSOR_PATH),
    }


def score_to_risk_level(score: float) -> str:
    """Map continuous score (0-1) to a human-readable risk label."""
    low_max = float(config.RISK_LEVEL_THRESHOLDS["low_max"])
    moderate_max = float(config.RISK_LEVEL_THRESHOLDS["moderate_max"])
    if score <= low_max:
        return "Low Risk"
    if score <= moderate_max:
        return "Moderate Risk"
    return "High Risk"


def predict_vendor_risk(profiles: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """Attach AI predictions to vendor profiles.

    Adds: ai_risk_level, ai_risk_score (0-1), ai_expected_delay_days.
    ai_risk_level is derived from ai_risk_score thresholds so labels and
    percentages stay consistent in UI and filtering.
    """
    models = models or load_models()
    X = _feature_frame(profiles)
    X_delay = profiles[[f for f in config.RISK_FEATURES
                        if f != config.DELAY_TARGET]].apply(
        pd.to_numeric, errors="coerce")

    out = profiles.copy()
    out["ai_risk_level_classifier"] = models["risk_classifier"].predict(X)
    out["ai_risk_score"] = np.clip(models["risk_regressor"].predict(X), 0, 1)
    out["ai_risk_level"] = out["ai_risk_score"].map(score_to_risk_level)
    out["ai_expected_delay_days"] = models["delay_regressor"].predict(X_delay)
    return out


def feature_importances(profiles: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """Importance table for report/dashboard explainability."""
    models = models or load_models()
    clf = models["risk_classifier"].named_steps["model"]
    return (pd.DataFrame({"feature": config.RISK_FEATURES,
                          "importance": clf.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
