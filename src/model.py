"""
model.py
--------
Trains Linear Regression (primary) and Random Forest (comparison) models
on the preprocessed TMDB feature matrix and evaluates both.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS_DIR = "models"
RANDOM_STATE = 42


def _regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R² in log-space."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_evaluate(model_df: pd.DataFrame) -> dict:
    """
    Split data, train both models, evaluate, and save artifacts.

    Args:
        model_df: Feature matrix + 'log_revenue' target from preprocessing.

    Returns:
        Dictionary with keys 'lr' and 'rf', each containing:
            - model, scaler, X_train, X_test, y_train, y_test,
              y_pred, metrics, feature_names
    """
    print("[4/5] Training models...")

    TARGET   = "log_revenue"
    FEATURES = [c for c in model_df.columns if c != TARGET]

    X = model_df[FEATURES].values
    y = model_df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ------------------------------------------------------------------ #
    # 1. Linear Regression  (with StandardScaler)
    # ------------------------------------------------------------------ #
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LinearRegression()),
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    lr_metrics = _regression_metrics(y_test, y_pred_lr)

    # Cross-validation (5-fold R²)
    cv_scores_lr = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring="r2")

    print(f"\n      ── Linear Regression ──────────────────────")
    print(f"         R²   : {lr_metrics['r2']:.4f}")
    print(f"         MAE  : {lr_metrics['mae']:.4f}  (log-scale)")
    print(f"         RMSE : {lr_metrics['rmse']:.4f}  (log-scale)")
    print(f"         CV R² (5-fold): {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")

    # ------------------------------------------------------------------ #
    # 2. Random Forest  (no scaling needed)
    # ------------------------------------------------------------------ #
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_metrics = _regression_metrics(y_test, y_pred_rf)

    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")

    print(f"\n      ── Random Forest ───────────────────────────")
    print(f"         R²   : {rf_metrics['r2']:.4f}")
    print(f"         MAE  : {rf_metrics['mae']:.4f}  (log-scale)")
    print(f"         RMSE : {rf_metrics['rmse']:.4f}  (log-scale)")
    print(f"         CV R² (5-fold): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
    print()

    # ------------------------------------------------------------------ #
    # 3. Save models
    # ------------------------------------------------------------------ #
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lr_pipeline, os.path.join(MODELS_DIR, "linear_regression_pipeline.pkl"))
    joblib.dump(rf,          os.path.join(MODELS_DIR, "random_forest.pkl"))
    print(f"      Models saved to {MODELS_DIR}/")

    return {
        "Linear Regression": {
            "model":         lr_pipeline.named_steps["lr"],
            "pipeline":      lr_pipeline,
            "scaler":        lr_pipeline.named_steps["scaler"],
            "X_train_raw":   X_train,
            "X_test_raw":    X_test,
            "X_train":       lr_pipeline.named_steps["scaler"].transform(X_train),
            "X_test":        lr_pipeline.named_steps["scaler"].transform(X_test),
            "y_train":       y_train,
            "y_test":        y_test,
            "y_pred":        y_pred_lr,
            "metrics":       lr_metrics,
            "feature_names": FEATURES,
            "cv_r2_mean":    float(cv_scores_lr.mean()),
            "cv_r2_std":     float(cv_scores_lr.std()),
        },
        "Random Forest": {
            "model":         rf,
            "pipeline":      None,
            "scaler":        None,
            "X_train_raw":   X_train,
            "X_test_raw":    X_test,
            "X_train":       X_train,
            "X_test":        X_test,
            "y_train":       y_train,
            "y_test":        y_test,
            "y_pred":        y_pred_rf,
            "metrics":       rf_metrics,
            "feature_names": FEATURES,
            "cv_r2_mean":    float(cv_scores_rf.mean()),
            "cv_r2_std":     float(cv_scores_rf.std()),
        },
    }
