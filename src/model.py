from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import OUTPUT_DIR

CANDIDATE_FEATURES = [
    "impervious_mean",
    "elev_mean",
    "slope_mean",
    "population",
    "housing_units",
    "median_income",
    "pct_renter",
    "pct_elderly",
    "pct_no_vehicle",
]


@dataclass
class ModelResults:
    regression_metrics: dict
    classification_metrics: dict
    feature_importance: dict


def _available_features(df: pd.DataFrame) -> list[str]:
    usable = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if not usable:
        raise ValueError("No usable model features were found in the dataset.")
    return usable


def run_baseline_models(df: pd.DataFrame) -> ModelResults:
    model_df = df.copy().replace([np.inf, -np.inf], np.nan)
    features = _available_features(model_df)

    reg_df = model_df.dropna(subset=["claim_rate_per_1000_units"]).copy()
    X_reg = reg_df[features]
    y_reg = reg_df["claim_rate_per_1000_units"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    linear_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    linear_pipe.fit(X_train_r, y_train_r)
    linear_preds = linear_pipe.predict(X_test_r)

    rf_reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
    ])
    rf_reg.fit(X_train_r, y_train_r)
    rf_reg_preds = rf_reg.predict(X_test_r)

    regression_metrics = {
        "linear_regression_r2": float(r2_score(y_test_r, linear_preds)),
        "linear_regression_rmse": float(np.sqrt(mean_squared_error(y_test_r, linear_preds))),
        "random_forest_r2": float(r2_score(y_test_r, rf_reg_preds)),
        "random_forest_rmse": float(np.sqrt(mean_squared_error(y_test_r, rf_reg_preds))),
    }

    clf_df = model_df.dropna(subset=["has_claim"]).copy()
    X_clf = clf_df[features]
    y_clf = clf_df["has_claim"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf,
        y_clf,
        test_size=0.2,
        random_state=42,
        stratify=y_clf if y_clf.nunique() > 1 else None,
    )

    logit = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ])
    logit.fit(X_train_c, y_train_c)
    logit_probs = logit.predict_proba(X_test_c)[:, 1]
    logit_preds = logit.predict(X_test_c)

    rf_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])
    rf_clf.fit(X_train_c, y_train_c)
    rf_probs = rf_clf.predict_proba(X_test_c)[:, 1]
    rf_preds = rf_clf.predict(X_test_c)

    classification_metrics = {
        "logistic_accuracy": float(accuracy_score(y_test_c, logit_preds)),
        "logistic_auc": float(roc_auc_score(y_test_c, logit_probs)) if len(np.unique(y_test_c)) > 1 else float("nan"),
        "rf_classifier_accuracy": float(accuracy_score(y_test_c, rf_preds)),
        "rf_classifier_auc": float(roc_auc_score(y_test_c, rf_probs)) if len(np.unique(y_test_c)) > 1 else float("nan"),
    }

    feature_importance = {
        feature: float(score)
        for feature, score in zip(features, rf_reg.named_steps["model"].feature_importances_)
    }

    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "features_used": features,
                "regression_metrics": regression_metrics,
                "classification_metrics": classification_metrics,
                "feature_importance": feature_importance,
            },
            f,
            indent=2,
        )

    return ModelResults(regression_metrics, classification_metrics, feature_importance)


def run_impervious_scenario(df: pd.DataFrame, reduction_fraction: float = 0.10) -> pd.DataFrame:
    model_df = df.copy().replace([np.inf, -np.inf], np.nan)
    features = _available_features(model_df)
    reg_df = model_df.dropna(subset=["claim_rate_per_1000_units"]).copy()

    if "impervious_mean" not in reg_df.columns:
        raise ValueError("Impervious scenario cannot run because 'impervious_mean' is missing.")

    X = reg_df[features]
    y = reg_df["claim_rate_per_1000_units"]

    rf_reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
    ])
    rf_reg.fit(X, y)

    baseline_pred = rf_reg.predict(X)
    scenario_df = reg_df.copy()
    scenario_df["impervious_mean"] = scenario_df["impervious_mean"] * (1 - reduction_fraction)
    scenario_pred = rf_reg.predict(scenario_df[features])

    results = reg_df[["GEOID", "claim_rate_per_1000_units", "impervious_mean"]].copy()
    results["predicted_baseline_claim_rate"] = baseline_pred
    results["predicted_scenario_claim_rate"] = scenario_pred
    results["predicted_change"] = results["predicted_scenario_claim_rate"] - results["predicted_baseline_claim_rate"]
    results["percent_change"] = np.where(
        results["predicted_baseline_claim_rate"] != 0,
        results["predicted_change"] / results["predicted_baseline_claim_rate"],
        np.nan,
    )
    results.to_csv(OUTPUT_DIR / "impervious_reduction_scenario.csv", index=False)
    return results
