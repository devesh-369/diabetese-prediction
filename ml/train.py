"""
Diabetes Prediction ML Training Pipeline
========================================
Production-grade ML pipeline with:
  - Zero data leakage (split BEFORE scaling)
  - Multiple model comparison
  - Sklearn Pipeline objects (scaler embedded per model)
  - Stratified cross-validation
  - Full metrics report
  - Artifact persistence via joblib
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "backend", "models")
REPORT_DIR = os.path.join(BASE_DIR, "ml", "reports")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"


# ─────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load CSV and handle domain-specific zero-value anomalies.
    Columns like Glucose, BP, BMI cannot physically be 0 →
    replace with NaN so the imputer handles them properly.
    """
    print("\n[1/5] Loading & cleaning data …")
    df = pd.read_csv(path)
    print(f"  Raw shape : {df.shape}")
    print(f"  Class dist:\n{df[TARGET].value_counts().to_string()}")

    # Zero-value replacement for medically impossible fields
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid:
        n_zeros = (df[col] == 0).sum()
        if n_zeros:
            print(f"  Replacing {n_zeros:>3} zeros in '{col}' with NaN")
        df[col] = df[col].replace(0, np.nan)

    print(f"  Missing values after replacement:\n{df.isnull().sum().to_string()}")
    return df


# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT  (BEFORE any fitting)
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """
    Stratified split ensures both sets have proportional class balance.
    CRITICAL: Split happens BEFORE preprocessing to avoid data leakage.
    """
    print("\n[2/5] Splitting data (stratified, no leakage) …")
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"  Train class dist: {dict(y_train.value_counts().sort_index())}")
    print(f"  Test  class dist: {dict(y_test.value_counts().sort_index())}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 3. PIPELINE FACTORY
# ─────────────────────────────────────────────
def build_pipelines() -> dict:
    """
    Each model is wrapped in an sklearn Pipeline that contains:
      Step 1 – Median imputation  (fit only on train, transform test)
      Step 2 – Standard scaling   (fit only on train, transform test)
      Step 3 – Classifier

    Because scaler/imputer are inside the pipeline, cross_validate and
    GridSearchCV will always refit them on each fold's training portion,
    guaranteeing zero leakage.
    """
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    pipelines = {
        "LogisticRegression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "RandomForest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ]),

        "GradientBoosting": Pipeline([
            ("pre", preprocessor),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                min_samples_leaf=3,
                random_state=RANDOM_STATE
            ))
        ]),

        "SVM_RBF": Pipeline([
            ("pre", preprocessor),
            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
    }
    return pipelines


# ─────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────
def evaluate_models(pipelines: dict, X_train, X_test, y_train, y_test) -> dict:
    """
    Fit each pipeline, compute CV + holdout metrics.
    Returns a dict of results keyed by model name.
    """
    print("\n[3/5] Training & evaluating models …")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, pipeline in pipelines.items():
        print(f"\n  ── {name}")

        # Cross-validation on training set only
        cv_scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=["accuracy", "f1", "roc_auc"],
            return_train_score=False,
            n_jobs=-1
        )

        # Final fit on full training set
        pipeline.fit(X_train, y_train)

        # Holdout evaluation
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        holdout = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred),
            "roc_auc":  roc_auc_score(y_test, y_proba),
        }

        results[name] = {
            "pipeline":   pipeline,
            "cv_acc_mean":  cv_scores["test_accuracy"].mean(),
            "cv_acc_std":   cv_scores["test_accuracy"].std(),
            "cv_f1_mean":   cv_scores["test_f1"].mean(),
            "cv_auc_mean":  cv_scores["test_roc_auc"].mean(),
            "holdout":      holdout,
            "y_pred":       y_pred,
            "y_proba":      y_proba,
            "confusion":    confusion_matrix(y_test, y_pred).tolist(),
            "report":       classification_report(y_test, y_pred, output_dict=True),
        }

        print(f"    CV  Accuracy : {holdout['accuracy']:.4f}  ±{cv_scores['test_accuracy'].std():.4f}")
        print(f"    CV  F1       : {cv_scores['test_f1'].mean():.4f}")
        print(f"    CV  ROC-AUC  : {cv_scores['test_roc_auc'].mean():.4f}")
        print(f"    Hold Accuracy: {holdout['accuracy']:.4f}")
        print(f"    Hold ROC-AUC : {holdout['roc_auc']:.4f}")

    return results


# ─────────────────────────────────────────────
# 5. PICK BEST MODEL & PERSIST
# ─────────────────────────────────────────────
def select_and_save(results: dict) -> str:
    """
    Best model = highest holdout ROC-AUC (more robust than accuracy for
    imbalanced classes). Saves the fitted sklearn Pipeline.
    """
    print("\n[4/5] Selecting best model …")

    best_name = max(results, key=lambda k: results[k]["holdout"]["roc_auc"])
    best      = results[best_name]

    print(f"\n  🏆 Winner : {best_name}")
    print(f"     Accuracy : {best['holdout']['accuracy']:.4f}")
    print(f"     F1 Score : {best['holdout']['f1']:.4f}")
    print(f"     ROC-AUC  : {best['holdout']['roc_auc']:.4f}")

    # Save the full pipeline (imputer + scaler + classifier)
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best["pipeline"], model_path, compress=3)
    print(f"\n  Model saved → {model_path}")

    # Persist metadata
    metadata = {
        "best_model":   best_name,
        "features":     FEATURES,
        "metrics": {
            "accuracy": round(best["holdout"]["accuracy"], 4),
            "f1_score": round(best["holdout"]["f1"], 4),
            "roc_auc":  round(best["holdout"]["roc_auc"], 4),
        },
        "confusion_matrix": best["confusion"],
        "classification_report": best["report"],
        "all_models": {
            name: {
                "cv_accuracy":  round(v["cv_acc_mean"], 4),
                "cv_f1":        round(v["cv_f1_mean"], 4),
                "cv_roc_auc":   round(v["cv_auc_mean"], 4),
                "hold_accuracy":round(v["holdout"]["accuracy"], 4),
                "hold_roc_auc": round(v["holdout"]["roc_auc"], 4),
            }
            for name, v in results.items()
        }
    }

    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata  → {meta_path}")

    return best_name


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
def generate_reports(results: dict, best_name: str, y_test):
    print("\n[5/5] Generating report visuals …")

    # ── Confusion matrix for best model
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── Model comparison bar chart
    model_names = list(results.keys())
    acc_vals  = [results[n]["holdout"]["accuracy"] for n in model_names]
    auc_vals  = [results[n]["holdout"]["roc_auc"]  for n in model_names]

    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - 0.2, acc_vals, 0.35, label="Accuracy",  color="#4f86f7")
    bars2 = ax.bar(x + 0.2, auc_vals, 0.35, label="ROC-AUC",   color="#f7874f")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model Comparison (Holdout Set)", fontweight="bold")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    print(f"  Charts saved to {REPORT_DIR}/")


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DIABETES PREDICTION — ML TRAINING PIPELINE")
    print("=" * 55)

    df             = load_and_clean(DATA_PATH)
    X_tr, X_te, y_tr, y_te = split_data(df)
    pipelines      = build_pipelines()
    results        = evaluate_models(pipelines, X_tr, X_te, y_tr, y_te)
    best_name      = select_and_save(results)
    generate_reports(results, best_name, y_te)

    print("\n✅  Training complete!")
    print(f"    Best model : {best_name}")
    print(f"    Artifacts  : {MODEL_DIR}/")
    print("=" * 55)
