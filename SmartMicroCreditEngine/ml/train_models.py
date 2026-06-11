"""
Train and persist the two machine learning models.

Model 1: RandomForestClassifier  -> Loan approval (APPROVED / REJECTED)
Model 2: RandomForestRegressor    -> Credit score (300 - 900)

Both models share the same feature set:
    income, expenses, bank_balance, existing_loans,
    credit_history, defaults, employment_type (encoded), age

Artifacts are saved to /models via joblib.
"""
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import Config
from ml.dataset import generate_dataset

FEATURE_COLUMNS = [
    "monthly_income",
    "monthly_expenses",
    "bank_balance",
    "existing_loans",
    "credit_history",
    "defaults",
    "employment_type_encoded",
    "age",
]


def _prepare(df, encoder=None, fit=False):
    df = df.copy()
    if fit:
        encoder = LabelEncoder()
        encoder.fit(Config.EMPLOYMENT_TYPES)
    df["employment_type_encoded"] = encoder.transform(df["employment_type"])
    X = df[FEATURE_COLUMNS].values
    return X, encoder


def train_and_save(n_samples=800, seed=42, verbose=True):
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    df = generate_dataset(n_samples=n_samples, seed=seed)

    # Encode employment type with a fixed vocabulary so inference is stable
    X, encoder = _prepare(df, fit=True)
    y_class = df["approved"].values
    y_reg = df["credit_score"].values

    Xtr, Xte, ytr_c, yte_c, ytr_r, yte_r = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=seed, stratify=y_class
    )

    # ---- Model 1: Loan approval classifier ----
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=seed,
        class_weight="balanced",
    )
    clf.fit(Xtr, ytr_c)
    acc = accuracy_score(yte_c, clf.predict(Xte))

    # ---- Model 2: Credit score regressor ----
    reg = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        random_state=seed,
    )
    reg.fit(Xtr, ytr_r)
    preds_r = reg.predict(Xte)
    mae = mean_absolute_error(yte_r, preds_r)
    r2 = r2_score(yte_r, preds_r)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "n_samples": n_samples,
        "classifier_accuracy": round(float(acc), 4),
        "regressor_mae": round(float(mae), 2),
        "regressor_r2": round(float(r2), 4),
        "approval_rate": round(float(df["approved"].mean()), 4),
    }

    joblib.dump(clf, Config.APPROVAL_MODEL_PATH)
    joblib.dump(reg, Config.CREDIT_MODEL_PATH)
    joblib.dump(encoder, Config.ENCODER_PATH)
    joblib.dump(metadata, Config.METADATA_PATH)

    if verbose:
        print("=" * 55)
        print(" Smart Micro-Credit Approval Engine - Model Training")
        print("=" * 55)
        print(f" Training samples       : {n_samples}")
        print(f" Approval rate          : {metadata['approval_rate'] * 100:.1f}%")
        print(f" Classifier accuracy    : {acc * 100:.2f}%")
        print(f" Regressor MAE (score)  : {mae:.2f}")
        print(f" Regressor R^2          : {r2:.4f}")
        print(f" Models saved to        : {Config.MODELS_DIR}")
        print("=" * 55)

    return metadata


def models_exist():
    return all(
        os.path.exists(p)
        for p in (
            Config.APPROVAL_MODEL_PATH,
            Config.CREDIT_MODEL_PATH,
            Config.ENCODER_PATH,
        )
    )


def ensure_models():
    """Train models on first run if they are not already present."""
    if not models_exist():
        train_and_save()
    return True


if __name__ == "__main__":
    train_and_save()
