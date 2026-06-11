"""
Inference layer that loads the trained models and produces predictions.

Combines ML output with the rule-based scoring, growth analysis, the Smart
Decision Engine, and recommendations into a single evaluation result.
"""
import json

import joblib
import numpy as np

from config import Config
from ml import scoring
from ml.train_models import FEATURE_COLUMNS, ensure_models

_CLASSIFIER = None
_REGRESSOR = None
_ENCODER = None
_METADATA = None


def _load():
    global _CLASSIFIER, _REGRESSOR, _ENCODER, _METADATA
    if _CLASSIFIER is None:
        ensure_models()
        _CLASSIFIER = joblib.load(Config.APPROVAL_MODEL_PATH)
        _REGRESSOR = joblib.load(Config.CREDIT_MODEL_PATH)
        _ENCODER = joblib.load(Config.ENCODER_PATH)
        try:
            _METADATA = joblib.load(Config.METADATA_PATH)
        except Exception:
            _METADATA = {}
    return _CLASSIFIER, _REGRESSOR, _ENCODER


def get_metadata():
    _load()
    return _METADATA or {}


def _features_from_app(app, encoder):
    emp = app.get("employment_type", "Salaried")
    if emp not in list(encoder.classes_):
        emp = "Salaried"
    emp_encoded = int(encoder.transform([emp])[0])

    feature_map = {
        "monthly_income": float(app["monthly_income"]),
        "monthly_expenses": float(app["monthly_expenses"]),
        "bank_balance": float(app["bank_balance"]),
        "existing_loans": float(app["existing_loans"]),
        "credit_history": int(app["credit_history"]),
        "defaults": int(app["defaults"]),
        "employment_type_encoded": emp_encoded,
        "age": int(app.get("age", 30)),
    }
    return np.array([[feature_map[c] for c in FEATURE_COLUMNS]], dtype=float)


def predict_ml(app):
    """Run both ML models. Returns (ml_prediction, ml_probability, ml_credit_score)."""
    clf, reg, encoder = _load()
    X = _features_from_app(app, encoder)

    pred = int(clf.predict(X)[0])
    try:
        prob = float(clf.predict_proba(X)[0][1])
    except Exception:
        prob = float(pred)

    raw_score = float(reg.predict(X)[0])
    ml_credit_score = round(float(np.clip(raw_score, Config.SCORE_MIN, Config.SCORE_MAX)), 2)

    ml_prediction = "APPROVED" if pred == 1 else "REJECTED"
    return ml_prediction, round(prob, 4), ml_credit_score


def evaluate_application(app, previous_score=None):
    """
    Full evaluation pipeline for a single application.

    Returns a dict with every value needed by the dashboard and the database.
    """
    ml_prediction, ml_prob, ml_credit_score = predict_ml(app)

    formula_score = scoring.formula_credit_score(app)
    final_score = scoring.hybrid_score(ml_credit_score, formula_score)
    dti = scoring.debt_to_income(app)

    growth_info = scoring.growth_analysis(final_score, previous_score)
    decision, reasons = scoring.smart_decision(ml_prediction, final_score, growth_info, app)
    recommendations = scoring.build_recommendations(app, final_score, growth_info, decision)

    return {
        "ml_prediction": ml_prediction,
        "ml_probability": ml_prob,
        "ml_credit_score": ml_credit_score,
        "formula_credit_score": formula_score,
        "final_score": final_score,
        "debt_to_income": dti,
        "growth": growth_info,
        "decision": decision,
        "reasons": reasons,
        "recommendations": recommendations,
    }


def serialize_reasons(reasons):
    return json.dumps(reasons)


def serialize_recommendations(recs):
    return json.dumps(recs)
