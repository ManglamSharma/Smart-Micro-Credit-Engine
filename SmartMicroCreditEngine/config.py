"""
Application configuration for the Smart Micro-Credit Approval Engine.
"""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Security
    SECRET_KEY = os.environ.get("SECRET_KEY", "fiserv-hackathon-smart-microcredit-dev-key-2026")

    # Database
    DATABASE_PATH = os.path.join(BASE_DIR, "database", "microcredit.db")

    # ML models
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    APPROVAL_MODEL_PATH = os.path.join(MODELS_DIR, "approval_model.joblib")
    CREDIT_MODEL_PATH = os.path.join(MODELS_DIR, "credit_score_model.joblib")
    ENCODER_PATH = os.path.join(MODELS_DIR, "employment_encoder.joblib")
    METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.joblib")

    # Session
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 8  # 8 hours

    # Domain constants
    SCORE_MIN = 300
    SCORE_MAX = 900

    # Employment types (must stay consistent across training + scoring)
    EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Business", "Freelancer", "Unemployed"]
