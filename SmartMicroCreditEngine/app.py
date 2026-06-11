"""
Smart Micro-Credit Approval Engine
==================================
A FinTech loan approval platform that evaluates applicants using:
    - Machine learning (Random Forest classifier + regressor)
    - Rule-based financial scoring formulas
    - Historical growth analysis
    - A Smart Decision Engine with explainable AI

Tech stack: Flask + SQLite + scikit-learn + pandas + numpy + Chart.js
"""
import json

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from auth import auth_bp, current_user, login_required
from config import Config
from database import db
from ml import predictor
from ml.train_models import ensure_models

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

app.register_blueprint(auth_bp)


# ----------------------------------------------------------------------------
# Startup: initialise DB and ensure ML models exist
# ----------------------------------------------------------------------------
def bootstrap():
    db.init_db()
    ensure_models()


bootstrap()


@app.context_processor
def inject_user():
    return {"current_user": current_user(), "score_min": Config.SCORE_MIN, "score_max": Config.SCORE_MAX}


# ----------------------------------------------------------------------------
# Public landing
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("index.html")


# ----------------------------------------------------------------------------
# Loan application
# ----------------------------------------------------------------------------
FIELD_SPECS = {
    "full_name": ("text", "Full Name"),
    "age": ("int", "Age"),
    "employment_type": ("choice", "Employment Type"),
    "monthly_income": ("float", "Monthly Income"),
    "monthly_expenses": ("float", "Monthly Expenses"),
    "bank_balance": ("float", "Bank Balance"),
    "existing_loans": ("float", "Existing Loans"),
    "credit_history": ("int", "Credit History (Months)"),
    "defaults": ("int", "Number Of Defaults"),
}


def _parse_application_form(form):
    data = {}
    errors = []

    full_name = (form.get("full_name") or "").strip()
    if len(full_name) < 2:
        errors.append("Full Name is required.")
    data["full_name"] = full_name

    employment_type = (form.get("employment_type") or "").strip()
    if employment_type not in Config.EMPLOYMENT_TYPES:
        errors.append("Please select a valid Employment Type.")
    data["employment_type"] = employment_type

    def parse_number(key, label, kind, min_val=0, max_val=None):
        raw = (form.get(key) or "").strip()
        if raw == "":
            errors.append(f"{label} is required.")
            return 0
        try:
            value = int(raw) if kind == "int" else float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return 0
        if value < min_val:
            errors.append(f"{label} cannot be negative.")
        if max_val is not None and value > max_val:
            errors.append(f"{label} value is too large.")
        return value

    data["age"] = parse_number("age", "Age", "int", min_val=18, max_val=100)
    data["monthly_income"] = parse_number("monthly_income", "Monthly Income", "float", min_val=0)
    data["monthly_expenses"] = parse_number("monthly_expenses", "Monthly Expenses", "float", min_val=0)
    data["bank_balance"] = parse_number("bank_balance", "Bank Balance", "float", min_val=0)
    data["existing_loans"] = parse_number("existing_loans", "Existing Loans", "float", min_val=0)
    data["credit_history"] = parse_number("credit_history", "Credit History (Months)", "int", min_val=0, max_val=600)
    data["defaults"] = parse_number("defaults", "Number Of Defaults", "int", min_val=0, max_val=50)

    return data, errors


@app.route("/apply", methods=["GET", "POST"])
@login_required
def apply():
    user = current_user()
    if request.method == "POST":
        data, errors = _parse_application_form(request.form)
        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template(
                "apply.html",
                employment_types=Config.EMPLOYMENT_TYPES,
                form=request.form,
            )

        # 1. Store the application
        application_id = db.create_application(user["id"], data)

        # 2. Find the previous final score for growth analysis
        prev = db.get_last_loan_history(user["id"])
        previous_score = prev["final_score"] if prev else None

        # 3. Full evaluation (ML + formula + growth + decision)
        result = predictor.evaluate_application(data, previous_score=previous_score)

        # 4. Persist the loan history snapshot
        db.create_loan_history(
            {
                "application_id": application_id,
                "user_id": user["id"],
                "ml_credit_score": result["ml_credit_score"],
                "formula_credit_score": result["formula_credit_score"],
                "final_score": result["final_score"],
                "ml_prediction": result["ml_prediction"],
                "decision": result["decision"],
                "growth_percentage": result["growth"]["growth_percentage"],
                "previous_score": result["growth"]["previous_score"],
                "debt_to_income": result["debt_to_income"],
                "reasons": predictor.serialize_reasons(result["reasons"]),
                "recommendations": predictor.serialize_recommendations(result["recommendations"]),
            }
        )

        flash("Application evaluated successfully.", "success")
        return redirect(url_for("dashboard"))

    return render_template(
        "apply.html",
        employment_types=Config.EMPLOYMENT_TYPES,
        form={},
    )


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
def _decode_json(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    history = db.get_loan_history_for_user(user["id"])

    latest = None
    if history:
        latest = dict(history[-1])
        latest["reasons"] = _decode_json(latest.get("reasons"), [])
        latest["recommendations"] = _decode_json(latest.get("recommendations"), [])

    # Build chart series (oldest -> newest)
    chart = {
        "labels": [f"#{i + 1}" for i in range(len(history))],
        "final_scores": [round(h["final_score"], 2) for h in history],
        "ml_scores": [round(h["ml_credit_score"], 2) for h in history],
        "formula_scores": [round(h["formula_credit_score"], 2) for h in history],
        "growth": [h["growth_percentage"] if h["growth_percentage"] is not None else 0 for h in history],
        "decisions": [h["decision"] for h in history],
        "dates": [h["created_at"][:10] for h in history],
    }

    return render_template(
        "dashboard.html",
        user=user,
        latest=latest,
        history=list(reversed(history)),  # newest first for the table
        chart=chart,
        metadata=predictor.get_metadata(),
        has_history=bool(history),
    )


@app.route("/api/history")
@login_required
def api_history():
    user = current_user()
    history = db.get_loan_history_for_user(user["id"])
    return jsonify(history)


@app.route("/application/<int:application_id>")
@login_required
def application_detail(application_id):
    user = current_user()
    app_row = db.get_application(application_id)
    if not app_row or app_row["user_id"] != user["id"]:
        flash("Application not found.", "danger")
        return redirect(url_for("dashboard"))

    history = db.get_loan_history_for_user(user["id"])
    record = next((h for h in history if h["application_id"] == application_id), None)
    if record:
        record = dict(record)
        record["reasons"] = _decode_json(record.get("reasons"), [])
        record["recommendations"] = _decode_json(record.get("recommendations"), [])

    return render_template(
        "application_detail.html",
        application=app_row,
        record=record,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
