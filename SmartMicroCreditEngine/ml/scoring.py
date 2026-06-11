"""
Rule-based (formula) credit scoring and the Smart Decision Engine.

This module is independent of the ML models and provides:
    - formula_credit_score()   : transparent rule-based score (300-900)
    - debt_to_income()          : DTI ratio helper
    - growth_analysis()         : compares current vs previous score
    - smart_decision()          : combines ML + formula + growth into a verdict
    - build_recommendations()   : personalized improvement suggestions
"""
from config import Config

SCORE_MIN = Config.SCORE_MIN
SCORE_MAX = Config.SCORE_MAX

EMPLOYMENT_STABILITY = {
    "Salaried": 1.0,
    "Business": 0.85,
    "Self-Employed": 0.75,
    "Freelancer": 0.6,
    "Unemployed": 0.2,
}


def _clip(value, low, high):
    return max(low, min(high, value))


def debt_to_income(app):
    """Debt-to-income ratio = (expenses + existing loan outflow) / income."""
    income = max(float(app["monthly_income"]), 1.0)
    obligations = float(app["monthly_expenses"]) + float(app["existing_loans"])
    return round(obligations / income, 4)


def formula_credit_score(app):
    """
    Transparent rule-based credit score between 300 and 900.

    Factors: income, debt-to-income ratio, credit history, existing loans, defaults.
    """
    income = float(app["monthly_income"])
    expenses = float(app["monthly_expenses"])
    bank_balance = float(app["bank_balance"])
    existing_loans = float(app["existing_loans"])
    credit_history = int(app["credit_history"])
    defaults = int(app["defaults"])
    employment_type = app.get("employment_type", "Salaried")

    dti = debt_to_income(app)
    stability = EMPLOYMENT_STABILITY.get(employment_type, 0.7)

    score = 500.0

    # Income contribution (up to +150)
    score += _clip((income - 20000) / 1000, -60, 150)

    # Employment stability (up to +110)
    score += stability * 110

    # Debt-to-income penalty (up to -240)
    score -= _clip(dti * 230, 0, 240)

    # Credit history reward (up to +130)
    score += _clip(credit_history * 1.3, 0, 130)

    # Savings buffer reward (up to +90)
    score += _clip(bank_balance / 8000, 0, 90)

    # Existing loans penalty (up to -70)
    score -= _clip(existing_loans / 2000, 0, 70)

    # Defaults penalty (heavy)
    score -= defaults * 55

    return round(_clip(score, SCORE_MIN, SCORE_MAX), 2)


def hybrid_score(ml_score, formula_score):
    """final_score = (ml_credit_score + formula_credit_score) / 2"""
    return round((float(ml_score) + float(formula_score)) / 2.0, 2)


def growth_analysis(current_score, previous_score):
    """
    Compare current credit score with the previous one.

    Returns a dict with previous_score, current_score, growth_percentage,
    trend ('positive' / 'negative' / 'neutral' / 'first'), and interpretation.
    """
    if previous_score is None:
        return {
            "previous_score": None,
            "current_score": round(current_score, 2),
            "growth_percentage": None,
            "trend": "first",
            "interpretation": "First application - no historical data to compare yet.",
        }

    previous_score = float(previous_score)
    if previous_score == 0:
        previous_score = 1.0  # avoid division by zero

    growth = (current_score - previous_score) / previous_score * 100.0
    growth = round(growth, 2)

    if growth > 5:
        trend = "positive"
        interpretation = "User is improving rapidly. Financial health is strengthening - increase approval confidence."
    elif growth < -5:
        trend = "negative"
        interpretation = "Financial health is deteriorating. Recent decline detected - reduce approval confidence."
    else:
        trend = "neutral"
        interpretation = "Financial health is stable with no significant change since the last application."

    return {
        "previous_score": round(previous_score, 2),
        "current_score": round(current_score, 2),
        "growth_percentage": growth,
        "trend": trend,
        "interpretation": interpretation,
    }


def smart_decision(ml_prediction, final_score, growth_info, app):
    """
    Smart Decision Engine.

    Combines:
        1. ML loan prediction (APPROVED / REJECTED)
        2. Final hybrid score
        3. Historical growth trend

    Returns (decision, reasons) where decision is one of:
        APPROVED, REJECTED, CONDITIONAL APPROVAL, MANUAL REVIEW
    """
    dti = debt_to_income(app)
    defaults = int(app["defaults"])
    credit_history = int(app["credit_history"])
    trend = growth_info["trend"]

    reasons = []

    # ---- Build explainability reasons (positive then negative) ----
    if float(app["monthly_income"]) >= 25000:
        reasons.append("✓ Stable income level")
    else:
        reasons.append("✗ Low monthly income")

    if dti <= 0.45:
        reasons.append("✓ Low debt-to-income ratio")
    elif dti <= 0.65:
        reasons.append("⚠ Moderate debt-to-income ratio")
    else:
        reasons.append("✗ High debt-to-income ratio")

    if credit_history >= 24:
        reasons.append("✓ Good credit history length")
    elif credit_history >= 6:
        reasons.append("⚠ Limited credit history")
    else:
        reasons.append("✗ Very short credit history")

    if defaults == 0:
        reasons.append("✓ No previous defaults")
    else:
        reasons.append(f"✗ {defaults} previous default(s) on record")

    if trend == "positive":
        reasons.append("✓ Positive financial growth trend")
    elif trend == "negative":
        reasons.append("✗ Declining financial health")
    elif trend == "neutral":
        reasons.append("✓ Stable financial trajectory")

    # ---- Decision logic ----
    score_strong = final_score >= 680
    score_mid = 560 <= final_score < 680
    score_weak = final_score < 560

    ml_approved = ml_prediction == "APPROVED"

    if score_strong and ml_approved and defaults == 0 and trend != "negative":
        decision = "APPROVED"
    elif score_weak or defaults >= 3 or (dti > 0.8 and not ml_approved):
        decision = "REJECTED"
    elif score_mid and ml_approved and trend == "positive":
        # Mid score but improving rapidly -> give a conditional chance
        decision = "CONDITIONAL APPROVAL"
    elif score_strong and trend == "negative":
        # Strong score but worsening -> needs a human look
        decision = "MANUAL REVIEW"
    elif ml_approved and score_mid:
        decision = "CONDITIONAL APPROVAL"
    elif not ml_approved and score_mid:
        decision = "MANUAL REVIEW"
    else:
        decision = "MANUAL REVIEW"

    # Override: rapidly improving applicant flagged for rejection on weak score
    if decision == "REJECTED" and trend == "positive" and defaults < 2 and final_score >= 520:
        decision = "MANUAL REVIEW"
        reasons.append("⚠ Weak score offset by strong improvement - flagged for review")

    return decision, reasons


def build_recommendations(app, final_score, growth_info, decision):
    """Generate personalized recommendations, especially when not fully approved."""
    recs = []
    income = float(app["monthly_income"])
    expenses = float(app["monthly_expenses"])
    existing_loans = float(app["existing_loans"])
    bank_balance = float(app["bank_balance"])
    defaults = int(app["defaults"])
    credit_history = int(app["credit_history"])
    dti = debt_to_income(app)

    # Expense reduction
    if dti > 0.5:
        target_expenses = income * 0.45
        reduce_by = max(1000, round((expenses + existing_loans - target_expenses) / 100) * 100)
        if reduce_by > 0:
            recs.append(f"Reduce monthly expenses by ~₹{int(reduce_by)} to lower your debt-to-income ratio.")

    # Existing loans
    if existing_loans > income * 0.2:
        recs.append("Close or consolidate an existing loan to free up monthly cash flow.")

    # Defaults / repayment
    if defaults > 0:
        recs.append("Improve repayment history by making all upcoming EMIs on time for 6+ months.")

    # Credit history
    if credit_history < 24:
        recs.append("Build a longer credit history by maintaining active, well-managed credit accounts.")

    # Savings
    if bank_balance < income * 3:
        target = round((income * 3 - bank_balance) / 1000) * 1000
        if target > 0:
            recs.append(f"Increase your savings balance by ~₹{int(target)} to strengthen your financial buffer.")

    # Growth-based
    if growth_info["trend"] == "negative":
        recs.append("Reverse the recent decline in financial health before reapplying.")

    if not recs and decision != "APPROVED":
        recs.append("Maintain your current financial discipline and reapply after building a longer track record.")

    if decision == "APPROVED":
        recs.append("Great profile! Keep your debt-to-income ratio low to qualify for higher limits next time.")

    return recs
