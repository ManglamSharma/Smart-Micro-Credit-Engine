"""
Synthetic dataset generation for the Smart Micro-Credit Approval Engine.

Generates realistic loan applicant records together with:
    - a continuous credit score target (300 - 900) for the regression model
    - a binary approval label (1 = APPROVED, 0 = REJECTED) for the classifier

The relationships are intentionally rule-influenced (with added noise) so the
trained models learn sensible, explainable patterns.
"""
import numpy as np
import pandas as pd

from config import Config

EMPLOYMENT_TYPES = Config.EMPLOYMENT_TYPES

# Stability weighting per employment type (higher = more financially stable)
EMPLOYMENT_STABILITY = {
    "Salaried": 1.0,
    "Business": 0.85,
    "Self-Employed": 0.75,
    "Freelancer": 0.6,
    "Unemployed": 0.2,
}


def _clip_score(value):
    return float(np.clip(value, Config.SCORE_MIN, Config.SCORE_MAX))


def generate_dataset(n_samples=800, seed=42):
    """Generate a synthetic applicant dataset as a pandas DataFrame."""
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_samples):
        employment_type = rng.choice(
            EMPLOYMENT_TYPES, p=[0.40, 0.20, 0.18, 0.15, 0.07]
        )
        stability = EMPLOYMENT_STABILITY[employment_type]

        age = int(rng.integers(21, 65))

        # Income correlated with employment stability
        base_income = rng.normal(35000, 15000) * (0.5 + stability)
        monthly_income = float(max(5000, base_income))

        # Expenses are a fraction of income plus noise
        expense_ratio = np.clip(rng.normal(0.55, 0.18), 0.2, 1.2)
        monthly_expenses = float(monthly_income * expense_ratio)

        # Savings / bank balance correlated with surplus
        surplus = max(0, monthly_income - monthly_expenses)
        bank_balance = float(max(0, rng.normal(surplus * 6, surplus * 3 + 5000)))

        # Existing loan obligations (monthly outflow)
        existing_loans = float(max(0, rng.normal(monthly_income * 0.15, monthly_income * 0.15)))

        # Credit history in months
        credit_history = int(np.clip(rng.normal(48 * stability, 24), 0, 360))

        # Number of defaults (more likely with low stability)
        default_lambda = (1.2 - stability) * 1.5
        defaults = int(min(8, rng.poisson(default_lambda)))

        # ---- Derived credit score (the regression target) ----
        dti = (monthly_expenses + existing_loans) / max(monthly_income, 1)

        score = 550
        score += stability * 120
        score += np.clip((monthly_income - 25000) / 1000, -80, 150)
        score -= np.clip(dti * 220, 0, 240)
        score += np.clip(credit_history * 1.2, 0, 130)
        score -= defaults * 55
        score += np.clip(bank_balance / 8000, 0, 90)
        score -= np.clip(existing_loans / 2000, 0, 70)
        score += rng.normal(0, 25)  # noise
        credit_score = _clip_score(score)

        # ---- Approval label ----
        # Probability of approval increases with score, decreases with risk.
        # The offset is tuned for a roughly balanced (~45%) approval rate.
        approval_logit = (
            (credit_score - 560) / 55
            - dti * 1.8
            - defaults * 0.9
            + stability * 0.9
        )
        approval_prob = 1 / (1 + np.exp(-approval_logit))
        approved = int(rng.random() < approval_prob)

        rows.append(
            {
                "monthly_income": round(monthly_income, 2),
                "monthly_expenses": round(monthly_expenses, 2),
                "bank_balance": round(bank_balance, 2),
                "existing_loans": round(existing_loans, 2),
                "credit_history": credit_history,
                "defaults": defaults,
                "employment_type": employment_type,
                "age": age,
                "credit_score": round(credit_score, 2),
                "approved": approved,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print("\nShape:", df.shape)
    print("Approval rate:", df["approved"].mean())
    print("Credit score range:", df["credit_score"].min(), "-", df["credit_score"].max())
