# Smart Micro-Credit Approval Engine

An intelligent loan approval platform built for the **Fiserv FinTech Hackathon**. It evaluates loan
applicants using **machine learning**, **rule-based financial scoring**, and **historical growth
analysis**, then produces a transparent, explainable decision.

---

## ✨ Key Features

| Feature | Description |
| --- | --- |
| 🔐 **Authentication** | Register / Login / Logout with hashed passwords and Flask session auth |
| 📝 **Loan Application** | Capture income, expenses, balance, loans, credit history, defaults, and more |
| 🤖 **Dual ML Models** | Random Forest **classifier** (approval) + Random Forest **regressor** (credit score 300–900) |
| 🧮 **Formula Credit Score** | Transparent rule-based score from income, DTI, history, loans, and defaults |
| ⚖️ **Hybrid Score** | `final_score = (ml_credit_score + formula_credit_score) / 2` |
| 📈 **Historical Growth Analysis** | Compares current vs previous score and adjusts approval confidence |
| 🧠 **Smart Decision Engine** | Combines ML + hybrid score + growth → `APPROVED / REJECTED / CONDITIONAL APPROVAL / MANUAL REVIEW` |
| 💡 **Explainable AI** | Every decision lists clear ✓ / ✗ reasons |
| 📌 **Recommendation Engine** | Personalized suggestions when not fully approved |
| 📊 **FinTech Dashboard** | Score gauge, trend lines, growth bars, and decision charts via Chart.js |

---

## 🛠 Tech Stack

- **Frontend:** HTML, CSS, JavaScript, [Chart.js](https://www.chartjs.org/)
- **Backend:** Python [Flask](https://flask.palletsprojects.com/)
- **Database:** SQLite
- **Machine Learning:** scikit-learn, pandas, NumPy, joblib
- **Auth:** Flask session authentication (Werkzeug password hashing)

---

## 📂 Project Structure

```
Arjun/
├── app.py                  # Flask app: routes, application flow, dashboard
├── auth.py                 # Registration / login / logout (session auth)
├── config.py               # Central configuration & constants
├── requirements.txt        # Python dependencies
├── README.md
│
├── database/
│   ├── __init__.py
│   └── db.py               # SQLite schema + data access (users, applications, loan_history)
│
├── ml/
│   ├── __init__.py
│   ├── dataset.py          # Synthetic dataset generator (800 records)
│   ├── scoring.py          # Formula score, growth analysis, decision engine, recommendations
│   ├── train_models.py     # Trains & saves both Random Forest models
│   └── predictor.py        # Loads models + runs the full evaluation pipeline
│
├── models/                 # Saved ML artifacts (.joblib) - auto-generated
│   ├── approval_model.joblib
│   ├── credit_score_model.joblib
│   ├── employment_encoder.joblib
│   └── model_metadata.joblib
│
├── static/
│   ├── css/style.css       # FinTech dark theme
│   └── js/dashboard.js     # Chart.js dashboard charts
│
└── templates/
    ├── base.html
    ├── index.html          # Landing page
    ├── login.html
    ├── register.html
    ├── apply.html          # Loan application form
    ├── dashboard.html      # Main FinTech dashboard
    └── application_detail.html
```

---

## 🗄 Database Schema

**users**
| column | type |
| --- | --- |
| id | INTEGER PK |
| name | TEXT |
| email | TEXT UNIQUE |
| password_hash | TEXT |
| created_at | TEXT |

**applications** (FK → users.id)
> full_name, age, employment_type, monthly_income, monthly_expenses, bank_balance,
> existing_loans, credit_history (months), defaults, created_at

**loan_history** (FK → applications.id, users.id)
> ml_credit_score, formula_credit_score, final_score, ml_prediction, decision,
> growth_percentage, previous_score, debt_to_income, reasons (JSON), recommendations (JSON)

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Train the ML models manually
The app trains models automatically on first run, but you can train them explicitly:
```bash
python -m ml.train_models
```
This generates an 800-record synthetic dataset and saves both models to `/models`.

### 3. Run the app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🔄 How It Works (Flow)

1. **Register / Login** → session is created.
2. **Submit a loan application** → stored in `applications`.
3. **ML inference**:
   - Classifier → `APPROVED` / `REJECTED`
   - Regressor → credit score (300–900)
4. **Formula score** computed from financial rules.
5. **Hybrid score** = average of ML + formula scores.
6. **Growth analysis** compares against the previous final score:
   - `growth_% = (current − previous) / previous × 100`
   - Positive growth → increases approval confidence
   - Negative growth → reduces approval confidence
7. **Smart Decision Engine** blends ML prediction + hybrid score + growth trend into one of:
   `APPROVED`, `REJECTED`, `CONDITIONAL APPROVAL`, `MANUAL REVIEW`.
8. **Explainable AI + Recommendations** are shown on the dashboard.

---

## 🧪 Demo Tip

To showcase **historical growth analysis**, submit two applications for the same user:
1. First with a weaker profile (low income, some defaults).
2. Then with a stronger profile (higher income, no defaults, longer history).

The dashboard will display a large **positive growth %** and increased approval confidence.

---

## 🤖 Machine Learning Details

- **Training data:** 800 synthetic records generated with realistic, rule-influenced relationships plus noise.
- **Model 1 — Loan Approval:** `RandomForestClassifier` (balanced class weights).
- **Model 2 — Credit Score:** `RandomForestRegressor` predicting a 300–900 score.
- **Features:** income, expenses, bank balance, existing loans, credit history, defaults, employment type (encoded), age.
- **Persistence:** models saved with `joblib` in `/models` and loaded lazily at inference time.

---

## 📝 Notes

- This is a hackathon prototype using a development server. For production, deploy behind a WSGI
  server (e.g., gunicorn/waitress) and set a strong `SECRET_KEY` via environment variable.
- The synthetic dataset is for demonstration; real deployments should train on vetted historical data.

---

Built with ❤️ for the **Fiserv FinTech Hackathon**.
