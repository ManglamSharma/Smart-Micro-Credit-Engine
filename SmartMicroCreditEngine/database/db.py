"""
SQLite database access layer for the Smart Micro-Credit Approval Engine.

Tables:
    users         - authentication / account info
    applications  - every loan application submitted by a user
    loan_history  - snapshot of the scoring + decision for each application
"""
import os
import sqlite3
from datetime import datetime

from config import Config


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    full_name       TEXT    NOT NULL,
    age             INTEGER NOT NULL,
    employment_type TEXT    NOT NULL,
    monthly_income  REAL    NOT NULL,
    monthly_expenses REAL   NOT NULL,
    bank_balance    REAL    NOT NULL,
    existing_loans  REAL    NOT NULL,
    credit_history  INTEGER NOT NULL,   -- months
    defaults        INTEGER NOT NULL,
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS loan_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id      INTEGER NOT NULL,
    user_id             INTEGER NOT NULL,
    ml_credit_score     REAL    NOT NULL,
    formula_credit_score REAL   NOT NULL,
    final_score         REAL    NOT NULL,
    ml_prediction       TEXT    NOT NULL,   -- APPROVED / REJECTED (raw ML)
    decision            TEXT    NOT NULL,   -- final smart decision
    growth_percentage   REAL,
    previous_score      REAL,
    debt_to_income      REAL,
    reasons             TEXT,               -- JSON list
    recommendations     TEXT,              -- JSON list
    created_at          TEXT    NOT NULL,
    FOREIGN KEY (application_id) REFERENCES applications (id) ON DELETE CASCADE,
    FOREIGN KEY (user_id)        REFERENCES users (id) ON DELETE CASCADE
);
"""


def get_connection():
    """Return a SQLite connection with row access by column name."""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Create all tables if they do not already exist."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# User helpers
# ----------------------------------------------------------------------------
def create_user(name, email, password_hash):
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_user_by_email(email):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Application helpers
# ----------------------------------------------------------------------------
def create_application(user_id, data):
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO applications
                (user_id, full_name, age, employment_type, monthly_income,
                 monthly_expenses, bank_balance, existing_loans, credit_history,
                 defaults, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                data["full_name"],
                data["age"],
                data["employment_type"],
                data["monthly_income"],
                data["monthly_expenses"],
                data["bank_balance"],
                data["existing_loans"],
                data["credit_history"],
                data["defaults"],
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_application(application_id):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM applications WHERE id = ?", (application_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Loan history helpers
# ----------------------------------------------------------------------------
def create_loan_history(record):
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO loan_history
                (application_id, user_id, ml_credit_score, formula_credit_score,
                 final_score, ml_prediction, decision, growth_percentage,
                 previous_score, debt_to_income, reasons, recommendations, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["application_id"],
                record["user_id"],
                record["ml_credit_score"],
                record["formula_credit_score"],
                record["final_score"],
                record["ml_prediction"],
                record["decision"],
                record.get("growth_percentage"),
                record.get("previous_score"),
                record.get("debt_to_income"),
                record.get("reasons"),
                record.get("recommendations"),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_last_loan_history(user_id, exclude_history_id=None):
    """Return the most recent loan_history row for a user (optionally excluding one)."""
    conn = get_connection()
    try:
        if exclude_history_id is not None:
            row = conn.execute(
                """
                SELECT * FROM loan_history
                WHERE user_id = ? AND id != ?
                ORDER BY id DESC LIMIT 1
                """,
                (user_id, exclude_history_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM loan_history WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_loan_history_for_user(user_id):
    """Return all loan history joined with the application, oldest first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT lh.*, a.full_name, a.monthly_income, a.monthly_expenses,
                   a.bank_balance, a.existing_loans, a.credit_history,
                   a.defaults, a.employment_type, a.age
            FROM loan_history lh
            JOIN applications a ON a.id = lh.application_id
            WHERE lh.user_id = ?
            ORDER BY lh.id ASC
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
