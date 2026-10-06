"""Generate seeded, reproducible sample datasets for tabchat.

Produces:
1. churn_sample.csv:
   - Binary classification, exactly 140 rows, 8 columns.
   - Target: churned (0 or 1).
   - Mild trap: arbitrary customer_id column (to be flagged/excluded by Planner).
2. housing_sample.csv:
   - Continuous regression, exactly 120 rows, 7 columns.
   - Target: median_house_value.
   - Mild trap: sold_date column requiring chronological split.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_SAMPLES_DIR = REPO_ROOT / "src" / "tabchat" / "frontend" / "static" / "samples"
FRONTEND_SAMPLES_DIR = REPO_ROOT / "src" / "tabchat" / "frontend" / "samples"


def generate_churn_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate reproducible churn classification dataset (140 rows, 8 cols)."""
    rng = np.random.RandomState(seed)
    n_rows = 140

    customer_ids = [f"CUST_{i+1:04d}" for i in range(n_rows)]
    ages = rng.randint(18, 75, size=n_rows)
    tenure_months = rng.randint(1, 72, size=n_rows)
    monthly_charges = np.round(rng.uniform(19.99, 119.99, size=n_rows), 2)
    total_charges = np.round(monthly_charges * tenure_months * rng.uniform(0.9, 1.1, size=n_rows), 2)
    # Ensure float values stay within float16 bounds (|x| <= 65504.0)
    total_charges = np.clip(total_charges, 10.0, 8500.0)

    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract_types = rng.choice(contract_choices, size=n_rows, p=[0.55, 0.25, 0.20])

    tech_support_choices = ["Yes", "No"]
    tech_support = rng.choice(tech_support_choices, size=n_rows, p=[0.35, 0.65])

    # Churn probability higher with high monthly charges, low tenure, month-to-month contract
    logits = (
        (monthly_charges - 60.0) / 40.0
        - (tenure_months - 24.0) / 20.0
        + (contract_types == "Month-to-month") * 0.8
        - (tech_support == "Yes") * 0.5
        + rng.normal(0, 0.4, size=n_rows)
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    churned = (probs >= 0.5).astype(int)

    # Ensure good class balance (at least 35 in minority class)
    if churned.sum() < 35 or (n_rows - churned.sum()) < 35:
        threshold = np.median(probs)
        churned = (probs >= threshold).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_types,
        "tech_support": tech_support,
        "churned": churned,
    })

    assert len(df) == 140, f"Expected 140 rows, got {len(df)}"
    assert len(df.columns) == 8, f"Expected 8 columns, got {len(df.columns)}"
    return df


def generate_housing_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate reproducible housing regression dataset (120 rows, 7 cols)."""
    rng = np.random.RandomState(seed)
    n_rows = 120

    # Mild trap: chronological sold_date column strictly ordered over time
    start_date = date(2023, 1, 15)
    # Consecutive ordered dates spaced by 2 to 4 days
    date_offsets = np.cumsum(rng.randint(2, 5, size=n_rows))
    sold_dates = [(start_date + timedelta(days=int(offset))).strftime("%Y-%m-%d") for offset in date_offsets]

    median_income = np.round(rng.uniform(1.8, 12.5, size=n_rows), 3)
    housing_median_age = rng.randint(3, 52, size=n_rows)
    total_rooms = rng.randint(450, 4800, size=n_rows)
    total_bedrooms = (total_rooms * rng.uniform(0.18, 0.28, size=n_rows)).astype(int)
    population = rng.randint(300, 3200, size=n_rows)

    # Target: continuous median_house_value (in thousands of dollars, e.g. 150.0 to 500.0, strictly <= 65504)
    trend_effect = np.linspace(0.0, 40.0, num=n_rows)  # Temporal appreciation over time
    raw_value = (
        80.0
        + median_income * 25.0
        - housing_median_age * 0.4
        + (total_rooms / total_bedrooms) * 8.0
        + trend_effect
        + rng.normal(0, 15.0, size=n_rows)
    )
    median_house_value = np.round(np.clip(raw_value, 95.0, 520.0), 2)

    df = pd.DataFrame({
        "sold_date": sold_dates,
        "median_income": median_income,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "median_house_value": median_house_value,
    })

    assert len(df) == 120, f"Expected 120 rows, got {len(df)}"
    assert len(df.columns) == 7, f"Expected 7 columns, got {len(df.columns)}"
    return df


def main() -> None:
    """Generate datasets and save to static sample directories."""
    STATIC_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    churn_df = generate_churn_dataset(seed=42)
    housing_df = generate_housing_dataset(seed=42)

    for target_dir in (STATIC_SAMPLES_DIR, FRONTEND_SAMPLES_DIR):
        churn_path = target_dir / "churn_sample.csv"
        housing_path = target_dir / "housing_sample.csv"

        churn_df.to_csv(churn_path, index=False, quoting=csv.QUOTE_MINIMAL)
        housing_df.to_csv(housing_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Wrote {churn_path} ({len(churn_df)} rows, {len(churn_df.columns)} cols)")
        print(f"Wrote {housing_path} ({len(housing_df)} rows, {len(housing_df.columns)} cols)")


if __name__ == "__main__":
    main()
