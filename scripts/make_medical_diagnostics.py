"""Generate a clean clinical diagnostics dataset for TabPFN-3.5 Interpretability demo."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_medical_diagnostics() -> pd.DataFrame:
    """Generate 120 clinical patient records with diabetes diagnostic biomarkers."""
    rng = np.random.default_rng(seed=42)
    n_samples = 120

    # Patient IDs (excluded from modeling)
    patient_ids = [f"PAT-{1001 + i:04d}" for i in range(n_samples)]

    # Clinical features
    age = rng.integers(22, 76, size=n_samples)
    bmi = np.round(rng.normal(27.8, 5.2, size=n_samples), 1)
    bmi = np.clip(bmi, 18.5, 43.0)

    # Fasting & postprandial glucose levels (mg/dL)
    glucose = np.round(rng.normal(118.0, 32.0, size=n_samples), 1)
    glucose = np.clip(glucose, 70.0, 198.0)

    # Serum insulin (uU/mL)
    insulin = np.round(rng.normal(85.0, 45.0, size=n_samples), 1)
    insulin = np.clip(insulin, 15.0, 260.0)

    # Diastolic blood pressure (mm Hg)
    blood_pressure = rng.integers(60, 96, size=n_samples)

    # Ground truth clinical outcome based on established physiology:
    # Strongest driver: glucose; secondary: BMI and age; minor: insulin and BP
    log_odds = (
        -11.8
        + 0.055 * glucose
        + 0.075 * bmi
        + 0.028 * age
        + 0.006 * insulin
        + 0.012 * blood_pressure
        + rng.normal(0, 0.45, size=n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    has_diabetes = (prob > 0.5).astype(int)

    # Ensure class balance: ~40% positive class
    pos_count = int(has_diabetes.sum())
    print(f"Generated {n_samples} patient records: {pos_count} positive ({pos_count/n_samples:.1%}), {n_samples - pos_count} negative")

    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": age,
            "bmi": bmi,
            "glucose_level": glucose,
            "insulin": insulin,
            "blood_pressure": blood_pressure,
            "has_diabetes": has_diabetes,
        }
    )
    return df


def main() -> None:
    df = generate_medical_diagnostics()

    # Save to demos/07_interpretability/
    demo_dir = Path("demos/07_interpretability")
    demo_dir.mkdir(parents=True, exist_ok=True)
    out_path = demo_dir / "medical_diagnostics.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path} ({len(df)} rows, {len(df.columns)} cols)")

    # Also save to static samples directory
    static_sample = Path("src/tabchat/frontend/static/samples/medical_diagnostics.csv")
    df.to_csv(static_sample, index=False)
    print(f"Saved copy to {static_sample}")


if __name__ == "__main__":
    main()
