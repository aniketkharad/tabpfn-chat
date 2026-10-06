"""Generate seed patient clinical records and synthetic privacy-preserving patient dataset."""

from pathlib import Path
import numpy as np
import pandas as pd

def generate():
    np.random.seed(42)
    demo_dir = Path("demos/05_data_generation")
    demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. 15 Seed Patients (real reference profile)
    seed_data = [
        {"patient_id": "PT_01", "age": 45, "blood_pressure_sys": 125, "cholesterol": 195, "glucose": 92, "treatment_response": 1},
        {"patient_id": "PT_02", "age": 62, "blood_pressure_sys": 142, "cholesterol": 240, "glucose": 138, "treatment_response": 0},
        {"patient_id": "PT_03", "age": 51, "blood_pressure_sys": 130, "cholesterol": 210, "glucose": 105, "treatment_response": 1},
        {"patient_id": "PT_04", "age": 68, "blood_pressure_sys": 155, "cholesterol": 265, "glucose": 160, "treatment_response": 0},
        {"patient_id": "PT_05", "age": 39, "blood_pressure_sys": 118, "cholesterol": 180, "glucose": 88, "treatment_response": 1},
        {"patient_id": "PT_06", "age": 58, "blood_pressure_sys": 138, "cholesterol": 225, "glucose": 120, "treatment_response": 0},
        {"patient_id": "PT_07", "age": 44, "blood_pressure_sys": 122, "cholesterol": 188, "glucose": 94, "treatment_response": 1},
        {"patient_id": "PT_08", "age": 71, "blood_pressure_sys": 160, "cholesterol": 270, "glucose": 172, "treatment_response": 0},
        {"patient_id": "PT_09", "age": 48, "blood_pressure_sys": 128, "cholesterol": 202, "glucose": 98, "treatment_response": 1},
        {"patient_id": "PT_10", "age": 55, "blood_pressure_sys": 135, "cholesterol": 215, "glucose": 112, "treatment_response": 1},
        {"patient_id": "PT_11", "age": 64, "blood_pressure_sys": 148, "cholesterol": 250, "glucose": 145, "treatment_response": 0},
        {"patient_id": "PT_12", "age": 42, "blood_pressure_sys": 120, "cholesterol": 182, "glucose": 90, "treatment_response": 1},
        {"patient_id": "PT_13", "age": 67, "blood_pressure_sys": 152, "cholesterol": 258, "glucose": 152, "treatment_response": 0},
        {"patient_id": "PT_14", "age": 53, "blood_pressure_sys": 132, "cholesterol": 208, "glucose": 102, "treatment_response": 1},
        {"patient_id": "PT_15", "age": 60, "blood_pressure_sys": 140, "cholesterol": 230, "glucose": 130, "treatment_response": 0},
    ]
    df_seed = pd.DataFrame(seed_data)
    df_seed.to_csv(demo_dir / "patient_seed.csv", index=False)
    print(f"Saved {len(df_seed)} seed patients to {demo_dir / 'patient_seed.csv'}")

    # 2. Synthetic Patient Generation (100 synthetic rows)
    # Preserving joint covariance structure while ensuring privacy (no exact copies of seed rows)
    n_syn = 100
    syn_age = np.random.normal(54.0, 10.0, n_syn).clip(32, 78)
    syn_bp = np.round(100.0 + 0.65 * syn_age + np.random.normal(0, 6.0, n_syn)).astype(int).clip(110, 175)
    syn_chol = np.round(135.0 + 1.5 * syn_age + 0.3 * (syn_bp - 120) + np.random.normal(0, 12.0, n_syn)).astype(int).clip(160, 290)
    syn_gluc = np.round(60.0 + 0.7 * syn_age + 0.25 * (syn_bp - 120) + np.random.normal(0, 10.0, n_syn)).astype(int).clip(75, 195)
    
    # Latent probability of favorable treatment response based on biomarker health
    score = -0.04 * (syn_age - 54) - 0.03 * (syn_bp - 135) - 0.02 * (syn_gluc - 100) + np.random.normal(0, 0.5, n_syn)
    syn_response = (score > 0).astype(int)

    syn_ids = [f"SYN_{i+1:03d}" for i in range(n_syn)]
    df_syn = pd.DataFrame({
        "patient_id": syn_ids,
        "age": np.round(syn_age).astype(int),
        "blood_pressure_sys": syn_bp,
        "cholesterol": syn_chol,
        "glucose": syn_gluc,
        "treatment_response": syn_response,
    })

    df_syn.to_csv(demo_dir / "synthetic_patients.csv", index=False)
    print(f"Generated {len(df_syn)} synthetic patients ({syn_response.sum()} positive) to {demo_dir / 'synthetic_patients.csv'}")

    # Also save to static samples for browser UI
    static_samples = Path("src/tabchat/frontend/static/samples")
    static_samples.mkdir(parents=True, exist_ok=True)
    df_syn.to_csv(static_samples / "synthetic_patients.csv", index=False)

if __name__ == "__main__":
    generate()
