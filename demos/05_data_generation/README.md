# Demo 05: Data Generation & Synthetic Clinical Validation (TabPFN-3.5 Hosted API)

This demo validates TabPFN-3.5's **Data Generation & Synthetic Tabular Validation** capability running live through the interactive `tabchat` Web UI, powered by **Gemini** (Planner & Narrator) and **Prior Labs' TabPFN-3.5 Hosted API** (Executor).

---

## 📋 Task & Dataset Profile

- **Seed Reference**: `patient_seed.csv` (15 real patient baseline clinical trial records)
- **Synthetic Dataset**: `synthetic_patients.csv` (100 synthetic patient records preserving biomarker covariance and treatment response relationships while eliminating patient-identifying data for HIPAA-compliant research)
- **Target**: `treatment_response` (Binary indicator: `0` = non-responsive, `1` = favorable treatment response)
- **Features Included (4)**: `age`, `blood_pressure_sys`, `cholesterol`, `glucose`
- **Features Excluded (1)**: `patient_id` (Excluded identifier to maintain privacy and prevent leakage)
- **Holdout Strategy**: Random 80/20 train/test split (Train N=80, Holdout N=20, seed=42)

---

## 💬 Human Interaction Flow

1. **Generation**: Seed patient distribution is expanded into 100 synthetic patient profiles preserving statistical dependencies without reproducing exact training instances.
2. **Upload**: User uploads `synthetic_patients.csv` via the web interface.
3. **User Prompt**:
   > *"I want to validate this privacy-preserving synthetic clinical dataset by predicting patient treatment response (treatment_response) from biomarkers age, blood_pressure_sys, cholesterol, and glucose. Exclude patient_id as it is an ID, and evaluate with accuracy."*
4. **Gemini Planner**: Formulates a validated `JobSpec` (Plan Card) selecting `task_type="classification"`, `target_column="treatment_response"`, excluding `patient_id`, and targeting `eval_metric="accuracy"`.
5. **Execution**: User confirms plan by clicking **🚀 Confirm & Run TabPFN**.
   - Zero LLM calls during training.
   - Evaluates naive 50% majority-class baseline.
   - Calls Prior Labs TabPFN-3.5 API (`POST /tabpfn/fit` and `POST /tabpfn/predict`).
   - Caches fit on disk to prevent duplicate token consumption.
6. **Gemini Narrator**: Confirms that synthetic feature distributions retain coherent predictive clinical signal and explains sample-level confidence calibration.

---

## 📊 Live Measured Results

| Metric | TabPFN-3.5 | Majority Baseline | Delta / Lift | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **0.8000** (80.0%) | 0.5000 (50.0%) | **+0.3000 (+30.0%)** | Synthetic data retains coherent learnable biological signal |
| **Log Loss** | **0.4132** | 0.6935 | **-0.2803** | Low predictive entropy, well-calibrated class probabilities |

### Sample Prediction Calibrations (Synthetic Holdout Patients)
- **Row #3**: Actual = `0`, Predicted = `0`, Confidence = **94.5%** (Age 69, BP 140, Chol 257, Gluc 120 -> High confidence non-responder)
- **Row #65**: Actual = `0`, Predicted = `0`, Confidence = **93.9%** (Age 68, BP 146, Chol 229, Gluc 110 -> High confidence non-responder)
- **Row #93**: Actual = `0`, Predicted = `0`, Confidence = **55.6%** (Age 51, BP 125, Chol 223, Gluc 115 -> Borderline biomarkers, calibrated uncertainty)

---

## 📁 Artifacts in this Demo

- [`generation_dashboard.png`](generation_dashboard.png): High-resolution full-page screenshot of the complete session workspace.
- [`generation_results_card.png`](generation_results_card.png): Detailed screenshot of the interactive Plan Card and Results Card.
- [`generation_session.html`](generation_session.html): Complete rendered DOM HTML capturing the session state.
- [`replay_audit.txt`](replay_audit.txt): CLI terminal audit trail extracted from `python -m tabchat.replay <session_id>`.
- [`synthetic_patients.csv`](synthetic_patients.csv): 100-row synthetic clinical dataset evaluated in this demonstration.
- [`patient_seed.csv`](patient_seed.csv): Reference seed patient dataset used to condition synthetic generation.
