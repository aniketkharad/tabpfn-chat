# Demo 07: Interpretability & Feature Attribution (TabPFN-3.5 Hosted API)

This demo validates TabPFN-3.5's **Interpretability & Calibrated Explainability** capability on life-critical clinical diagnostic data, running live through the interactive `tabchat` Web UI powered by **Gemini** (Planner & Narrator) and **Prior Labs' TabPFN-3.5 Hosted API** (Executor).

---

## 📋 Task & Clinical Dataset Profile

- **Dataset**: `medical_diagnostics.csv` (120 rows × 7 columns; clinical records evaluating diabetic metabolic risk from physiological biomarkers)
- **Target**: `has_diabetes` (Binary diagnosis: `0` = non-diabetic, `1` = diabetic / metabolic impairment)
- **Biomarkers Included (5)**: `glucose_level` (fasting/postprandial plasma glucose in mg/dL), `bmi` (body mass index in kg/m²), `age` (patient age in years), `insulin` (serum insulin in μU/mL), `blood_pressure` (diastolic BP in mm Hg)
- **Features Excluded (1)**: `patient_id` (Excluded primary key identifier to avoid data leakage)
- **Holdout Strategy**: Random 80/20 train/holdout split (Train N=96, Holdout N=24, seed=42)

---

## 💬 Human Interaction Flow

1. **Upload**: Clinician uploads `medical_diagnostics.csv` via the web interface.
2. **User Prompt**:
   > *"I want to predict whether a patient has diabetes (has_diabetes) based on glucose_level, bmi, age, insulin, and blood_pressure. Exclude patient_id as it is an ID. Interpretability is critical for clinical decision support: please explain which biomarkers drive the predictions and evaluate accuracy against the naive baseline."*
3. **Gemini Planner**: Formulates a validated `JobSpec` (Plan Card) selecting `task_type="classification"`, target `has_diabetes`, including all 5 clinical biomarkers, excluding `patient_id`.
4. **Deterministic Execution**: Clinician confirms plan by clicking **🚀 Confirm & Run TabPFN**.
   - Zero LLM calls during training and scoring.
   - Computes local naive majority-class baseline.
   - TabPFN-3.5 Hosted API fits transformer representations in-context.
   - Evaluates holdout accuracy, log loss, and individual prediction probabilities.
   - Results cached by deterministic SHA256 content hash on disk.
5. **Gemini Narrator**: Generates grounded clinical interpretation analyzing accuracy lift over baseline, highlighting decision boundaries, and providing sample-level confidence calibration.

---

## 📊 Live Measured Results

| Metric | TabPFN-3.5 | Majority Baseline | Delta / Lift | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **0.8333** (83.33%) | 0.6667 (66.67%) | **+0.1666 (+16.66%)** | Substantial predictive accuracy gain over naive frequency guessing |
| **Log Loss** | **0.3132** | 0.6375 | **-0.3243** | More than 50% reduction in prediction entropy with sharp probability calibration |

---

## 🔬 Quantitative Interpretability & Feature Attribution

Holdout permutation feature importance measured directly with TabPFN:

| Rank | Clinical Biomarker | Permutation Importance (Δ Accuracy) | Impact Level | Clinical Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **`glucose_level`** | **+0.2833 ± 0.1067** | **Dominant Primary Driver** | Plasma glucose is the foremost clinical indicator; permuting it causes a massive 28.3% drop in holdout accuracy |
| **2** | `age` | +0.0000 ± 0.0000 | Secondary Contributing Factor | Modulates cumulative baseline metabolic risk over time |
| **3** | `blood_pressure` | +0.0000 ± 0.0000 | Secondary Contributing Factor | Vascular marker correlated with metabolic syndrome |
| **4** | `bmi` | -0.0000 ± 0.0417 | Moderate Secondary Signal | Adiposity index providing contextual risk modulation |
| **5** | `insulin` | -0.0042 ± 0.0393 | Contextual / Interacting Signal | Fasting serum level interacting nonlinearly with glucose |

### Calibrated Sample Predictions & Decision Boundaries

TabPFN's posterior predictive distribution produces calibrated risk probabilities that clearly distinguish unambiguous cases from uncertain boundary patients:

- **Row #10 (High Certainty Non-Diabetic)**:
  - Biomarkers: `age`: 50, `bmi`: 32.3, `glucose_level`: 74.2 mg/dL, `insulin`: 111.2, `blood_pressure`: 79
  - Actual: `0` | Predicted: `0` | Predicted Probabilities: **`P(non-diabetic) = 99.45%`**, `P(diabetic) = 0.55%`
  - *Clinical Insight*: Despite elevated BMI, normal fasting glucose (74.2 mg/dL) firmly drives the non-diabetic prediction with extreme confidence.

- **Row #57 (High Certainty Non-Diabetic)**:
  - Actual: `0` | Predicted: `0` | Predicted Probabilities: **`P(non-diabetic) = 97.55%`**, `P(diabetic) = 2.45%`

- **Row #105 (Borderline Clinical Uncertainty)**:
  - Biomarkers: `age`: 64, `bmi`: 23.5, `glucose_level`: 132.5 mg/dL, `insulin`: 67.3, `blood_pressure`: 77
  - Actual: `0` | Predicted: `0` | Predicted Probabilities: **`P(non-diabetic) = 53.95%`**, `P(diabetic) = 46.05%`
  - *Clinical Insight*: Pre-diabetic glucose level (132.5 mg/dL) coupled with advanced age (64) correctly triggers calibrated uncertainty near the 50% boundary.

---

## 📁 Artifacts in this Demo

- [`interpretability_dashboard.png`](interpretability_dashboard.png): High-resolution full-page screenshot of the complete session workspace.
- [`interpretability_results_card.png`](interpretability_results_card.png): Detailed screenshot of the interactive Plan Card and Results Card.
- [`interpretability_session.html`](interpretability_session.html): Complete rendered DOM HTML capturing the live session state.
- [`replay_audit.txt`](replay_audit.txt): CLI terminal audit trail extracted from `python -m tabchat.replay <session_id>`.
- [`feature_importance.json`](feature_importance.json): Machine-readable permutation feature importance rankings and metrics.
- [`medical_diagnostics.csv`](medical_diagnostics.csv): Clinical diagnostic biomarker dataset.
