# Demo 01: Binary Classification (TabPFN-3.5 Hosted API)

This demo validates TabPFN-3.5's **Classification** capability running live through the interactive `tabchat` Web UI, powered by **Gemini** (Planner & Narrator) and **Prior Labs' TabPFN-3.5 Hosted API** (Executor).

---

## 📋 Task & Dataset Profile

- **Dataset**: `churn_sample.csv` (140 rows × 8 columns)
- **Target**: `churned` (Binary Classification: `0` = retained, `1` = churned)
- **Features Included (6)**: `age`, `tenure_months`, `monthly_charges`, `total_charges`, `contract_type`, `tech_support`
- **Features Excluded (1)**: `customer_id` (Excluded as an uninformative primary key identifier to avoid leakage)
- **Holdout Strategy**: Random 80/20 train/test split (Train N=112, Holdout N=28, seed=42)

---

## 💬 Human Interaction Flow

1. **Upload**: User loads `churn_sample.csv` via the web interface.
2. **User Prompt**:
   > *"I want to predict whether a customer will churn (churned). Exclude customer_id as it is an ID, and evaluate with balanced accuracy."*
3. **Gemini Planner**: Automatically generates a mathematically validated `JobSpec` (Plan Card) identifying target, features, and excluding `customer_id`.
4. **Execution**: User confirms plan by clicking **🚀 Confirm & Run TabPFN**.
   - Zero LLM calls during training.
   - Fits local `MajorityClassBaseline`.
   - Calls Prior Labs TabPFN-3.5 API (`POST /tabpfn/fit` and `POST /tabpfn/predict`).
   - Caches fit on disk to prevent duplicate token consumption.
5. **Gemini Narrator**: Delivers grounded, uncertainty-calibrated explanation with baseline lift comparisons.

---

## 📊 Live Measured Results

| Metric | TabPFN-3.5 | Majority Baseline | Delta / Lift | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **0.9286** (92.9%) | 0.5357 (53.6%) | **+0.3929 (+39.3%)** | Major accuracy improvement over trivial baseline |
| **Log Loss** | **0.1615** | 0.6906 | **-0.5291** | Sharp reduction in entropy, well-calibrated probabilities |

### Sample Prediction Calibrations (First 5 Holdout Rows)
- **Row #14**: Actual = `0`, Predicted = `0`, Confidence = **99.5%**
- **Row #118**: Actual = `0`, Predicted = `0`, Confidence = **55.5%** (Calibrated uncertainty for borderline customer profile)
- **Row #25**: Actual = `0`, Predicted = `0`, Confidence = **99.7%**
- **Row #0**: Actual = `0`, Predicted = `0`, Confidence = **99.6%**
- **Row #59**: Actual = `0`, Predicted = `0`, Confidence = **57.3%**

---

## 📁 Artifacts in this Demo

- [`classification_dashboard.png`](classification_dashboard.png): High-resolution full-page screenshot of the complete session workspace.
- [`classification_results_card.png`](classification_results_card.png): Detailed screenshot of the interactive Plan Card and Results Card.
- [`classification_session.html`](classification_session.html): Complete rendered DOM HTML capturing the session state.
- [`replay_audit.txt`](replay_audit.txt): CLI terminal audit trail extracted from `python -m tabchat.replay <session_id>`.
- [`churn_sample.csv`](churn_sample.csv): Input dataset used for this demonstration run.
