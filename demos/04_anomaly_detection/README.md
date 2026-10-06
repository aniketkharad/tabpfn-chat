# Demo 04: Anomaly & Rare Event Detection (TabPFN-3.5 Hosted API)

This demo validates TabPFN-3.5's **Anomaly Detection** capability on high-imbalance rare failure events running live through the interactive `tabchat` Web UI, powered by **Gemini** (Planner & Narrator) and **Prior Labs' TabPFN-3.5 Hosted API** (Executor).

---

## 📋 Task & Dataset Profile

- **Dataset**: `equipment_anomalies.csv` (120 rows × 7 columns; industrial sensor telemetry with ~6.7% anomalous failures)
- **Target**: `is_failure` (Binary indicator: `0` = normal operation, `1` = critical equipment anomaly)
- **Features Included (5)**: `temperature_c`, `vibration_mms`, `pressure_bar`, `rpm`, `operating_hours`
- **Features Excluded (1)**: `sensor_id` (Excluded as an uninformative identifier to prevent leakage)
- **Holdout Strategy**: Random 80/20 train/test split (Train N=96, Holdout N=24, seed=42)

---

## 💬 Human Interaction Flow

1. **Upload**: User uploads `equipment_anomalies.csv` via the web interface.
2. **User Prompt**:
   > *"I want to detect rare equipment failure anomalies (is_failure) using temperature, vibration, pressure, rpm, and operating hours. Exclude sensor_id as it is an identifier, and evaluate with roc_auc."*
3. **Gemini Planner**: Formulates a validated `JobSpec` (Plan Card) selecting `task_type="classification"`, `target_column="is_failure"`, excluding `sensor_id`, and targeting `eval_metric="roc_auc"`.
4. **Execution**: User confirms plan by clicking **🚀 Confirm & Run TabPFN**.
   - Zero LLM calls during training.
   - Evaluates naive majority-class baseline (predicting no anomalies).
   - Calls Prior Labs TabPFN-3.5 API (`POST /tabpfn/fit` and `POST /tabpfn/predict`).
   - Caches fit on disk to prevent duplicate token consumption.
5. **Gemini Narrator**: Explains discrimination of rare anomaly patterns, log loss certainty, sample probabilities, and class imbalance trade-offs.

---

## 📊 Live Measured Results

| Metric | TabPFN-3.5 | Majority Baseline | Delta / Lift | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **1.0000** (100.0%) | 0.9167 (91.7%) | **+0.0833 (+8.3%)** | Flawlessly identifies rare anomaly events |
| **Log Loss** | **0.0001** | 0.2902 | **-0.2901** | Exceptional probabilistic certainty and calibration |

### Sample Prediction Probabilities (Holdout Cases)
- **Row #45**: Actual = `0`, Predicted = `0`, Probabilities = `[1.0, 0.0]` (Clear normal operation: 64.9°C, 2.11 mm/s, 5.27 bar)
- **Row #72**: Actual = `0`, Predicted = `0`, Probabilities = `[1.0, 0.0]` (Clear normal operation: 68.4°C, 2.62 mm/s, 5.01 bar)
- **Row #5**: Actual = `0`, Predicted = `0`, Probabilities = `[1.0, 0.0]`
- **Row #52**: Actual = `0`, Predicted = `0`, Probabilities = `[1.0, 0.0]`

---

## 📁 Artifacts in this Demo

- [`anomaly_dashboard.png`](anomaly_dashboard.png): High-resolution full-page screenshot of the complete session workspace.
- [`anomaly_results_card.png`](anomaly_results_card.png): Detailed screenshot of the interactive Plan Card and Results Card.
- [`anomaly_session.html`](anomaly_session.html): Complete rendered DOM HTML capturing the session state.
- [`replay_audit.txt`](replay_audit.txt): CLI terminal audit trail extracted from `python -m tabchat.replay <session_id>`.
- [`equipment_anomalies.csv`](equipment_anomalies.csv): Synthetic telemetry dataset with injected rare anomalies used for this demo.
