# TabPFN-3.5 Capability Demos & Case Studies

This directory contains standalone, end-to-end demonstrations of Prior Labs' **TabPFN-3.5** foundation model capabilities evaluated through the `tabchat` local-first web application.

For the complete benchmark summary, empirical metrics, and detailed clinical/business insights, see **[`../CASE_STUDIES.md`](../CASE_STUDIES.md)**.

---

## 📁 Directory Structure

| Folder | Capability | Dataset | Key Finding / Artifacts |
| :--- | :--- | :--- | :--- |
| [**`01_classification/`**](01_classification/) | Binary Classification | `churn_sample.csv` (140 rows) | **+39.29%** Accuracy Lift over baseline; automatic ID leakage prevention. |
| [**`02_regression/`**](02_regression/) | Continuous Regression & Quantiles | `housing_sample.csv` (120 rows) | **67%** RMSE reduction; $q_{10}$–$q_{90}$ predictive interval calibration. |
| [**`03_time_series_forecasting/`**](03_time_series_forecasting/) | Time-Series Forecasting | `air_passengers.csv` (144 rows) | **0.9542** $R^2$ score; Box-Jenkins seasonal trend extrapolation. |
| [**`04_anomaly_detection/`**](04_anomaly_detection/) | Rare Event Anomaly Detection | `equipment_anomalies.csv` (120 rows) | **100%** precision on 6.7% rare industrial failures without false positives. |
| [**`05_data_generation/`**](05_data_generation/) | Synthetic Data Generation | `synthetic_patients.csv` (100 rows) | Preserves clinical correlation structure from 15 seeds while protecting privacy. |
| [**`06_text_integration/`**](06_text_integration/) | Multimodal Text Integration | `customer_feedback_escalation.csv` (120 rows) | **+41.67%** Accuracy Lift combining natural language feedback with finances. |
| [**`07_interpretability/`**](07_interpretability/) | Interpretability & Attribution | `medical_diagnostics.csv` (120 rows) | Permutation importance isolates `glucose_level` (+0.2833 Δ) as primary driver. |

---

## 📦 Contents of Each Demo Folder

Each demo subdirectory is completely self-contained with reproducible artifacts:
- **`*.csv`**: Benchmark tabular dataset.
- **`*_dashboard.png`**: Full-page high-resolution screenshot of the interactive workspace.
- **`*_results_card.png`**: Detailed screenshot of the interactive Plan Card and Results Card.
- **`*_session.html`**: Rendered DOM HTML capturing the active session state.
- **`replay_audit.txt`**: Complete CLI terminal audit log from `python -m tabchat.replay <session_id>`.
- **`README.md`**: Dedicated case study documentation with measured baseline comparisons.
