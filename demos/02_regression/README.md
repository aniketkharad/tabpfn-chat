# Demo 02: Continuous Regression & Quantiles (TabPFN-3.5 Hosted API)

This demo validates TabPFN-3.5's **Regression** capability running live through the interactive `tabchat` Web UI, powered by **Gemini** (Planner & Narrator) and **Prior Labs' TabPFN-3.5 Hosted API** (Executor).

---

## 📋 Task & Dataset Profile

- **Dataset**: `housing_sample.csv` (120 rows × 7 columns)
- **Target**: `median_house_value` (Continuous regression: housing prices)
- **Features Included (5)**: `median_income`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`
- **Features Excluded (1)**: `sold_date` (Used strictly as the chronological partitioning column)
- **Holdout Strategy**: Chronological split on `sold_date` (Train N=96, Holdout N=24) to prevent lookahead temporal leakage

---

## 💬 Human Interaction Flow

1. **Upload**: User loads `housing_sample.csv` via the web interface.
2. **User Prompt**:
   > *"I want to predict continuous house values (median_house_value) based on income, rooms, and other features. Use sold_date for chronological validation split, and evaluate with RMSE."*
3. **Gemini Planner**: Automatically configures a `JobSpec` (Plan Card) with `task_type="regression"`, `target_column="median_house_value"`, `split_strategy="chronological"`, `split_column="sold_date"`, and `eval_metric="rmse"`.
4. **Execution**: User confirms plan by clicking **🚀 Confirm & Run TabPFN**.
   - Zero LLM calls during training.
   - Fits local `MeanTargetBaseline`.
   - Calls Prior Labs TabPFN-3.5 API with `output_type="mean"` and `output_type="quantiles"` (`[0.1, 0.5, 0.9]`).
   - Caches fit on disk to prevent duplicate token consumption.
5. **Gemini Narrator**: Explains continuous variance captured ($R^2$), error reduction ($\Delta \text{RMSE}$), and examines $q_{10}$–$q_{90}$ predictive intervals.

---

## 📊 Live Measured Results

| Metric | TabPFN-3.5 | Mean Target Baseline | Delta / Lift | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **RMSE** | **30.8973** | 93.6386 | **-62.7413** | Massive 67% reduction in root-mean-squared error |
| **$R^2$ Score** | **0.8406** | -0.4642 | **+1.3048** | Explains 84.1% of continuous temporal price variance |

### Sample Prediction Quantiles (First 5 Chronological Holdout Rows)
- **Row #96**: Actual = `412.38`, Predicted = `398.71` | $q_{10}$: `374.12`, $q_{50}$: `397.80`, $q_{90}$: `421.45`
- **Row #97**: Actual = `429.09`, Predicted = `402.64` | $q_{10}$: `377.96`, $q_{50}$: `401.88`, $q_{90}$: `425.77`
- **Row #98**: Actual = `391.24`, Predicted = `384.52` | $q_{10}$: `359.80`, $q_{50}$: `383.91`, $q_{90}$: `407.60`
- **Row #99**: Actual = `385.19`, Predicted = `379.10` | $q_{10}$: `354.21`, $q_{50}$: `378.44`, $q_{90}$: `402.18`
- **Row #100**: Actual = `368.45`, Predicted = `362.30` | $q_{10}$: `338.90`, $q_{50}$: `361.75`, $q_{90}$: `385.10`

---

## 📁 Artifacts in this Demo

- [`regression_dashboard.png`](regression_dashboard.png): High-resolution full-page screenshot of the complete session workspace.
- [`regression_results_card.png`](regression_results_card.png): Detailed screenshot of the interactive Plan Card and Results Card.
- [`regression_session.html`](regression_session.html): Complete rendered DOM HTML capturing the session state.
- [`replay_audit.txt`](replay_audit.txt): CLI terminal audit trail extracted from `python -m tabchat.replay <session_id>`.
- [`housing_sample.csv`](housing_sample.csv): Input dataset used for this demonstration run.
