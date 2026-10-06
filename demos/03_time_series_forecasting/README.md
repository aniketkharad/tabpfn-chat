# Demo 03: Time-Series Forecasting (TabPFN-3.5 Hosted API)

This demo validates TabPFN-3.5's **Time-Series Forecasting** capability running live through the interactive `tabchat` Web UI, powered by **Gemini** (Planner & Narrator) and **Prior Labs' TabPFN-3.5 Hosted API** (Executor).

---

## 📋 Task & Dataset Profile

- **Dataset**: `air_passengers.csv` (144 rows × 4 columns, canonical Box-Jenkins monthly international airline passenger data from 1949 to 1960)
- **Target**: `passengers` (Continuous monthly demand)
- **Features Included (2)**: `year`, `month_num` (Capturing secular growth trend and monthly annual seasonality)
- **Features Excluded (1)**: `date` (Used strictly as the chronological partitioning column)
- **Holdout Strategy**: Chronological split on `date` (Train N=115, Holdout N=29) to forecast future unseen monthly horizons without lookahead bias

---

## 💬 Human Interaction Flow

1. **Upload**: User uploads `air_passengers.csv` via the web interface.
2. **User Prompt**:
   > *"I want to forecast international airline passenger demand (passengers) forward in time based on year and month_num. Use date for chronological validation split to evaluate future holdout accuracy with RMSE."*
3. **Gemini Planner**: Formulates a validated `JobSpec` (Plan Card) selecting `task_type="regression"`, `target_column="passengers"`, `features=['year', 'month_num']`, `split_strategy="chronological"`, `split_column="date"`, and `eval_metric="rmse"`.
4. **Execution**: User confirms plan by clicking **🚀 Confirm & Run TabPFN**.
   - Zero LLM calls during training.
   - Evaluates historical `MeanTargetBaseline`.
   - Calls Prior Labs TabPFN-3.5 API with `output_type="mean"` and `output_type="quantiles"` (`[0.1, 0.5, 0.9]`).
   - Caches fit on disk to prevent duplicate token consumption.
5. **Gemini Narrator**: Explains trend capture, seasonality modeling, error metrics, and sample-level $q_{10}$–$q_{90}$ predictive interval calibration.

---

## 📊 Live Measured Results

| Metric | TabPFN-3.5 | Mean Target Baseline | Delta / Lift | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **RMSE** | **16.7262** | 215.0573 | **-198.3311** | Error slashed by over 92% compared to historical mean |
| **$R^2$ Score** | **0.9542** | -6.5763 | **+7.5305** | Successfully captures 95.4% of future temporal variance |

### Sample Forecast Predictions (Future Holdout Window)
- **Row #117** (1958-10): Actual = `359.0`, Predicted = `361.57` | $q_{10}$: `346.91`, $q_{50}$: `360.82`, $q_{90}$: `376.59` (Near-exact forecast within narrow confidence bounds)
- **Row #120** (1959-01): Actual = `360.0`, Predicted = `382.05` | $q_{10}$: `320.92`, $q_{50}$: `380.11`, $q_{90}$: `445.85` (Appropriately reflects wider uncertainty at start of new yearly cycle)
- **Row #118** (1958-11): Actual = `310.0`, Predicted = `324.81` | $q_{10}$: `308.20`, $q_{50}$: `323.95`, $q_{90}$: `338.40`

---

## 📁 Artifacts in this Demo

- [`forecasting_dashboard.png`](forecasting_dashboard.png): High-resolution full-page screenshot of the complete session workspace.
- [`forecasting_results_card.png`](forecasting_results_card.png): Detailed screenshot of the interactive Plan Card and Results Card.
- [`forecasting_session.html`](forecasting_session.html): Complete rendered DOM HTML capturing the session state.
- [`replay_audit.txt`](replay_audit.txt): CLI terminal audit trail extracted from `python -m tabchat.replay <session_id>`.
- [`air_passengers.csv`](air_passengers.csv): Canonical Box-Jenkins benchmark dataset used for this demonstration run.
