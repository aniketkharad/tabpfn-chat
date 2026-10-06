# Demo 06: Context-Aware Text Integration (TabPFN-3.5 Hosted API)

This demo validates TabPFN-3.5's **Native Text Integration & Multimodal Tabular Learning** capability running live through the interactive `tabchat` Web UI, powered by **Gemini** (Planner & Narrator) and **Prior Labs' TabPFN-3.5 Hosted API** (Executor).

---

## 📋 Task & Dataset Profile

- **Dataset**: `customer_feedback_escalation.csv` (120 rows × 6 columns; customer support records mixing natural language feedback sentences with categorical and numeric financial metrics)
- **Target**: `needs_escalation` (Binary indicator: `0` = normal support resolution, `1` = urgent management/technical escalation required)
- **Features Included (4)**: `feedback_text` (natural language text sentences $\le 64$ chars), `account_tier` (categorical: Basic/Pro/Enterprise), `monthly_spend` (continuous financial metric), `tenure_months` (customer age)
- **Features Excluded (1)**: `ticket_id` (Excluded primary key to prevent data leakage)
- **Holdout Strategy**: Random 80/20 train/test split (Train N=96, Holdout N=24, seed=42)

---

## 💬 Human Interaction Flow

1. **Upload**: User uploads `customer_feedback_escalation.csv` via the web interface.
2. **User Prompt**:
   > *"I want to predict whether a customer ticket needs urgent escalation (needs_escalation) using customer feedback_text, account_tier, monthly_spend, and tenure_months. Exclude ticket_id as it is an ID, and evaluate with balanced accuracy."*
3. **Gemini Planner**: Formulates a validated `JobSpec` (Plan Card) selecting `task_type="classification"`, `target_column="needs_escalation"`, including the unstructured `feedback_text` column alongside structured columns, excluding `ticket_id`.
4. **Execution**: User confirms plan by clicking **🚀 Confirm & Run TabPFN**.
   - Zero LLM calls during training.
   - Evaluates naive majority-class baseline.
   - TabPFN-3.5 API processes natural language text tokens directly with transformer embeddings in-context.
   - Evaluates holdout predictions, log loss, and accuracy.
   - Caches fit on disk to prevent duplicate token consumption.
5. **Gemini Narrator**: Explains how TabPFN extracts contextual semantic signals from feedback phrases in synergy with financial spending tier attributes.

---

## 📊 Live Measured Results

| Metric | TabPFN-3.5 | Majority Baseline | Delta / Lift | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **1.0000** (100.0%) | 0.5833 (58.3%) | **+0.4167 (+41.7%)** | Text embeddings provide unambiguous classification separation |
| **Log Loss** | **0.0275** | 0.6801 | **-0.6526** | Near-zero predictive entropy with calibrated class probabilities |

### Sample Prediction Probabilities (Holdout Customer Tickets)
- **Row #7**: `feedback_text`: *"Polite representative and clear steps"*, `spend`: \$87.72 | Actual = `0`, Predicted = `0`, Confidence = **99.99%**
- **Row #0**: `feedback_text`: *"Works as expected after update"*, `spend`: \$263.95 | Actual = `0`, Predicted = `0`, Confidence = **99.99%**
- **Row #80**: `feedback_text`: *"Feature works nicely thank you"*, `spend`: \$99.01 | Actual = `0`, Predicted = `0`, Confidence = **99.95%**

---

## 📁 Artifacts in this Demo

- [`text_integration_dashboard.png`](text_integration_dashboard.png): High-resolution full-page screenshot of the complete session workspace.
- [`text_integration_results_card.png`](text_integration_results_card.png): Detailed screenshot of the interactive Plan Card and Results Card.
- [`text_integration_session.html`](text_integration_session.html): Complete rendered DOM HTML capturing the session state.
- [`replay_audit.txt`](replay_audit.txt): CLI terminal audit trail extracted from `python -m tabchat.replay <session_id>`.
- [`customer_feedback_escalation.csv`](customer_feedback_escalation.csv): Multimodal tabular dataset combining natural language text and structured numbers.
