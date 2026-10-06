# TabPFN Chat (`tabpfn-chat`)

A local-first, interactive tabular analysis tool powered by **Gemini** and Prior Labs' **TabPFN-3.5** API. Upload small CSV tables, clarify modeling goals via natural language, automatically generate verified job specifications, and run fast zero-shot tabular predictions with uncertainty estimation—all without burning excessive LLM tokens or API limits.

---

## ⚡ Architecture: The 3-Stage Pipeline

```text
+-----------------------------------------------+
|             1. Planner (Gemini)               |
|  Sees: Metadata card + 5 preview rows ONLY    |
|  Does: Clarifies goals -> Emits JobSpec       |
+-----------------------+-----------------------+
                        | (Validated JSON)
                        v
+-----------------------------------------------+
|            2. Executor (Deterministic)        |
|  Sees: Full data table                        |
|  Does: ZERO LLM calls. Computes baselines,    |
|        checks disk cache, calls TabPFN API    |
+-----------------------+-----------------------+
                        | (Metrics & Predictions)
                        v
+-----------------------------------------------+
|            3. Narrator (Gemini)               |
|  Sees: TabPFN results + baseline comparisons  |
|  Does: Explains results & calibrated intervals|
+-----------------------------------------------+
```

---

## 🚀 Key Highlights

- **Local-First & Resilient**: All session state persists to disk (`data/sessions/<session_id>/`). Survives server reloads without data loss.
- **Deterministic Fit Caching**: Fits are cached via `sha256(raw_csv + canonical_job_spec)` under `data/cache/tabpfn/`. Never pay API tokens twice for identical data and specs.
- **Zero-LLM Execution**: Baseline calculation, holdout scoring, and TabPFN fitting run purely in deterministic Python.
- **Privacy & Rate Governance**: The LLM only sees column metadata and 5 preview rows. Requests are paced at 12 RPM (semaphore-gated) with automatic Flash-Lite fallback.

---

## 🏆 Empirical Case Studies & Benchmarks

Explore 7 complete, end-to-end capability evaluations comparing TabPFN-3.5 zero-shot predictions against deterministic baselines across real-world tabular domains:

| # | Capability | Domain | TabPFN-3.5 vs. Baseline | Highlights |
| :-: | :--- | :--- | :-: | :--- |
| **01** | [**Classification**](demos/01_classification/) | Customer Churn | **0.9286** vs 0.5357 Acc | **+39.3% Lift**, ID leakage prevention |
| **02** | [**Regression**](demos/02_regression/) | Housing Valuation | **30.90** vs 93.64 RMSE | **67% Error Reduction**, $q_{10}$–$q_{90}$ quantiles |
| **03** | [**Forecasting**](demos/03_time_series_forecasting/) | Air Passengers | **16.73** vs 215.06 RMSE | **0.9542 $R^2$**, Box-Jenkins monthly trend |
| **04** | [**Anomaly Detection**](demos/04_anomaly_detection/) | Equipment Failure | **1.0000** vs 0.9167 Acc | **0.0001 Log Loss**, 6.7% rare failure isolation |
| **05** | [**Data Generation**](demos/05_data_generation/) | Clinical Patients | **0.8000** vs 0.5000 Acc | Privacy-preserving synthetic table expansion |
| **06** | [**Text Integration**](demos/06_text_integration/) | Support Escalations | **1.0000** vs 0.5833 Acc | **+41.7% Lift**, multimodal text embeddings |
| **07** | [**Interpretability**](demos/07_interpretability/) | Diabetes Diagnosis | **0.8333** vs 0.6667 Acc | Permutation importance: `glucose_level` driver |

👉 **[Read the Full Empirical Report (CASE_STUDIES.md)](CASE_STUDIES.md)** for detailed clinical/business insights, high-res UI screenshots, rendered HTML snapshots, and terminal audit logs.

---

## 🛠️ Quickstart

### Prerequisites
- **Python 3.12**
- **uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 1. Clone & Setup
```bash
git clone https://github.com/aniketkharad/tabpfn-chat.git
cd tabpfn-chat
uv sync
```

### 2. Configure Environment
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```
```dotenv
GEMINI_API_KEY=your_gemini_api_key
TABPFN_TOKEN=your_priorlabs_token
LLM_PRIMARY_MODEL=gemini-3.8-flash
LLM_FALLBACK_MODEL=gemini-3.8-flash-lite
```

### 3. Run Web Application
```bash
uv run uvicorn tabchat.main:app --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

> **💡 Zero-Token Mock Mode**: Test the complete UI, chat, and results flow offline without using API quotas:  
> **[http://localhost:8000?mock=1](http://localhost:8000?mock=1)**

---

## 📊 Sample Datasets

Generate reproducible test datasets locally:
```bash
uv run python scripts/make_samples.py
```
- **`churn_sample.csv`** (140 rows × 8 cols): Binary classification (tests ID leakage detection).
- **`housing_sample.csv`** (120 rows × 7 cols): Continuous regression (tests chronological split logic).

*(Or click the 1-click **⚡ Churn Sample** / **🏡 Housing Sample** buttons directly in the UI)*

---

## 🔍 Terminal Audit & Replay

Inspect session history, generated specs, baseline comparisons, and metrics directly from your CLI:
```bash
uv run python -m tabchat.replay <session_id>
```

---

## 🧪 Testing (Offline-First)

All 84 automated tests run offline with zero external network or API key dependencies:
```bash
uv run pytest
```

---

## 🛡️ Operational Limits

| Constraint | Limit | Violation Handling |
| :--- | :--- | :--- |
| **File Size** | $\le$ 1 MB | HTTP 422 with granular validation errors |
| **Dimensions** | 1–200 rows, 2–50 columns | HTTP 422 boundary rejection |
| **Cell Boundaries** | Strings $\le$ 64 chars, finite floats ($\|x\| \le 65504$), int32 | HTTP 422 with row & col indices |
| **Chat Turns** | 15 turns per session | HTTP 400 turn cap warning |
| **Rate Limit** | 12 RPM sliding window | Sequential queue delay / Flash-Lite fallback |

---

## 📄 License & Terms

Released under the [MIT License](LICENSE) ([GitHub](https://github.com/aniketkharad/tabpfn-chat/blob/main/LICENSE)).

- **TabPFN-Rel & TabPFN-3.5 API**: Subject to Prior Labs terms of service and model licenses. See [Prior Labs Documentation](https://docs.priorlabs.ai/) and [PriorLabs/tabpfn-rel](https://github.com/PriorLabs/tabpfn-rel).
- **Google Gemini API**: Subject to the [Google APIs Terms of Service](https://developers.google.com/terms) and [Gemini API Additional Terms of Service](https://ai.google.dev/terms). In compliance with free-tier terms, TabPFN Chat transmits only structural schema metadata and 5 preview rows to the LLM—never full raw tabular datasets.
