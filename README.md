# tabchat

> **A lightweight, local-first chat & prediction web app for TabPFN-3.5.**

`tabchat` enables users to upload small tabular datasets, converse with an LLM (Gemini 3.8 Flash with Flash-Lite fallback) to clarify intent, compile a mathematically validated `JobSpec`, execute it deterministically against **TabPFN-3.5** via Prior Labs' API with **ZERO LLM calls during training**, and receive grounded, uncertainty-calibrated explanations.

---

## Key Architectural Principles

1. **Local-First & Disk-Backed Persistence**: All session artifacts reside under `data/sessions/<session_id>/` (`raw.csv`, `dataset_card.json`, `history.jsonl`, `job_spec.json`, `results.json`). State survives server restarts (`uvicorn --reload`) and hard process termination with zero data loss.
2. **Zero LLM Calls in Execution**: The LLM plans (`Planner`) and narrates (`Narrator`). Partitioning, baseline metrics, model fitting, holdout scoring, and calibration (`Executor`) are 100% deterministic Python code.
3. **Deterministic Disk Fit Caching**: Every TabPFN run is keyed by `sha256(raw_csv_bytes + canonical_job_spec_json)`. Results are cached under `data/cache/tabpfn/<hash>.json`. You never pay API tokens twice for identical data and specs.
4. **Strict Concurrency & Rate Limiting**: All LLM calls are gated behind an `asyncio.Semaphore(1)` and a sliding-window token-bucket limiter capped at **12 RPM** (below the 15 RPM free cap). Burst 429s automatically fall back to Flash-Lite. Quota exhaustion (500 RPD) trips a fatal 503 kill switch.
5. **No Framework Bloat**: Pure FastAPI backend (`uvicorn`, `pydantic-settings`, standard library) and vanilla HTML/CSS/JS frontend (< 14 KB gzipped). Zero Node.js build steps, zero client-side frameworks, and zero CDN dependencies.

---

## 1. Quickstart & Local Setup

### Prerequisites

- **Python 3.12** (pinned)
- **uv** (Astral's fast Python package manager):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Installation

1. **Clone the repository and enter the directory**:
   ```bash
   cd tabpfn_chat
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure Environment Variables**:
   Copy the example environment template:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to supply your API credentials:
   ```dotenv
   GEMINI_API_KEY=your_gemini_api_key_here
   TABPFN_TOKEN=your_priorlabs_token_here
   LLM_PRIMARY_MODEL=gemini-3.8-flash
   LLM_FALLBACK_MODEL=gemini-3.8-flash-lite
   ```
   *(Note: API keys are validated on startup with non-empty checks. Keys are NEVER logged, committed, or exposed in client responses).*

4. **Launch the Local Web Server**:
   ```bash
   uv run uvicorn tabchat.main:app --reload
   ```
   Open your browser at **[http://localhost:8000](http://localhost:8000)**.

5. **Mock Mode (Zero API Token Consumption)**:
   Test the entire interactive UI without consuming Gemini or TabPFN API quotas:
   **[http://localhost:8000?mock=1](http://localhost:8000?mock=1)**

---

## 2. Seeded Sample Datasets

Reproducible sample datasets can be generated locally:
```bash
uv run python scripts/make_samples.py
```

This generates two sample CSVs under `src/tabchat/frontend/static/samples/`:

| Dataset | Rows × Cols | Task Type | Mild Trap Description |
| :--- | :---: | :---: | :--- |
| `churn_sample.csv` | 140 × 8 | Binary Classification | Arbitrary `customer_id` column that the Planner must identify and exclude as an uninformative primary key / leakage. |
| `housing_sample.csv` | 120 × 7 | Continuous Regression | Ordered `sold_date` column requiring chronological holdout split instead of standard random permutation. |

**1-Click UI Buttons**: The upload pane in the web interface includes instant 1-click buttons (`⚡ Churn Sample` and `🏡 Housing Sample`) that automatically load and parse these files without manual downloads.

---

## 3. CLI Replay & Audit Trail Utility

Inspect and audit any active or completed session directly from your terminal:

```bash
uv run python -m tabchat.replay <session_id>
```

Optional arguments:
```bash
uv run python -m tabchat.replay <session_id> --data-dir data
```

### Audit Trail Output Sections:
1. **Session Initialization**: Session ID, disk storage path, and ISO creation timestamp.
2. **Dataset Shape & Column Metadata**: Dimensions, file size, column list with inferred types, null counts, and unique values.
3. **Conversational Chat Turns**: Chronological log of user messages and assistant responses.
4. **Analytical Job Spec**: Formatted schema details (task type, target, features, excluded columns, split strategy, metric, and rationale) plus canonical JSON.
5. **Execution Metrics vs Baseline**: Sample partitions (train vs holdout), side-by-side metric comparison table (Baseline vs TabPFN-3.5 vs Delta Lift), sample predictions preview, and execution warnings.
6. **Final Calibrated Narration**: Grounded textual synthesis with uncertainty caveats.
7. **Diagnostic Warnings**: If any session file is missing or corrupted, the utility outputs explicit warnings while continuing graceful inspection.

---

## 4. Running the Test Suite (Offline-First)

All tests run completely offline with mocked LLM and TabPFN providers. No network calls or API keys are required:

```bash
uv run pytest -v
```

### Test Coverage (84 Automated Tests):
- `tests/test_platform.py`: Config loading, fail-fast validations, storage engine disk persistence, rate limiter sliding window.
- `tests/test_spec.py`: Pydantic JobSpec constraints, canonical JSON serialization, hashing determinism.
- `tests/test_metrics.py`: Pure-NumPy baseline metrics, classification scoring (accuracy, log loss), regression scoring (RMSE, R²).
- `tests/test_pipeline.py`: Pipeline executor, holdout splits, baseline comparisons.
- `tests/test_engine_integration.py`: End-to-end FastAPI endpoints, 3-stage engine, server restart persistence, turn limits (15 turns), auto-repair loop.
- `tests/test_frontend_routes.py`: Static file mounts, single-page application delivery, asset integrity.
- `tests/test_tabpfn_pipeline.py`: Disk fit caching, cache hit/miss behavior, regression quantile handling.
- `tests/test_robustness.py`:
  - **Adversarial CSV Injection**: Verifies that prompt injections embedded in cell values (`"IGNORE PREVIOUS INSTRUCTIONS..."`) are isolated as passive tabular metadata and cannot hijack execution or leak API keys.
  - **Crash Recovery Simulation**: Simulates abrupt process death (`os._exit(1)`) after planning; verifies resumption on a fresh process via disk state.
  - **Extreme Values**: High class imbalance (98 negative, 2 positive) triggers low support warnings; multi-collinear identical columns execute safely.
- `tests/test_replay.py`: Full audit report generation, corrupted file handling, and missing artifact diagnostics.

---

## 5. Hard Operational Boundaries

| Boundary | Limit | Behavior on Violation |
| :--- | :--- | :--- |
| **Max File Upload** | 1 MB (`1,048,576` bytes) | 422 Unprocessable Entity with error list |
| **Row Limits** | 1 to 200 rows | 422 Unprocessable Entity with row count message |
| **Column Limits** | 2 to 50 columns | 422 Unprocessable Entity with column count message |
| **String Cell Length** | &le; 64 characters | 422 Unprocessable Entity specifying row and column |
| **Integer Values** | Signed 32-bit (`-2,147,483,648` to `2,147,483,647`) | 422 Unprocessable Entity |
| **Float Values** | Finite IEEE 754 half-precision (`\|x\| <= 65504.0`) | 422 Unprocessable Entity (rejects `inf`, `-inf`, NaN) |
| **Chat Turns** | 15 turns per session | 400 Bad Request with turn exhaustion message |
| **LLM Concurrency** | 1 active call (`asyncio.Semaphore(1)`) | Sequential queueing |
| **LLM Rate Limit** | 12 Requests Per Minute (RPM) | Paced delay before execution |
| **TabPFN Timeout** | 90 seconds | Operation aborted with timeout retry button |

---

## 6. Troubleshooting & Operational Runbook

### Rate Limit Errors (HTTP 429)
- **Cause**: Gemini free tier enforces a strict limit of 15 RPM.
- **Handling**:
  - `tabchat` paces all outbound LLM requests at **12 RPM** using a token-bucket governor.
  - On temporary burst 429 errors from Google, the client automatically retries once using `LLM_FALLBACK_MODEL` (`gemini-3.8-flash-lite`).
  - If the rate limit is exceeded, the web UI displays a top yellow alert banner: *"Rate limit hit. Waiting for cooldown..."* for 8 seconds. Wait a few seconds before sending another chat message.

### Daily Quota Exhaustion (HTTP 503)
- **Cause**: Google GenAI free tier enforces a cap of 500 Requests Per Day (RPD).
- **Handling**:
  - When the API returns `RESOURCE_EXHAUSTED` or `GenerateRequestsPerDay`, the LLM client immediately trips a fatal kill switch (`DailyQuotaExhaustedError`).
  - The API returns HTTP 503, and the web interface displays an unclosable modal: *"Daily API budget exhausted. Please resume tomorrow."*
  - **Resolution**: Wait until quota resets (midnight Pacific Time) or configure a paid API key in `.env`.

### CSV Validation Errors (HTTP 422)
- If your upload fails, the red error box details the exact violation (up to 20 granular line items).
- Common causes:
  - Header with empty or duplicate column names.
  - Column with 100% missing values.
  - Cell strings exceeding 64 characters (e.g. lengthy text descriptions).
  - Non-standard float values exceeding $\pm 65504$.

### TabPFN API Timeout
- TabPFN hosted fits typically complete within 15–30 seconds.
- If Prior Labs' API experiences high load, the backend enforces a 90-second timeout (`TABPFN_FIT_TIMEOUT_SECONDS=90`).
- If an execution aborts due to timeout, the UI displays a `Retry` button to re-trigger the fit.

---

## 7. Project Structure

```
tabpfn_chat/
├── AGENTS.md                          # Architectural invariants and mandates
├── README.md                          # Local runbook and documentation
├── pyproject.toml                     # Python dependencies (uv pinned)
├── scripts/
│   └── make_samples.py                # Deterministic sample dataset generator
├── src/
│   └── tabchat/
│       ├── app.py                     # FastAPI application factory and routes mount
│       ├── main.py                    # Server entrypoint (uvicorn tabchat.main:app)
│       ├── config.py                  # Pydantic Settings (.env configuration)
│       ├── storage.py                 # Disk-backed session store
│       ├── validation.py              # CSV boundary checks and Dataset Card generator
│       ├── spec.py                    # JobSpec schema & canonical JSON hashing
│       ├── metrics.py                 # Pure-NumPy local baselines and metrics
│       ├── tabpfn.py                  # TabPFN client and disk fit caching
│       ├── replay.py                  # CLI replay & terminal audit utility
│       ├── engine/
│       │   ├── planner.py             # Analytical planner with auto-repair
│       │   ├── executor.py            # Zero-LLM deterministic pipeline executor
│       │   └── narrator.py            # Grounded narrator with hallucination guard
│       ├── frontend/
│       │   ├── index.html             # Semantic SPA layout
│       │   ├── style.css              # Dark slate stylesheet (< 170 lines)
│       │   ├── app.js                 # Vanilla JS SPA (< 500 lines)
│       │   └── samples/               # Seeded CSV sample datasets
│       ├── llm/
│       │   └── client.py              # Semaphore-gated Gemini client with token bucket
│       ├── prompts/                   # Planner and Narrator system prompts
│       ├── routes/
│       │   └── api.py                 # FastAPI REST API endpoints
│       ├── schemas/                   # Pydantic JobSpec schemas
│       ├── services/                  # Spec compiler and cached TabPFN runner
│       └── skills/                    # Domain heuristic recipe registry
└── tests/                             # 84 offline unit, integration, and robustness tests
```

---

## 8. License

This project is licensed under the [MIT License](LICENSE).
TabPFN-3.5 is subject to Prior Labs terms of service and model licenses.
