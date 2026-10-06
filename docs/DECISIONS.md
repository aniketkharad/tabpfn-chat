# Architectural Decisions & Confirmed Consensus

Date: **2026-10-06**  
Status: **Approved & Aligned with updated AGENTS.md**

---

## 1. Confirmed Architectural Decisions

### 1.1 Direct REST Architecture (`httpx`)
- **Decision**: Adopt the Direct REST API integration via `httpx` to communicate directly with Prior Labs' endpoints (`https://api.priorlabs.ai/tabpfn/*`).
- **Rationale**:
  - Eliminates heavy C-extensions and large frameworks (`torch`, `tabpfn-rel`, `relbench`, `featuretools`, `lightgbm`, `duckdb`).
  - Keeps Docker image size < 150 MB and idle memory RSS < 40 MB.
  - Zero developmental setback: The direct REST endpoints are the foundational, official JSON contract that Prior Labs itself maintains for cloud execution.

### 1.2 Model Selection & Quota Strategy
- **Decision**:
  - **Standard Primary Model**: `gemini-3.8-flash` (loaded via env `LLM_PRIMARY_MODEL`).
  - **Fallback Model**: `gemini-3.5-flash-lite` / `gemini-flash-lite-latest` (loaded via env `LLM_FALLBACK_MODEL`).
- **Rate Limiting & Safety**:
  - Enforce concurrency lock `asyncio.Semaphore(1)`.
  - Token-bucket rate limiter capped at $\le 12$ RPM.
  - On 429 burst errors, fall back to Flash-Lite.
  - On daily quota exhaustion, trip fatal 503 kill switch.

### 1.3 LLM Input & Data Privacy Boundaries
- **Decision**: The LLM will only receive the `dataset_card`:
  - Sanitized column names, inferred data types, null counts, unique counts.
  - Preview of the top 1 to 5 rows for semantic comprehension of table schema and intent.
- **Safety**: Raw table bytes are NEVER sent to the LLM and cell values are NEVER logged to terminal or disk log files. Only deterministic executor code handles full CSV bytes.

### 1.4 Web Stack & Local-First Persistence
- **Decision**:
  - Backend: **FastAPI**, **pydantic-settings**, **httpx**, and Python standard library.
  - Frontend: Vanilla HTML5, CSS3, JavaScript (single file, no npm, no build step, no framework bloat, < 60 KB).
  - Local-First Persistence: Sessions live under `data/sessions/<session_id>/` (raw data, dataset card, chat history, job specs, results). State survives `uvicorn --reload`.
  - Disk Fit Caching: TabPFN runs cached at `data/cache/tabpfn/<hash>.json` keyed by `sha256(raw_csv_bytes + canonical_job_spec_json)`. Never re-run identical jobs on the API.

### 1.5 Hosting Scope
- **Decision**: Develop and test locally first (`localhost` via `uvicorn`). No cloud service setup or deployment for now until end-to-end local testing is complete.
- **Legacy Dependencies Benchmark**: Skipped. Keep workspace lean without downloading 2.5 GB of legacy wheels.
