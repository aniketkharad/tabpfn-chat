# PROGRESS: tabpfn-chat

Session Date: **2026-10-06**  
Current Status: **Prompt 4 Complete & Tested. 3-Stage Intelligence Engine & FastAPI Routes Fully Operational.**

---

## Completed Tasks

### Prompt 1: Preparation, Audit & Verification
- Cloned commit history from `tabpfn-prj` into `tabpfn_chat` for attribution; disconnected legacy remote.
- Completed comprehensive codebase audit in `docs/AUDIT.md`.
- Completed empirical verification in `docs/FACTS.md`:
  - TabPFN capability matrix with live classification and quantile regression tests via direct REST API.
  - Quota verification (10,000 minimum billable tokens; 5M daily and 20M monthly quota pools).
  - Gemini models (`gemini-3.8-flash`, `gemini-3.5-flash-lite` / `gemini-3.8-flash-lite`).
- Documented decisions and confirmed user consensus in `docs/DECISIONS.md`.

### Prompt 2: Hardened Local Infrastructure Layer
1. **Configuration (`src/tabchat/config.py`)**:
   - Built with `pydantic-settings` to load from `.env` and environment variables.
   - Pinned settings: `GEMINI_API_KEY` (fail-fast), `TABPFN_TOKEN` (fail-fast), `LLM_PRIMARY_MODEL` ("gemini-3.8-flash"), `LLM_FALLBACK_MODEL` ("gemini-3.8-flash-lite"), `DATA_DIR` (Path("data")), `MAX_UPLOAD_BYTES` (1 MB), `MAX_ROWS` (200), `MAX_COLS` (50), `MAX_STR_LEN` (64), `MAX_SESSION_TURNS` (15), `GEMINI_RPM_LIMIT` (12), `GEMINI_CONCURRENCY` (1), `TABPFN_FIT_TIMEOUT_SECONDS` (90).
   - Validates writable `DATA_DIR` on startup with clean error reporting.

2. **Disk-Backed Storage Engine (`src/tabchat/storage.py`)**:
   - Persists all state under `data/sessions/<session_id>/`:
     - `raw.csv`: Original uploaded table bytes.
     - `dataset_card.json`: Structural summary and preview rows.
     - `history.jsonl`: Append-only chat messages with role, content, and ISO timestamp.
     - `job_spec.json`: Analysis specification (null until generated).
     - `results.json`: Execution output, TabPFN predictions, and metrics.
   - Methods: `create_session()`, `session_exists()`, `save_dataset()`, `get_dataset()`, `append_history()`, `get_history()`, `save_job_spec()`, `get_job_spec()`, `save_results()`, `get_results()`, `delete_session()`.
   - Concurrency: `get_lock(session_id)` providing per-session `asyncio.Lock` protection against write collisions.

3. **CSV Validation & Dataset Card Generator (`src/tabchat/validation.py`)**:
   - `validate_and_parse_csv(file_bytes)`:
     - Pre-parse 1 MB byte check.
     - Enforces $1 \le \text{rows} \le 200$, $2 \le \text{cols} \le 50$.
     - Detects empty headers and duplicate headers before pandas de-duplication.
     - Rejects 100% null/empty columns.
     - Rejects cell strings > 64 chars.
     - Enforces signed int32 bounds ($-2,147,483,648$ to $2,147,483,647$).
     - Enforces float16 bounds ($|x| \le 65504$) and rejects non-finite floats (`inf`/`-inf`).
     - Collects up to 20 granular validation errors with explicit row and column indexes.
   - Generates compact dataset card: `row_count`, `col_count`, `column_names`, inferred types (`numeric`, `categorical`, `datetime`), `null_counts`, `unique_counts`, and top 5 JSON-sanitized preview rows.

4. **Rate-Limited LLM Client (`src/tabchat/llm/client.py`)**:
   - `GeminiLLMClient`:
     - Concurrency Governor: Module-level `asyncio.Semaphore(1)`.
     - Rate Governor: Sliding-window token bucket capped at 12 RPM over 60s.
     - Fallback: Retries burst 429 errors once with `LLM_FALLBACK_MODEL` (`gemini-3.8-flash-lite`).
     - Fatal Kill Switch: Detects daily exhaustion (`RESOURCE_EXHAUSTED` / `GenerateRequestsPerDay`) and raises `DailyQuotaExhaustedError`.
     - `generate_structured()` using Pydantic `response_schema`.
     - `generate_text()` with system instruction support.
   - `FakeLLMClient`: Offline mock client sharing identical signatures, governed by semaphore and rate limiter.

5. **Platform Unit Tests (`tests/test_platform.py`)**:
   - Covered:
     - Config fail-fast on missing keys and valid loading.
     - CSV row and col boundaries, string and numeric bounds.
     - Storage engine lifecycle and restart persistence.
     - Sliding window rate limiter pacing and sequentiality.

### docs/SPEC.md : Local TabPFN Chat Specification
1. **Specification Document (`docs/SPEC.md`)**:
   - Codified full local-first web app contracts: system overview, platform and storage rules, JobSpec schema contract, and pipeline execution contract.

2. **JobSpec Schema Contract & Validation (`src/tabchat/spec.py`)**:
   - Pydantic model `JobSpec`:
     - Fields: `task_type`, `target_column`, `feature_columns`, `excluded_columns`, `split_strategy`, `split_column`, `holdout_fraction`, `random_seed`, `eval_metric`, `confidence_intervals`, `rationale`.
     - Cross-field validation enforcing target absence from features, non-empty features, split column requirements, and task-metric compatibility.
   - Canonical Hashing: deterministic SHA-256 keys for caching.

3. **Deterministic Local Baselines & Pure-NumPy Metrics (`src/tabchat/metrics.py`)**:
   - Zero heavy dependencies (no scikit-learn, no torch).
   - Local Baselines: `MajorityClassBaseline`, `MeanTargetBaseline`.
   - Evaluation & Uncertainty Calibration: ROC-AUC via rank statistic, balanced accuracy, log loss, ECE with 10 bins, Wilson confidence intervals, RMSE, MAE, R², 80% prediction interval coverage.

4. **TabPFN Client & Disk Fit Cache (`src/tabchat/tabpfn.py`)**:
   - `DiskFitCache`: Persists TabPFN fit outputs under `data/cache/tabpfn/<dataset_hash>_<spec_hash>.json`.
   - `TabPFNClient` and `FakeTabPFNClient`.

5. **Deterministic Pipeline Executor (`src/tabchat/pipeline/executor.py`)**:
   - Zero LLM calls during training or inference.
   - Deterministic splitting (random, chronological, group).
   - Cache hit checks before API calls.

6. **LLM Planner with 2-Turn Auto-Repair (`src/tabchat/pipeline/planner.py`)**:
   - Privacy boundary: sees only `dataset_card` and chat history, never raw rows.
   - Auto-repair loop with explicit error feedback.

7. **LLM Narrator (`src/tabchat/pipeline/narrator.py`)**:
   - Sees only `results.json` and `dataset_card.json`. Grounded narrative generation.

### Prompt 3: TabPFN Service Bridge, Spec Compiler, Skill Recipes & Cached Runner
1. **Job Spec Schema (`src/tabchat/schemas/job_spec.py`)**:
   - Pydantic V2 model with strict intra-spec validation and canonical JSON generation.
2. **Spec Compiler & Dataset Verifier (`src/tabchat/services/spec_compiler.py`)**:
   - Validates target existence, feature existence, non-null targets, class sample counts, and numeric types for regression.
3. **Skill Recipes & Registry (`src/tabchat/skills/`)**:
   - Markdown recipes (`task_and_target.md`, `leakage_and_splits.md`, `uncertainty_and_metrics.md`).
   - Pure standard library YAML frontmatter parser and compact index generator (< 150 tokens).
4. **Cached TabPFN Service (`src/tabchat/services/tabpfn_runner.py`)**:
   - Deterministic disk cache under `data/cache/tabpfn/<hash>.json`.
   - Local baselines, TabPFN call, holdout scoring, and sample predictions preview.

### Prompt 4: 3-Stage Intelligence Engine & FastAPI Endpoints
1. **System Prompts (`src/tabchat/prompts/`)**:
   - `planner_system.md`: Ruthless DS planner. Inputs: `dataset_card` + Skill Index + History. Clear vs ambiguous intent rules.
   - `narrator_system.md`: Grounded narrator. Inputs: `results.json` + `dataset_card.json`. Baseline comparisons, sample size variance caveats, zero hallucinations.
2. **The Planner Engine (`src/tabchat/engine/planner.py`)**:
   - `plan_analysis(session_id, user_message)`:
     - Appends user message to `history.jsonl`.
     - Invocates LLM with structured `PlannerOutput` schema.
     - Compiles and verifies `JobSpec` against `df`.
     - Auto-repair: feeds validation errors back to LLM once. Converts to clarification if still invalid.
     - Persists valid `JobSpec` to `job_spec.json`.
     - Appends assistant response to `history.jsonl`.
3. **The Executor Engine (`src/tabchat/engine/executor.py`)**:
   - STRICT CONSTRAINT: ZERO LLM CALLS. Deterministic execution only.
   - `execute_plan(session_id)`:
     - Loads `job_spec.json` and `raw.csv`.
     - Invokes `tabpfn_runner.run_tabpfn_job(spec, df, raw_bytes)`.
     - Persists execution output to `results.json`.
4. **The Narrator Engine (`src/tabchat/engine/narrator.py`)**:
   - `narrate_results(session_id)`:
     - Loads `results.json` and `dataset_card.json`.
     - Calls LLM via `generate_text()`.
     - **Hallucination Guard**: Regex scanner verifying all percentages and metric floats against `results.json` values within tolerance rounding; logs severe warning on phantom metrics.
     - Appends narration to `history.jsonl`.
5. **FastAPI Application & HTTP Routes (`src/tabchat/routes/api.py`, `src/tabchat/app.py`)**:
   - `POST /api/session`: Creates session, sets HttpOnly cookie.
   - `GET /api/session`: Resumes session state from disk.
   - `POST /api/upload`: Multipart CSV upload, validation, dataset card generation (200 / 422).
   - `POST /api/chat`: Runs Planner, enforces 15-turn cap, returns reply, plan_card, turns_left.
   - `POST /api/run`: Runs Executor -> Narrator, returns results and narrative.
   - `GET /api/log`: Returns plaintext session log/history.
   - `DELETE /api/session`: Deletes session from disk and clears cookie.
6. **Integration & Server Restart Tests (`tests/test_engine_integration.py`)**:
   - Complete end-to-end flow tested offline: Session -> Upload -> Chat -> Run -> Narrate.
   - Full server restart / client reconnect test proving zero state loss from disk.
   - Auto-repair loop verification.
   - Zero LLM calls assertion during execution.
   - Hallucination Guard detection test.
   - Turn limit cap enforcement (15 turns).
   - 422 structured error response on invalid CSV.
   - **Total test suite: 69 passed in 1.19s** (`uv run --env-file .env pytest -v`).

### Prompt 5: Responsive Single-Page Frontend & Mock Mode
1. **Frontend Architecture & Static Assets (`src/tabchat/frontend/`)**:
   - `index.html`: Semantic HTML5 layout with split two-pane workspace, dataset preview, upload error callouts, and dynamic plan & results card slots.
   - `style.css`: Clean, dark slate responsive stylesheet with system fonts, custom properties, and desktop/mobile breakpoint layouts (160 lines).
   - `app.js`: Vanilla JavaScript SPA with zero external dependencies, mock mode interceptor, XSS-safe micro-markdown parser, 15s/90s timers, inline SVG chart generator, and session persistence (447 lines).
2. **FastAPI StaticFiles Mount (`src/tabchat/app.py`)**:
   - Mounted `src/tabchat/frontend/` at `/static`.
   - Served `index.html` at `/` and `/index.html` via `FileResponse`.
   - Preserves all `/api/*` endpoints without shadowing.
3. **Mock Mode (`?mock=1`)**:
   - Client-side interceptor routing all `/api/*` requests to deterministic mock responses.
   - Supports immediate testing of: CSV drag-and-drop, 422 error display on trigger files, conversational planning, dynamic Plan Card, TabPFN execution, dynamic Results Card with inline SVG confidence bars, 429 rate limit banner, 503 quota modal, and session resets.
4. **Safety & XSS Prevention**:
   - Micro-markdown parser strictly escapes HTML entities (`&`, `<`, `>`, `"`, `'`) before applying formatting tags (`**bold**`, `*italics*`, `` `inline code` ``, `<br>`).
   - Pure `textContent` used for user and system error messages.
5. **Persistence & State Restoration**:
   - Automatically polls `GET /api/session` on load.
   - Restores preview table, chat feed history, plan cards, and execution results seamlessly across browser refreshes.
6. **UI Verification Checklist (`docs/UI_CHECKLIST.md`)**:
   - Documented comprehensive manual testing steps for mock mode, live mode, error scenarios, responsiveness, and security.
7. **Verification & Testing (`tests/test_frontend_routes.py`)**:
   - Automated tests for root endpoint, HTML alias, static CSS/JS serving, and API coexistence.
   - Total test suite: **74 passed in 1.34s** (`uv run --env-file .env pytest -v`).
   - Total frontend size: **12.5 KB gzipped** (well below the 60 KB limit).

---

## Next Steps

- System verification and final end-to-end user checks.
