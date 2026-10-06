# PART 0: `AGENTS.md` (paste into the new repo)

## Mission

tabpfn-chat is a lightweight public web app. A user uploads a small table and chats with an LLM (Gemini now, swappable) to clarify what they want. The backend turns the agreed intent into a validated job spec, runs it on the TabPFN-3.5 hosted API, and explains results with calibrated uncertainty. Everything that happens is logged in order.

## Working rules (non-negotiable)

1. **Human in the loop.**
   - Before ANY dependency install or change, any dataset download, any new external service or account, or any deploy: STOP. State what, why, size, and wait for my "proceed".
   - Keys: `GEMINI_API_KEY`, `TABPFN_TOKEN`. Ask me to add them to `.env` and wait for "done". Never print, log, echo, commit or request secrets in chat. fCheck presence with booleans only.
   - Stop at every CHECKPOINT and summarize: done / next / needs from me.
   - Before commands that touch the network or files outside the repo, write one line: "About to do X because Y".
2. **uv only.** `uv init/add/sync/run`; Python pinned; project-local venv; never `pip`, `sudo` or global installs; never touch files outside the repo. Prefer `uv run --env-file .env`.
3. **Light by budget.** Targets (measure and report, justify any miss): backend idle RSS <= 256 MB, Docker image <= 400 MB, backend runtime dependencies as few as possible, frontend <= 60 KB gzipped with no framework and no CDN, hand-written backend Python <= ~1,500 lines, frontend <= ~600 lines. Ask before exceeding.
4. **Verify, don't recall.** Never hard-code API, model, quota or pricing facts from memory. Record verified facts with source URLs and dates in `docs/FACTS.md`. Model ids come from env (`LLM_PRIMARY_MODEL`, `LLM_FALLBACK_MODEL`).
5. **Secrets and logs.** Keys only in env or the host secret store, never in code, images, git, logs, error messages or the browser. All logging goes through one redacting logger. Uploaded cell values are never logged or persisted.
6. **Offline-first tests.** Unit tests use fake LLM and fake TabPFN providers and need no network. Live tests are opt-in.
7. **Honesty.** README and results contain only measured numbers from runs actually executed.
8. **Progress.** Keep `docs/PROGRESS.md` updated (done, open, exact commands) so a fresh session can continue.

---

# updated AGENTS.md;

## Mission

Build a production-grade, local-first web app (`tabchat`) running on FastAPI and vanilla HTML/CSS/JS. A user uploads a CSV (<=200 rows, <=50 columns), chats with an LLM (Gemini 3.8 Flash with Flash-Lite fallback) to clarify intent, generates a validated Pydantic Job Spec, executes it deterministically against TabPFN-3.5 via Prior Labs' API with ZERO LLM calls during training, and receives a grounded explanation with calibrated uncertainty.

## Architectural Mandates

1. Local-First & Disk-Backed: Everything lives under `data/sessions/<session_id>/` (raw data, dataset cards, chat history, job specs, results). The backend must survive process restarts (`uvicorn --reload`) without losing active session state or cached artifacts.
2. Zero LLM Calls in Execution: The LLM plans (Planner) and narrates (Narrator). The training, split validation, baseline computation, metric scoring, and TabPFN API invocation (Executor) are 100% deterministic Python code.
3. Disk Fit Caching: Every TabPFN run is hashed by `sha256(raw_csv_bytes + canonical_job_spec_json)`. Results are stored in `data/cache/tabpfn/<hash>.json`. Never pay API tokens twice for the exact same dataset and spec.
4. Strict Concurrency & Rate Limiting: Lock all LLM API invocations behind an `asyncio.Semaphore(1)`. Enforce a token-bucket rate limiter of <= 12 RPM (against the 15 RPM free cap). On 429 burst errors, fall back to Flash-Lite. On daily quota exhaustion (500 RPD), immediately trip a fatal 503 kill switch.
5. No Framework Bloat: Pure FastAPI, pydantic-settings, and standard library. Vanilla JS/HTML/CSS for frontend. No Node.js build step, no Docker multi-stage cloud bloat, no remote log shippers, no hash-chain gimmicks.

## Hard Operating Boundaries

- Upload Caps: File size <= 1 MB, UTF-8 encoded, <= 200 rows, <= 50 columns. String cells <= 64 characters. Integers within signed int32 (-2,147,483,648 to 2,147,483,647). Floats finite and within IEEE 754 half-precision range (|x| <= 65504).
- Turn Caps: Maximum 15 conversation turns per session. Maximum 2 auto-repair attempts for invalid job specs.
- Safety & Privacy: LLM never sees the full raw CSV table—only the `dataset_card` (column names, inferred dtypes, null counts, unique counts, and top 5 preview rows). Cell values are NEVER logged to terminal or disk log files.
