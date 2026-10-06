# Codebase Audit: RelPilot to tabpfn-chat

Date: **2026-10-06**  
Audited Codebase: `tabpfn-prj` (RelPilot)  
Target Application: `tabpfn-chat` (Lightweight Public Web App)

---

## 1. Module-by-Module Audit Matrix

| File / Module | Disposition | Rationale & Security Assessment |
| :--- | :---: | :--- |
| `src/relpilot/__init__.py` | **DROP** | Trivial boilerplate ("Hello from relpilot!"). |
| `src/relpilot/server.py` | **DROP** | **High Server Risk**: MCP stdio server built for local Claude/Cursor desktop tools. Directly reads from and writes to arbitrary local paths (`runs/`, `outbox.jsonl`). Exposes internal server paths in JSON responses. Not suitable for a stateless public web service. |
| `src/relpilot/engine.py` | **REWRITE** | **High Server Risk**: Imports heavy dependencies (`tabpfn_rel`, `torch`, `duckdb`, `scikit-learn`, `pandas`). Writes `.parquet` and `.json` artifacts to local disk (`runs/`). Deep Feature Synthesis consumes hundreds of megabytes of RAM and saturates CPU.<br>**Valuable Core**: Keep the mathematical evaluation logic (`compute_ece`, `compute_precision_and_lift_at_k`, baseline comparisons) and re-implement as lean, in-process functions operating on predictions returned from the hosted REST API. |
| `src/relpilot/gate.py` | **KEEP / REWRITE** | **Core Algorithm (KEEP)**: `wilson_lower_bound` and `find_act_threshold` are pure mathematical routines that operate on numpy/array slices with zero side effects, no network calls, and no disk writes. Excellent for calibrated uncertainty gating.<br>**Server Risk (DROP)**: `record_outbox` appends to a shared local `outbox.jsonl` file. On a multi-tenant public server, this causes file lock contention, disk exhaustion, and privacy leaks. Replace with in-memory session history and structured audit logging. |
| `src/relpilot/agent.py` | **REWRITE** | **Architecture Mismatch**: Built as a CLI REPL (`input()`) executing MCP tool calls over stdio. In `tabpfn-chat`, the agent runs as a stateless web chat controller using Gemini (`gemini-3.8-flash` / `gemini-3.5-flash-lite`) via SSE streaming or structured JSON schema to clarify user intent, extract the target column, and synthesize explanations. |
| `src/relpilot/workspace.py` | **DROP** | **High Server Risk**: Assumes persistent local filesystem hierarchy (`schema.yaml`, `tasks/*.yaml`, `skills/*.md`, `data/*.csv`). On a public web app, datasets are uploaded small tables dynamically passed in memory; schema introspection must happen dynamically in memory without touching disk. |
| `src/relpilot/demo.py` | **DROP** | Scripted CLI demonstration runner. Irrelevant for the web app. |
| `src/relpilot/report.py` | **REWRITE** | **Valuable Asset**: Standalone, clean SVG generator (< 60 lines, no external CDNs) creating reliability calibration diagrams and HTML reports. Can be adapted for server-rendered SVG or transferred directly to the lean vanilla frontend (< 60 KB). |
| `src/relpilot/smoke.py` | **DROP** | CLI smoke test script relying on `tabpfn-rel` and local files. |
| `tests/test_relpilot.py` | **REWRITE** | Mathematical unit tests for Wilson bounds, threshold monotonicity, ECE calculation, and calibration are excellent offline tests that must be kept. Tests for `Workspace` filesystem validation and MCP server can be dropped. |
| `scripts/prepare_data.py` | **DROP** | **Severe Server Risk**: Uses `kagglehub` to download multi-gigabyte datasets from Kaggle to disk. Public servers must NEVER execute unauthenticated or untrusted data downloads. |
| `spike/*` | **DROP** | One-off relational exploration scripts (`run_spike2.py`, `run_step4.py`, task YAMLs). |
| `workspaces/*` | **DROP** | Legacy local database tables and task definitions. |
| `docs/NOTES.md` | **KEEP** | Historical research documentation (preserved for attribution). |
| `docs/RESULTS.md` | **KEEP** | Benchmark metrics on Olist dataset (preserved for attribution). |
| `docs/demo_transcript.md`| **KEEP** | Log of legacy run execution. |

---

## 2. Server Safety & Threat Model for a Public Web App

The legacy `tabpfn-prj` codebase assumed a **trusted, single-user, local machine** environment with no auth and no web UI. Deploying that design onto a public web server introduces critical vulnerabilities:

1. **Arbitrary File System Writes & Traversal**:
   - Old code: Writes to `runs/<run_id>/predictions.parquet`, `runs/<run_id>/report.html`, and `outbox.jsonl`.
   - Threat: Server disk exhaustion, path traversal attacks, unauthenticated data exposure, and cross-session data leaks.
   - Mitigation: Stateless design. Zero disk persistence for uploaded table rows or predictions. All transient state is kept strictly in-memory per request/session and garbage collected immediately.
2. **Data Privacy & Cell Value Leakage**:
   - Rule 5 of `AGENTS.md` strictly specifies: *"Uploaded cell values are never logged or persisted."*
   - Old code: Merged raw CSV tables, logged data rows to stdout/stderr, and persisted parquet files.
   - Mitigation: Redacting logger that strips cell values. Only table metadata (row count, column count, sanitized column names, data types, statistical metrics) may appear in telemetry.
3. **Unbounded Compute & Memory Exhaustion (DoS)**:
   - Old code: Loaded full PyTorch models, deep feature synthesis, and DuckDB multi-table joins.
   - Threat: A single malicious or oversized CSV upload will crash the container via Out-Of-Memory (OOM) or lock CPU cores indefinitely.
   - Mitigation: Strict pre-upload validation: max rows (e.g. <= 2,000), max columns (e.g. <= 30), max file size (e.g. <= 2 MB), timeout limits (<= 30s).
4. **Subprocess & Interactive Blocking**:
   - Old code: Relied on `sys.stdin.read` and stdio MCP server subprocesses.
   - Threat: Blocks the asynchronous web server event loop.
   - Mitigation: Pure asynchronous async/await architecture using lightweight ASGI (e.g. Starlette / FastAPI or minimal asyncio HTTP).

---

## 3. Dependency Footprint & Forbidden Imports

### 3.1 What the Web Runtime Must NOT Import

To strictly meet the non-negotiable budget targets (Docker <= 400 MB, backend idle RSS <= 256 MB, minimal dependencies), the web runtime must **EXCLUDE** the following packages:

| Forbidden Dependency | Installed Weight | Why Forbidden in Web Runtime |
| :--- | :---: | :--- |
| `torch` / `pytorch` | ~1,200 MB | Pulls heavy CUDA/CPU binaries; spikes idle RSS by > 200 MB; unnecessary when using hosted API. |
| `tabpfn` (OSS) | ~150 MB | Requires PyTorch and model weights; designed for local GPU inference. |
| `tabpfn-rel` / `relarena-core` | ~250 MB | Deep Feature Synthesis, RelBench, multi-table relation engines; not needed for single-table chat. |
| `featuretools` / `woodwork` | ~120 MB | Heavy automated feature engineering library; severe memory overhead. |
| `lightgbm` | ~80 MB | C-extensions, OpenMP thread contention issues on macOS/Linux. |
| `duckdb` | ~65 MB | Local embedded analytical database; unnecessary for simple CSV uploads. |
| `mcp` | ~20 MB | Model Context Protocol SDK; designed for LLM desktop tool hosts, not public web APIs. |
| `kagglehub` | ~15 MB | Dataset downloader; public server must never download external datasets. |
| `matplotlib` / `seaborn` | ~110 MB | Heavy plotting libraries; generate pure SVG dynamically or let frontend render canvas/SVG. |

### 3.2 Permitted Lean Runtime Stack

| Package | Purpose | Est. Size |
| :--- | :--- | :---: |
| `starlette` / `uvicorn` (or minimal asyncio) | Ultra-lightweight ASGI web server & SSE streaming | ~5 MB |
| `httpx` | Fast asynchronous HTTP client for TabPFN REST API & Gemini API | ~3 MB |
| `pydantic` | Strict request validation & job spec schemas | ~8 MB |
| `python-multipart` | Streaming CSV file upload parsing | ~1 MB |
| **Total Runtime Dependencies** | **~4 packages** | **< 20 MB** |

---

## 4. Environment & Benchmark Measurements

### Legacy Environment Snapshot (`tabpfn-prj` lockfile)
- Python Version: `3.12.14`
- Total Packages in `uv.lock`: **142 packages**
- Core bloat drivers: `torch==2.14.1`, `scipy==1.18.1`, `pandas==2.3.3`, `duckdb==1.5.6`, `featuretools==1.31.0`, `lightgbm==4.7.0`.

*(Note: Per Rule 1, running a full `uv sync` on the legacy dependency tree to measure live download/sync duration, `.venv` disk usage, and import benchmarks is staged pending user approval).*
