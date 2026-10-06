# docs/SPEC.md: Local TabPFN Chat Specification

## 1. System Overview
A local-first, disk-backed web application (`tabchat`) using FastAPI and vanilla JS/HTML/CSS.
Flow: User uploads a CSV -> Chat with Gemini to clarify intent -> Gemini outputs a validated `JobSpec` -> Deterministic execution of TabPFN on the backend (ZERO LLM calls during training) -> Gemini narrates results with calibrated uncertainty.

## 2. Platform & Storage Contracts
- Storage root: `data/sessions/<session_id>/`
- Stored session artifacts: `raw.csv`, `dataset_card.json`, `history.jsonl`, `job_spec.json`, `results.json`
- TabPFN cache root: `data/cache/tabpfn/<dataset_hash>_<spec_hash>.json`
- Hard data limits: file <= 1 MB, <= 200 rows, <= 50 columns, string cells <= 64 chars, signed int32, finite float16 (|x| <= 65504).
- LLM guards: Global concurrency locked to 1 (`asyncio.Semaphore(1)`), rate limit <= 12 RPM, max 15 turns per session.

## 3. JobSpec Schema Contract
The `JobSpec` is the compiled contract passed from the Planner to the Executor:
- `task_type`: "classification" | "regression"
- `target_column`: str (must exist in data, cannot be in feature_columns)
- `feature_columns`: list[str] (must exist in data, >= 1 column)
- `excluded_columns`: list[str]
- `split_strategy`: "random" | "chronological" | "group"
- `split_column`: Optional[str] (required for chronological/group splits)
- `holdout_fraction`: float (0.1 to 0.4, default 0.2)
- `random_seed`: int (default 42)
- `eval_metric`: "auto" | "roc_auc" | "accuracy" | "log_loss" | "r2" | "rmse"
- `confidence_intervals`: bool (default True)
- `rationale`: str

## 4. Pipeline Execution Contract
1. Planner: Sees only `dataset_card.json` + chat history + skill index. Never sees raw table rows beyond the 5 preview rows.
2. Executor: Pure Python. 0 LLM calls. Splits data, runs local baseline (majority class or mean), checks cache, invokes TabPFN API, records holdout metrics and sample predictions.
3. Narrator: Sees only `results.json` + `dataset_card.json`. Grounded narration strictly comparing model metrics to baseline metrics.
