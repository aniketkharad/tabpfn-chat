"""Deterministic executor module for tabchat: executes JobSpec with ZERO LLM calls."""

from __future__ import annotations

import logging
from typing import Any, Callable

from tabchat.schemas.job_spec import JobSpec
from tabchat.services.tabpfn_runner import run_tabpfn_job
from tabchat.storage import SessionStore

logger = logging.getLogger("tabchat.executor")


async def execute_plan(
    session_id: str,
    store: SessionStore | None = None,
    api_caller: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a validated JobSpec deterministically against TabPFN.

    STRICT MANDATE: ZERO LLM CALLS.
    Steps:
    1. Load job_spec.json and raw.csv from session store.
    2. Invoke tabpfn_runner.run_tabpfn_job(spec, df, raw_bytes).
    3. Save returned result dictionary to data/sessions/<session_id>/results.json.
    4. Return results dict.
    """
    store = store or SessionStore()

    # 1. Load job_spec.json
    spec_dict = store.get_job_spec(session_id)
    if spec_dict is None:
        raise ValueError(
            f"No job_spec found for session '{session_id}'. Please define a plan before running."
        )
    spec = JobSpec.model_validate(spec_dict)

    # Load raw.csv and DataFrame
    df, _ = store.get_dataset(session_id)
    raw_path = store.sessions_dir / session_id / "raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file missing for session '{session_id}'.")
    raw_bytes = raw_path.read_bytes()

    # 2. Invoke cached TabPFN runner (Zero LLM calls, deterministic baselines & scoring)
    logger.info("Executing deterministic plan for session %s (task: %s)", session_id, spec.task_type)
    results = await run_tabpfn_job(
        spec=spec,
        df=df,
        raw_bytes=raw_bytes,
        api_caller=api_caller,
    )

    # 3. Save returned result dictionary to results.json
    store.save_results(session_id, results)

    # 4. Return results dict
    return results
