"""Narrator module for tabchat: grounded explanation and calibrated uncertainty reporting."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from tabchat.llm.client import FakeLLMClient, GeminiLLMClient
from tabchat.storage import SessionStore

logger = logging.getLogger("tabchat.narrator")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
NARRATOR_PROMPT_PATH = PROMPTS_DIR / "narrator_system.md"

# Whitelisted standard versions or constants that may appear in narrative text
WHITELISTED_FLOATS = {3.5, 3.8, 3.12, 1.0, 0.0, 0.1, 0.2, 0.5, 0.9, 0.8}


def load_narrator_system_prompt() -> str:
    """Load narrator system prompt from markdown file."""
    if NARRATOR_PROMPT_PATH.exists():
        return NARRATOR_PROMPT_PATH.read_text(encoding="utf-8")
    return "You are the TabPFN Analytical Narrator. Compare against baselines and explain uncertainty."


def _collect_numeric_values(data: Any) -> set[float]:
    """Recursively collect all numeric floats and ints from a nested structure."""
    numbers: set[float] = set()
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        val = float(data)
        numbers.add(val)
        numbers.add(round(val, 1))
        numbers.add(round(val, 2))
        numbers.add(round(val, 3))
        numbers.add(round(val, 4))
        # Add percentage representations (e.g. 0.85 -> 85.0)
        if -10.0 <= val <= 10.0:
            pct = val * 100.0
            numbers.add(pct)
            numbers.add(round(pct, 0))
            numbers.add(round(pct, 1))
            numbers.add(round(pct, 2))
    elif isinstance(data, dict):
        for v in data.values():
            numbers.update(_collect_numeric_values(v))
    elif isinstance(data, (list, tuple)):
        for item in data:
            numbers.update(_collect_numeric_values(item))
    return numbers


def verify_hallucination_guard(narration: str, results: dict[str, Any]) -> list[float]:
    """Inspect narration text to verify numbers against results.json.

    Returns a list of phantom metrics detected (empty if all verified).
    Logs severe warnings for any undetected numbers.
    """
    known_numbers = _collect_numeric_values(results)
    known_numbers.update(WHITELISTED_FLOATS)

    phantom_metrics: list[float] = []

    # 1. Check percentage expressions: e.g. "78.5%", "80%"
    pct_matches = re.findall(r"(\b\d+(?:\.\d+)?)\s*%", narration)
    for m in pct_matches:
        try:
            val = float(m)
            # Check if percentage or decimal exists in known numbers
            matched = any(abs(val - k) < 0.2 for k in known_numbers) or any(
                abs(val / 100.0 - k) < 0.02 for k in known_numbers
            )
            if not matched and val not in WHITELISTED_FLOATS:
                phantom_metrics.append(val)
                logger.warning(
                    "HALLUCINATION GUARD: Phantom percentage detected in narration: %s%% not found in results.json",
                    val,
                )
        except ValueError:
            continue

    # 2. Check floating point decimals: e.g. "0.84", "12.35"
    float_matches = re.findall(r"\b(\d+\.\d{1,4})\b", narration)
    for m in float_matches:
        try:
            val = float(m)
            if val in WHITELISTED_FLOATS:
                continue
            matched = any(abs(val - k) < 0.02 for k in known_numbers) or any(
                abs(val * 100.0 - k) < 0.5 for k in known_numbers
            )
            if not matched:
                phantom_metrics.append(val)
                logger.warning(
                    "HALLUCINATION GUARD: Phantom floating-point metric detected in narration: %s not found in results.json",
                    val,
                )
        except ValueError:
            continue

    return phantom_metrics


async def narrate_results(
    session_id: str,
    llm_client: GeminiLLMClient | FakeLLMClient | None = None,
    store: SessionStore | None = None,
) -> str:
    """Generate a grounded analytical narrative explaining deterministic results.

    Steps:
    1. Load results.json and dataset_card.json from session store.
    2. Format prompt with results and metadata.
    3. Call LLM via generate_text.
    4. Hallucination Guard: Verify all percentage and metric numbers against results.json.
    5. Append narration to history.jsonl.
    6. Return narration text.
    """
    store = store or SessionStore()
    llm = llm_client or GeminiLLMClient()

    # 1. Load results.json and dataset_card.json
    results = store.get_results(session_id)
    if results is None:
        raise ValueError(f"No results found for session '{session_id}'. Run the executor first.")

    try:
        _, dataset_card = store.get_dataset(session_id)
    except FileNotFoundError:
        dataset_card = {}

    # 2. Format context for narration
    context = {
        "dataset_card": {
            "row_count": dataset_card.get("row_count"),
            "col_count": dataset_card.get("col_count"),
            "column_names": dataset_card.get("column_names", []),
        },
        "execution_results": results,
    }

    user_prompt = (
        f"Below are the verified execution results and dataset metadata:\n\n"
        f"```json\n{json.dumps(context, indent=2)}\n```\n\n"
        "Synthesize a rigorous, grounded analytical report comparing TabPFN against the naive baseline, "
        "noting holdout sample size, calibrated uncertainty, sample inspections, and any variance caveats."
    )

    messages = [{"role": "user", "content": user_prompt}]
    system_prompt = load_narrator_system_prompt()

    # 3. Call LLM via generate_text
    narration = await llm.generate_text(
        messages=messages,
        system_prompt=system_prompt,
    )

    # 4. Hallucination Guard
    verify_hallucination_guard(narration, results)

    # 5. Append narration to history.jsonl
    store.append_history(session_id, role="assistant", message=narration)

    # 6. Return narration text
    return narration
