"""Planner module for tabchat: turns user intent into a validated JobSpec or clarifying questions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from tabchat.llm.client import FakeLLMClient, GeminiLLMClient
from tabchat.schemas.job_spec import JobSpec
from tabchat.services.spec_compiler import compile_and_verify_spec
from tabchat.skills.registry import get_registry
from tabchat.storage import SessionStore

logger = logging.getLogger("tabchat.planner")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PLANNER_PROMPT_PATH = PROMPTS_DIR / "planner_system.md"


def load_planner_system_prompt() -> str:
    """Load planner system prompt from markdown file."""
    if PLANNER_PROMPT_PATH.exists():
        return PLANNER_PROMPT_PATH.read_text(encoding="utf-8")
    return "You are the TabPFN Analytical Planner. Formulate JobSpec or ask clarifying questions."


class PlannerOutput(BaseModel):
    """Structured response container for the Planner LLM."""

    is_clarification: bool = Field(
        ...,
        description="True if intent is ambiguous and clarifying questions are needed; False if JobSpec is ready",
    )
    clarification_message: Optional[str] = Field(
        default=None,
        description="Clarifying questions or feedback when intent is ambiguous",
    )
    selected_skills: list[str] = Field(
        default_factory=list,
        description="List of skill recipe names selected from the skills index",
    )
    proposed_spec: Optional[JobSpec] = Field(
        default=None,
        description="The structured JobSpec when intent is clear and actionable",
    )


def _format_planner_context(dataset_card: dict[str, Any], skills_index: str) -> str:
    """Format dataset card summary and available skills for LLM system context."""
    card_summary = {
        "row_count": dataset_card.get("row_count"),
        "col_count": dataset_card.get("col_count"),
        "column_names": dataset_card.get("column_names", []),
        "inferred_types": dataset_card.get("inferred_types", {}),
        "null_counts": dataset_card.get("null_counts", {}),
        "unique_counts": dataset_card.get("unique_counts", {}),
        "preview_rows": dataset_card.get("preview_rows", [])[:5],
    }
    return (
        f"\n\n## Available Skills\n{skills_index}\n\n"
        f"## Active Dataset Card\n```json\n{json.dumps(card_summary, indent=2)}\n```\n"
    )


async def plan_analysis(
    session_id: str,
    user_message: str,
    llm_client: GeminiLLMClient | FakeLLMClient | None = None,
    store: SessionStore | None = None,
) -> dict[str, Any]:
    """Turn user message into a validated JobSpec or clarification output.

    Steps:
    1. Append user message to history.jsonl.
    2. Check dataset presence; format prompt with dataset_card + skills + history.
    3. Call LLM with PlannerOutput schema.
    4. If proposed_spec is provided, compile_and_verify_spec.
       - If invalid, auto-repair by feeding error back to LLM once.
       - If still invalid, convert to clarification explaining why dataset cannot support request.
    5. Save valid proposed_spec to session store.
    6. Append assistant response to history.jsonl.
    7. Return planner output dictionary.
    """
    store = store or SessionStore()
    llm = llm_client or GeminiLLMClient()

    # 1. Append user message to history.jsonl
    store.append_history(session_id, role="user", message=user_message)

    # 2. Retrieve dataset card and DataFrame
    try:
        df, dataset_card = store.get_dataset(session_id)
    except FileNotFoundError:
        clarification = (
            "Please upload a CSV dataset before defining an analytical modeling task."
        )
        output = PlannerOutput(
            is_clarification=True,
            clarification_message=clarification,
            selected_skills=[],
            proposed_spec=None,
        )
        store.append_history(session_id, role="assistant", message=clarification)
        return output.model_dump()

    # Retrieve skills index and chat history
    skills_index = get_registry().get_skills_index()
    raw_history = store.get_history(session_id)
    llm_messages = [{"role": msg["role"], "content": msg["content"]} for msg in raw_history]

    system_prompt = load_planner_system_prompt() + _format_planner_context(dataset_card, skills_index)

    # 3. Call LLM for initial PlannerOutput
    try:
        planner_res: PlannerOutput = await llm.generate_structured(
            messages=llm_messages,
            response_schema=PlannerOutput,
            system_prompt=system_prompt,
        )
    except Exception as e:
        logger.error("Planner LLM invocation failed: %s", e)
        # Fallback to clarifying question on LLM parse error
        planner_res = PlannerOutput(
            is_clarification=True,
            clarification_message="I could not interpret your request into a valid plan. Could you clarify your target column?",
            selected_skills=[],
            proposed_spec=None,
        )

    # 4. If a proposed_spec is provided, compile and verify against DataFrame
    if not planner_res.is_clarification and planner_res.proposed_spec is not None:
        is_valid, errors = compile_and_verify_spec(planner_res.proposed_spec, df)
        if not is_valid:
            logger.warning(
                "Initial JobSpec validation failed with errors: %s. Initiating auto-repair...",
                errors,
            )
            # Auto-repair: Feed error back to LLM once with the validation error message
            error_msg = (
                "The proposed JobSpec failed dataset validation with the following errors:\n"
                + "\n".join(f"- {err}" for err in errors)
                + "\n\nPlease correct the JobSpec parameters (target_column, feature_columns, split_column, etc.) "
                "or explain why the dataset cannot support this request."
            )
            repair_messages = list(llm_messages) + [
                {
                    "role": "assistant",
                    "content": f"Proposed spec: {planner_res.proposed_spec.canonical_json()}",
                },
                {"role": "user", "content": error_msg},
            ]

            try:
                repaired_res: PlannerOutput = await llm.generate_structured(
                    messages=repair_messages,
                    response_schema=PlannerOutput,
                    system_prompt=system_prompt,
                )
                if not repaired_res.is_clarification and repaired_res.proposed_spec is not None:
                    re_valid, re_errors = compile_and_verify_spec(repaired_res.proposed_spec, df)
                    if re_valid:
                        planner_res = repaired_res
                    else:
                        # Auto-repair failed: convert to clarification explaining why data cannot support request
                        planner_res = PlannerOutput(
                            is_clarification=True,
                            clarification_message=(
                                "The requested analysis cannot be run on this dataset because: "
                                + "; ".join(re_errors)
                            ),
                            selected_skills=repaired_res.selected_skills,
                            proposed_spec=None,
                        )
                else:
                    planner_res = repaired_res
            except Exception as repair_exc:
                logger.error("Auto-repair LLM invocation failed: %s", repair_exc)
                planner_res = PlannerOutput(
                    is_clarification=True,
                    clarification_message=(
                        "The proposed analysis specification encountered validation errors: "
                        + "; ".join(errors)
                    ),
                    selected_skills=[],
                    proposed_spec=None,
                )

    # 5. Persist valid JobSpec if present
    if not planner_res.is_clarification and planner_res.proposed_spec is not None:
        store.save_job_spec(session_id, planner_res.proposed_spec.model_dump())
        assistant_reply = (
            f"I have configured an analysis plan to predict '{planner_res.proposed_spec.target_column}' "
            f"using {len(planner_res.proposed_spec.feature_columns)} features ({planner_res.proposed_spec.task_type}). "
            f"{planner_res.proposed_spec.rationale}"
        )
    else:
        assistant_reply = (
            planner_res.clarification_message
            or "Could you clarify what you'd like to predict or analyze in this dataset?"
        )

    # 6. Append assistant response to history.jsonl
    store.append_history(session_id, role="assistant", message=assistant_reply)

    # 7. Return planner output dictionary
    return planner_res.model_dump()
