"""LLM Planner for conversational intent clarification and JobSpec compilation."""

from __future__ import annotations

import json
import logging
from typing import Any
from pydantic import BaseModel, Field

from tabchat.llm.client import FakeLLMClient, GeminiLLMClient
from tabchat.spec import JobSpec, validate_job_spec

logger = logging.getLogger("tabchat.planner")

PLANNER_SYSTEM_PROMPT = """You are the TabPFN Analytical Planner.
Your role is to help the user define a rigorous machine learning job for a small tabular dataset.

Hard Rules:
1. You only see the Dataset Card (column names, inferred data types, null/unique counts, and top 5 preview rows).
2. NEVER assume or invent columns not present in the Dataset Card.
3. Target column must exist in the Dataset Card and must NEVER be included in feature_columns.
4. feature_columns must contain at least 1 column and all must exist in the Dataset Card.
5. If recommending chronological or group splits, split_column must exist in the Dataset Card.
6. For classification, target should be categorical or binary. For regression, target must be numeric.
7. Be concise, mathematically grounded, and transparent about limitations.
"""


class JobSpecPlanResult(BaseModel):
    """Container for planned JobSpec or clarifying request."""

    is_spec_ready: bool = Field(
        ...,
        description="True if user intent is sufficiently clear to compile a JobSpec, False if clarification is needed",
    )
    message: str = Field(
        ...,
        description="Conversational response or explanation to the user",
    )
    job_spec: JobSpec | None = Field(
        default=None,
        description="The compiled JobSpec if is_spec_ready is True, otherwise null",
    )


class JobSpecRepairError(ValueError):
    """Raised when JobSpec validation fails after maximum auto-repair attempts."""


class Planner:
    """Conversational intent clarifier and JobSpec compiler using Gemini."""

    def __init__(self, llm_client: GeminiLLMClient | FakeLLMClient) -> None:
        self.llm = llm_client

    def _format_context_prompt(
        self,
        dataset_card: dict[str, Any],
        extra_instructions: str = "",
    ) -> str:
        card_summary = {
            "row_count": dataset_card.get("row_count"),
            "col_count": dataset_card.get("col_count"),
            "column_names": dataset_card.get("column_names", []),
            "inferred_types": dataset_card.get("inferred_types", {}),
            "null_counts": dataset_card.get("null_counts", {}),
            "unique_counts": dataset_card.get("unique_counts", {}),
            "preview_rows": dataset_card.get("preview_rows", [])[:5],
        }
        prompt = (
            f"Dataset Card:\n```json\n{json.dumps(card_summary, indent=2)}\n```\n\n"
        )
        if extra_instructions:
            prompt += f"{extra_instructions}\n"
        return prompt

    async def chat_turn(
        self,
        history: list[dict[str, Any]],
        dataset_card: dict[str, Any],
    ) -> str:
        """Conduct a conversational turn to answer questions and clarify intent."""
        system_instruction = (
            PLANNER_SYSTEM_PROMPT
            + "\n\n"
            + self._format_context_prompt(dataset_card)
        )
        return await self.llm.generate_text(
            messages=history,
            system_prompt=system_instruction,
        )

    async def generate_job_spec(
        self,
        history: list[dict[str, Any]],
        dataset_card: dict[str, Any],
        max_repairs: int = 2,
    ) -> JobSpec:
        """Compile conversation into a validated JobSpec, repairing up to max_repairs times."""
        system_instruction = (
            PLANNER_SYSTEM_PROMPT
            + "\n\n"
            + self._format_context_prompt(
                dataset_card,
                "Output a valid JobSpec matching the user intent and dataset structure.",
            )
        )

        messages = list(history)
        # Request structured generation
        current_spec: JobSpec | None = None
        attempt = 0

        while attempt <= max_repairs:
            logger.info("Planner generating JobSpec (attempt %d of %d)...", attempt + 1, max_repairs + 1)
            try:
                current_spec = await self.llm.generate_structured(
                    messages=messages,
                    response_schema=JobSpec,
                    system_prompt=system_instruction,
                )
            except Exception as e:
                logger.warning("Structured generation exception on attempt %d: %s", attempt + 1, e)
                if attempt == max_repairs:
                    raise JobSpecRepairError(f"Failed to generate valid JobSpec structure: {e}") from e
                attempt += 1
                messages.append({
                    "role": "user",
                    "content": f"The JobSpec structure was invalid: {e}. Please correct it.",
                })
                continue

            # Validate against dataset card
            validation_errors = validate_job_spec(current_spec, dataset_card)
            if not validation_errors:
                logger.info("Planner JobSpec successfully validated.")
                return current_spec

            logger.warning(
                "JobSpec validation failed on attempt %d: %s",
                attempt + 1,
                "; ".join(validation_errors),
            )

            if attempt == max_repairs:
                raise JobSpecRepairError(
                    f"JobSpec failed validation after {max_repairs} repair attempts: {'; '.join(validation_errors)}"
                )

            # Auto-repair prompt turn
            attempt += 1
            repair_prompt = (
                f"Your proposed JobSpec had validation errors:\n"
                + "\n".join(f"- {err}" for err in validation_errors)
                + "\n\nPlease correct the JobSpec and ensure it strictly respects the dataset columns and constraints."
            )
            messages.append({"role": "assistant", "content": current_spec.model_dump_json(indent=2)})
            messages.append({"role": "user", "content": repair_prompt})

        raise JobSpecRepairError("Exceeded maximum JobSpec repair attempts.")
