"""LLM Narrator for grounded explanation and calibrated uncertainty reporting."""

from __future__ import annotations

import json
import logging
from typing import Any

from tabchat.llm.client import FakeLLMClient, GeminiLLMClient

logger = logging.getLogger("tabchat.narrator")

NARRATOR_SYSTEM_PROMPT = """You are the TabPFN Analytical Narrator.
Your goal is to provide a clear, rigorous, and grounded explanation of machine learning results on a small tabular dataset.

Mandatory Narration Rules:
1. Grounded in Evidence: You only see results.json and dataset_card.json. NEVER invent numbers or metrics.
2. Baseline Comparison: You must strictly contrast model performance against the local naive baseline:
   - Classification: Compare accuracy and ROC-AUC against majority-class baseline (0.5 AUC). Report lift.
   - Regression: Compare RMSE against naive mean-target RMSE. Report RMSE reduction and R².
3. Calibrated Uncertainty:
   - For classification: Discuss calibration (ECE), Wilson score confidence intervals, and probability reliability.
   - For regression: Discuss 80% prediction interval coverage (q10 to q90) and average interval width.
4. Concrete Inspections: Reference specific sample predictions from sample_predictions to illustrate where the model was confident, where it was uncertain, and any errors.
5. Caveats & Honesty: Emphasize sample size limitations (<= 200 rows), split strategy implications, and potential confounders.
6. Privacy: Do not expose raw individual rows beyond what is summarized in the provided results.
"""


class Narrator:
    """Narrates deterministic execution results using Gemini with grounded uncertainty reporting."""

    def __init__(self, llm_client: GeminiLLMClient | FakeLLMClient) -> None:
        self.llm = llm_client

    def _build_narration_context(
        self,
        results: dict[str, Any],
        dataset_card: dict[str, Any],
    ) -> str:
        # Build clean grounded context without leaking raw CSV rows
        context = {
            "dataset_overview": {
                "total_rows": dataset_card.get("row_count"),
                "total_cols": dataset_card.get("col_count"),
                "column_names": dataset_card.get("column_names", []),
            },
            "job_results": results,
        }
        return json.dumps(context, indent=2, ensure_ascii=False)

    async def narrate_results(
        self,
        results: dict[str, Any],
        dataset_card: dict[str, Any],
    ) -> str:
        """Synthesize a grounded explanation comparing TabPFN to local baseline with calibrated uncertainty."""
        context_json = self._build_narration_context(results, dataset_card)

        user_prompt = (
            f"Here are the deterministic results and dataset summary:\n\n"
            f"```json\n{context_json}\n```\n\n"
            f"Please synthesize a grounded analytical explanation according to the narration rules."
        )

        messages = [{"role": "user", "content": user_prompt}]

        return await self.llm.generate_text(
            messages=messages,
            system_prompt=NARRATOR_SYSTEM_PROMPT,
        )
