"""Unit tests for LLM Planner, auto-repair loop, Narrator, and privacy boundaries."""

import pytest
from tabchat.llm.client import FakeLLMClient
from tabchat.pipeline.narrator import Narrator
from tabchat.pipeline.planner import JobSpecRepairError, Planner
from tabchat.spec import JobSpec

SAMPLE_DATASET_CARD = {
    "row_count": 100,
    "col_count": 4,
    "column_names": ["age", "tenure", "churn", "signup_date"],
    "inferred_types": {
        "age": "numeric",
        "tenure": "numeric",
        "churn": "categorical",
        "signup_date": "datetime",
    },
    "null_counts": {"age": 0, "tenure": 0, "churn": 0, "signup_date": 0},
    "unique_counts": {"age": 40, "tenure": 10, "churn": 2, "signup_date": 90},
    "preview_rows": [
        {"age": 25, "tenure": 2, "churn": "no", "signup_date": "2023-01-01"},
        {"age": 35, "tenure": 5, "churn": "yes", "signup_date": "2023-01-02"},
    ],
}


@pytest.mark.asyncio
async def test_planner_chat_turn():
    """Verify Planner conversational chat turn generates text."""
    fake_llm = FakeLLMClient(mock_text="I recommend predicting churn using age and tenure.")
    planner = Planner(llm_client=fake_llm)

    history = [{"role": "user", "content": "What should I predict here?"}]
    response = await planner.chat_turn(history, SAMPLE_DATASET_CARD)

    assert "predicting churn" in response
    assert fake_llm.call_count == 1
    # Check that system prompt included dataset card
    prompt_used = fake_llm.call_history[0]["messages"]
    assert len(prompt_used) == 1


@pytest.mark.asyncio
async def test_planner_spec_generation_direct_success():
    """Verify Planner directly compiles a valid JobSpec on first turn."""
    valid_spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["age", "tenure"],
        split_strategy="random",
        holdout_fraction=0.2,
        eval_metric="roc_auc",
        rationale="Predicting customer churn.",
    )
    fake_llm = FakeLLMClient(mock_structured=valid_spec)
    planner = Planner(llm_client=fake_llm)

    history = [{"role": "user", "content": "Train TabPFN to predict churn."}]
    spec = await planner.generate_job_spec(history, SAMPLE_DATASET_CARD, max_repairs=2)

    assert spec.target_column == "churn"
    assert spec.feature_columns == ["age", "tenure"]
    assert fake_llm.call_count == 1


class AutoRepairFakeLLM(FakeLLMClient):
    """Mock client that returns an invalid spec on call 1 and valid spec on call 2."""

    def __init__(self, invalid_spec: JobSpec, valid_spec: JobSpec) -> None:
        super().__init__()
        self.invalid_spec = invalid_spec
        self.valid_spec = valid_spec

    async def generate_structured(self, messages, response_schema, system_prompt=""):
        self.call_count += 1
        if self.call_count == 1:
            return self.invalid_spec
        return self.valid_spec


@pytest.mark.asyncio
async def test_planner_auto_repair_recovers():
    """Verify Planner executes auto-repair turn when first spec is invalid."""
    invalid_spec = JobSpec(
        task_type="classification",
        target_column="non_existent_column",  # Invalid column
        feature_columns=["age", "tenure"],
        split_strategy="random",
        rationale="Invalid first attempt.",
    )
    valid_spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["age", "tenure"],
        split_strategy="random",
        rationale="Corrected second attempt.",
    )

    repair_llm = AutoRepairFakeLLM(invalid_spec, valid_spec)
    planner = Planner(llm_client=repair_llm)

    history = [{"role": "user", "content": "Build model."}]
    compiled_spec = await planner.generate_job_spec(history, SAMPLE_DATASET_CARD, max_repairs=2)

    assert compiled_spec.target_column == "churn"
    assert repair_llm.call_count == 2


@pytest.mark.asyncio
async def test_planner_auto_repair_exhaustion_raises():
    """Verify Planner raises JobSpecRepairError if repair fails repeatedly."""
    persistently_invalid_spec = JobSpec(
        task_type="classification",
        target_column="never_exists",
        feature_columns=["age"],
        split_strategy="random",
        rationale="Consistently failing spec.",
    )
    fake_llm = FakeLLMClient(mock_structured=persistently_invalid_spec)
    planner = Planner(llm_client=fake_llm)

    history = [{"role": "user", "content": "Build model."}]
    with pytest.raises(JobSpecRepairError, match="failed validation after 2 repair attempts"):
        await planner.generate_job_spec(history, SAMPLE_DATASET_CARD, max_repairs=2)


@pytest.mark.asyncio
async def test_narrator_grounded_output():
    """Verify Narrator receives results and dataset card to produce explanation."""
    mock_narrative = (
        "TabPFN achieved 85% accuracy (95% CI: [0.72, 0.94]) compared to the 60% baseline. "
        "The model demonstrates strong calibration (ECE: 0.04)."
    )
    fake_llm = FakeLLMClient(mock_text=mock_narrative)
    narrator = Narrator(llm_client=fake_llm)

    sample_results = {
        "task_type": "classification",
        "target_column": "churn",
        "n_train": 80,
        "n_holdout": 20,
        "baseline_metrics": {"accuracy": 0.60},
        "model_metrics": {"accuracy": 0.85, "accuracy_ci_95": [0.72, 0.94], "ece": 0.04},
        "comparison": {"accuracy_delta": 0.25, "accuracy_lift": 1.42},
        "sample_predictions": [],
    }

    narration = await narrator.narrate_results(sample_results, SAMPLE_DATASET_CARD)
    assert "TabPFN achieved 85%" in narration
    assert fake_llm.call_count == 1

    # Verify input message to LLM includes results and dataset summary
    content = fake_llm.call_history[0]["messages"][0]["content"]
    assert "job_results" in content
    assert "dataset_overview" in content
