"""Integration tests for the 3-stage intelligence engine and FastAPI routes."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import pandas as pd
import pytest
from starlette.testclient import TestClient

from tabchat.app import app
from tabchat.engine.executor import execute_plan
from tabchat.engine.narrator import narrate_results, verify_hallucination_guard
from tabchat.engine.planner import PlannerOutput, plan_analysis
from tabchat.llm.client import FakeLLMClient, SlidingWindowRateLimiter
from tabchat.schemas.job_spec import JobSpec
from tabchat.storage import SessionStore


def create_sample_classification_csv(n_rows: int = 30) -> bytes:
    """Generate sample churn dataset with continuous features and binary target."""
    data = {
        "customer_id": [f"CUST_{i:04d}" for i in range(n_rows)],
        "age": [25 + (i % 40) for i in range(n_rows)],
        "tenure": [1 + (i % 10) for i in range(n_rows)],
        "balance": [1000.0 + (i * 150.5) for i in range(n_rows)],
        "churn": ["yes" if i % 3 == 0 else "no" for i in range(n_rows)],
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")


def mock_tabpfn_api_caller(
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    random_seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Mock TabPFN API caller matching _fit_and_predict_tabpfn signature."""
    n_test = len(X_holdout)
    if task_type == "classification":
        preds = ["yes" if i % 2 == 0 else "no" for i in range(n_test)]
        probs = [[0.8, 0.2] if p == "yes" else [0.2, 0.8] for p in preds]
        return {
            "predictions": preds,
            "probabilities": probs,
            "classes": ["no", "yes"],
            "model_path": "v3.5_default",
        }
    else:
        preds = [42.0 + (i * 1.5) for i in range(n_test)]
        return {
            "predictions": preds,
            "quantiles": {
                "q10": [p - 5.0 for p in preds],
                "q50": preds,
                "q90": [p + 5.0 for p in preds],
            },
            "model_path": "v3.5_default",
        }


@pytest.fixture
def test_env(tmp_path: Path):
    """Setup isolated session store and fake LLM client on FastAPI app state."""
    store = SessionStore(data_dir=tmp_path / "data")
    mock_spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["age", "tenure", "balance"],
        excluded_columns=["customer_id"],
        split_strategy="random",
        holdout_fraction=0.2,
        random_seed=42,
        eval_metric="accuracy",
        confidence_intervals=True,
        rationale="Predicting churn while excluding unique ID.",
    )

    fake_planner_output = PlannerOutput(
        is_clarification=False,
        clarification_message=None,
        selected_skills=["task_and_target", "leakage_and_splits"],
        proposed_spec=mock_spec,
    )

    fake_narration_text = (
        "The TabPFN classifier was evaluated on 6 holdout samples from the customer churn dataset. "
        "The model achieved an accuracy of 83.3%, representing a 16.6% lift over the majority-class "
        "baseline of 66.7%. Because the dataset has fewer than 100 rows, results should be interpreted "
        "with caution due to wider sample variance."
    )

    # Use a high-capacity rate limiter in tests to prevent synthetic delays
    test_rate_limiter = SlidingWindowRateLimiter(rpm_limit=1000)
    llm = FakeLLMClient(
        mock_text=fake_narration_text,
        mock_structured=fake_planner_output,
        rate_limiter=test_rate_limiter,
    )

    # Attach to app.state
    app.state.store = store
    app.state.llm_client = llm
    app.state.tabpfn_api_caller = mock_tabpfn_api_caller

    client = TestClient(app)
    return {
        "client": client,
        "store": store,
        "llm": llm,
        "mock_spec": mock_spec,
        "tmp_path": tmp_path,
        "rate_limiter": test_rate_limiter,
    }


def test_full_session_flow_and_restart_persistence(test_env):
    """Test full flow: Create Session -> Upload -> Chat -> Run -> Kill Client -> Resume from Disk."""
    client: TestClient = test_env["client"]
    store: SessionStore = test_env["store"]

    # 1. Initialize session via POST /api/session
    res_session = client.post("/api/session")
    assert res_session.status_code == 200
    data = res_session.json()
    assert "session_id" in data
    assert data["status"] == "created"
    session_id = data["session_id"]
    assert client.cookies.get("session_id") == session_id

    # 2. Upload dataset via POST /api/upload
    csv_bytes = create_sample_classification_csv(n_rows=30)
    files = {"file": ("customers.csv", io.BytesIO(csv_bytes), "text/csv")}
    res_upload = client.post("/api/upload", files=files)
    assert res_upload.status_code == 200
    card = res_upload.json()
    assert card["row_count"] == 30
    assert card["col_count"] == 5
    assert "churn" in card["column_names"]

    # 3. Chat with Planner via POST /api/chat
    chat_payload = {"message": "I want to predict churn based on customer features"}
    res_chat = client.post("/api/chat", json=chat_payload)
    assert res_chat.status_code == 200
    chat_data = res_chat.json()
    assert "reply" in chat_data
    assert chat_data["plan_card"] is not None
    assert chat_data["plan_card"]["target_column"] == "churn"
    assert chat_data["turns_left"] == 14

    # Verify job_spec.json was persisted to disk
    persisted_spec = store.get_job_spec(session_id)
    assert persisted_spec is not None
    assert persisted_spec["target_column"] == "churn"

    # 4. POST /api/run: Execute plan and narrate
    res_run = client.post("/api/run")
    assert res_run.status_code == 200
    run_data = res_run.json()
    assert "results" in run_data
    assert "narration" in run_data
    assert run_data["results"]["holdout_size"] == 6
    assert "baseline_metrics" in run_data["results"]
    assert "model_metrics" in run_data["results"]
    assert "83.3%" in run_data["narration"]

    # Verify results.json persisted to disk
    persisted_results = store.get_results(session_id)
    assert persisted_results is not None
    assert persisted_results["holdout_size"] == 6

    # 5. KILL TEST CLIENT AND RECREATE (Simulating full server restart / client reconnect)
    del client
    new_client = TestClient(app, cookies={"session_id": session_id})

    # Verify session resumes with full state from disk
    res_resume = new_client.get("/api/session")
    assert res_resume.status_code == 200
    resumed_data = res_resume.json()
    assert resumed_data["session_id"] == session_id
    assert resumed_data["dataset_card"]["row_count"] == 30
    assert len(resumed_data["history"]) >= 3  # user chat, assistant plan, narrator response
    assert resumed_data["job_spec"]["target_column"] == "churn"
    assert resumed_data["results"]["holdout_size"] == 6

    # 6. Test GET /api/log
    res_log = new_client.get("/api/log")
    assert res_log.status_code == 200
    assert res_log.headers["content-type"].startswith("text/plain")
    log_text = res_log.text
    assert "I want to predict churn" in log_text

    # 7. Test DELETE /api/session
    res_del = new_client.delete("/api/session")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"
    assert not store.session_exists(session_id)


def test_clarification_mode_in_chat(test_env):
    """Test that when intent is ambiguous, planner returns clarifying question and no plan card."""
    client: TestClient = test_env["client"]
    llm: FakeLLMClient = test_env["llm"]

    # Configure FakeLLM to return a clarification
    llm.mock_structured = PlannerOutput(
        is_clarification=True,
        clarification_message="Which column would you like to predict? Available columns are age, tenure, balance, churn.",
        selected_skills=[],
        proposed_spec=None,
    )

    client.post("/api/session")
    csv_bytes = create_sample_classification_csv(20)
    client.post("/api/upload", files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")})

    res = client.post("/api/chat", json={"message": "Help me do machine learning"})
    assert res.status_code == 200
    data = res.json()
    assert data["plan_card"] is None
    assert "Which column would you like to predict?" in data["reply"]


def test_planner_auto_repair_mechanism(test_env):
    """Test auto-repair loop: invalid initial spec is fed back to LLM and repaired."""
    store: SessionStore = test_env["store"]
    session_id = store.create_session()
    csv_bytes = create_sample_classification_csv(30)
    card = {
        "row_count": 30,
        "col_count": 5,
        "column_names": ["customer_id", "age", "tenure", "balance", "churn"],
        "inferred_types": {},
        "null_counts": {},
        "unique_counts": {},
        "preview_rows": [],
    }
    store.save_dataset(session_id, csv_bytes, card)

    # Initial output has an invalid non-existent column
    invalid_spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["age", "nonexistent_feature"],
        split_strategy="random",
        holdout_fraction=0.2,
        rationale="Invalid initial plan",
    )
    initial_output = PlannerOutput(
        is_clarification=False,
        proposed_spec=invalid_spec,
    )

    # Repaired output fixes the column
    valid_spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["age", "tenure"],
        split_strategy="random",
        holdout_fraction=0.2,
        rationale="Repaired plan",
    )
    repaired_output = PlannerOutput(
        is_clarification=False,
        proposed_spec=valid_spec,
    )

    # Supply queue of structured responses with fast rate limiter
    test_rate_limiter = test_env["rate_limiter"]
    llm = FakeLLMClient(
        mock_structured=[initial_output, repaired_output],
        rate_limiter=test_rate_limiter,
    )

    import asyncio

    res = asyncio.run(
        plan_analysis(
            session_id=session_id,
            user_message="Predict churn",
            llm_client=llm,
            store=store,
        )
    )

    assert res["is_clarification"] is False
    assert res["proposed_spec"]["feature_columns"] == ["age", "tenure"]
    # Check that LLM was invoked twice (initial + auto-repair turn)
    assert llm.call_count == 2


def test_executor_strict_zero_llm_calls(test_env):
    """STRICT REQUIREMENT: Ensure Executor performs ZERO LLM calls during execution."""
    store: SessionStore = test_env["store"]
    llm: FakeLLMClient = test_env["llm"]
    session_id = store.create_session()

    csv_bytes = create_sample_classification_csv(30)
    card = {
        "row_count": 30,
        "col_count": 5,
        "column_names": ["customer_id", "age", "tenure", "balance", "churn"],
        "inferred_types": {},
        "null_counts": {},
        "unique_counts": {},
        "preview_rows": [],
    }
    store.save_dataset(session_id, csv_bytes, card)

    spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["age", "tenure", "balance"],
        split_strategy="random",
        holdout_fraction=0.2,
        rationale="Zero LLM test",
    )
    store.save_job_spec(session_id, spec.model_dump())

    llm_calls_before = llm.call_count

    import asyncio

    results = asyncio.run(
        execute_plan(
            session_id=session_id,
            store=store,
            api_caller=mock_tabpfn_api_caller,
        )
    )

    llm_calls_after = llm.call_count
    # Verify ZERO LLM calls were made during execution
    assert llm_calls_after == llm_calls_before
    assert "model_metrics" in results
    assert "baseline_metrics" in results


def test_narrator_hallucination_guard_catches_phantom_metrics(caplog):
    """Test that narrator's Hallucination Guard logs a severe warning on hallucinated metrics."""
    mock_results = {
        "baseline_metrics": {"accuracy": 0.60},
        "model_metrics": {"accuracy": 0.80},
        "delta": {"accuracy": 0.20},
        "holdout_size": 10,
    }

    # Narration containing hallucinated number 99.4% and decimal 4.567
    hallucinated_text = (
        "The model scored an unbelievable 99.4% accuracy with a loss of 4.567 on 10 holdout cases."
    )

    with caplog.at_level("WARNING"):
        phantoms = verify_hallucination_guard(hallucinated_text, mock_results)

    assert len(phantoms) >= 1
    assert any("HALLUCINATION GUARD" in record.message for record in caplog.records)


def test_turn_limit_enforcement(test_env):
    """Test that sending more than 15 messages triggers turn limit error."""
    client: TestClient = test_env["client"]
    client.post("/api/session")
    csv_bytes = create_sample_classification_csv(20)
    client.post("/api/upload", files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")})

    # Send 15 messages
    for i in range(15):
        res = client.post("/api/chat", json={"message": f"Turn {i + 1}"})
        assert res.status_code == 200

    # 16th message must fail with 400
    res_16 = client.post("/api/chat", json={"message": "Turn 16"})
    assert res_16.status_code == 400
    assert "turn limit" in res_16.json()["detail"].lower()


def test_upload_validation_error_structured_response(test_env):
    """Test that invalid CSV returns 422 with structured error list."""
    client: TestClient = test_env["client"]
    client.post("/api/session")

    # CSV with duplicate column headers
    bad_csv = "age,age,churn\n25,30,yes\n40,45,no\n".encode("utf-8")
    files = {"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 422
    data = res.json()
    assert "detail" in data
    assert any("Duplicate" in err for err in data["detail"])
