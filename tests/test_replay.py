"""Unit tests for the CLI replay utility (src/tabchat/replay.py)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import pytest

from tabchat.replay import main, replay_session
from tabchat.storage import SessionStore


@pytest.fixture
def sample_complete_session(tmp_path: Path) -> tuple[Path, str]:
    """Create a fully executed sample session on disk."""
    data_dir = tmp_path / "data"
    store = SessionStore(data_dir=data_dir)
    session_id = store.create_session()

    # 1. Dataset card & raw CSV
    card = {
        "row_count": 50,
        "col_count": 3,
        "column_names": ["age", "income", "churn"],
        "inferred_types": {"age": "numeric", "income": "numeric", "churn": "categorical"},
        "null_counts": {"age": 0, "income": 0, "churn": 0},
        "unique_counts": {"age": 30, "income": 45, "churn": 2},
        "preview_rows": [
            {"age": 25, "income": 50000, "churn": "no"},
            {"age": 40, "income": 80000, "churn": "yes"},
        ],
    }
    raw_csv = b"age,income,churn\n25,50000,no\n40,80000,yes\n"
    store.save_dataset(session_id, raw_csv, card)

    # 2. History
    store.append_history(session_id, "user", "Predict customer churn")
    store.append_history(
        session_id,
        "assistant",
        "I have configured an analysis plan to predict churn using age and income.",
    )

    # 3. JobSpec
    spec = {
        "task_type": "classification",
        "target_column": "churn",
        "feature_columns": ["age", "income"],
        "excluded_columns": [],
        "split_strategy": "random",
        "holdout_fraction": 0.2,
        "random_seed": 42,
        "eval_metric": "accuracy",
        "rationale": "Predicting churn using demographic and financial attributes.",
    }
    store.save_job_spec(session_id, spec)

    # 4. Results
    results = {
        "spec": spec,
        "train_size": 40,
        "holdout_size": 10,
        "baseline_metrics": {"accuracy": 0.6000, "log_loss": 0.6730},
        "model_metrics": {"accuracy": 0.9000, "log_loss": 0.3120},
        "delta": {"accuracy": 0.3000, "log_loss": -0.3610},
        "warnings": ["Sample warning check"],
        "sample_predictions": [
            {"row_index": 12, "actual": "yes", "predicted": "yes", "probabilities": [0.15, 0.85]},
            {"row_index": 15, "actual": "no", "predicted": "no", "probabilities": [0.90, 0.10]},
        ],
    }
    store.save_results(session_id, results)

    # 5. Final Narration in history
    store.append_history(
        session_id,
        "assistant",
        "TabPFN achieved 90.0% holdout accuracy (+30.0% lift over baseline of 60.0%).",
    )

    return data_dir, session_id


def test_replay_complete_session(sample_complete_session):
    """Test full replay output with all artifacts present."""
    data_dir, session_id = sample_complete_session
    buf = io.StringIO()
    code = replay_session(session_id, data_dir=data_dir, out=buf)

    assert code == 0
    output = buf.getvalue()

    # Section 1
    assert "TABCHAT SESSION AUDIT TRAIL" in output
    assert session_id in output

    # Section 2
    assert "50 rows × 3 columns" in output
    assert "income" in output
    assert "numeric" in output

    # Section 3
    assert "[Turn 1 - USER]" in output
    assert "Predict customer churn" in output

    # Section 4
    assert "Task Type       : classification" in output
    assert "Target Column   : churn" in output

    # Section 5
    assert "Train = 40 rows | Holdout = 10 rows" in output
    assert "accuracy" in output
    assert "+0.3000" in output
    assert "Sample warning check" in output
    assert "Row Index | Actual     | Predicted" in output

    # Section 6
    assert "TabPFN achieved 90.0% holdout accuracy" in output


def test_replay_partial_session_warnings(tmp_path: Path):
    """Test replay on a session that has only uploaded a dataset but not planned or executed."""
    data_dir = tmp_path / "data"
    store = SessionStore(data_dir=data_dir)
    session_id = store.create_session()

    card = {
        "row_count": 20,
        "col_count": 2,
        "column_names": ["x", "y"],
        "inferred_types": {"x": "numeric", "y": "numeric"},
        "null_counts": {"x": 0, "y": 0},
        "unique_counts": {"x": 20, "y": 20},
    }
    store.save_dataset(session_id, b"x,y\n1,2\n", card)

    buf = io.StringIO()
    code = replay_session(session_id, data_dir=data_dir, out=buf)

    assert code == 0
    output = buf.getvalue()

    assert "[WARNING] job_spec.json missing" in output
    assert "[WARNING] results.json missing" in output


def test_replay_corrupted_json_diagnostic(tmp_path: Path):
    """Test that corrupted files yield explicit diagnostic warnings instead of unhandled crashes."""
    data_dir = tmp_path / "data"
    store = SessionStore(data_dir=data_dir)
    session_id = store.create_session()

    sess_dir = data_dir / "sessions" / session_id
    (sess_dir / "dataset_card.json").write_text("{invalid_json: 123", encoding="utf-8")
    (sess_dir / "job_spec.json").write_text("not-a-json", encoding="utf-8")
    (sess_dir / "results.json").write_text("{broken", encoding="utf-8")
    (sess_dir / "history.jsonl").write_text("{bad json line\n", encoding="utf-8")

    buf = io.StringIO()
    code = replay_session(session_id, data_dir=data_dir, out=buf)

    assert code == 0
    output = buf.getvalue()

    assert "[WARNING] Corrupted JSON in dataset_card.json" in output
    assert "[WARNING] Corrupted record at history.jsonl line 1" in output
    assert "[WARNING] Corrupted JSON in job_spec.json" in output
    assert "[WARNING] Corrupted JSON in results.json" in output


def test_replay_nonexistent_session(tmp_path: Path):
    """Test that requesting a missing session directory returns error exit code 1."""
    data_dir = tmp_path / "data"
    buf = io.StringIO()
    code = replay_session("nonexistent_session_id", data_dir=data_dir, out=buf)

    assert code == 1
    output = buf.getvalue()
    assert "[ERROR] Session directory does not exist" in output


def test_replay_cli_main(sample_complete_session, monkeypatch):
    """Test CLI main() invocation with sys.argv."""
    data_dir, session_id = sample_complete_session
    monkeypatch.setattr("sys.argv", ["replay", session_id, "--data-dir", str(data_dir)])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
