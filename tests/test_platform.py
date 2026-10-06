"""Platform infrastructure tests for tabchat (offline-first, zero network)."""

import asyncio
import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from tabchat.config import Settings
from tabchat.llm.client import (
    DailyQuotaExhaustedError,
    FakeLLMClient,
    LLMRateLimitError,
    SlidingWindowRateLimiter,
)
from tabchat.storage import SessionNotFoundError, SessionStore
from tabchat.validation import (
    CSVValidationError,
    validate_and_parse_csv,
)


# ==============================================================================
# 1. Configuration Unit Tests
# ==============================================================================

def test_config_fail_fast_on_missing_keys(monkeypatch):
    """Verify settings raise clean validation error when API keys are missing or empty."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)

    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,  # Do not read .env
            GEMINI_API_KEY="",
            TABPFN_TOKEN="",
        )


def test_config_valid_loading(tmp_path):
    """Verify valid settings with defaults and directory creation."""
    data_dir = tmp_path / "custom_data"
    s = Settings(
        _env_file=None,
        GEMINI_API_KEY="test_gemini_key",
        TABPFN_TOKEN="test_tabpfn_token",
        DATA_DIR=data_dir,
    )
    assert s.GEMINI_API_KEY == "test_gemini_key"
    assert s.TABPFN_TOKEN == "test_tabpfn_token"
    assert s.LLM_PRIMARY_MODEL == "gemini-3.8-flash"
    assert s.LLM_FALLBACK_MODEL == "gemini-3.8-flash-lite"
    assert s.MAX_ROWS == 200
    assert s.MAX_COLS == 50
    assert data_dir.is_dir()


# ==============================================================================
# 2. CSV Validation & Boundary Tests
# ==============================================================================

def test_csv_row_count_boundaries():
    """Test boundary conditions for row count: exactly 200 rows passes, 201 fails, 0 fails."""
    # 200 rows -> PASS
    lines_200 = ["col_a,col_b"] + [f"{i},val_{i}" for i in range(200)]
    csv_200 = "\n".join(lines_200).encode("utf-8")
    df_200, card_200 = validate_and_parse_csv(csv_200)
    assert len(df_200) == 200
    assert card_200["row_count"] == 200

    # 201 rows -> FAIL
    lines_201 = ["col_a,col_b"] + [f"{i},val_{i}" for i in range(201)]
    csv_201 = "\n".join(lines_201).encode("utf-8")
    with pytest.raises(CSVValidationError) as exc:
        validate_and_parse_csv(csv_201)
    assert any("exceeds maximum limit of 200 rows" in e for e in exc.value.errors)

    # 0 data rows -> FAIL
    csv_0 = b"col_a,col_b\n"
    with pytest.raises(CSVValidationError) as exc:
        validate_and_parse_csv(csv_0)
    assert any("at least 1 row" in e for e in exc.value.errors)


def test_csv_column_count_boundaries():
    """Test boundary conditions for column count: 2 passes, 1 fails, 51 fails."""
    # 1 column -> FAIL (minimum is 2)
    csv_1_col = b"col_a\n1\n2\n"
    with pytest.raises(CSVValidationError) as exc:
        validate_and_parse_csv(csv_1_col)
    assert any("below minimum requirement of 2 columns" in e for e in exc.value.errors)

    # 50 columns -> PASS
    cols_50 = [f"c_{i}" for i in range(50)]
    csv_50_col = (",".join(cols_50) + "\n" + ",".join(["1"] * 50) + "\n").encode("utf-8")
    df_50, card_50 = validate_and_parse_csv(csv_50_col)
    assert len(df_50.columns) == 50

    # 51 columns -> FAIL
    cols_51 = [f"c_{i}" for i in range(51)]
    csv_51_col = (",".join(cols_51) + "\n" + ",".join(["1"] * 51) + "\n").encode("utf-8")
    with pytest.raises(CSVValidationError) as exc:
        validate_and_parse_csv(csv_51_col)
    assert any("exceeds maximum limit of 50 columns" in e for e in exc.value.errors)


def test_csv_string_length_boundaries():
    """Test string length boundaries: 64 characters passes, 65 characters fails."""
    str_64 = "a" * 64
    str_65 = "a" * 65

    # 64 chars -> PASS
    csv_64 = f"col_a,col_b\n1,{str_64}\n".encode("utf-8")
    df_64, card_64 = validate_and_parse_csv(csv_64)
    assert df_64.iloc[0]["col_b"] == str_64

    # 65 chars -> FAIL
    csv_65 = f"col_a,col_b\n1,{str_65}\n".encode("utf-8")
    with pytest.raises(CSVValidationError) as exc:
        validate_and_parse_csv(csv_65)
    assert any("cell length 65 chars exceeds limit of 64" in e for e in exc.value.errors)


def test_csv_numeric_boundaries():
    """Test signed int32 bounds and float16 magnitude boundaries."""
    # Max signed int32 (2147483647) -> PASS
    csv_int_ok = b"col_a,col_b\n2147483647,-2147483648\n"
    df_ok, _ = validate_and_parse_csv(csv_int_ok)
    assert len(df_ok) == 1

    # Int32 overflow (2147483648) -> FAIL
    csv_int_overflow = b"col_a,col_b\n2147483648,1\n"
    with pytest.raises(CSVValidationError) as exc:
        validate_and_parse_csv(csv_int_overflow)
    assert any("exceeds signed int32 bounds" in e for e in exc.value.errors)

    # Float16 max (65504.0) -> PASS
    csv_float_ok = b"col_a,col_b\n65504.0,-65504.0\n"
    df_float_ok, _ = validate_and_parse_csv(csv_float_ok)
    assert len(df_float_ok) == 1

    # Float16 overflow (65505.0) -> FAIL
    csv_float_overflow = b"col_a,col_b\n65505.0,1.0\n"
    with pytest.raises(CSVValidationError) as exc:
        validate_and_parse_csv(csv_float_overflow)
    assert any("exceeds float16 range" in e for e in exc.value.errors)


def test_csv_header_and_null_checks():
    """Test duplicate headers, empty headers, and 100% null columns."""
    # Duplicate headers -> FAIL
    csv_dup = b"col_a,col_a\n1,2\n"
    with pytest.raises(CSVValidationError) as exc:
        validate_and_parse_csv(csv_dup)
    assert any("Duplicate column header" in e for e in exc.value.errors)

    # 100% null column -> FAIL
    csv_null_col = b"col_a,col_b\n1,\n2,\n"
    with pytest.raises(CSVValidationError) as exc:
        validate_and_parse_csv(csv_null_col)
    assert any("Column 'col_b' is 100% null or empty" in e for e in exc.value.errors)


def test_dataset_card_preview_rows_and_metadata():
    """Verify dataset card structure, types, null counts, and preview rows."""
    raw_csv = (
        "id,date,score,label\n"
        "1,2026-01-01,98.5,cat\n"
        "2,2026-01-02,95.0,dog\n"
        "3,2026-01-03,,cat\n"
        "4,2026-01-04,91.2,bird\n"
        "5,2026-01-05,88.0,cat\n"
        "6,2026-01-06,85.5,dog\n"
    ).encode("utf-8")

    df, card = validate_and_parse_csv(raw_csv)
    assert card["row_count"] == 6
    assert card["col_count"] == 4
    assert card["column_names"] == ["id", "date", "score", "label"]
    assert card["inferred_types"]["id"] == "numeric"
    assert card["inferred_types"]["date"] == "datetime"
    assert card["inferred_types"]["score"] == "numeric"
    assert card["inferred_types"]["label"] == "categorical"
    assert card["null_counts"]["score"] == 1
    assert card["unique_counts"]["label"] == 3
    # Preview rows must be capped at 5
    assert len(card["preview_rows"]) == 5
    assert card["preview_rows"][0]["id"] == 1
    assert card["preview_rows"][2]["score"] is None  # Null sanitized to None


# ==============================================================================
# 3. Storage Engine Persistence & Server Restart Tests
# ==============================================================================

def test_storage_engine_lifecycle_and_restart(tmp_path):
    """Test session creation, persistence, simulated restart, and retrieval."""
    sessions_dir = tmp_path / "sessions_test"
    store1 = SessionStore(data_dir=sessions_dir)

    # 1. Create session
    session_id = store1.create_session()
    assert len(session_id) == 32
    assert store1.session_exists(session_id)

    # 2. Save dataset
    raw_bytes = b"col_x,col_y\n1,alpha\n2,beta\n"
    card = {"row_count": 2, "col_count": 2, "column_names": ["col_x", "col_y"]}
    store1.save_dataset(session_id, raw_bytes, card)

    # 3. Append history
    store1.append_history(session_id, "user", "Hello assistant!")
    store1.append_history(session_id, "assistant", "Hello! How can I help?")

    # 4. Save job spec and results
    spec = {"task": "classification", "target_col": "col_y"}
    results = {"auroc": 0.88, "status": "success"}
    store1.save_job_spec(session_id, spec)
    store1.save_results(session_id, results)

    # 5. SIMULATE SERVER RESTART: Create a brand new SessionStore pointing to same dir
    store2 = SessionStore(data_dir=sessions_dir)
    assert store2.session_exists(session_id)

    # Verify retrieved dataset
    df_loaded, card_loaded = store2.get_dataset(session_id)
    assert len(df_loaded) == 2
    assert list(df_loaded.columns) == ["col_x", "col_y"]
    assert card_loaded["row_count"] == 2

    # Verify retrieved history
    history = store2.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello assistant!"
    assert "timestamp" in history[0]
    assert history[1]["role"] == "assistant"

    # Verify retrieved spec & results
    spec_loaded = store2.get_job_spec(session_id)
    assert spec_loaded == spec

    results_loaded = store2.get_results(session_id)
    assert results_loaded == results

    # Verify delete session
    store2.delete_session(session_id)
    assert not store2.session_exists(session_id)

    # Non-existent session raises SessionNotFoundError
    with pytest.raises(SessionNotFoundError):
        store2.get_dataset(session_id)


def test_storage_zero_leaks_outside_data_dir(tmp_path):
    """Memory check: Verify zero temp files or artifacts leak outside data_dir/sessions/."""
    base_dir = tmp_path / "leak_check_dir"
    store = SessionStore(data_dir=base_dir)
    sid = store.create_session()
    store.save_dataset(sid, b"a,b\n1,2\n", {"row_count": 1})

    # All files in base_dir must be strictly within base_dir/sessions/<sid>/
    all_files = list(base_dir.rglob("*"))
    for f in all_files:
        assert str(f).startswith(str(base_dir / "sessions"))


# ==============================================================================
# 4. Concurrency & Rate Limiter Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_fake_llm_client_strict_sequentiality():
    """Verify that FakeLLMClient with concurrency=1 executes concurrent calls strictly sequentially."""
    sem = asyncio.Semaphore(1)
    client = FakeLLMClient(
        mock_text="Simulated completion",
        latency_seconds=0.10,  # 100ms simulated latency
        semaphore=sem,
    )

    # Launch 5 concurrent calls
    tasks = [
        client.generate_text([{"role": "user", "content": f"msg {i}"}])
        for i in range(5)
    ]
    responses = await asyncio.gather(*tasks)

    assert len(responses) == 5
    assert client.call_count == 5
    assert len(client.call_history) == 5

    # Assert strict sequentiality: call[i].start must be >= call[i-1].end
    history = client.call_history
    for i in range(1, len(history)):
        prev_end = history[i - 1]["end"]
        curr_start = history[i]["start"]
        # Allow tiny float jitter (1ms)
        assert curr_start >= prev_end - 0.005, (
            f"Call {i} started at {curr_start} before call {i-1} finished at {prev_end}"
        )


@pytest.mark.asyncio
async def test_fake_llm_structured_output():
    """Verify FakeLLMClient returns schema-conforming Pydantic models."""
    class SampleJobSpec(BaseModel):
        task_type: str = "classification"
        target_column: str = "churn"

    client = FakeLLMClient(
        mock_structured=SampleJobSpec(task_type="regression", target_column="price")
    )
    result = await client.generate_structured(
        [{"role": "user", "content": "plan this"}],
        response_schema=SampleJobSpec,
    )
    assert isinstance(result, SampleJobSpec)
    assert result.task_type == "regression"
    assert result.target_column == "price"


@pytest.mark.asyncio
async def test_sliding_window_rate_limiter_pacing():
    """Verify SlidingWindowRateLimiter limits throughput within time window."""
    limiter = SlidingWindowRateLimiter(rpm_limit=2, window_seconds=0.1)

    t0 = time.monotonic()
    await limiter.acquire()  # slot 1
    await limiter.acquire()  # slot 2

    # 3rd acquire must be delayed until oldest timestamp rolls off
    await limiter.acquire()  # slot 3
    elapsed = time.monotonic() - t0

    assert elapsed >= 0.09, f"Rate limiter failed to pace requests (elapsed: {elapsed:.3f}s)"
