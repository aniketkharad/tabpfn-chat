"""Comprehensive offline unit tests for the TabPFN pipeline: JobSpec, compiler, skills, and runner."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tabchat.schemas.job_spec import JobSpec
from tabchat.services.spec_compiler import compile_and_verify_spec
from tabchat.services.tabpfn_runner import (
    compute_cache_key,
    run_tabpfn_job,
    run_tabpfn_job_sync,
)
from tabchat.skills.registry import get_skill_content, get_skills_index, get_registry


# -------------------------------------------------------------------------
# 1. JobSpec Pydantic Validators Tests
# -------------------------------------------------------------------------

def test_job_spec_valid():
    """Verify that a well-formed JobSpec instantiates without error."""
    spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["tenure", "monthly_charges"],
        excluded_columns=["customer_id"],
        split_strategy="random",
        holdout_fraction=0.2,
        random_seed=42,
        eval_metric="roc_auc",
        confidence_intervals=True,
        rationale="Standard customer churn classification with stratified holdout.",
    )
    assert spec.task_type == "classification"
    assert spec.target_column == "churn"
    assert len(spec.feature_columns) == 2
    assert "customer_id" in spec.canonical_json()


def test_job_spec_target_inside_features_raises():
    """Target column cannot be present in feature_columns."""
    with pytest.raises(ValidationError) as exc_info:
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=["churn", "tenure"],
            rationale="Invalid spec with target inside features.",
        )
    assert "cannot" in str(exc_info.value) and "feature_columns" in str(exc_info.value)


def test_job_spec_empty_feature_columns_raises():
    """feature_columns must contain at least 1 column."""
    with pytest.raises(ValidationError) as exc_info:
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=[],
            rationale="Invalid spec with empty features.",
        )
    assert "feature_columns must contain at least 1 column" in str(exc_info.value)


def test_job_spec_missing_split_column_for_group_raises():
    """Group split requires a valid split_column."""
    with pytest.raises(ValidationError) as exc_info:
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=["tenure"],
            split_strategy="group",
            split_column=None,
            rationale="Invalid spec with missing group column.",
        )
    assert "split_column is required when split_strategy is 'group'" in str(exc_info.value)


def test_job_spec_missing_split_column_for_chronological_raises():
    """Chronological split requires a valid split_column."""
    with pytest.raises(ValidationError) as exc_info:
        JobSpec(
            task_type="regression",
            target_column="price",
            feature_columns=["sqft"],
            split_strategy="chronological",
            split_column="",
            rationale="Invalid spec with empty chronological column.",
        )
    assert "split_column is required when split_strategy is 'chronological'" in str(exc_info.value)


def test_job_spec_split_column_cannot_be_target_raises():
    """split_column cannot be the target column."""
    with pytest.raises(ValidationError) as exc_info:
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=["tenure"],
            split_strategy="group",
            split_column="churn",
            rationale="Invalid spec with split_column equal to target.",
        )
    assert "cannot be the target_column" in str(exc_info.value)


def test_job_spec_holdout_fraction_bounds():
    """holdout_fraction must be between 0.1 and 0.4."""
    with pytest.raises(ValidationError):
        JobSpec(
            task_type="regression",
            target_column="price",
            feature_columns=["sqft"],
            holdout_fraction=0.05,
            rationale="Fraction too small.",
        )

    with pytest.raises(ValidationError):
        JobSpec(
            task_type="regression",
            target_column="price",
            feature_columns=["sqft"],
            holdout_fraction=0.45,
            rationale="Fraction too large.",
        )


def test_job_spec_metric_task_mismatch_raises():
    """Classification cannot use regression metrics and vice versa."""
    with pytest.raises(ValidationError) as exc1:
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=["tenure"],
            eval_metric="rmse",
            rationale="Mismatch classification with rmse.",
        )
    assert "only valid for regression" in str(exc1.value)

    with pytest.raises(ValidationError) as exc2:
        JobSpec(
            task_type="regression",
            target_column="price",
            feature_columns=["sqft"],
            eval_metric="roc_auc",
            rationale="Mismatch regression with roc_auc.",
        )
    assert "only valid for classification" in str(exc2.value)


# -------------------------------------------------------------------------
# 2. Spec Compiler & Dataset Verifier Tests
# -------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture providing a well-behaved toy DataFrame."""
    return pd.DataFrame({
        "customer_id": [f"c_{i}" for i in range(20)],
        "tenure": [i * 2 for i in range(20)],
        "monthly_charges": [20.0 + i * 3.5 for i in range(20)],
        "churn": [0, 1] * 10,
        "price": [100.0 + i * 15.0 for i in range(20)],
        "group_id": [f"grp_{i % 4}" for i in range(20)],
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="D"),
    })


def test_compile_and_verify_valid_spec(sample_df):
    """A valid spec against matching DataFrame returns (True, [])."""
    spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["tenure", "monthly_charges"],
        excluded_columns=["customer_id"],
        rationale="Valid classification spec.",
    )
    is_valid, errors = compile_and_verify_spec(spec, sample_df)
    assert is_valid is True
    assert errors == []


def test_compile_and_verify_nonexistent_columns(sample_df):
    """Spec with missing target and missing features returns explicit errors."""
    spec = JobSpec(
        task_type="classification",
        target_column="nonexistent_target",
        feature_columns=["tenure", "missing_feat_1", "missing_feat_2"],
        rationale="Spec with nonexistent columns.",
    )
    is_valid, errors = compile_and_verify_spec(spec, sample_df)
    assert is_valid is False
    assert any("Target column 'nonexistent_target' does not exist" in e for e in errors)
    assert any("missing_feat_1" in e for e in errors)


def test_compile_and_verify_null_target(sample_df):
    """Target column with null values returns a validation error."""
    df_null = sample_df.copy()
    df_null.loc[0, "churn"] = np.nan

    spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["tenure"],
        rationale="Spec testing null target.",
    )
    is_valid, errors = compile_and_verify_spec(spec, df_null)
    assert is_valid is False
    assert any("contains 1 null values" in e for e in errors)


def test_compile_and_verify_single_class_target():
    """Classification target with only 1 class returns an error."""
    df_single = pd.DataFrame({
        "feat": [1, 2, 3, 4],
        "label": ["A", "A", "A", "A"],
    })
    spec = JobSpec(
        task_type="classification",
        target_column="label",
        feature_columns=["feat"],
        rationale="Spec testing single class.",
    )
    is_valid, errors = compile_and_verify_spec(spec, df_single)
    assert is_valid is False
    assert any("must contain at least 2 classes" in e for e in errors)


def test_compile_and_verify_sparse_class_target():
    """Classification target where a class has < 2 samples returns an error."""
    df_sparse = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5],
        "label": ["A", "A", "A", "A", "B"],  # B has only 1 sample
    })
    spec = JobSpec(
        task_type="classification",
        target_column="label",
        feature_columns=["feat"],
        rationale="Spec testing sparse class.",
    )
    is_valid, errors = compile_and_verify_spec(spec, df_sparse)
    assert is_valid is False
    assert any("Every class must have at least 2 samples" in e for e in errors)


def test_compile_and_verify_regression_non_numeric():
    """Regression target must be numeric."""
    df_text = pd.DataFrame({
        "feat": [1, 2, 3, 4],
        "target": ["low", "high", "medium", "low"],
    })
    spec = JobSpec(
        task_type="regression",
        target_column="target",
        feature_columns=["feat"],
        rationale="Regression on string column.",
    )
    is_valid, errors = compile_and_verify_spec(spec, df_text)
    assert is_valid is False
    assert any("must be numeric" in e for e in errors)


def test_compile_and_verify_missing_split_column(sample_df):
    """Missing split column returns an error."""
    spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["tenure"],
        split_strategy="group",
        split_column="ghost_column",
        rationale="Missing split column.",
    )
    is_valid, errors = compile_and_verify_spec(spec, sample_df)
    assert is_valid is False
    assert any("Split column 'ghost_column' does not exist" in e for e in errors)


# -------------------------------------------------------------------------
# 3. Skill Recipes & Registry Tests
# -------------------------------------------------------------------------

def test_skill_registry_index():
    """Registry provides a compact, informative index for system prompts."""
    index_str = get_skills_index()
    assert "task_and_target" in index_str
    assert "leakage_and_splits" in index_str
    assert "uncertainty_and_metrics" in index_str
    # Verify token/word compactness (< 150 words)
    assert len(index_str.split()) < 150


def test_skill_registry_content_retrieval():
    """Registry returns complete recipe content on demand."""
    content = get_skill_content("task_and_target")
    assert "Classification vs. Regression" in content
    assert "Binary Targets Encoded as Text" in content

    leak_content = get_skill_content("leakage_and_splits")
    assert "Data Leakage Prevention" in leak_content

    uncertainty_content = get_skill_content("uncertainty_and_metrics")
    assert "Majority-Class Baseline" in uncertainty_content


def test_skill_registry_missing_skill_raises():
    """Requesting an unknown skill raises KeyError."""
    with pytest.raises(KeyError) as exc_info:
        get_skill_content("unknown_recipe")
    assert "Skill 'unknown_recipe' not found" in str(exc_info.value)


# -------------------------------------------------------------------------
# 4. Cached TabPFN Runner Tests (Offline Mocking & Deterministic Caching)
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tabpfn_runner_caching_and_mock_execution(sample_df, tmp_path):
    """Execute TabPFN runner with mock API, verify cache creation, and assert zero repeat calls."""
    raw_bytes = sample_df.to_csv(index=False).encode("utf-8")
    spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["tenure", "monthly_charges"],
        split_strategy="random",
        holdout_fraction=0.2,
        random_seed=42,
        rationale="Classification run to verify caching and metrics.",
    )

    # Construct mock API caller
    mock_api = MagicMock()
    mock_api.return_value = {
        "predictions": np.array([0, 1, 0, 1]),
        "probabilities": np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]),
        "classes": [0, 1],
    }

    cache_dir = tmp_path / "cache" / "tabpfn"
    cache_key = compute_cache_key(raw_bytes, spec)
    expected_cache_file = cache_dir / f"{cache_key}.json"

    assert not expected_cache_file.exists()

    # First Execution: Cache Miss -> Mock API called
    result1 = await run_tabpfn_job(
        spec=spec,
        df=sample_df,
        raw_bytes=raw_bytes,
        cache_dir=cache_dir,
        api_caller=mock_api,
    )

    assert mock_api.call_count == 1
    assert expected_cache_file.exists()
    assert result1["train_size"] + result1["holdout_size"] == len(sample_df)
    assert "baseline_metrics" in result1
    assert "model_metrics" in result1
    assert "delta" in result1
    assert len(result1["sample_predictions"]) == result1["holdout_size"]
    assert "accuracy" in result1["baseline_metrics"]

    # Second Execution: Cache Hit -> Mock API MUST NOT be called again
    result2 = await run_tabpfn_job(
        spec=spec,
        df=sample_df,
        raw_bytes=raw_bytes,
        cache_dir=cache_dir,
        api_caller=mock_api,
    )

    # Assert zero network / API overhead on repeated execution
    assert mock_api.call_count == 1
    assert result2 == result1


@pytest.mark.asyncio
async def test_tabpfn_runner_regression_caching(sample_df, tmp_path):
    """Execute regression pipeline, verify quantiles and baseline delta computation."""
    raw_bytes = sample_df.to_csv(index=False).encode("utf-8")
    spec = JobSpec(
        task_type="regression",
        target_column="price",
        feature_columns=["tenure", "monthly_charges"],
        split_strategy="random",
        holdout_fraction=0.2,
        random_seed=42,
        rationale="Regression run with point predictions and quantiles.",
    )

    mock_api = MagicMock()
    mock_api.return_value = {
        "predictions": np.array([120.0, 150.0, 180.0, 210.0]),
        "quantiles": {
            "q10": np.array([110.0, 140.0, 170.0, 200.0]),
            "q50": np.array([120.0, 150.0, 180.0, 210.0]),
            "q90": np.array([130.0, 160.0, 190.0, 220.0]),
        },
    }

    cache_dir = tmp_path / "cache" / "tabpfn"

    result = await run_tabpfn_job(
        spec=spec,
        df=sample_df,
        raw_bytes=raw_bytes,
        cache_dir=cache_dir,
        api_caller=mock_api,
    )

    assert mock_api.call_count == 1
    assert "rmse" in result["baseline_metrics"]
    assert "r2" in result["baseline_metrics"]
    assert "rmse" in result["model_metrics"]
    assert "r2" in result["model_metrics"]
    assert "rmse" in result["delta"]
    assert "q10" in result["sample_predictions"][0]["quantiles"]

    # Run again: verify cache hit
    result_cached = await run_tabpfn_job(
        spec=spec,
        df=sample_df,
        raw_bytes=raw_bytes,
        cache_dir=cache_dir,
        api_caller=mock_api,
    )
    assert mock_api.call_count == 1
    assert result_cached == result


def test_tabpfn_runner_sync_wrapper(sample_df, tmp_path):
    """Verify synchronous runner wrapper functions identically."""
    raw_bytes = sample_df.to_csv(index=False).encode("utf-8")
    spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["tenure"],
        split_strategy="random",
        holdout_fraction=0.2,
        random_seed=42,
        rationale="Testing sync runner wrapper.",
    )

    mock_api = MagicMock()
    mock_api.return_value = {
        "predictions": np.array([0, 1, 0, 1]),
        "probabilities": np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]),
        "classes": [0, 1],
    }

    cache_dir = tmp_path / "cache" / "tabpfn"
    res1 = run_tabpfn_job_sync(spec, sample_df, raw_bytes, cache_dir=cache_dir, api_caller=mock_api)
    assert mock_api.call_count == 1

    res2 = run_tabpfn_job_sync(spec, sample_df, raw_bytes, cache_dir=cache_dir, api_caller=mock_api)
    assert mock_api.call_count == 1
    assert res1 == res2
