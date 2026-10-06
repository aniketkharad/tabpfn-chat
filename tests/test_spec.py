"""Unit tests for JobSpec schema, validation rules, and canonical hashing."""

import pytest
from tabchat.spec import (
    JobSpec,
    compute_dataset_hash,
    compute_spec_hash,
    get_tabpfn_cache_key,
    validate_job_spec,
)

SAMPLE_CARD = {
    "row_count": 100,
    "col_count": 4,
    "column_names": ["age", "income", "churn", "signup_date"],
    "inferred_types": {
        "age": "numeric",
        "income": "numeric",
        "churn": "categorical",
        "signup_date": "datetime",
    },
    "null_counts": {"age": 0, "income": 0, "churn": 0, "signup_date": 0},
    "unique_counts": {"age": 45, "income": 80, "churn": 2, "signup_date": 95},
    "preview_rows": [],
}


def test_valid_job_spec_creation():
    """Verify standard valid classification JobSpec creation."""
    spec = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["age", "income"],
        split_strategy="random",
        holdout_fraction=0.2,
        random_seed=42,
        eval_metric="roc_auc",
        rationale="Predict customer churn from demographic and financial features.",
    )
    assert spec.target_column == "churn"
    assert spec.feature_columns == ["age", "income"]
    assert spec.split_strategy == "random"
    assert spec.holdout_fraction == 0.2

    errors = validate_job_spec(spec, SAMPLE_CARD)
    assert len(errors) == 0


def test_spec_target_in_features_raises_error():
    """Target column cannot also be in feature_columns."""
    with pytest.raises(ValueError, match="cannot also be in feature_columns"):
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=["churn", "age"],
            rationale="Invalid spec with target in features.",
        )


def test_spec_empty_feature_columns_raises_error():
    """Feature columns list must not be empty."""
    with pytest.raises(ValueError, match="must contain at least 1 column"):
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=[],
            rationale="Invalid spec with empty features.",
        )


def test_spec_missing_split_column_for_chronological():
    """Chronological split requires non-empty split_column."""
    with pytest.raises(ValueError, match="split_column is required"):
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=["age", "income"],
            split_strategy="chronological",
            split_column=None,
            rationale="Missing split column.",
        )


def test_spec_split_column_cannot_be_target():
    """split_column cannot be identical to target_column."""
    with pytest.raises(ValueError, match="cannot be the target_column"):
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=["age", "income"],
            split_strategy="chronological",
            split_column="churn",
            rationale="Split column matches target.",
        )


def test_spec_metric_task_mismatch_raises_error():
    """Regression metric with classification task must be rejected."""
    with pytest.raises(ValueError, match="only valid for regression"):
        JobSpec(
            task_type="classification",
            target_column="churn",
            feature_columns=["age", "income"],
            eval_metric="rmse",
            rationale="Mismatched metric.",
        )

    with pytest.raises(ValueError, match="only valid for classification"):
        JobSpec(
            task_type="regression",
            target_column="income",
            feature_columns=["age"],
            eval_metric="roc_auc",
            rationale="Mismatched metric.",
        )


def test_validate_job_spec_nonexistent_columns():
    """validate_job_spec detects columns not present in the dataset card."""
    spec = JobSpec(
        task_type="classification",
        target_column="non_existent_target",
        feature_columns=["age", "ghost_feature"],
        split_strategy="chronological",
        split_column="phantom_time",
        rationale="Spec with nonexistent columns.",
    )
    errors = validate_job_spec(spec, SAMPLE_CARD)
    assert any("Target column 'non_existent_target' does not exist" in e for e in errors)
    assert any("Feature column 'ghost_feature' does not exist" in e for e in errors)
    assert any("split_column 'phantom_time' does not exist" in e for e in errors)


def test_validate_job_spec_regression_on_categorical_target():
    """validate_job_spec flags regression tasks attempted on categorical targets."""
    spec = JobSpec(
        task_type="regression",
        target_column="churn",  # Categorical in SAMPLE_CARD
        feature_columns=["age", "income"],
        eval_metric="rmse",
        rationale="Trying regression on categorical target.",
    )
    errors = validate_job_spec(spec, SAMPLE_CARD)
    assert any("incompatible with task_type 'regression'" in e for e in errors)


def test_spec_canonical_hashing_invariance():
    """Canonical spec hash is identical regardless of JSON key ordering."""
    spec1 = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["age", "income"],
        random_seed=42,
        rationale="Consistent hash test.",
    )
    spec2 = JobSpec(
        rationale="Consistent hash test.",
        target_column="churn",
        feature_columns=["age", "income"],
        random_seed=42,
        task_type="classification",
    )
    hash1 = compute_spec_hash(spec1)
    hash2 = compute_spec_hash(spec2)
    assert hash1 == hash2

    # Modified field alters hash
    spec3 = JobSpec(
        task_type="classification",
        target_column="churn",
        feature_columns=["age", "income"],
        random_seed=999,  # Changed
        rationale="Consistent hash test.",
    )
    assert compute_spec_hash(spec3) != hash1


def test_dataset_hash_and_cache_key():
    """Verify raw bytes hashing and formatted cache keys."""
    raw_data = b"col1,col2\n1,2\n3,4\n"
    d_hash = compute_dataset_hash(raw_data)
    assert len(d_hash) == 64

    s_hash = "abcdef123456"
    cache_key = get_tabpfn_cache_key(d_hash, s_hash)
    assert cache_key == f"{d_hash}_{s_hash}"
