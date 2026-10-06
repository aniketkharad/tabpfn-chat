"""Unit tests for TabPFN deterministic executor, data splitting, and disk caching."""

import io
from pathlib import Path
import pandas as pd
import pytest

from tabchat.pipeline.executor import TabPFNExecutor, split_dataframe
from tabchat.spec import JobSpec
from tabchat.tabpfn import DiskFitCache, FakeTabPFNClient


@pytest.fixture
def sample_classification_csv() -> bytes:
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        "income": [30, 45, 50, 65, 70, 85, 90, 105, 110, 125],
        "group_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5"],
        "timestamp": [f"2023-01-{i:02d}" for i in range(1, 11)],
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def sample_dataset_card() -> dict:
    return {
        "row_count": 10,
        "col_count": 5,
        "column_names": ["age", "income", "group_id", "timestamp", "target"],
        "inferred_types": {
            "age": "numeric",
            "income": "numeric",
            "group_id": "categorical",
            "timestamp": "datetime",
            "target": "numeric",
        },
        "null_counts": {c: 0 for c in ["age", "income", "group_id", "timestamp", "target"]},
        "unique_counts": {"age": 10, "income": 10, "group_id": 5, "timestamp": 10, "target": 2},
        "preview_rows": [],
    }


def test_split_dataframe_random(sample_classification_csv):
    """Verify random splitting produces expected proportions and is reproducible."""
    df = pd.read_csv(io.BytesIO(sample_classification_csv))
    spec = JobSpec(
        task_type="classification",
        target_column="target",
        feature_columns=["age", "income"],
        split_strategy="random",
        holdout_fraction=0.3,
        random_seed=42,
        rationale="Random split test.",
    )

    train_1, holdout_1 = split_dataframe(df, spec)
    assert len(train_1) == 7
    assert len(holdout_1) == 3
    # Check disjoint
    assert set(train_1.index).isdisjoint(set(holdout_1.index))

    # Reproducibility check with same seed
    train_2, holdout_2 = split_dataframe(df, spec)
    assert train_1.index.equals(train_2.index)
    assert holdout_1.index.equals(holdout_2.index)


def test_split_dataframe_chronological(sample_classification_csv):
    """Verify chronological split sorts strictly by temporal column."""
    df = pd.read_csv(io.BytesIO(sample_classification_csv))
    spec = JobSpec(
        task_type="classification",
        target_column="target",
        feature_columns=["age", "income"],
        split_strategy="chronological",
        split_column="timestamp",
        holdout_fraction=0.2,
        rationale="Chronological split test.",
    )

    train_df, holdout_df = split_dataframe(df, spec)
    assert len(train_df) == 8
    assert len(holdout_df) == 2
    # Ensure all timestamps in train are <= timestamps in holdout
    assert train_df["timestamp"].max() <= holdout_df["timestamp"].min()


def test_split_dataframe_group(sample_classification_csv):
    """Verify group split guarantees zero group overlap between partitions."""
    df = pd.read_csv(io.BytesIO(sample_classification_csv))
    spec = JobSpec(
        task_type="classification",
        target_column="target",
        feature_columns=["age", "income"],
        split_strategy="group",
        split_column="group_id",
        holdout_fraction=0.4,
        random_seed=42,
        rationale="Group split test.",
    )

    train_df, holdout_df = split_dataframe(df, spec)
    assert len(train_df) > 0
    assert len(holdout_df) > 0
    train_groups = set(train_df["group_id"].unique())
    holdout_groups = set(holdout_df["group_id"].unique())
    assert train_groups.isdisjoint(holdout_groups)


def test_disk_fit_caching(tmp_path):
    """Verify disk fit cache storage and retrieval."""
    cache = DiskFitCache(cache_dir=tmp_path / "cache")
    dataset_hash = "dataset123"
    spec_hash = "spec456"

    # Initially empty
    assert cache.get(dataset_hash, spec_hash) is None

    # Set and retrieve
    payload = {"status": "success", "score": 0.95}
    cache.set(dataset_hash, spec_hash, payload)

    retrieved = cache.get(dataset_hash, spec_hash)
    assert retrieved == payload


@pytest.mark.asyncio
async def test_tabpfn_executor_end_to_end_classification(
    tmp_path, sample_classification_csv, sample_dataset_card
):
    """Verify full deterministic classification execution, metrics, and caching."""
    fake_client = FakeTabPFNClient()
    cache = DiskFitCache(cache_dir=tmp_path / "cache")
    executor = TabPFNExecutor(tabpfn_client=fake_client, cache=cache)

    spec = JobSpec(
        task_type="classification",
        target_column="target",
        feature_columns=["age", "income"],
        split_strategy="random",
        holdout_fraction=0.3,
        random_seed=42,
        eval_metric="roc_auc",
        rationale="End to end classification test.",
    )

    # First execution: cache miss, invokes client
    results_1 = await executor.execute(sample_classification_csv, sample_dataset_card, spec)
    assert results_1["cache_hit"] is False
    assert fake_client.call_count == 1
    assert "baseline_metrics" in results_1
    assert "model_metrics" in results_1
    assert "comparison" in results_1
    assert len(results_1["sample_predictions"]) == 3  # 3 holdout rows

    # Second execution: cache hit, zero new client calls
    results_2 = await executor.execute(sample_classification_csv, sample_dataset_card, spec)
    assert results_2["cache_hit"] is True
    assert fake_client.call_count == 1  # Unchanged
    assert results_2["model_metrics"] == results_1["model_metrics"]


@pytest.mark.asyncio
async def test_tabpfn_executor_end_to_end_regression(tmp_path):
    """Verify full deterministic regression execution and prediction intervals."""
    df = pd.DataFrame({
        "sqft": [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000],
        "bedrooms": [1, 1, 2, 2, 3, 3, 4, 4, 4, 5],
        "price": [150.0, 200.0, 280.0, 320.0, 390.0, 440.0, 520.0, 580.0, 610.0, 750.0],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    raw_csv = buf.getvalue()

    card = {
        "row_count": 10,
        "col_count": 3,
        "column_names": ["sqft", "bedrooms", "price"],
        "inferred_types": {"sqft": "numeric", "bedrooms": "numeric", "price": "numeric"},
        "null_counts": {"sqft": 0, "bedrooms": 0, "price": 0},
        "unique_counts": {"sqft": 10, "bedrooms": 5, "price": 10},
        "preview_rows": [],
    }

    spec = JobSpec(
        task_type="regression",
        target_column="price",
        feature_columns=["sqft", "bedrooms"],
        split_strategy="random",
        holdout_fraction=0.3,
        random_seed=42,
        eval_metric="rmse",
        rationale="End to end regression test.",
    )

    fake_client = FakeTabPFNClient()
    executor = TabPFNExecutor(tabpfn_client=fake_client, cache=DiskFitCache(cache_dir=tmp_path / "cache"))

    results = await executor.execute(raw_csv, card, spec)
    assert results["task_type"] == "regression"
    assert "baseline_rmse" in results["comparison"]
    assert "model_rmse" in results["comparison"]
    assert "interval_coverage_80" in results["model_metrics"]
    assert len(results["sample_predictions"]) == 3
    assert "interval_80" in results["sample_predictions"][0]
