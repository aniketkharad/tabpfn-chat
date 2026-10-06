"""JobSpec schema contract and validation for tabchat."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


from tabchat.schemas.job_spec import JobSpec


def validate_job_spec(spec: JobSpec, card: dict[str, Any]) -> list[str]:
    """Validate a JobSpec against a Dataset Card and return a list of error strings.

    Args:
        spec: The JobSpec instance to validate.
        card: The dataset card dict containing column_names, inferred_types, etc.

    Returns:
        List of validation error messages. Empty list indicates spec is valid.
    """
    errors: list[str] = []
    columns: set[str] = set(card.get("column_names", []))
    inferred_types: dict[str, str] = card.get("inferred_types", {})

    # 1. Target column must exist
    if spec.target_column not in columns:
        errors.append(
            f"Target column '{spec.target_column}' does not exist in dataset columns: {sorted(columns)}."
        )

    # 2. Target column cannot be in feature columns
    if spec.target_column in spec.feature_columns:
        errors.append(
            f"Target column '{spec.target_column}' cannot also be included in feature_columns."
        )

    # 3. Feature columns must exist and be non-empty
    if not spec.feature_columns:
        errors.append("feature_columns must contain at least 1 column.")
    else:
        for feat in spec.feature_columns:
            if feat not in columns:
                errors.append(
                    f"Feature column '{feat}' does not exist in dataset columns."
                )

    # 4. Split column validations
    if spec.split_strategy in ("chronological", "group"):
        if not spec.split_column:
            errors.append(
                f"split_column is required when split_strategy is '{spec.split_strategy}'."
            )
        elif spec.split_column not in columns:
            errors.append(
                f"split_column '{spec.split_column}' does not exist in dataset columns."
            )
        elif spec.split_column == spec.target_column:
            errors.append(
                f"split_column '{spec.split_column}' cannot be the same as target_column."
            )

    # 5. Type compatibility check for regression
    if spec.target_column in columns and spec.task_type == "regression":
        t_type = inferred_types.get(spec.target_column)
        if t_type == "categorical":
            errors.append(
                f"Target column '{spec.target_column}' is inferred as categorical, which is incompatible with task_type 'regression'."
            )

    # 6. Holdout fraction range check
    if not (0.1 <= spec.holdout_fraction <= 0.4):
        errors.append(
            f"holdout_fraction {spec.holdout_fraction} must be between 0.1 and 0.4."
        )

    return errors


def compute_dataset_hash(file_bytes: bytes) -> str:
    """Compute deterministic SHA-256 hex digest of raw CSV bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def compute_spec_hash(spec: JobSpec | dict[str, Any]) -> str:
    """Compute canonical SHA-256 hex digest of a JobSpec."""
    if isinstance(spec, JobSpec):
        data = spec.model_dump(mode="json")
    else:
        data = dict(spec)
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_tabpfn_cache_key(dataset_hash: str, spec_hash: str) -> str:
    """Generate cache filename identifier conforming to <dataset_hash>_<spec_hash>."""
    return f"{dataset_hash}_{spec_hash}"
