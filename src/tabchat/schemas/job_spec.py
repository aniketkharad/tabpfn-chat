"""Pydantic V2 JobSpec schema contract for tabchat."""

from __future__ import annotations

import json
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class JobSpec(BaseModel):
    """The compiled job specification contract passed to the executor."""

    task_type: Literal["classification", "regression"] = Field(
        ..., description="Supervised learning task type"
    )
    target_column: str = Field(
        ..., description="Name of the target column to predict"
    )
    feature_columns: list[str] = Field(
        ..., description="List of feature column names (must contain at least 1)"
    )
    excluded_columns: list[str] = Field(
        default_factory=list, description="Columns explicitly excluded from modeling"
    )
    split_strategy: Literal["random", "chronological", "group"] = Field(
        default="random", description="Strategy used to partition train and holdout data"
    )
    split_column: str | None = Field(
        default=None,
        description="Column used for chronological or group splits",
    )
    holdout_fraction: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
        description="Fraction of data reserved for holdout evaluation (0.1 to 0.4)",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducible dataset partitioning",
    )
    eval_metric: Literal["auto", "roc_auc", "accuracy", "log_loss", "r2", "rmse"] = Field(
        default="auto",
        description="Evaluation metric for model assessment",
    )
    confidence_intervals: bool = Field(
        default=True,
        description="Whether to compute calibrated intervals or predictive quantiles",
    )
    rationale: str = Field(
        ...,
        description="Short 1-sentence reasoning for choices",
    )

    @model_validator(mode="after")
    def validate_spec_rules(self) -> "JobSpec":
        """Validate intra-spec semantic integrity."""
        # 1. target_column is NOT in feature_columns
        if self.target_column in self.feature_columns:
            raise ValueError(
                f"target_column '{self.target_column}' cannot also be in feature_columns"
            )

        # 2. feature_columns must contain at least 1 column
        if not self.feature_columns or len(self.feature_columns) < 1:
            raise ValueError("feature_columns must contain at least 1 column")

        # 3. If split_strategy is 'chronological' or 'group', split_column MUST be specified and cannot be target_column
        if self.split_strategy in ("chronological", "group"):
            if not self.split_column or not self.split_column.strip():
                raise ValueError(
                    f"split_column is required when split_strategy is '{self.split_strategy}'"
                )
            if self.split_column == self.target_column:
                raise ValueError(
                    f"split_column '{self.split_column}' cannot be the target_column"
                )

        # 4. Metric compatibility check (if not auto)
        if self.task_type == "classification" and self.eval_metric in ("r2", "rmse"):
            raise ValueError(
                f"Metric '{self.eval_metric}' is only valid for regression, not classification"
            )
        if self.task_type == "regression" and self.eval_metric in ("roc_auc", "accuracy", "log_loss"):
            raise ValueError(
                f"Metric '{self.eval_metric}' is only valid for classification, not regression"
            )

        return self

    def canonical_json(self) -> str:
        """Produce a canonical, deterministically sorted JSON string for hashing."""
        data = self.model_dump()
        return json.dumps(data, sort_keys=True, separators=(",", ":"))
