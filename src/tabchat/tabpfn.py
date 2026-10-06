"""Prior Labs TabPFN direct REST client, offline mock client, and disk cache."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any
import httpx
import numpy as np
import pandas as pd

from tabchat.config import get_settings

logger = logging.getLogger("tabchat.tabpfn")


class TabPFNError(Exception):
    """Base exception for TabPFN execution failures."""


class TabPFNRateLimitError(TabPFNError):
    """Raised when TabPFN API quota or rate limits are exceeded (HTTP 429)."""


def _sanitize_for_cache(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class DiskFitCache:
    """Manages disk-backed caching of TabPFN model executions under data/cache/tabpfn/."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        if cache_dir is None:
            self.cache_dir = get_settings().DATA_DIR / "cache" / "tabpfn"
        else:
            self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, dataset_hash: str, spec_hash: str) -> Path:
        filename = f"{dataset_hash}_{spec_hash}.json"
        return self.cache_dir / filename

    def get(self, dataset_hash: str, spec_hash: str) -> dict[str, Any] | None:
        """Retrieve cached execution results if available, else None."""
        c_path = self._cache_path(dataset_hash, spec_hash)
        if c_path.exists():
            try:
                data = json.loads(c_path.read_text(encoding="utf-8"))
                logger.info("TabPFN Cache Hit: %s", c_path.name)
                return data
            except Exception as e:
                logger.warning("Corrupted cache file %s, discarding: %e", c_path.name, e)
                c_path.unlink(missing_ok=True)
                return None
        return None

    def set(self, dataset_hash: str, spec_hash: str, results: dict[str, Any]) -> None:
        """Persist execution results under the canonical hash key."""
        c_path = self._cache_path(dataset_hash, spec_hash)
        c_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=_sanitize_for_cache),
            encoding="utf-8",
        )
        logger.info("TabPFN Cache Stored: %s", c_path.name)


class TabPFNClient:
    """Production asynchronous TabPFN client invoking Prior Labs REST API."""

    BASE_URL = "https://api.priorlabs.ai"

    def __init__(self, token: str | None = None, timeout_seconds: float = 90.0) -> None:
        settings = get_settings()
        self.token = token or settings.TABPFN_TOKEN
        self.timeout = timeout_seconds or float(settings.TABPFN_FIT_TIMEOUT_SECONDS)
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def fit_and_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        task: str,  # "classification" or "regression"
    ) -> dict[str, Any]:
        """Execute full fit and predict pipeline against Prior Labs REST API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            train_csv = train_df.to_csv(index=False).encode("utf-8")
            test_csv = test_df.to_csv(index=False).encode("utf-8")

            # 1. Prepare train set upload
            prep_train_resp = await client.post(
                f"{self.BASE_URL}/tabpfn/prepare_train_set_upload",
                headers=self.headers,
                json={"file_name": "train.csv"},
            )
            if prep_train_resp.status_code == 429:
                raise TabPFNRateLimitError(f"TabPFN quota exceeded: {prep_train_resp.text}")
            prep_train_resp.raise_for_status()
            prep_train_data = prep_train_resp.json()
            train_upload_id = prep_train_data["train_set_upload_id"]
            train_put_url = prep_train_data["upload_url"]

            # 2. Upload train CSV bytes
            put_train = await client.put(
                train_put_url,
                content=train_csv,
                headers={"Content-Type": "text/csv"},
            )
            put_train.raise_for_status()

            # 3. Fit model
            fit_resp = await client.post(
                f"{self.BASE_URL}/tabpfn/fit",
                headers=self.headers,
                json={
                    "train_set_upload_id": train_upload_id,
                    "target_column": target_col,
                    "task": task,
                    "tabpfn_config": {"model_path": "v3.5_default"},
                },
            )
            fit_resp.raise_for_status()
            fitted_data = fit_resp.json()
            fitted_train_set_id = fitted_data["fitted_train_set_id"]

            # 4. Prepare test set upload
            prep_test_resp = await client.post(
                f"{self.BASE_URL}/tabpfn/prepare_test_set_upload",
                headers=self.headers,
                json={"file_name": "test.csv"},
            )
            prep_test_resp.raise_for_status()
            prep_test_data = prep_test_resp.json()
            test_upload_id = prep_test_data["test_set_upload_id"]
            test_put_url = prep_test_data["upload_url"]

            # 5. Upload test CSV bytes
            put_test = await client.put(
                test_put_url,
                content=test_csv,
                headers={"Content-Type": "text/csv"},
            )
            put_test.raise_for_status()

            # 6. Predict
            predict_payload: dict[str, Any] = {
                "fitted_train_set_id": fitted_train_set_id,
                "test_set_upload_id": test_upload_id,
            }
            if task == "regression":
                predict_payload["output_type"] = "quantiles"
                predict_payload["quantiles"] = [0.1, 0.5, 0.9]
            else:
                predict_payload["output_type"] = "probas"

            pred_resp = await client.post(
                f"{self.BASE_URL}/tabpfn/predict",
                headers=self.headers,
                json=predict_payload,
            )
            pred_resp.raise_for_status()
            return pred_resp.json()


class FakeTabPFNClient:
    """Deterministic offline mock TabPFN client for testing without external API calls."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.call_count = 0

    async def fit_and_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        task: str,
    ) -> dict[str, Any]:
        """Simulate realistic predictions based on training data distribution and features."""
        self.call_count += 1
        rng = np.random.RandomState(self.seed + self.call_count)
        n_test = len(test_df)

        if task == "classification":
            y_train = train_df[target_col].to_numpy()
            classes = list(np.unique(y_train))
            n_classes = len(classes)

            # Generate confident, well-calibrated probabilities
            raw_scores = rng.normal(loc=0.0, scale=1.0, size=(n_test, n_classes))
            # If ground truth target exists in test_df, add positive signal towards true class
            if target_col in test_df.columns:
                y_test = test_df[target_col].to_numpy()
                for i, true_val in enumerate(y_test):
                    if true_val in classes:
                        c_idx = classes.index(true_val)
                        raw_scores[i, c_idx] += 1.8  # Strong signal boost

            exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            predicted_indices = np.argmax(probs, axis=1)
            predicted_classes = [
                int(classes[idx]) if isinstance(classes[idx], (np.integer, int))
                else float(classes[idx]) if isinstance(classes[idx], (np.floating, float))
                else str(classes[idx])
                for idx in predicted_indices
            ]

            return {
                "task": "classification",
                "classes": [str(c) for c in classes],
                "predictions": predicted_classes,
                "probabilities": probs.tolist(),
                "model_version": "fake-tabpfn-3.5",
            }

        else:  # regression
            y_train = train_df[target_col].to_numpy(dtype=float)
            mean_y = float(np.mean(y_train))
            std_y = float(np.std(y_train)) if len(y_train) > 1 else 1.0

            # Generate synthetic point predictions with signal
            if target_col in test_df.columns:
                y_test = test_df[target_col].to_numpy(dtype=float)
                # Blend true target with small Gaussian noise
                noise = rng.normal(0.0, 0.25 * std_y, size=n_test)
                point_preds = y_test + noise
            else:
                point_preds = mean_y + rng.normal(0.0, std_y, size=n_test)

            q10 = point_preds - 1.28 * (0.3 * std_y)
            q50 = point_preds
            q90 = point_preds + 1.28 * (0.3 * std_y)

            return {
                "task": "regression",
                "predictions": point_preds.tolist(),
                "quantiles": {
                    "q10": q10.tolist(),
                    "q50": q50.tolist(),
                    "q90": q90.tolist(),
                },
                "model_version": "fake-tabpfn-3.5",
            }
