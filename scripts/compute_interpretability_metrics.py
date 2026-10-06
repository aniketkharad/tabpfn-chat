"""Compute exact feature importance and attribution metrics using TabPFN."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import tabpfn_client
from tabpfn_client import TabPFNClassifier
from tabchat.config import get_settings

DATA_PATH = Path("demos/07_interpretability/medical_diagnostics.csv")
OUT_JSON = Path("demos/07_interpretability/feature_importance.json")


def main() -> None:
    settings = get_settings()
    if settings.TABPFN_TOKEN:
        tabpfn_client.set_access_token(settings.TABPFN_TOKEN)
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["age", "bmi", "glucose_level", "insulin", "blood_pressure"]
    target_col = "has_diabetes"

    # Match the standard 80/20 train/holdout split with seed=42
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(df))
    split_idx = int(0.8 * len(df))
    train_idx = indices[:split_idx]
    holdout_idx = indices[split_idx:]

    X_train = df.loc[train_idx, feature_cols]
    y_train = df.loc[train_idx, target_col]
    X_holdout = df.loc[holdout_idx, feature_cols]
    y_holdout = df.loc[holdout_idx, target_col]

    print(f"[*] Fitting TabPFNClassifier on {len(X_train)} training rows...")
    clf = TabPFNClassifier(random_state=42)
    clf.fit(X_train, y_train)

    holdout_acc = float(clf.score(X_holdout, y_holdout))
    print(f"[+] TabPFN Holdout Accuracy: {holdout_acc:.4f}")

    print("[*] Computing holdout permutation feature importance...")
    perm_result = permutation_importance(
        clf,
        X_holdout,
        y_holdout,
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
    )

    importance_data = []
    for i, col in enumerate(feature_cols):
        mean_imp = float(perm_result.importances_mean[i])
        std_imp = float(perm_result.importances_std[i])
        importance_data.append(
            {
                "feature": col,
                "importance_mean": round(mean_imp, 4),
                "importance_std": round(std_imp, 4),
            }
        )

    # Sort descending by importance
    importance_data.sort(key=lambda x: x["importance_mean"], reverse=True)

    summary = {
        "holdout_accuracy": round(holdout_acc, 4),
        "feature_importances": importance_data,
        "primary_driver": importance_data[0]["feature"],
        "secondary_driver": importance_data[1]["feature"],
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] Saved feature importance summary to {OUT_JSON}")
    for item in importance_data:
        print(f"  - {item['feature']:16s}: {item['importance_mean']:+.4f} (±{item['importance_std']:.4f})")


if __name__ == "__main__":
    main()
