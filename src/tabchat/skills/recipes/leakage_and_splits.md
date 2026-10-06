---
name: leakage_and_splits
description: Spotting ID columns, detecting sequential date leakage, group split guidelines.
tags: [data-leakage, split-strategy, time-series, groups]
---

# Data Leakage Prevention and Dataset Splitting Recipe

## 1. Spotting ID and Identifier Columns
- Columns representing unique record identifiers (e.g., `id`, `user_id`, `patient_id`, `uuid`, `row_num`, `order_number`) have nearly 100% cardinality.
- Identifiers carry spurious correlation and lead to catastrophic overfitting if used as features. They must be placed in `excluded_columns` unless designated as a `split_column` in a group split.

## 2. Detecting Sequential and Temporal Date Leakage
- When a dataset contains dates, timestamps, or ordered sequence indices (e.g., `transaction_date`, `created_at`, `timestamp`), performing a random split causes future information to leak into past predictions.
- For temporal data, designate `split_strategy: "chronological"` and supply the temporal column in `split_column`.
- The dataset is sorted by `split_column`, placing the earlier historical records into the training partition and future records into the holdout partition.

## 3. Group Split Guidelines
- If individual subjects or entities have multiple recorded entries (e.g., multiple visits per patient, multiple sessions per customer), records from the same entity must never appear in both train and holdout splits simultaneously.
- Set `split_strategy: "group"` and assign `split_column` to the entity grouping identifier. Entire groups are allocated together to either train or holdout.

## 4. Holdout Fraction Selection
- Maintain `holdout_fraction` between 0.1 and 0.4 (default 0.2). TabPFN performs best when trained on representative sample sizes while preserving enough holdout rows for reliable empirical metric evaluation.
