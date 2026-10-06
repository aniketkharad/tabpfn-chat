---
name: task_and_target
description: Rules for identifying continuous vs. categorical targets, handling binary targets encoded as text.
tags: [task, target, classification, regression]
---

# Task and Target Selection Recipe

## 1. Classification vs. Regression Target Identification
- **Continuous Targets**: When the target column consists of floating-point numbers or real-valued measurements (e.g., price, revenue, temperature, blood pressure, duration), designate `task_type: "regression"`. If integer values span a broad, continuous range with many unique values (> 10 distinct values), treat as regression.
- **Categorical Targets**: When the target consists of distinct classes, labels, categories, or status codes (e.g., churned/active, fraud/normal, grade A/B/C), designate `task_type: "classification"`.

## 2. Binary Targets Encoded as Text
- Text columns containing binary values (e.g., `"yes"`/`"no"`, `"true"`/`"false"`, `"churn"`/`"retained"`, `"pass"`/`"fail"`, `"default"`/`"paid"`) are classified as `task_type: "classification"`.
- TabPFN accepts string/categorical targets directly without requiring manual integer encoding.

## 3. Class Balance and Feasibility Constraints
- **Minimum Classes**: A classification problem must possess at least 2 distinct target classes.
- **Minimum Class Representation**: Every class must contain at least 2 instances in the uploaded dataset to ensure that both training and holdout partitions have representation.
- **Missing Values**: Target columns must be free of `null` or `NaN` values. Rows with missing targets cannot be used for supervised training or evaluation.
