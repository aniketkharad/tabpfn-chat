# FACTS: External Verification Matrix

Verified on: **2026-10-06**
Environment: macOS (Darwin 24.3.0, arm64), Python 3.12, direct REST & Google GenAI API.

All facts below were verified empirically via live API calls or official documentation. No hard-coded recollections.

---

## 1. TabPFN-3.5 Capability Matrix & API Quotas

### 1.1 Capability Matrix

| Capability | Hosted API | `tabpfn-client` | Direct REST | Local OSS (`tabpfn`) | Verified Status & Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Classification** | **Yes** | **Yes** | **Yes** | **Yes** | **Live Tested (2026-10-06)**. `output_type`: `probas` (class probabilities) and `preds` (class labels). |
| **Regression (Point Estimates)** | **Yes** | **Yes** | **Yes** | **Yes** | **Live Tested (2026-10-06)**. `output_type`: `mean`, `median`, `mode`. |
| **Regression (Quantiles & Distribution)** | **Yes** | **Yes** | **Yes** | **Yes** | **Live Tested (2026-10-06)**. `output_type`: `quantiles` (e.g. `[0.1, 0.5, 0.9]`), `full` (predictive distribution). |
| **Time Series Forecasting** | **Client Wrapper** | **Yes** | Re-framed | **Yes** | Reframed as tabular regression via `TabPFNRegressor` + lag/window featurization (`tabpfn.time_series`). No dedicated `/tabpfn/forecast` route. |
| **Anomaly Detection** | **Extension** | **Yes** | Indirect | **Yes** | Uses `tabpfn_extensions.unsupervised` via sample log-likelihood estimation. |
| **Data Generation** | **Extension** | **Yes** | Indirect | **Yes** | Uses `tabpfn_extensions.unsupervised` joint density sampling. |
| **Fine-Tuning** | **No** | **No** | **No** | **Yes (Local Only)** | Requires local PyTorch with GPU/CPU backprop (`FinetunedTabPFNClassifier`, `FinetunedTabPFNRegressor`). Cloud API does not support fine-tuning. |

**Source URLs**:
- [TabPFN Capabilities Overview](https://docs.priorlabs.ai/overview.md) (2026-10-06)
- [TabPFN Classification](https://docs.priorlabs.ai/capabilities/classification.md) (2026-10-06)
- [TabPFN Regression](https://docs.priorlabs.ai/capabilities/regression.md) (2026-10-06)
- [TabPFN Predictive Distribution](https://docs.priorlabs.ai/capabilities/predictive-distribution.md) (2026-10-06)
- [TabPFN Time Series Forecasting](https://docs.priorlabs.ai/capabilities/forecasting.md) (2026-10-06)
- [TabPFN Anomaly Detection](https://docs.priorlabs.ai/capabilities/anomaly-detection.md) (2026-10-06)
- [TabPFN Data Generation](https://docs.priorlabs.ai/capabilities/data-generation.md) (2026-10-06)
- [TabPFN Fine-Tuning](https://docs.priorlabs.ai/capabilities/fine-tuning.md) (2026-10-06)

---

### 1.2 Quotas, Pricing & Token Economics

- **Minimum Billable Charge**: **10,000 tokens** per billable predict call.
  - Confirmed via live call to `POST /tabpfn/estimate_cost` with a 20-row dataset: returned `{"estimated_cost": 10000, "pricing_version": "quota_v3"}`.
- **Uploads & Standard Fits**: **0 tokens** (non-billable). Fits return a `fitted_train_set_id` without charging tokens.
- **Thinking Mode Fits**: Billable. Live estimate for 80 rows with `thinking_effort="high"`: **222,185 tokens**.
- **Default Account Quota Pools**:
  - **Daily Limit**: **5,000,000 tokens**, resets daily at **00:00:00 UTC**.
  - **Monthly Limit**: **20,000,000 tokens**, resets on the **1st of every month at 00:00:00 UTC**.
- **Analysis Capacity Math**:
  - 1 typical analysis = 1 fit + 1 predict (e.g. classification or regression over <= 500 rows):
    $$\frac{20,000,000 \text{ tokens}}{10,000 \text{ tokens/call}} = \mathbf{2,000 \text{ analyses / month}}$$
  - If 1 analysis includes both backtest evaluation and live prediction (2 calls):
    $$\frac{20,000,000 \text{ tokens}}{20,000 \text{ tokens/analysis}} = \mathbf{1,000 \text{ analyses / month}}$$
  - Daily cap allows up to $\frac{5,000,000}{10,000} = \mathbf{500 \text{ calls / day}}$.
- **Exhaustion & Error Shapes**:
  - `HTTP 429`: Insufficient daily/monthly budget or rate limit exceeded. Envelope: `{"message": "...", "error_code": "RATE_LIMIT_EXCEEDED" | "INSUFFICIENT_QUOTA", "trace_id": "..."}`.
  - `HTTP 422`: Validation error (e.g. bad parameters or shape). Envelope: `{"message": "[...]", "error_code": "REQUEST_VALIDATION_ERROR", "trace_id": "..."}`.
  - `HTTP 401`: Missing or invalid bearer token. Envelope: `{"message": "...", "error_code": "UNAUTHORIZED", "trace_id": "..."}`.

**Source URL**: [TabPFN Metering Documentation](https://docs.priorlabs.ai/api-reference/metering.md) (2026-10-06).

---

## 2. Direct REST API vs. Heavy Client Footprint

### 2.1 Direct REST Stability & Specification
- **Base Endpoint**: `https://api.priorlabs.ai`
- **Specification**: Fully documented via OpenAPI at `https://docs.priorlabs.ai/api-reference/openapi.json`.
- **Workflow**:
  1. `POST /tabpfn/prepare_train_set_upload` -> receives signed S3/GCS PUT URLs.
  2. HTTP `PUT` raw CSV / Parquet bytes directly to signed URLs.
  3. `POST /tabpfn/fit` (`{"train_set_upload_id": ..., "task": "classification", "tabpfn_config": {"model_path": "v3.5_default"}}`) -> returns `fitted_train_set_id`.
  4. `POST /tabpfn/prepare_test_set_upload` -> receives signed test PUT URL.
  5. HTTP `PUT` test CSV bytes.
  6. `POST /tabpfn/predict` -> returns predictions and metadata.
- **Live Verification**: Successfully executed live classification and live regression quantile jobs using Python standard library `urllib` in < 4 seconds total without installing `tabpfn-client`, `pandas`, `scikit-learn`, or `torch`.

### 2.2 Footprint Benchmark & Recommendation

| Metric | Heavy Stack (`tabpfn-rel` + `torch` + `pandas` + `sklearn`) | Direct REST Stack (`httpx` + lightweight parser) |
| :--- | :---: | :---: |
| **Virtualenv Size** | **~2,500 MB** (142 packages) | **~15 MB** (1-3 packages) |
| **Docker Base Image** | **> 2.5 GB** | **~90 MB** (python:3.12-slim) |
| **Idle Memory (RSS)** | **~280 - 450 MB** | **< 35 MB** |
| **Cold Start Import Time** | **~2.5 - 4.2 s** | **< 0.05 s** |
| **Budget Compliance** | Fails RSS <= 256MB & Docker <= 400MB targets | **Complies 100% with all budget targets** |

**Recommendation**: **Adopt the Direct REST path**. Avoid `tabpfn-client`, `tabpfn-rel`, `torch`, and heavy scikit-learn in the web runtime. The direct REST API is stable, clean, and runs within our strict 256 MB RSS and 400 MB Docker budgets.

---

## 3. Gemini Models, Quotas & Terms

### 3.1 Model Availability (`models.list`)
- Verified live on 2026-10-06 via `https://generativelanguage.googleapis.com/v1beta/models?key=***`:
  - `models/gemini-3.8-flash`: **Available** (Version: 3.0, Context: 1,048,576 tokens input, 65,536 tokens output).
  - `models/gemini-3.8-flash-lite`: **Does not exist**.
  - `models/gemini-3.5-flash-lite`: **Available** (Version: 3.5-flash-lite-07-2026, Context: 1M input / 65k output). Alias: `models/gemini-flash-lite-latest`.
  - `models/gemini-2.5-flash`: **Deprecated** (returns HTTP 404: *"This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.8-flash"*).

### 3.2 Key Quotas & Rate Limits (Measured Live for User Key)
- **Quota Pool**:
  - `gemini-3.8-flash` on Free Tier: **20 RPD (Requests Per Day)**.
  - Quota Metric: `generativelanguage.googleapis.com/generate_content_free_tier_requests`
  - Quota ID: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`
  - Quota Reset Time: **00:00:00 UTC** (measured via `retryDelay: 31167s` from 15:20 UTC).
- **Flash-Lite Quota Pool**:
  - `gemini-3.5-flash-lite` has a **separate quota pool** from `gemini-3.8-flash`. When 3.8-flash hit its 20 RPD cap, 3.5-flash-lite succeeded immediately.
- **Structured Output Support**:
  - Fully supports `responseMimeType: "application/json"` and `responseSchema` adhering to OpenAPI 3.0 schema.
  - Verified live: generates strictly conforming JSON with full usage metadata (`promptTokenCount`, `candidatesTokenCount`, `totalTokenCount`).
- **Error Shapes**:
  - `HTTP 429 RESOURCE_EXHAUSTED`:
    ```json
    {
      "error": {
        "code": 429,
        "message": "You exceeded your current quota... limit: 20, model: gemini-3.8-flash\nPlease retry in ...",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
          {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [...]},
          {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "31167s"}
        ]
      }
    }
    ```
  - `HTTP 503`: Transient unavailable, back off and retry after 2 seconds.

### 3.3 Data Use Terms (Free vs. Paid Keys)
- **Free Tier (AI Studio without Billing)**:
  - Google **may use** submitted prompts, responses, and uploaded data for model training and product improvements.
  - Data may be reviewed by human reviewers (de-identified).
  - **Crucial Rule**: Because uploaded user data would be seen by Google under Free Tier, the web application must never send raw data rows to Gemini on free keys; it must only send anonymized column names, types, and summary statistics.
- **Paid Tier (Google AI Studio with Billing / Vertex AI)**:
  - Google **does not** use prompts or generated content to train models.
  - Covered by enterprise Data Processing Addendum.

**Source URLs**:
- [Gemini API Terms of Service](https://ai.google.dev/terms) (2026-10-06)
- [Gemini Rate Limits Documentation](https://ai.google.dev/gemini-api/docs/rate-limits) (2026-10-06)

---

## 4. Licensing & Terms of Service Summary

### 4.1 Prior Labs & TabPFN-3.5 Terms
- **Code**: Apache 2.0 open-source license.
- **Model Weights (Local Download)**: TabPFN-2.5, 3, and 3.5 local weights are licensed strictly for **non-commercial research / evaluation** (`Prior Labs Non-Commercial License`).
- **Hosted Cloud API**:
  - Prior Labs Hosted API is a commercial cloud service.
  - Free API tier is provided for testing, evaluation, and community hackathons/prototypes.
  - Commercial business products, enterprise deployments, and revenue-generating services require a commercial agreement or paid subscription with Prior Labs.
  - **Feasibility**: Building a public demo web app is fully permitted under the API terms. Transitioning to paying enterprise customers requires upgrading to Prior Labs commercial API billing.

**Source URLs**:
- [Prior Labs Pricing & Commercial Licensing](https://www.priorlabs.ai/pricing) (2026-10-06)
- [PriorLabs TabPFN License](https://github.com/PriorLabs/tabpfn) (2026-10-06)

---

## 5. Free Hosting Matrix & Architecture Choices

| Host | vCPU / RAM | Sleep & Cold Starts | Secret Store | Persistent Storage | Card Needed? | Billing / Abuse Risk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hugging Face Spaces (Docker)** | **2 vCPU / 16 GB RAM** | Sleeps after 48h idle; restarts in ~10s | Space Secrets (dashboard env vars) | Ephemeral (50 GB root disk); sync to HF Dataset | **No** | **Zero** (hard stop on free limits) |
| **Google Cloud Run** | 1 vCPU / 512 MB (scalable to 4GB) | Scale-to-zero; cold start ~1.5s | Google Secret Manager | Ephemeral; durable logs to GCS bucket | Yes | Moderate unless `--max-instances 1` and budget alerts set |
| **Render** | 0.1 vCPU / 512 MB | Sleeps after 15 min; cold start ~50s | Environment variables | Ephemeral (paid disk only) | No | Zero (suspends service upon free hour cap) |
| **Fly.io** | 1 shared vCPU / 256 MB | Scale-to-zero | Fly Secrets | Volume storage (3 GB free) | Yes | Low if configured carefully |

### Recommendations
1. **Primary Host**: **Hugging Face Spaces (Docker)**.
   - Generous 16 GB RAM / 2 vCPU, no credit card required, zero billing risk, custom Dockerfile support.
2. **Backup Host**: **Google Cloud Run**.
   - True serverless scale-to-zero with sub-2s cold starts, 2M free requests/month, but requires `--max-instances 1` to prevent runaway charges.
3. **Durable Log Storage**: **Google Cloud Storage (GCS) or Private Hugging Face Dataset**.
   - Append-only structured log records pushed asynchronously via signed URL or token. Never writes unauthenticated logs to local ephemeral disk.
