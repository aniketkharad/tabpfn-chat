# UI Verification Checklist (`docs/UI_CHECKLIST.md`)

This document outlines manual and automated verification procedures for the **tabchat** single-page web interface (`src/tabchat/frontend/`).

---

## 1. Mock Mode Immediate Verification (`?mock=1`)
Open `http://localhost:8000/?mock=1` in any modern web browser.
Verify that the purple **Mock Mode** badge appears in the top header. In this mode, zero external API keys or background services are required.

- [ ] **Initial Page Load**:
  - Session ID is initialized (e.g., `mock-session-demo`).
  - LLM concurrency badge displays green `Idle`.
  - Turns badge displays `15/15 remaining`.
  - Welcome system message is visible in the chat feed.
  - Chat input is enabled and focused.

- [ ] **Drag-and-Drop Valid CSV**:
  - Drag and drop any `.csv` file into the upload zone (or click "Browse Files").
  - Verification:
    - In-flight text displays: "Uploading & parsing dataset...".
    - Dataset Card appears in the left pane showing shape (`30 rows × 5 cols`).
    - Schema & Types list displays chips for `customer_id` (categorical), `age` (numeric), `tenure` (numeric), `balance` (numeric), `churn` (categorical).
    - Scrollable preview table renders the top 5 rows with zebra striping.
    - System message in chat confirms: "Dataset loaded: 30 rows, 5 columns."

- [ ] **Upload CSV Exceeding Limits (Simulated 422 Error)**:
  - Drag or select a file named `over200_sample.csv` or `error_test.csv`.
  - Verification:
    - Red callout box appears below the upload box with header: "❌ Validation Errors:".
    - Displays exact granular validation errors:
      - `Row 201: Exceeds row limit of 200 (found 250 rows)`
      - `Cell [row 5, col 3]: String length 72 exceeds 64 chars limit`
    - No broken state or freeze occurs.
    - Re-uploading a valid file clears the error box immediately.

- [ ] **Conversational Planning & Plan Card**:
  - In the chat bar, enter: "Predict whether customers will churn based on their account info" and press <kbd>Enter</kbd>.
  - Verification:
    - User message appears right-aligned in blue.
    - Concurrency badge pulses amber `Processing`.
    - Loading indicator displays: "Gemini is planning...".
    - Input bar is disabled during planning.
    - Turns counter decreases to `14/15 remaining`.
    - Assistant message appears with micro-markdown formatted text.
    - Dynamic **Plan Card** renders in the timeline:
      - Target Column: `churn`
      - Validation Split: `random (80/20)`
      - Features Included: `age`, `tenure`, `balance`
      - Features Excluded: `customer_id` (strikethrough)
      - Prominent green button: `🚀 Confirm & Run TabPFN`
      - Edit hint: "Want to change features or target? Reply in chat to adjust the plan."

- [ ] **Run Analysis & Dynamic Results Card**:
  - Click `🚀 Confirm & Run TabPFN`.
  - Verification:
    - Loading spinner displays: "TabPFN is training...".
    - Dynamic **Results Card** renders in the feed:
      - Holdout sample size badge: `Holdout N=6`.
      - **Metrics Comparison Table**:
        - Accuracy: TabPFN `0.8333` vs Baseline `0.5000` -> Lift `+0.3333` (highlighted green).
        - Log Loss: TabPFN `0.3842` vs Baseline `0.6931` -> Lift `-0.3089` (highlighted green).
      - **Inline SVG Chart**:
        - Horizontal bars render holdout sample predictions (#4, #7, #12, #18, #23).
        - Confidence bars display percentages (e.g. `82% (yes)`, `89% (no)`) with green/amber classification coding.
      - **Holdout Predictions Preview**:
        - Table showing row indexes, actual values, predicted values, and probability splits.
      - **Grounded Narration**:
        - Markdown-formatted narrative explaining lift and calibration.

- [ ] **Error Handling Simulation**:
  - Chat input `test 429`:
    - Rate limit warning banner drops down: "⚠️ Rate limit hit. Waiting for cooldown...".
    - Banner auto-dismisses after 8 seconds.
  - Chat input `test 503`:
    - Fullscreen modal appears: "Daily API budget exhausted. Please resume tomorrow." with "Acknowledge" button.

- [ ] **Session Reset**:
  - Click `Reset Session` button in the top left.
  - Confirm browser prompt.
  - Verification:
    - Session ID regenerates.
    - Dataset preview clears.
    - Chat feed resets to initial welcome greeting.
    - Turns badge resets to `15/15 remaining`.

---

## 2. Live Backend Verification (`http://localhost:8000`)
With the FastAPI server running (`uv run --env-file .env uvicorn tabchat.app:app --host 0.0.0.0 --port 8000`):

- [ ] **Live File Upload**:
  - Upload a valid CSV file (e.g., sample churn or tabular data &le; 200 rows).
  - Confirm HTTP 200 response from `POST /api/upload`.
  - Confirm preview table and column types populate from live dataset card JSON.

- [ ] **Browser Refresh Persistence**:
  - Refresh the browser (<kbd>F5</kbd> or <kbd>Cmd+R</kbd>).
  - Verification:
    - The active session is retrieved from disk via HttpOnly cookie `GET /api/session`.
    - Dataset preview table persists exactly.
    - Full chat message history persists in feed.
    - Active Plan Card and Results Card persist.
    - Turns remaining counter accurately reflects used turns.

- [ ] **In-Flight Feedback Timers**:
  - Verify that long-running operations (>15s) update text to: "Still working. TabPFN fits may take up to 60s...".
  - Verify that 90s timeout triggers abort and shows explicit `Retry` button.

---

## 3. Responsive Layout Verification
- [ ] **Desktop View (&ge; 800px)**:
  - Two-pane workspace: Left inspector fixed at 380px, Right workspace fills remaining width.
  - Independent scrolling for left inspector pane and right chat stream.
- [ ] **Mobile View (< 800px)**:
  - Layout stacks vertically into a single column.
  - Inspector is at the top, chat timeline flows naturally underneath.
  - Inputs, buttons, and tables remain accessible without overflow distortion.

---

## 4. Security & Safety Verification
- [ ] **XSS Prevention**:
  - Messages containing `<script>alert('xss')</script>` or `<img src=x onerror=alert(1)>` are strictly escaped to text.
  - Only safe micro-markdown tokens (`**bold**`, `*italics*`, `` `inline code` ``, `<br>`) are rendered.
  - Zero raw `innerHTML` from untrusted assistant or user strings without sanitization.
