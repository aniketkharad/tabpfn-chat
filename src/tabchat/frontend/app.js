/**
 * tabchat — Frontend Application
 * Pure vanilla JS, zero external dependencies, mock mode support, XSS-safe micro-markdown.
 */
(function () {
  'use strict';

  const IS_MOCK_MODE = new URLSearchParams(window.location.search).get('mock') === '1';
  const state = { sessionId: null, datasetCard: null, jobSpec: null, results: null, turnsLeft: 15, isInFlight: false, lastAction: null };

  const $ = (id) => document.getElementById(id);
  const el = {
    mockBadge: $('mock-badge'), concurrencyBadge: $('concurrency-badge'), turnsBadge: $('turns-badge'),
    sessionIdDisplay: $('session-id-display'), btnResetSession: $('btn-reset-session'),
    dropZone: $('drop-zone'), fileInput: $('file-input'),
    btnSampleChurn: $('btn-sample-churn'), btnSampleHousing: $('btn-sample-housing'),
    uploadErrors: $('upload-errors'), uploadErrorsList: $('upload-errors-list'),
    datasetCardSection: $('dataset-card-section'), datasetShapeBadge: $('dataset-shape-badge'),
    columnTypesList: $('column-types-list'), previewTableHead: $('preview-table-head'), previewTableBody: $('preview-table-body'),
    chatFeed: $('chat-feed'), inFlightContainer: $('in-flight-container'), inFlightText: $('in-flight-text'),
    btnRetryTimeout: $('btn-retry-timeout'), chatForm: $('chat-form'), chatInput: $('chat-input'), btnSend: $('btn-send'),
    rateLimitBanner: $('rate-limit-banner'), quotaModal: $('quota-modal'), btnCloseQuotaModal: $('btn-close-quota-modal'),
  };

  // --- XSS-Safe Micro-Markdown ---
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }

  function renderMicroMarkdown(rawText) {
    return escapeHtml(rawText)
      .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
  }

  // --- Mock Mode Dispatcher ---
  let mockState = { sessionId: 'mock-session-demo', datasetCard: null, history: [], jobSpec: null, results: null, turnsLeft: 15 };

  function mockFetch(url, options = {}) {
    const method = (options.method || 'GET').toUpperCase();
    return new Promise((resolve) => {
      setTimeout(() => {
        if (url === '/api/session' && method === 'GET') {
          resolve(new Response(JSON.stringify({ session_id: mockState.sessionId, dataset_card: mockState.datasetCard, history: mockState.history, job_spec: mockState.jobSpec, results: mockState.results }), { status: 200 }));
        } else if (url === '/api/session' && method === 'POST') {
          mockState = { sessionId: 'mock-session-' + Math.random().toString(36).substring(2, 8), datasetCard: null, history: [], jobSpec: null, results: null, turnsLeft: 15 };
          resolve(new Response(JSON.stringify({ session_id: mockState.sessionId, status: 'created' }), { status: 200 }));
        } else if (url === '/api/session' && method === 'DELETE') {
          mockState.datasetCard = null; mockState.history = []; mockState.jobSpec = null; mockState.results = null; mockState.turnsLeft = 15;
          resolve(new Response(JSON.stringify({ status: 'deleted' }), { status: 200 }));
        } else if (url === '/api/upload' && method === 'POST') {
          let fileName = (options.body && options.body.get && options.body.get('file')) ? options.body.get('file').name : '';
          if (fileName.includes('error') || fileName.includes('over200')) {
            resolve(new Response(JSON.stringify({ detail: ['Row 201: Exceeds row limit of 200', 'Cell [row 5, col 3]: String length 72 exceeds 64 chars limit'], errors: ['Row 201: Exceeds row limit of 200', 'Cell [row 5, col 3]: String length 72 exceeds 64 chars limit'] }), { status: 422 }));
          } else if (fileName.includes('housing')) {
            const card = {
              row_count: 120, col_count: 7,
              column_names: ['sold_date', 'median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'median_house_value'],
              column_types: { sold_date: 'datetime', median_income: 'numeric', housing_median_age: 'numeric', total_rooms: 'numeric', total_bedrooms: 'numeric', population: 'numeric', median_house_value: 'numeric' },
              null_counts: { sold_date: 0, median_income: 0, housing_median_age: 0, total_rooms: 0, total_bedrooms: 0, population: 0, median_house_value: 0 },
              unique_counts: { sold_date: 120, median_income: 118, housing_median_age: 44, total_rooms: 119, total_bedrooms: 112, population: 116, median_house_value: 120 },
              preview_rows: [
                { sold_date: '2023-01-17', median_income: 8.325, housing_median_age: 41, total_rooms: 880, total_bedrooms: 129, population: 322, median_house_value: 452.6 },
                { sold_date: '2023-01-20', median_income: 8.301, housing_median_age: 21, total_rooms: 7099, total_bedrooms: 1106, population: 2401, median_house_value: 358.5 },
                { sold_date: '2023-01-24', median_income: 7.257, housing_median_age: 52, total_rooms: 1467, total_bedrooms: 190, population: 496, median_house_value: 352.1 },
                { sold_date: '2023-01-27', median_income: 5.643, housing_median_age: 52, total_rooms: 1274, total_bedrooms: 235, population: 558, median_house_value: 341.3 },
                { sold_date: '2023-01-30', median_income: 3.846, housing_median_age: 52, total_rooms: 1627, total_bedrooms: 280, population: 565, median_house_value: 269.7 },
              ],
            };
            mockState.datasetCard = card;
            resolve(new Response(JSON.stringify(card), { status: 200 }));
          } else {
            const card = {
              row_count: 140, col_count: 8,
              column_names: ['customer_id', 'age', 'tenure_months', 'monthly_charges', 'total_charges', 'contract_type', 'tech_support', 'churned'],
              column_types: { customer_id: 'categorical', age: 'numeric', tenure_months: 'numeric', monthly_charges: 'numeric', total_charges: 'numeric', contract_type: 'categorical', tech_support: 'categorical', churned: 'numeric' },
              null_counts: { customer_id: 0, age: 0, tenure_months: 0, monthly_charges: 0, total_charges: 0, contract_type: 0, tech_support: 0, churned: 0 },
              unique_counts: { customer_id: 140, age: 53, tenure_months: 68, monthly_charges: 135, total_charges: 140, contract_type: 3, tech_support: 2, churned: 2 },
              preview_rows: [
                { customer_id: 'CUST_0001', age: 34, tenure_months: 3, monthly_charges: 65.5, total_charges: 196.5, contract_type: 'Month-to-month', tech_support: 'No', churned: 0 },
                { customer_id: 'CUST_0002', age: 45, tenure_months: 27, monthly_charges: 89.2, total_charges: 2408.4, contract_type: 'One year', tech_support: 'Yes', churned: 0 },
                { customer_id: 'CUST_0003', age: 29, tenure_months: 1, monthly_charges: 102.1, total_charges: 102.1, contract_type: 'Month-to-month', tech_support: 'No', churned: 1 },
                { customer_id: 'CUST_0004', age: 52, tenure_months: 49, monthly_charges: 45.0, total_charges: 2205.0, contract_type: 'Two year', tech_support: 'Yes', churned: 0 },
                { customer_id: 'CUST_0005', age: 38, tenure_months: 8, monthly_charges: 78.4, total_charges: 627.2, contract_type: 'Month-to-month', tech_support: 'No', churned: 1 },
              ],
            };
            mockState.datasetCard = card;
            resolve(new Response(JSON.stringify(card), { status: 200 }));
          }
        } else if (url === '/api/chat' && method === 'POST') {
          let userMsg = '';
          try { userMsg = JSON.parse(options.body).message; } catch (_) {}
          if (userMsg.includes('429')) { resolve(new Response(JSON.stringify({ detail: 'Rate limit' }), { status: 429 })); return; }
          if (userMsg.includes('503')) { resolve(new Response(JSON.stringify({ detail: 'Quota exhausted' }), { status: 503 })); return; }
          mockState.turnsLeft = Math.max(0, mockState.turnsLeft - 1);
          mockState.history.push({ role: 'user', content: userMsg });
          const proposed = {
            task_type: 'classification', target_column: 'churn', feature_columns: ['age', 'tenure', 'balance'],
            excluded_columns: ['customer_id'], split_strategy: 'random', holdout_fraction: 0.2, random_seed: 42,
            eval_metric: 'accuracy', rationale: 'Excluded customer_id to avoid leakage. 80/20 train/holdout split.',
          };
          mockState.jobSpec = proposed;
          const reply = "I've configured an analytical plan to predict 'churn'. Excluded customer_id to avoid leakage.";
          mockState.history.push({ role: 'assistant', content: reply });
          resolve(new Response(JSON.stringify({ reply, plan_card: proposed, turns_left: mockState.turnsLeft }), { status: 200 }));
        } else if (url === '/api/run' && method === 'POST') {
          const results = {
            spec: mockState.jobSpec || { task_type: 'classification', target_column: 'churn', feature_columns: ['age', 'tenure', 'balance'], excluded_columns: ['customer_id'], split_strategy: 'random', holdout_fraction: 0.2 },
            holdout_size: 6, train_size: 24, baseline_metrics: { accuracy: 0.5000, log_loss: 0.6931 },
            model_metrics: { accuracy: 0.8333, log_loss: 0.3842 }, delta: { accuracy: 0.3333, log_loss: -0.3089 },
            sample_predictions: [
              { row_index: 4, actual: 'yes', predicted: 'yes', probabilities: [0.18, 0.82] },
              { row_index: 7, actual: 'no', predicted: 'no', probabilities: [0.89, 0.11] },
              { row_index: 12, actual: 'no', predicted: 'no', probabilities: [0.75, 0.25] },
              { row_index: 18, actual: 'yes', predicted: 'yes', probabilities: [0.22, 0.78] },
              { row_index: 23, actual: 'no', predicted: 'yes', probabilities: [0.45, 0.55] },
            ],
            warnings: [],
          };
          mockState.results = results;
          const narration = "TabPFN achieved **83.33%** holdout accuracy, demonstrating a **+33.33%** lift over baseline (50.00%). Prediction uncertainty is well calibrated.";
          mockState.history.push({ role: 'assistant', content: narration });
          resolve(new Response(JSON.stringify({ results, narration }), { status: 200 }));
        } else {
          resolve(new Response(JSON.stringify({ error: 'Mock 404' }), { status: 404 }));
        }
      }, 300);
    });
  }

  const nativeFetch = window.fetch;
  window.fetch = function (url, options) {
    if (IS_MOCK_MODE && typeof url === 'string' && url.startsWith('/api')) return mockFetch(url, options);
    return nativeFetch.apply(this, arguments);
  };

  // --- Network Wrapper with Timeout & In-Flight Tracking ---
  let inFlightTimer = null;
  async function apiCall(url, options = {}, actionName = 'Working...') {
    if (state.isInFlight) return;
    setInFlight(true, actionName);
    state.lastAction = () => apiCall(url, options, actionName);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);

    inFlightTimer = setTimeout(() => {
      if (state.isInFlight) el.inFlightText.textContent = 'Still working. TabPFN fits may take up to 60s...';
    }, 15000);

    try {
      const response = await window.fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeoutId); clearTimeout(inFlightTimer);
      if (response.status === 429) { showRateLimitBanner(); throw new Error('Rate limit exceeded (429)'); }
      if (response.status === 503) { showQuotaModal(); throw new Error('Daily quota exhausted (503)'); }
      return response;
    } catch (err) {
      clearTimeout(timeoutId); clearTimeout(inFlightTimer);
      if (err.name === 'AbortError') {
        el.inFlightText.textContent = 'Request timed out (90s limit).';
        el.btnRetryTimeout.classList.remove('hidden');
        appendSystemMessage('Operation timed out after 90 seconds. You may retry.', true);
      }
      throw err;
    } finally {
      if (el.btnRetryTimeout.classList.contains('hidden')) setInFlight(false);
      else { state.isInFlight = false; updateConcurrencyUI(false); toggleInputState(false); }
    }
  }

  function setInFlight(inFlight, label = 'Processing...') {
    state.isInFlight = inFlight;
    updateConcurrencyUI(inFlight);
    toggleInputState(inFlight);
    if (inFlight) {
      el.inFlightText.textContent = label;
      el.btnRetryTimeout.classList.add('hidden');
      el.inFlightContainer.classList.remove('hidden');
    } else {
      el.inFlightContainer.classList.add('hidden');
      el.btnRetryTimeout.classList.add('hidden');
    }
  }

  function updateConcurrencyUI(isBusy) {
    el.concurrencyBadge.textContent = isBusy ? 'Processing' : 'Idle';
    el.concurrencyBadge.className = isBusy ? 'status-val processing' : 'status-val idle';
  }

  function toggleInputState(disabled) {
    el.chatInput.disabled = disabled;
    el.btnSend.disabled = disabled;
    const runBtn = $('btn-run-tabpfn');
    if (runBtn) runBtn.disabled = disabled;
  }

  function showRateLimitBanner() {
    el.rateLimitBanner.classList.remove('hidden');
    setTimeout(() => el.rateLimitBanner.classList.add('hidden'), 8000);
  }
  function showQuotaModal() { el.quotaModal.classList.remove('hidden'); }

  // --- UI Renderers ---
  function updateSessionIdUI(id) { state.sessionId = id; el.sessionIdDisplay.textContent = id || 'None'; }
  function updateTurnsUI(turnsLeft) { state.turnsLeft = turnsLeft; el.turnsBadge.textContent = `${turnsLeft}/15 remaining`; }

  function renderDatasetCard(card) {
    state.datasetCard = card;
    if (!card) { el.datasetCardSection.classList.add('hidden'); return; }
    el.datasetShapeBadge.textContent = `${card.row_count} rows × ${card.col_count} cols`;

    el.columnTypesList.innerHTML = (card.column_names || []).map(col => {
      const type = (card.column_types || {})[col] || 'unknown';
      return `<div class="col-chip"><span class="col-name">${escapeHtml(col)}</span><span class="col-type type-${type}">${escapeHtml(type)}</span></div>`;
    }).join('');

    el.previewTableHead.innerHTML = `<tr>${(card.column_names || []).map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr>`;
    el.previewTableBody.innerHTML = (card.preview_rows || []).map(row =>
      `<tr>${(card.column_names || []).map(c => `<td>${escapeHtml(row[c] !== null && row[c] !== undefined ? String(row[c]) : '')}</td>`).join('')}</tr>`
    ).join('');
    el.datasetCardSection.classList.remove('hidden');
  }

  function showUploadErrors(errors) {
    el.uploadErrorsList.innerHTML = '';
    if (!errors || errors.length === 0) { el.uploadErrors.classList.add('hidden'); return; }
    errors.forEach(err => {
      const li = document.createElement('li'); li.textContent = err;
      el.uploadErrorsList.appendChild(li);
    });
    el.uploadErrors.classList.remove('hidden');
  }

  function appendUserMessage(text) {
    const msg = document.createElement('div'); msg.className = 'message message-user';
    const c = document.createElement('div'); c.className = 'message-content'; c.textContent = text;
    msg.appendChild(c); el.chatFeed.appendChild(msg); scrollToBottom();
  }

  function appendAssistantMessage(text) {
    const msg = document.createElement('div'); msg.className = 'message message-assistant';
    const c = document.createElement('div'); c.className = 'message-content'; c.innerHTML = renderMicroMarkdown(text);
    msg.appendChild(c); el.chatFeed.appendChild(msg); scrollToBottom();
  }

  function appendSystemMessage(text, isError = false) {
    const msg = document.createElement('div'); msg.className = isError ? 'message message-error' : 'message message-system';
    const c = document.createElement('div'); c.className = 'message-content'; c.textContent = text;
    msg.appendChild(c); el.chatFeed.appendChild(msg); scrollToBottom();
  }

  function scrollToBottom() { el.chatFeed.scrollTop = el.chatFeed.scrollHeight; }

  // --- Dynamic Plan Card Component ---
  function renderPlanCard(spec) {
    state.jobSpec = spec;
    if (!spec) return;
    const existing = $('active-plan-card'); if (existing) existing.remove();

    const card = document.createElement('div'); card.id = 'active-plan-card'; card.className = 'plan-card';
    const holdoutPct = Math.round((spec.holdout_fraction || 0.2) * 100);
    const incPills = (spec.feature_columns || []).map(f => `<span class="feature-pill">${escapeHtml(f)}</span>`).join('');
    const excPills = (spec.excluded_columns || []).map(f => `<span class="feature-pill excluded">${escapeHtml(f)}</span>`).join('');

    card.innerHTML = `
      <div class="plan-card-header">
        <span class="plan-title">📋 Proposed Analysis Plan</span>
        <span class="badge badge-info">${escapeHtml(spec.task_type.toUpperCase())}</span>
      </div>
      <div class="plan-details-grid">
        <div class="plan-item"><span class="plan-item-label">Target Column</span><span class="plan-item-val">${escapeHtml(spec.target_column)}</span></div>
        <div class="plan-item"><span class="plan-item-label">Validation Split</span><span class="plan-item-val">${escapeHtml(spec.split_strategy)} (${100 - holdoutPct}/${holdoutPct})</span></div>
        <div class="plan-item"><span class="plan-item-label">Features Included (${(spec.feature_columns || []).length})</span><div class="pill-list">${incPills || '<em>None</em>'}</div></div>
        <div class="plan-item"><span class="plan-item-label">Features Excluded (${(spec.excluded_columns || []).length})</span><div class="pill-list">${excPills || '<em>None</em>'}</div></div>
      </div>
      <div class="plan-card-actions">
        <button id="btn-run-tabpfn" class="btn btn-success">🚀 Confirm & Run TabPFN</button>
        <span class="plan-edit-hint">Want to change features or target? Reply in chat to adjust the plan.</span>
      </div>
    `;
    card.querySelector('#btn-run-tabpfn').addEventListener('click', handleRunAnalysis);
    el.chatFeed.appendChild(card);
    scrollToBottom();
  }

  // --- Dynamic Results Card Component with Inline SVG ---
  function renderResultsCard(results, narrationText) {
    state.results = results;
    if (!results) return;
    const existing = $('active-results-card'); if (existing) existing.remove();

    const card = document.createElement('div'); card.id = 'active-results-card'; card.className = 'results-card';
    const mRows = Object.entries(results.model_metrics || {}).map(([k, mVal]) => {
      const bVal = (results.baseline_metrics || {})[k];
      const dVal = (results.delta || {})[k];
      let liftCls = '', dStr = '-';
      if (dVal !== undefined) {
        dStr = (dVal > 0 ? '+' : '') + dVal;
        const higherIsBetter = k === 'accuracy' || k === 'r2';
        liftCls = (higherIsBetter ? dVal > 0 : dVal < 0) ? 'lift-positive' : 'lift-negative';
      }
      return `<tr><td><strong>${escapeHtml(k.toUpperCase())}</strong></td><td class="metric-val">${escapeHtml(String(mVal))}</td><td class="metric-val">${bVal !== undefined ? escapeHtml(String(bVal)) : '-'}</td><td class="metric-val ${liftCls}">${escapeHtml(dStr)}</td></tr>`;
    }).join('');

    const pRows = (results.sample_predictions || []).slice(0, 5).map(item => {
      const det = item.probabilities ? item.probabilities.map(p => `${Math.round(p * 100)}%`).join(' / ') : (item.quantiles ? `q10: ${item.quantiles.q10}, q90: ${item.quantiles.q90}` : '-');
      return `<tr><td>${escapeHtml(String(item.row_index))}</td><td>${escapeHtml(String(item.actual))}</td><td><strong>${escapeHtml(String(item.predicted))}</strong></td><td>${escapeHtml(det)}</td></tr>`;
    }).join('');

    card.innerHTML = `
      <div class="results-header">
        <span class="results-title">📊 TabPFN-3.5 Evaluation Results</span>
        <span class="badge badge-info">Holdout N=${escapeHtml(String(results.holdout_size || 0))}</span>
      </div>
      <div>
        <table class="metrics-table"><thead><tr><th>Metric</th><th>TabPFN-3.5</th><th>Baseline</th><th>Lift / Delta</th></tr></thead><tbody>${mRows}</tbody></table>
      </div>
      <div class="chart-container">
        <span class="chart-title">Holdout Prediction Confidences</span>
        ${buildSvgChartHtml(results)}
      </div>
      <div class="table-preview-wrapper">
        <span class="section-subtitle">Holdout Predictions (Top 5):</span>
        <div class="table-scroll-container">
          <table class="data-table"><thead><tr><th>Row</th><th>Actual</th><th>Predicted</th><th>Details</th></tr></thead><tbody>${pRows}</tbody></table>
        </div>
      </div>
      ${narrationText ? `<div class="narration-block">${renderMicroMarkdown(narrationText)}</div>` : ''}
    `;
    el.chatFeed.appendChild(card);
    scrollToBottom();
  }

  function buildSvgChartHtml(results) {
    const preds = (results.sample_predictions || []).slice(0, 5);
    if (preds.length === 0) return '<svg viewBox="0 0 500 40" class="chart-svg"><text x="250" y="24" fill="#94a3b8" text-anchor="middle" font-size="12">No predictions</text></svg>';
    const bars = preds.map((item, idx) => {
      const y = 14 + idx * 22;
      const match = String(item.actual) === String(item.predicted);
      const color = match ? '#10b981' : '#f59e0b';
      const conf = (item.probabilities && item.probabilities.length > 0) ? Math.max(...item.probabilities) : 0.8;
      const w = Math.max(10, Math.round(conf * 260));
      return `
        <text x="10" y="${y + 11}" fill="#94a3b8" font-size="11" font-family="monospace">#${escapeHtml(String(item.row_index))}</text>
        <rect x="50" y="${y}" width="${w}" height="14" rx="3" fill="${color}" />
        <text x="${55 + w}" y="${y + 11}" fill="#e2e8f0" font-size="11" font-weight="bold">${Math.round(conf * 100)}% (${escapeHtml(String(item.predicted))})</text>
      `;
    }).join('');
    return `<svg viewBox="0 0 500 ${preds.length * 24 + 14}" class="chart-svg">${bars}</svg>`;
  }

  // --- Actions ---
  async function loadSampleDataset(sampleName) {
    if (state.isInFlight) return;
    try {
      const res = await window.fetch(`/static/samples/${sampleName}`);
      if (!res.ok) {
        appendSystemMessage(`Could not fetch sample dataset ${sampleName} (HTTP ${res.status}).`, true);
        return;
      }
      const text = await res.text();
      const blob = new Blob([text], { type: 'text/csv' });
      const file = new File([blob], sampleName, { type: 'text/csv' });
      await handleFileUpload(file);
    } catch (err) {
      appendSystemMessage(`Failed to load sample dataset: ${err.message}`, true);
    }
  }

  async function handleFileUpload(file) {
    if (!file) return;
    el.uploadErrors.classList.add('hidden');
    const formData = new FormData(); formData.append('file', file);
    try {
      const res = await apiCall('/api/upload', { method: 'POST', body: formData }, 'Uploading & parsing dataset...');
      if (res.status === 422) {
        const errorData = await res.json();
        showUploadErrors(errorData.errors || errorData.detail || ['Validation failed.']);
        return;
      }
      if (!res.ok) { appendSystemMessage(`Upload failed (HTTP ${res.status}).`, true); return; }
      const card = await res.json();
      renderDatasetCard(card);
      appendSystemMessage(`Dataset loaded: ${card.row_count} rows, ${card.col_count} columns.`);
    } catch (err) { console.error(err); }
  }

  async function handleChatSubmit(e) {
    e.preventDefault();
    const message = el.chatInput.value.trim();
    if (!message || state.isInFlight) return;
    el.chatInput.value = ''; appendUserMessage(message);
    try {
      const res = await apiCall('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message }) }, 'Gemini is planning...');
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        appendSystemMessage(errJson.detail || `Chat error (HTTP ${res.status})`, true);
        return;
      }
      const data = await res.json();
      if (data.turns_left !== undefined) updateTurnsUI(data.turns_left);
      if (data.reply) appendAssistantMessage(data.reply);
      if (data.plan_card) renderPlanCard(data.plan_card);
    } catch (err) { console.error(err); }
  }

  async function handleRunAnalysis() {
    if (state.isInFlight) return;
    try {
      const res = await apiCall('/api/run', { method: 'POST' }, 'TabPFN is training...');
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        appendSystemMessage(errJson.detail || `Execution failed (HTTP ${res.status})`, true);
        return;
      }
      const data = await res.json();
      renderResultsCard(data.results, data.narration);
    } catch (err) { console.error(err); }
  }

  async function handleResetSession() {
    if (!confirm('Are you sure you want to reset the active session?')) return;
    try {
      await apiCall('/api/session', { method: 'DELETE' }, 'Resetting session...');
      const createRes = await apiCall('/api/session', { method: 'POST' });
      const createData = await createRes.json();
      updateSessionIdUI(createData.session_id);
      state.datasetCard = null; state.jobSpec = null; state.results = null;
      updateTurnsUI(15); renderDatasetCard(null); el.uploadErrors.classList.add('hidden');
      el.chatFeed.innerHTML = '<div class="message message-system"><div class="message-content">Session reset. Upload a CSV to start.</div></div>';
    } catch (err) { console.error(err); }
  }

  async function restoreSession() {
    if (IS_MOCK_MODE) el.mockBadge.classList.remove('hidden');
    try {
      const res = await window.fetch('/api/session');
      if (res.status === 404) {
        const initRes = await window.fetch('/api/session', { method: 'POST' });
        const initData = await initRes.json();
        updateSessionIdUI(initData.session_id);
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      updateSessionIdUI(data.session_id);
      if (data.dataset_card) renderDatasetCard(data.dataset_card);
      if (data.history && data.history.length > 0) {
        el.chatFeed.innerHTML = '';
        data.history.forEach(m => (m.role === 'user' ? appendUserMessage(m.content) : appendAssistantMessage(m.content)));
      }
      if (data.job_spec) renderPlanCard(data.job_spec);
      if (data.results) renderResultsCard(data.results, null);
      if (data.history) {
        const userTurns = data.history.filter(m => m.role === 'user').length;
        updateTurnsUI(Math.max(0, 15 - userTurns));
      }
    } catch (err) { console.error('Restore failed:', err); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    el.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); el.dropZone.classList.add('dragover'); });
    el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('dragover'));
    el.dropZone.addEventListener('drop', (e) => {
      e.preventDefault(); el.dropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
    });
    el.fileInput.addEventListener('change', () => {
      if (el.fileInput.files && el.fileInput.files[0]) handleFileUpload(el.fileInput.files[0]);
    });
    if (el.btnSampleChurn) el.btnSampleChurn.addEventListener('click', () => loadSampleDataset('churn_sample.csv'));
    if (el.btnSampleHousing) el.btnSampleHousing.addEventListener('click', () => loadSampleDataset('housing_sample.csv'));
    el.chatForm.addEventListener('submit', handleChatSubmit);
    el.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); el.chatForm.dispatchEvent(new Event('submit', { cancelable: true })); }
    });
    el.btnResetSession.addEventListener('click', handleResetSession);
    el.btnCloseQuotaModal.addEventListener('click', () => el.quotaModal.classList.add('hidden'));
    el.btnRetryTimeout.addEventListener('click', () => { if (state.lastAction) state.lastAction(); });
    restoreSession();
  });
})();
