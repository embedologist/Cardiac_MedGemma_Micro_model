/**
 * MedGemma-Micro Interactive Test & Chat Interface Engine
 * =======================================================
 * Handles:
 *  - Real-time animated canvas oscilloscope for 90s PPG signals
 *  - REST interaction with FastAPI model backend
 *  - Arrhythmia classification & telemetry updates
 *  - Multimodal chat with soft-prompt prefix conditioning
 */

const STATE = {
  condition: 0,
  conditionNames: {
    0: 'Normal Sinus Rhythm',
    1: 'Atrial Fibrillation (AFib)',
    2: 'Sinus Bradycardia',
    3: 'Sinus Tachycardia',
    4: 'Premature Ventricular Contractions (PVC)'
  },
  waveform: [],
  metrics: { estimated_bpm: 72, rmssd_ms: 38.4, sdnn_ms: 41.2 },
  isSweeping: true,
  sweepIndex: 0,
  sweepSpeed: 3, // points per frame
  isClassifying: false,
  isGenerating: false,
  useMultimodal: true,
  chatHistory: []
};

// DOM Elements
const canvas = document.getElementById('ppg-canvas');
const ctx = canvas.getContext('2d');
const conditionChips = document.getElementById('condition-chips');
const probBarsContainer = document.getElementById('prob-bars-container');
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const btnSend = document.getElementById('btn-send');
const btnToggleSweep = document.getElementById('btn-toggle-sweep');
const btnRegenPpg = document.getElementById('btn-regen-ppg');
const toggleNoise = document.getElementById('toggle-noise');
const toggleMultimodal = document.getElementById('toggle-multimodal');
const bridgeIndicator = document.getElementById('bridge-indicator');
const presetsContainer = document.getElementById('presets-container');

const metricHr = document.getElementById('metric-hr');
const metricRmssd = document.getElementById('metric-rmssd');
const metricSdnn = document.getElementById('metric-sdnn');
const metricLatency = document.getElementById('metric-latency');
const badgeRhythmName = document.getElementById('badge-rhythm-name');
const currentRhythmBadge = document.getElementById('current-rhythm-badge');
const statusPulseDot = document.getElementById('status-pulse-dot');
const chatTps = document.getElementById('chat-tps');

// Initialize Canvas Size
function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}
window.addEventListener('resize', resizeCanvas);

// Color Themes per condition
const CONDITION_COLORS = {
  0: { stroke: '#00f0ff', glow: 'rgba(0, 240, 255, 0.4)', badgeClass: '' },
  1: { stroke: '#ff4757', glow: 'rgba(255, 71, 87, 0.4)', badgeClass: 'badge-afib' },
  2: { stroke: '#38bdf8', glow: 'rgba(56, 189, 248, 0.4)', badgeClass: '' },
  3: { stroke: '#ffa502', glow: 'rgba(255, 165, 2, 0.4)', badgeClass: 'badge-tachy' },
  4: { stroke: '#a855f7', glow: 'rgba(168, 85, 247, 0.4)', badgeClass: 'badge-afib' },
};

// =====================================================================
// Oscilloscope Renderer
// =====================================================================

let lastFrameTime = performance.now();
let frameCount = 0;
let fpsTimer = 0;

function drawOscilloscope(timestamp) {
  requestAnimationFrame(drawOscilloscope);

  // FPS calculation
  frameCount++;
  if (timestamp - fpsTimer >= 1000) {
    const fpsEl = document.getElementById('canvas-fps');
    if (fpsEl) fpsEl.textContent = `${frameCount} FPS`;
    frameCount = 0;
    fpsTimer = timestamp;
  }

  const w = canvas.width;
  const h = canvas.height;
  if (w === 0 || h === 0) return;

  const pts = STATE.waveform;
  if (!pts || pts.length === 0) return;

  // Background clear with slight decay trail
  ctx.fillStyle = 'rgba(4, 7, 13, 0.25)';
  ctx.fillRect(0, 0, w, h);

  // Baseline mid-line
  ctx.strokeStyle = 'rgba(0, 240, 255, 0.1)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  const theme = CONDITION_COLORS[STATE.condition] || CONDITION_COLORS[0];

  // Draw Waveform line
  ctx.save();
  ctx.shadowColor = theme.glow;
  ctx.shadowBlur = 10;
  ctx.strokeStyle = theme.stroke;
  ctx.lineWidth = 2.2;
  ctx.lineJoin = 'round';
  ctx.beginPath();

  const numPoints = pts.length;
  const stepX = w / (numPoints - 1);
  const paddingY = 24;
  const usableH = h - paddingY * 2;

  // If sweeping, draw up to sweepIndex, plus sweep head beam
  const limit = STATE.isSweeping ? Math.min(numPoints, STATE.sweepIndex) : numPoints;

  for (let i = 0; i < limit; i++) {
    const x = i * stepX;
    // Z-score normalized signal (mean ~0.0, std ~1.0): center at h / 2
    // Map ±3 standard deviations to usable canvas height
    const y = h / 2 - pts[i] * (usableH / 6.0);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  // Draw Sweep Head Cursor
  if (STATE.isSweeping && limit > 0 && limit < numPoints) {
    const headX = (limit - 1) * stepX;
    const headY = h / 2 - pts[limit - 1] * (usableH / 6.0);

    // Glowing head dot
    ctx.shadowBlur = 16;
    ctx.shadowColor = '#ffffff';
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(headX, headY, 4, 0, Math.PI * 2);
    ctx.fill();

    // Vertical sweep guide line
    ctx.shadowBlur = 4;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(headX, 0);
    ctx.lineTo(headX, h);
    ctx.stroke();

    // Advance sweep index
    STATE.sweepIndex = (STATE.sweepIndex + STATE.sweepSpeed);
    if (STATE.sweepIndex >= numPoints) {
      STATE.sweepIndex = 0;
      // Instant clear on loop
      ctx.fillStyle = '#04070d';
      ctx.fillRect(0, 0, w, h);
    }
  }

  ctx.restore();
}

// =====================================================================
// API Integrations
// =====================================================================

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'ready') {
      const hudSize = document.getElementById('hud-size');
      if (hudSize) hudSize.textContent = `${data.size_mb} MB`;
    }
  } catch (err) {
    console.warn('Status check pending:', err);
  }
}

async function generateWaveform(condition, noise = 0.04) {
  try {
    const res = await fetch('/api/ppg/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ condition, noise_level: noise })
    });
    const data = await res.json();
    STATE.condition = data.condition_idx;
    STATE.waveform = data.waveform_preview;
    STATE.metrics = data.metrics;
    STATE.sweepIndex = 0;

    // Update Telemetry Displays
    updateTelemetry(data.metrics, data.condition_idx, data.condition_name);

    // Automatically trigger classification on new signal
    await runClassification();
  } catch (err) {
    console.error('Failed to generate PPG:', err);
  }
}

async function runClassification() {
  if (STATE.isClassifying) return;
  STATE.isClassifying = true;
  const btn = document.getElementById('btn-run-classifier');
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/ppg/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ condition: STATE.condition })
    });
    const data = await res.json();

    // Update Latency
    metricLatency.textContent = data.inference_time_ms;

    // Render Probability Bars
    renderProbabilityBars(data.probabilities, data.predicted_idx);
  } catch (err) {
    console.error('Classification failed:', err);
  } finally {
    STATE.isClassifying = false;
    if (btn) btn.disabled = false;
  }
}

function updateTelemetry(metrics, condIdx, condName) {
  if (!metrics) return;
  const bpm = metrics.estimated_bpm ?? 72;
  const rmssd = metrics.rmssd_ms ?? 38;
  const sdnn = metrics.sdnn_ms ?? 42;

  metricHr.textContent = bpm.toFixed(1);
  metricRmssd.textContent = rmssd.toFixed(1);
  metricSdnn.textContent = sdnn.toFixed(1);

  if (condName) badgeRhythmName.textContent = condName;

  // Update badge styling
  currentRhythmBadge.className = 'rhythm-status-badge';
  const theme = CONDITION_COLORS[condIdx] || CONDITION_COLORS[0];
  if (theme && theme.badgeClass) {
    currentRhythmBadge.classList.add(theme.badgeClass);
  }

  // Update HR sub label
  const hrSub = document.getElementById('metric-hr-sub');
  if (hrSub) {
    if (bpm < 50) hrSub.textContent = 'Severe Bradycardia';
    else if (bpm > 100) hrSub.textContent = 'Tachycardic State';
    else hrSub.textContent = 'Resting Normal Rhythm';
  }
}

function renderProbabilityBars(probs, predictedIdx) {
  probBarsContainer.innerHTML = '';
  const entries = Object.entries(probs);

  entries.forEach(([name, prob], idx) => {
    const isMax = idx === predictedIdx;
    const pct = (prob * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = `prob-row ${isMax ? 'highlight' : ''}`;
    if (isMax && (idx === 1 || idx === 3 || idx === 4)) {
      row.classList.add('danger');
    }

    row.innerHTML = `
      <div class="prob-meta">
        <span class="prob-name">${name}</span>
        <span class="prob-pct">${pct}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${pct}%"></div>
      </div>
    `;
    probBarsContainer.appendChild(row);
  });
}

// =====================================================================
// Presets Loader
// =====================================================================

async function loadPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    presetsContainer.innerHTML = '';

    data.presets.forEach(preset => {
      const chip = document.createElement('button');
      chip.className = 'preset-chip';
      chip.textContent = `${preset.title}`;
      chip.title = preset.prompt;
      chip.addEventListener('click', () => {
        // Set condition if different
        if (STATE.condition !== preset.condition) {
          selectCondition(preset.condition);
        }
        userInput.value = preset.prompt;
        userInput.focus();
      });
      presetsContainer.appendChild(chip);
    });
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

// =====================================================================
// Chat Conversation Logic
// =====================================================================

function appendMessage(role, content, meta = null) {
  const msgEl = document.createElement('div');
  msgEl.className = `message-bubble ${role === 'user' ? 'user-msg' : 'assistant-msg'}`;

  const isUser = role === 'user';
  const avatar = isUser ? '👤' : '🩺';
  const authorName = isUser ? 'Physician / User' : 'MedGemma-Micro';
  const tagText = isUser ? 'Query' : (meta ? `${meta.tps} tok/s · ${meta.tokens} tokens` : 'Edge Inference');

  // Simple markdown formatting
  let formatted = escapeHtml(content)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');

  msgEl.innerHTML = `
    <div class="msg-avatar">
      <span>${avatar}</span>
    </div>
    <div class="msg-body">
      <div class="msg-author">
        <span class="name">${authorName}</span>
        <span class="tag">${tagText}</span>
      </div>
      <div class="msg-content">
        <p>${formatted}</p>
      </div>
    </div>
  `;

  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return msgEl;
}

function appendThinkingMessage() {
  const msgEl = document.createElement('div');
  msgEl.className = 'message-bubble assistant-msg thinking-bubble';
  msgEl.innerHTML = `
    <div class="msg-avatar"><span>🩺</span></div>
    <div class="msg-body">
      <div class="msg-author">
        <span class="name">MedGemma-Micro</span>
        <span class="tag">Computing Multimodal Soft Prefix...</span>
      </div>
      <div class="msg-content">
        <div class="loading-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>
  `;
  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return msgEl;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

async function handleChatSubmit(e) {
  if (e) e.preventDefault();
  const text = userInput.value.trim();
  if (!text || STATE.isGenerating) return;

  userInput.value = '';
  STATE.isGenerating = true;
  btnSend.disabled = true;

  // Append User message
  appendMessage('user', text);
  const priorHistory = STATE.chatHistory.slice(-4);
  STATE.chatHistory.push({ role: 'user', content: text });

  // Append Thinking placeholder
  const thinkingEl = appendThinkingMessage();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        history: priorHistory,
        use_ppg_context: STATE.useMultimodal,
        condition: STATE.condition,
        metrics: STATE.metrics,
        temperature: 0.65,
        max_tokens: 180
      })
    });

    const data = await res.json();
    thinkingEl.remove();

    if (data.reply) {
      appendMessage('assistant', data.reply, {
        tps: data.tokens_per_sec,
        tokens: data.tokens_generated
      });
      STATE.chatHistory.push({ role: 'assistant', content: data.reply });
      // Prevent unbounded memory accumulation during prolonged testing
      if (STATE.chatHistory.length > 50) {
        STATE.chatHistory = STATE.chatHistory.slice(-50);
      }

      chatTps.textContent = `${data.tokens_per_sec} tok/s (${data.elapsed_sec}s)`;
    } else {
      appendMessage('assistant', 'Error: Failed to generate response from model.');
    }
  } catch (err) {
    console.error('Chat error:', err);
    thinkingEl.remove();
    appendMessage('assistant', `Inference request failed: ${err.message}`);
  } finally {
    STATE.isGenerating = false;
    btnSend.disabled = false;
    userInput.focus();
  }
}

// =====================================================================
// Event Listeners
// =====================================================================

function appendConditionChangeNotification(name) {
  const notifEl = document.createElement('div');
  notifEl.className = 'session-divider';
  notifEl.style.cssText = 'text-align: center; margin: 10px 0; padding: 4px 12px; background: rgba(0, 240, 255, 0.08); border-radius: 20px; color: #00f0ff; font-size: 11px; font-weight: 600;';
  notifEl.innerHTML = `<span>⚡ Telemetry switched to: <strong>${escapeHtml(name)}</strong></span>`;
  chatMessages.appendChild(notifEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function selectCondition(condIdx) {
  condIdx = parseInt(condIdx);
  const isChanged = STATE.condition !== condIdx;
  STATE.condition = condIdx;

  // Update chip active states
  const chips = conditionChips.querySelectorAll('.chip');
  chips.forEach(c => {
    c.classList.toggle('active', parseInt(c.dataset.condition) === condIdx);
  });

  // If switching condition, clear old conversational history to prevent rhythm cross-contamination
  if (isChanged) {
    STATE.chatHistory = [];
    appendConditionChangeNotification(STATE.conditionNames[condIdx]);
  }

  const noise = toggleNoise.checked ? 0.04 : 0.0;
  generateWaveform(condIdx, noise);
}

conditionChips.addEventListener('click', e => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  selectCondition(chip.dataset.condition);
});

btnToggleSweep.addEventListener('click', () => {
  STATE.isSweeping = !STATE.isSweeping;
  const sweepIcon = document.getElementById('sweep-icon');
  const sweepText = document.getElementById('sweep-text');
  if (STATE.isSweeping) {
    sweepIcon.textContent = '⏸';
    sweepText.textContent = 'Pause Monitor';
  } else {
    sweepIcon.textContent = '▶';
    sweepText.textContent = 'Resume Sweep';
  }
});

btnRegenPpg.addEventListener('click', () => {
  const noise = toggleNoise.checked ? 0.04 : 0.0;
  generateWaveform(STATE.condition, noise);
});

toggleNoise.addEventListener('change', () => {
  const noise = toggleNoise.checked ? 0.04 : 0.0;
  generateWaveform(STATE.condition, noise);
});

toggleMultimodal.addEventListener('change', () => {
  STATE.useMultimodal = toggleMultimodal.checked;
  bridgeIndicator.classList.toggle('active', STATE.useMultimodal);
  bridgeIndicator.querySelector('span:last-child').textContent = STATE.useMultimodal
    ? 'Prefix K=4 (896-dim) Active'
    : 'Multimodal Bridge Off';
});

document.getElementById('btn-run-classifier').addEventListener('click', () => {
  runClassification();
});

const btnClearChat = document.getElementById('btn-clear-chat');
if (btnClearChat) {
  btnClearChat.addEventListener('click', () => {
    STATE.chatHistory = [];
    chatMessages.innerHTML = '';
    const condName = STATE.conditionNames[STATE.condition] || 'Normal Sinus Rhythm';
    const bpm = STATE.metrics.estimated_bpm || 72;
    appendMessage('assistant', `Conversation history cleared. Actively monitoring **${condName}** (${bpm} BPM). How can I assist with your telemetry or cardiology questions?`);
  });
}

chatForm.addEventListener('submit', handleChatSubmit);
userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleChatSubmit();
  }
});

// =====================================================================
// Wear OS (Samsung Galaxy Watch 4) Streaming Logic
// =====================================================================

async function streamWearOSScenario(cond, fs) {
  const sqiBadge = document.getElementById('wearos-sqi-badge');
  const sqiText = document.getElementById('wearos-sqi-text');
  const bufferFill = document.getElementById('wearos-progress-fill');
  const bufferPct = document.getElementById('wearos-buffer-pct');
  const bufferCount = document.getElementById('wearos-buffer-count');

  sqiText.textContent = `Streaming ${fs}Hz...`;
  sqiBadge.className = 'wearos-sqi-badge';

  // Highlight active button
  document.querySelectorAll('.wearos-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.cond) === cond && parseInt(btn.dataset.fs) === fs);
  });

  try {
    const res = await fetch('/api/wearos/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        condition: cond,
        sampling_rate: fs,
        duration_sec: 90.0,
      })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Simulation failed');

    // Update buffer HUD
    bufferFill.style.width = `${data.buffer_fill_pct}%`;
    bufferPct.textContent = `${data.buffer_fill_pct}% Full`;
    bufferCount.textContent = `${data.total_points_ingested.toLocaleString()} / 2,250 samples`;

    const q = data.quality || {};
    sqiText.textContent = `SQI: ${q.sqi || 0.0} (${q.quality_flag || 'OK'})`;
    if (!q.is_usable) {
      sqiBadge.classList.add('warning');
    }

    // If signal is usable, run classification on Wear OS buffer
    if (q.is_usable && cond <= 4) {
      const clsRes = await fetch('/api/wearos/classify', { method: 'POST' });
      const clsData = await clsRes.json();
      if (clsData.success) {
        STATE.condition = clsData.predicted_idx;
        renderProbabilityBars(clsData.probabilities, clsData.predicted_idx);
        metricLatency.textContent = clsData.inference_time_ms;
      }
    } else if (cond === 5) {
      // Off-wrist lead-off rejection demonstration
      probBarsContainer.innerHTML = `
        <div style="padding: 12px; background: rgba(255, 71, 87, 0.12); border: 1px solid rgba(255, 71, 87, 0.3); border-radius: 8px; color: #ff4757; font-size: 0.78rem;">
          <strong>🚫 Lead-Off Detected (GREEN_STATUS = -1)</strong><br>
          Galaxy Watch 4 sensor is detached from wrist. MedGemma-Micro safety guard rejected inference to prevent erroneous diagnosis.
        </div>
      `;
    }

    // Refresh full waveform for canvas
    if (data.waveform_preview && data.waveform_preview.length > 0) {
      // Repeat preview across 90s window for smooth sweep
      const full = [];
      while (full.length < 2250) {
        full.push(...data.waveform_preview);
      }
      STATE.waveform = full.slice(0, 2250);
      STATE.sweepIndex = 0;
    }

    if (data.metrics) {
      STATE.metrics = data.metrics;
      updateTelemetry(data.metrics, cond, data.condition_name);
    }
  } catch (err) {
    console.error('Wear OS streaming error:', err);
    sqiText.textContent = 'Stream Error';
    sqiBadge.classList.add('warning');
  }
}

// Bind Wear OS scenario buttons
document.getElementById('btn-stream-w4-normal').addEventListener('click', () => streamWearOSScenario(0, 25));
document.getElementById('btn-stream-w4-afib').addEventListener('click', () => streamWearOSScenario(1, 25));
document.getElementById('btn-stream-w4-100hz').addEventListener('click', () => streamWearOSScenario(3, 100));
document.getElementById('btn-stream-w4-detached').addEventListener('click', () => streamWearOSScenario(5, 25));

// =====================================================================
// App Initialization
// =====================================================================

async function init() {
  resizeCanvas();
  await fetchStatus();
  await loadPresets();
  // Initial normal sinus waveform
  await generateWaveform(0, 0.04);
  // Start render loop
  requestAnimationFrame(drawOscilloscope);
}

document.addEventListener('DOMContentLoaded', init);
