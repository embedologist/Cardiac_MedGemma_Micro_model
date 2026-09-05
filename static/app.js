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
    // Invert normalized 0..1 to canvas y coordinates
    const y = h - paddingY - pts[i] * usableH;
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
    const headY = h - paddingY - pts[limit - 1] * usableH;

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
  metricHr.textContent = metrics.estimated_bpm.toFixed(1);
  metricRmssd.textContent = metrics.rmssd_ms.toFixed(1);
  metricSdnn.textContent = metrics.sdnn_ms.toFixed(1);

  badgeRhythmName.textContent = condName;

  // Update badge styling
  currentRhythmBadge.className = 'rhythm-status-badge';
  const theme = CONDITION_COLORS[condIdx];
  if (theme && theme.badgeClass) {
    currentRhythmBadge.classList.add(theme.badgeClass);
  }

  // Update HR sub label
  const hrSub = document.getElementById('metric-hr-sub');
  if (hrSub) {
    if (metrics.estimated_bpm < 50) hrSub.textContent = 'Severe Bradycardia';
    else if (metrics.estimated_bpm > 100) hrSub.textContent = 'Tachycardic State';
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
  STATE.chatHistory.push({ role: 'user', content: text });

  // Append Thinking placeholder
  const thinkingEl = appendThinkingMessage();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        history: STATE.chatHistory.slice(-4),
        use_ppg_context: STATE.useMultimodal,
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

function selectCondition(condIdx) {
  condIdx = parseInt(condIdx);
  STATE.condition = condIdx;

  // Update chip active states
  const chips = conditionChips.querySelectorAll('.chip');
  chips.forEach(c => {
    c.classList.toggle('active', parseInt(c.dataset.condition) === condIdx);
  });

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
    ? 'Prefix K=4 (960-dim) Active'
    : 'Multimodal Bridge Off';
});

document.getElementById('btn-run-classifier').addEventListener('click', () => {
  runClassification();
});

chatForm.addEventListener('submit', handleChatSubmit);
userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleChatSubmit();
  }
});

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
