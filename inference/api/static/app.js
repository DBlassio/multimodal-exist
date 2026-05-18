// app.js — Shared utilities for Multimodal Demo

// ── Fetch helper ──────────────────────────────────────────────────────────
async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('fetchJSON error:', url, err);
    return null;
  }
}

// ── Badge helpers ─────────────────────────────────────────────────────────
function pred21Badge(pred) {
  const cls = pred === 'SEXIST' ? 'sexist' : 'not';
  return `<span class="badge badge-${cls}">${pred}</span>`;
}

function pred22Badge(pred) {
  if (pred === 'DIRECT')      return `<span class="badge badge-direct">DIRECT</span>`;
  if (pred === 'JUDGEMENTAL') return `<span class="badge badge-judg">JUDGEMENTAL</span>`;
  return `<span class="badge badge-not">NO</span>`;
}

function agreeBadge(agree) {
  return agree
    ? `<span class="badge badge-agree">✓ Agree</span>`
    : `<span class="badge badge-disagree">✗ Disagree</span>`;
}

// ── Image URL ─────────────────────────────────────────────────────────────
function imageUrl(filename) {
  return `/images/${encodeURIComponent(filename)}`;
}

// ── Gate bar HTML ─────────────────────────────────────────────────────────
function gateBar(label, value, cls) {
  const v   = (value != null && !isNaN(value)) ? value : 0;
  const pct = Math.round(v * 100);
  return `
    <div class="gate-bar-row">
      <div class="gate-bar-label">${label}</div>
      <div class="gate-bar-track">
        <div class="gate-bar-fill ${cls}" style="width:${pct}%"></div>
      </div>
      <div class="gate-bar-value">${v.toFixed(3)}</div>
    </div>`;
}

// ── Meme detail modal ─────────────────────────────────────────────────────
let currentModal = null;

function openModal(memeId) {
  const overlay = document.getElementById('modal-overlay');
  if (!overlay) return;
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';
  loadModalContent(memeId);
}

function closeModal() {
  const overlay = document.getElementById('modal-overlay');
  if (!overlay) return;
  overlay.classList.remove('open');
  document.body.style.overflow = '';
}

async function loadModalContent(memeId) {
  const body = document.getElementById('modal-body');
  body.innerHTML = '<div class="loading"><div class="spinner"></div> Loading…</div>';

  const data = await fetchJSON(`/api/memes/${memeId}`);
  if (!data) {
    body.innerHTML = '<div class="error-msg">Failed to load meme details.</div>';
    return;
  }

  const preds = data.predictions;
  const models = Object.keys(preds);

  // Model rows
  const modelRowsHtml = models.map(m => {
    const p = preds[m];
    return `
      <div class="model-row">
        <div class="model-row-name">${p.label}</div>
        ${pred21Badge(p.pred21)}
        <div class="prob-bar">
          <div class="prob-bar-fill" style="width:${(p.p21*100).toFixed(1)}%"></div>
        </div>
        <div class="prob-value">${(p.p21*100).toFixed(1)}%</div>
        ${pred22Badge(p.pred22)}
      </div>`;
  }).join('');

  // Categories from first available model
  const firstPred = preds[models[0]];
  const catHtml = firstPred ? Object.entries(firstPred.cat_probs || {}).map(([label, prob]) => {
    const active = prob >= 0.3;
    return `<span class="tag ${active ? '' : 'inactive'}" title="${(prob*100).toFixed(1)}%">${label.split('-').slice(0,2).join('-')}</span>`;
  }).join('') : '';

  // Gates
const gates = data.gates;
const gatesHtml = gates ? `
  ${gateBar('β (Image)', gates.beta   ?? 0, 'beta')}
  ${gateBar('α (EEG)',   gates.alpha  ?? 0, 'alpha')}
  ${gateBar('λ (Eye-Tracking)', gates.lambda ?? 0, 'lambda')}
` : '';

  body.innerHTML = `
    <div style="display:grid;grid-template-columns:200px 1fr;gap:1.5rem;align-items:start;">
      <img src="${imageUrl(data.image_file)}"
           style="width:100%;border-radius:8px;border:1px solid var(--border);"
           onerror="this.src='';this.style.background='var(--navy-mid)';this.style.height='150px';">
      <div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.75rem;">
          <span style="font-family:var(--font-mono);font-size:11px;color:var(--white-dim);">${data.lang.toUpperCase()}</span>
          ${agreeBadge(data.models_agree_21)}
        </div>
        <p style="font-size:13px;color:var(--white-dim);line-height:1.6;margin-bottom:1rem;">${data.text}</p>
        <div class="tag-list">${catHtml}</div>
      </div>
    </div>
    <hr class="divider">
    <div class="section-title">Task 2.1 — Sexism detection</div>
    ${modelRowsHtml}
    ${gatesHtml}
  `;
}

// Close modal on overlay click
document.addEventListener('DOMContentLoaded', () => {
  const overlay = document.getElementById('modal-overlay');
  if (overlay) {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeModal();
    });
  }
  // ESC key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
});
