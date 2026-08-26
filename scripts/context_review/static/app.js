// Tiny review UI. State lives entirely in memory; persistence is the
// JSONL written server-side at docs/context_review_feedback.jsonl.

let current = null; // last rolled scenario JSON
let currentSeed = null;
let maxPlayhead = 0;

const $ = (sel) => document.querySelector(sel);

const STATE_GLYPH = {
  CONTEXT:       'CTX',
  GENERATED:     'GEN',
  TO_GENERATE:   '→T←',
  GENERATING:    '···',
  PENDING:       '···',
  WASTED:        'WST',
  OUTSIDE:       '',
  MASKED_FUTURE: 'M·F',
  MASKED_PAST:   'M·P',
  MASKED_FAILED: 'M·X',
};

// Studio/review detail states → the three states the model actually sees in
// its prompt: a bar is either CONTEXT (real content), MASKED (a MaskBar
// placeholder), or TO_GENERATE (the in-flight target). PENDING means "no
// prompt has been built yet" so it's neither.
const MODEL_STATE = {
  CONTEXT:       'M_CONTEXT',
  GENERATED:     'M_CONTEXT',
  TO_GENERATE:   'M_TO_GEN',
  WASTED:        'M_TO_GEN',     // AR-sampled then discarded
  PENDING:       'PENDING',
  MASKED_FUTURE: 'M_MASKED',
  MASKED_PAST:   'M_MASKED',
  MASKED_FAILED: 'M_MASKED',
  OUTSIDE:       'OUTSIDE',
};

const MODEL_STATE_GLYPH = {
  M_CONTEXT:  'CTX',
  M_MASKED:   'MSK',
  M_TO_GEN:   '→T←',
  PENDING:    '···',
  OUTSIDE:    '',
};

let simpleMode = false;

const LEGEND = [
  ['CONTEXT',       'context'],
  ['GENERATED',     'generated'],
  ['TO_GENERATE',   'to-generate (this tick)'],
  ['PENDING',       'pending / no gen yet'],
  ['WASTED',        'wasted (warmup tail)'],
  ['MASKED_FUTURE', 'masked · future (>= playhead)'],
  ['MASKED_PAST',   'masked · pre-history (policy)'],
  ['MASKED_FAILED', 'masked · failed gen'],
];

const MODEL_LEGEND = [
  ['M_CONTEXT', 'context (real content in prompt)'],
  ['M_MASKED',  'masked (MaskBar in prompt)'],
  ['M_TO_GEN',  'to-generate (in-flight target)'],
  ['PENDING',   'pending (no prompt built yet)'],
];

function renderLegend() {
  const el = $('#legend');
  el.innerHTML = '';
  const items = simpleMode ? MODEL_LEGEND : LEGEND;
  for (const [state, label] of items) {
    const div = document.createElement('div');
    div.className = 'legend-item';
    div.innerHTML = `<span class="legend-sw s-${state}"></span><span>${label}</span>`;
    el.appendChild(div);
  }
}

function renderScenario(s) {
  current = s;
  const p = s.params, d = s.derived;
  $('#scenario-summary').textContent =
    `seed=${s.seed}\n` +
    `tracks=${s.grid.length}  model_dim=${p.model_dim}  lookahead=${p.lookahead}  ` +
    `num_anticip=${p.num_anticip}  buffer=${p.buffer_bars}  adapt_buffer=${p.adapt_buffer}\n` +
    `policy=${p.warmup_policy}  mask_gap=${p.mask_gap}  playhead=${p.playhead}  regime=${d.regime}\n` +
    `steady_col=${d.steady_col}  playhead_col=${d.playhead_col}  target_col=${d.target_col}  ` +
    `toGen=[${d.to_gen_cols.join(',')}]  lookahead=[${d.lookahead_cols.join(',')}]\n` +
    `first_gen_playhead=${d.first_gen_playhead}  first_target_bar=${d.first_target_bar}  ` +
    `failed_gen_bars=${d.failed_gen_bars.length ? '['+d.failed_gen_bars.join(',')+']' : '∅'}`;

  const ncols = p.model_dim;
  const playheadCol = d.playhead_col;

  const tbl = document.createElement('table');
  tbl.className = 'grid-table';

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  headRow.appendChild(document.createElement('th')); // corner
  for (let c = 0; c < ncols; c++) {
    const th = document.createElement('th');
    const bar = s.grid[0]?.cells[c]?.bar ?? c;
    th.textContent = bar;
    if (c === playheadCol) th.style.color = '#4e9eff';
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  tbl.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (const row of s.grid) {
    const tr = document.createElement('tr');
    const td0 = document.createElement('td');
    td0.className = 'label';
    td0.textContent = (row.track.is_agent ? '▶ ' : '  ') + row.track.label;
    tr.appendChild(td0);
    for (const cell of row.cells) {
      const td = document.createElement('td');
      const div = document.createElement('div');
      const shown = simpleMode ? (MODEL_STATE[cell.state] ?? cell.state) : cell.state;
      const glyph = simpleMode ? MODEL_STATE_GLYPH[shown] : STATE_GLYPH[shown];
      div.className   = `cell s-${shown}`;
      div.textContent = glyph ?? '';
      div.title       = `bar=${cell.bar}  detail=${cell.state}  model=${MODEL_STATE[cell.state]}`;
      td.appendChild(div);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);

  const wrap = $('#grid-wrap');
  wrap.innerHTML = '';
  wrap.appendChild(tbl);
  $('#seed-input').value = s.seed;
  $('#playhead-val').textContent = s.params.playhead;
}

async function roll(seed) {
  $('#status').textContent = 'rolling…';
  const url = seed != null ? `/roll?seed=${seed}` : '/roll';
  const r   = await fetch(url);
  const s   = await r.json();
  currentSeed = s.seed;
  maxPlayhead = s.max_playhead ?? 64;
  const slider = $('#playhead-slider');
  slider.max   = maxPlayhead;
  slider.value = s.params.playhead;
  renderScenario(s);
  $('#status').textContent = '';
  $('#feedback').value = '';
}

async function setPlayhead(ph) {
  if (currentSeed == null) return;
  $('#playhead-val').textContent = ph;
  const r = await fetch(`/state?seed=${currentSeed}&playhead=${ph}`);
  const s = await r.json();
  renderScenario(s);
}

async function save() {
  if (!current) return;
  const text = $('#feedback').value.trim();
  if (!text) {
    $('#save-status').textContent = 'feedback is empty';
    return;
  }
  $('#save-status').textContent = 'saving…';
  const r = await fetch('/feedback', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      seed:     current.seed,
      params:   current.params,
      derived:  current.derived,
      grid:     current.grid,
      feedback: text,
    }),
  });
  const j = await r.json();
  if (j.ok) {
    $('#save-status').textContent = `saved · seed ${current.seed}`;
    $('#feedback').value = '';
    await refreshHistory();
  } else {
    $('#save-status').textContent = j.error || 'error';
  }
}

async function refreshHistory() {
  const r = await fetch('/feedback');
  const j = await r.json();
  const el = $('#history');
  el.innerHTML = '';
  const list = (j.entries || []).slice(-10).reverse();
  for (const e of list) {
    const div = document.createElement('div');
    div.className = 'entry';
    div.innerHTML = `
      <div class="meta">${e.ts} · seed=${e.seed} · policy=${e.params?.warmup_policy} · playhead=${e.params?.playhead}</div>
      <div class="body">${escapeHtml(e.feedback || '')}</div>`;
    div.addEventListener('click', () => roll(e.seed));
    div.style.cursor = 'pointer';
    el.appendChild(div);
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;',
  })[c]);
}

$('#btn-roll').addEventListener('click', () => roll());
$('#btn-replay').addEventListener('click', () => {
  const v = $('#seed-input').value;
  if (v !== '') roll(parseInt(v, 10));
});
$('#btn-save').addEventListener('click', save);
$('#playhead-slider').addEventListener('input', (e) => {
  setPlayhead(parseInt(e.target.value, 10));
});
$('#simple-toggle').addEventListener('change', (e) => {
  simpleMode = e.target.checked;
  if (current) renderScenario(current);
  renderLegend();
});

renderLegend();
roll();
refreshHistory();
