// Neat Change Studio — CYWS before/after change detection on the Modalix MLA.
// Pick a test pair (or add two photos), run it on the MLA, inspect the result.
// Vanilla JS, no build step. Talks to webapp/server.py (/api/*).
"use strict";

/* ------------------------------------------------------------------ helpers */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage blocked */ } },
};
const mq = {
  phone: matchMedia("(max-width: 760px)"),
  tablet: matchMedia("(max-width: 1023px)"),
  desktop: matchMedia("(min-width: 1024px)"),
  coarse: matchMedia("(pointer: coarse)"),
  dark: matchMedia("(prefers-color-scheme: dark)"),
  still: matchMedia("(prefers-reduced-motion: reduce)"),
};

const LABELS = ["appeared", "disappeared", "changed"];
const LABEL_TEXT = { appeared: "Appeared", disappeared: "Disappeared", changed: "Changed" };
const cap = (s) => String(s).replace(/(^|[\s/(·+-])([a-z])/g, (m, a, b) => a + b.toUpperCase());
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const PAIR_NAMES = {
  "20240918_004758": "Three Cups Removed",
  "20240916_175631": "Pillows Swapped",
  "20240919_171750": "Kitchen Counter",
  "20240916_180158": "Sofa · No Change",
};
const SUGGEST = [
  ["20240918_004758", "Three Cups Removed"],
  ["20240916_175631", "Pillows Swapped"],
  ["20240919_171750", "Kitchen Counter"],
  ["20240916_180158", "No Change"],
];
const TIME_COLOR = { load: "#3a86ec", register: "#a78bfa", stream: "#16c8a6", decode: "#a9c81c", postprocess: "#f2801d", other: "#8b95a1",
  preprocess_cpu: "#16c8a6", preprocess_wait: "#5fd8c4", infer_wait: "#0f8f80" };
const TIME_LABEL = { load: "Decode JPEGs", register: "Align Photos", stream: "Preprocess + MLA", decode: "Box Decode",
  postprocess: "Merge + Masks", other: "Other", preprocess_cpu: "Crop / Resize CPU (Sum)", preprocess_wait: "Waiting for Crops",
  infer_wait: "Waiting for the MLA" };
const LEVEL_NAME = { "0": "Global", "0,1": "Tiled", "0,1,2": "Fine" };

const state = {
  status: null, pairs: [], filter: "all",
  current: null,                      // { id, kind: "pair" | "upload" }
  slots: { before: null, after: null }, // { kind: "pair", id, url } | { kind: "file", file, url }
  uploadId: null,
  result: null, mode: "overlay",
  layers: { boxes: true, heat: true, mask: false, gt: false, tiles: false },
  opts: { preset: null, levels: "0,1", thr: 0.3, register: false, tiles: "full", mask_mode: "box" },
  hashKeys: new Set(), presets: {}, presetsLoaded: false, defaults: null,
  inflight: null, want: null, runAll: null, pairResults: {}, winCount: { "0": 1 }, lastTotal: 1200,
  hl: null, pinned: null, zoom: { s: 1, x: 0, y: 0 }, swipe: 50, suppressClick: false, modelSwitching: null, idleWaiters: [],
};

const NET_ERR = "Can't reach the board — check the connection";
async function api(path, opts = {}) {
  let r;
  try { r = await fetch(path, opts); }
  catch (e) { throw new Error(NET_ERR); }       // fetch rejects (TypeError) only when the server can't be reached
  let body = null;
  try { body = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error((body && body.error) || `${r.status} ${r.statusText}`);
  return body;
}
const postJSON = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

/* ------------------------------------------------------------------ formatting */
function pairName(id) {
  if (PAIR_NAMES[id]) return PAIR_NAMES[id];
  const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/.exec(id || "");
  return m ? `${MONTHS[+m[2] - 1]} ${+m[3]} · ${m[4]}:${m[5]}` : (id || "");
}
function fmtMs(v) {
  if (v == null || !isFinite(v)) return "—";
  if (v >= 10000) return `${(v / 1000).toFixed(1)}<small>s</small>`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)}<small>s</small>`;
  return `${v < 10 ? v.toFixed(1) : Math.round(v)}<small>ms</small>`;
}
const fmtInt = (v) => (v == null || !isFinite(v) ? "—" : String(Math.round(v)));
const fmtPct = (v) => (v == null || !isFinite(v) ? "—" : `${Math.round(v)}<small>%</small>`);
const pct = (v) => (v == null ? "—" : `${Math.round(100 * v)}%`);

function countTo(el, value, fmt) {
  if (!el) return;
  cancelAnimationFrame(el._raf);
  if (value == null || !isFinite(value)) { el.innerHTML = "—"; el._v = null; return; }
  const from = el._v == null ? 0 : el._v;
  el._v = value;
  const item = el.closest(".metric-item");
  if (item && from !== value) { item.classList.remove("flash"); void item.offsetWidth; item.classList.add("flash"); clearTimeout(item._t); item._t = setTimeout(() => item.classList.remove("flash"), 700); }
  if (mq.still.matches || from === value) { el.innerHTML = fmt(value); return; }
  const t0 = performance.now(), dur = 650;
  const step = (t) => {
    const k = clamp((t - t0) / dur, 0, 1), e = 1 - Math.pow(1 - k, 3);   // a frame's timestamp can precede t0
    el.innerHTML = fmt(from + (value - from) * e);
    if (k < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}
function setText(el, html) { if (el) { cancelAnimationFrame(el._raf); el._v = null; el.innerHTML = html; } }

/* ------------------------------------------------------------------ toasts */
function toast(msg, type = "info", ms) {
  const box = $("#toasts");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.setAttribute("role", type === "error" ? "alert" : "status");
  const icon = type === "error" ? "i-alert" : type === "ok" ? "i-check" : "i-info";
  el.innerHTML = `<svg><use href="#${icon}"/></svg><div class="t-msg"></div><button type="button" class="t-x" aria-label="Dismiss"><svg><use href="#i-close"/></svg></button>`;
  el.querySelector(".t-msg").textContent = msg;
  const close = () => { if (el._gone) return; el._gone = true; el.classList.add("out"); setTimeout(() => el.remove(), 260); };
  el.querySelector(".t-x").addEventListener("click", close);
  box.appendChild(el);
  while (box.children.length > 4) box.firstElementChild.remove();
  setTimeout(close, ms || (type === "error" ? 8000 : 4000));
}

/* ------------------------------------------------------------------ theme */
function effectiveTheme() {
  const t = document.documentElement.getAttribute("data-theme");
  return t || (mq.dark.matches ? "dark" : "light");
}
function syncThemeMeta() {
  const dark = effectiveTheme() === "dark";
  $("#meta-theme").setAttribute("content", dark ? "#0c0f14" : "#f4f6f8");
  $("#btn-theme").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
}
function toggleTheme() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  const root = document.documentElement;
  if (!mq.still.matches) {
    root.classList.add("theme-anim");
    clearTimeout(toggleTheme._t);
    toggleTheme._t = setTimeout(() => root.classList.remove("theme-anim"), 420);
  }
  root.setAttribute("data-theme", next);
  store.set("cyws-theme", next);
  syncThemeMeta();
}

/* ------------------------------------------------------------------ status / model */
let modelsSig = "";
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    state.status = s;
    renderStatus(s);
    if (!state.presetsLoaded) {
      state.presetsLoaded = true;
      state.presets = s.presets || {};
      const d = normalizeOpts(s.defaults || {});
      state.defaults = { ...state.defaults, ...d };
      for (const k of Object.keys(d)) if (!state.hashKeys.has(k)) state.opts[k] = d[k];
      applyHashPreset();
      buildPresets();
      syncControls();
    }
    return s;
  } catch (e) {
    state.status = null;
    renderStatus(null);
    return null;
  }
}

function renderStatus(s) {
  const dot = $("#status-dot"), text = $("#status-text"), pill = $("#status-pill"), hero = $("#hero-model");
  const st = s ? (state.modelSwitching ? "loading" : s.state) : "error";
  dot.className = `dot ${st}`;
  let label, plain;
  if (!s) { label = plain = "Server Unreachable"; }
  else if (st === "ready") {
    const lat = s.latency && s.latency.mean_ms;
    label = `<span class="st-long">MLA Ready · </span><b>${esc(s.model)}</b>${lat ? ` · ${lat < 10 ? lat.toFixed(1) : Math.round(lat)} ms<span class="st-long"> per Window</span>` : ""}`;
    plain = `MLA ready, model ${s.model}`;
  } else if (st === "error") { label = plain = `Model Error${s.error ? `: ${s.error}` : ""}`; }
  else { label = plain = `Loading ${esc(state.modelSwitching || s.model || "the Model")}…`; }
  text.innerHTML = label;
  pill.title = plain;
  pill.setAttribute("aria-label", `Model status: ${plain}`);
  hero.className = `hero-model ${st}`;
  const m = s && (s.models || []).find((x) => x.name === s.model);
  $("#hero-model-text").textContent = !s ? "Server Unreachable" : st === "ready" ? `${m ? m.label : s.model} · Ready on the MLA`
    : st === "error" ? (/not on this board/.test(s.error || "") ? "No Model on This Board Yet — Download One in Settings" : "Model Failed to Load — Open Settings")
    : "Loading the Model onto the MLA…";
  $("#chip-model").textContent = s && s.model ? s.model : "";
  if (s && s.latency && s.latency.mean_ms) countTo($("#m-mla"), s.latency.mean_ms, fmtMs);
  if (s) renderModels(s);
  updateComposer();
}

/* Models: the ones on the board, plus the catalog on Hugging Face (status.models and
   /api/hub). A model on the board loads when picked; one that isn't offers a download,
   with progress, cancel and resume, like Neat GenAI Studio and Decision Studio. */
const fmtMB = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${Math.round(b / 1e6)} MB`);
const fmtLeft = (sec) => (sec == null ? "" : sec >= 90 ? `${Math.round(sec / 60)} min left` : `${Math.max(1, sec)} s left`);
let hubTimer = null;

async function refreshHub() {
  clearTimeout(hubTimer);
  try {
    state.hub = await api("/api/hub");
  } catch (e) {
    state.hub = state.hub || { error: e.message, models: {} };
  }
  modelsSig = "";
  if (state.status) renderModels(state.status);
  const busy = Object.values((state.hub && state.hub.models) || {}).some((m) => m.state === "downloading");
  hubTimer = setTimeout(refreshHub, busy ? 800 : 60000);
  const downloading = Object.keys((state.hub && state.hub.models) || {}).find((n) => state.hub.models[n].state === "downloading");
  if (state.hubWasBusy && !busy) {                     // a download finished: the model list changed
    const name = state.hubWasBusy, m = state.hub.models[name];
    await refreshStatus();
    if (m && m.state === "on_board") {
      toast(`${name} is on the board`, "ok", 3000);
      // nothing loaded yet (a fresh board): load what just arrived
      if (state.status && state.status.state !== "ready" && !state.modelSwitching) switchModel(name);
    }
  }
  state.hubWasBusy = downloading || null;
}

async function hubAction(action, name) {
  try {
    state.hub = await postJSON(`/api/hub/${action}`, { name });
    if (action === "download") toast(`Downloading ${name} from Hugging Face`, "info", 2500);
  } catch (e) {
    toast(e.message, "error");
  }
  refreshHub();
}

function renderModels(s) {
  const hub = state.hub || { models: {} };
  const sig = JSON.stringify([s.model, s.state, state.modelSwitching, s.models, hub]);
  if (sig === modelsSig) return;
  modelsSig = sig;
  const names = [...new Set([...(s.models || []).map((m) => m.name), ...Object.keys(hub.models || {})])];
  $("#models").innerHTML = names.map((name) => {
    const local = (s.models || []).find((m) => m.name === name) || {};
    const remote = (hub.models || {})[name];
    const onBoard = !!local.available;
    const on = name === (state.modelSwitching || s.model);
    const loading = on && (state.modelSwitching || s.state === "loading");
    const [, ...rest] = (local.label || name).split("·");
    const about = remote && remote.about ? remote.about : rest.join("·").trim() || "Compiled for the MLA";
    const acc = (remote && remote.accuracy) || {};
    const stats = remote ? [remote.latency_ms ? `${remote.latency_ms} ms per Window` : null,
      acc.changes_found ? `${acc.changes_found} Found` : null, acc.pixel_f1 != null ? `F1 ${acc.pixel_f1.toFixed(3)}` : null,
      fmtMB(remote.bytes)].filter(Boolean) : [];
    let badge = "", action = "";
    if (on && !loading) badge = `<span class="mm-badge on">Loaded</span>`;
    else if (loading) badge = `<span class="mm-badge">Loading…</span>`;
    else if (onBoard) badge = `<span class="mm-badge">On Board</span>`;
    if (remote && remote.state === "downloading") {
      const pr = remote.progress || {};
      action = `<div class="mm-dl"><div class="mm-bar"><i style="width:${(100 * (pr.fraction || 0)).toFixed(1)}%"></i></div>
        <div class="mm-dl-row"><span>${Math.round(100 * (pr.fraction || 0))}% · ${fmtMB(pr.done_bytes || 0)} of ${fmtMB(remote.bytes)}${pr.bytes_per_second ? ` · ${fmtMB(pr.bytes_per_second)}/s` : ""}${pr.seconds_left != null ? ` · ${fmtLeft(pr.seconds_left)}` : ""}</span>
        <button type="button" class="mm-act ghost" data-act="cancel">Cancel</button></div></div>`;
    } else if (remote && !onBoard) {
      const part = remote.partial_bytes || 0;
      action = `${remote.error ? `<p class="mm-err">${esc(remote.error)}</p>` : ""}<div class="mm-dl-row">
        <button type="button" class="mm-act" data-act="download"><svg><use href="#i-download"/></svg>${part ? `Continue · ${fmtMB(part)} of ${fmtMB(remote.bytes)}` : `Download · ${fmtMB(remote.bytes)}`}</button>
        ${part ? `<button type="button" class="mm-act ghost" data-act="discard">Discard</button>` : ""}</div>`;
    }
    const pick = onBoard && !on && !state.modelSwitching;
    const role = onBoard ? `role="radio" aria-checked="${on}" tabindex="${pick || on ? 0 : -1}"`
      : `role="group" aria-label="${esc(name)}: not on this board yet"`;
    return `<div class="model-opt ${loading ? "loading" : ""} ${onBoard ? "" : "remote"}" data-name="${esc(name)}" data-onboard="${onBoard ? 1 : 0}"
        ${role} title="${pick ? `Load ${esc(name)} on the MLA` : ""}">
      <span class="mm-dot"></span><span class="mm-body">
        <span class="mm-top"><span class="mm-name">${esc(name)}</span>${remote && remote.precision ? `<span class="mm-prec">${esc(remote.precision)}</span>` : ""}${badge}</span>
        <span class="mm-sub">${esc(about)}</span>
        ${stats.length ? `<span class="mm-stats">${stats.map((x) => `<span>${esc(x)}</span>`).join(" · ")}</span>` : ""}
        ${action}</span></div>`;
  }).join("");
  const src = $("#models-src");
  if (hub.repo) {
    src.innerHTML = `${hub.error ? `<span class="mm-err">${esc(hub.error)}</span> ` : ""}Models download from Hugging Face:
      <a href="${esc(hub.page)}" target="_blank" rel="noopener">${esc(hub.repo)}<svg><use href="#i-external"/></svg></a>${hub.collection
        ? ` · <a href="${esc(hub.collection)}" target="_blank" rel="noopener">Collection<svg><use href="#i-external"/></svg></a>` : ""}${hub.free_bytes ? ` · ${fmtMB(hub.free_bytes)} Free` : ""}${hub.directory ? `<br>Folder: <code>${esc(hub.directory)}</code>` : ""}`;
  } else src.textContent = "";
}

/* ------------------------------------------------------------------ side panel */
const railCollapsed = () => $("#workspace").classList.contains("rail-collapsed");
function setRailCollapsed(on, { remember = true } = {}) {
  const ws = $("#workspace"), rail = $("#rail"), t = $("#rail-toggle");
  ws.classList.toggle("rail-collapsed", on);
  rail.inert = on && mq.desktop.matches;
  t.setAttribute("aria-expanded", String(!on));
  t.title = on ? "Show Panel" : "Hide Panel";
  t.setAttribute("aria-label", on ? "Show the side panel" : "Hide the side panel");
  $("#btn-settings").setAttribute("aria-pressed", String(!on && mq.desktop.matches));
  if (remember) store.set("cyws-rail", on ? "collapsed" : "open");
}
/* The header's Settings button: show the panel at Settings; pressed again there, hide it. */
function toggleSettings() {
  if (mq.phone.matches) { setTab(document.body.dataset.tab === "settings" ? "detect" : "settings"); return; }
  if (mq.tablet.matches) { if ($("#workspace").classList.contains("rail-open")) setDrawer(false); else openSettings(); return; }
  const rail = $("#rail"), card = $('.card[data-card="settings"]');
  const atSettings = !card.classList.contains("collapsed") && Math.abs(card.getBoundingClientRect().top - rail.getBoundingClientRect().top) < 40;
  if (!railCollapsed() && atSettings) setRailCollapsed(true);
  else openSettings();
}

/* Model switch: a full-screen card follows the server's real stages
   (release the MLA → load the model → warm up), with a progress bar paced from the
   last load time of that model. It stays up at least MSW_MIN_MS so a quick switch
   still reads as a deliberate step, then shows a check and fades out. */
const MSW_STEPS = ["release", "load", "warm", "ready"];
const MSW_SPAN = { release: [0, 0.12], load: [0.12, 0.82], warm: [0.82, 0.97], ready: [1, 1] };
const MSW_MIN_MS = 1500;
const msw = { t0: 0, stageT0: 0, stage: "", fill: 0 };

function mswShow(from, to) {
  const el = $("#mswitch");
  el.classList.remove("out");
  el.hidden = false;
  el.dataset.s = from ? "release" : "load";
  $("#msw-from").textContent = from || "";
  $("#msw-to").textContent = to;
  $("#msw-title").textContent = `Switching to ${to}`;
  $("#msw-err").hidden = true;
  $("#msw-close").hidden = true;
  $('#msw-steps li[data-s="release"]').classList.toggle("skip", !from);
  msw.t0 = msw.stageT0 = performance.now(); msw.stage = el.dataset.s; msw.fill = 0;
  mswStage(msw.stage, 0, null);
}
function mswStage(stage, elapsed, expected) {
  const el = $("#mswitch");
  if (stage !== msw.stage) { msw.stage = stage; msw.stageT0 = performance.now(); }
  el.dataset.s = stage;
  const k = MSW_STEPS.indexOf(stage);
  $$("#msw-steps li").forEach((li) => {
    const i = MSW_STEPS.indexOf(li.dataset.s);
    li.classList.toggle("done", i < k || stage === "ready");
    li.classList.toggle("cur", i === k && stage !== "ready");
    li.classList.remove("fail");
  });
  // progress inside the stage: linear against the last load time when known, else easing toward its end
  const [a, b] = MSW_SPAN[stage] || [0, 0];
  const inStage = (performance.now() - msw.stageT0) / 1000;
  const want = stage === "ready" ? 1 : expected ? clamp(elapsed / expected, a, b) : a + (b - a) * (1 - Math.exp(-inStage / 3));
  msw.fill = Math.max(msw.fill, want);
  $("#msw-bar").style.width = `${(100 * msw.fill).toFixed(1)}%`;
  $(".msw-fill").style.setProperty("--fill", msw.fill.toFixed(3));
  $("#msw-time").textContent = `${elapsed.toFixed(1)} s`;
  $("#msw-eta").textContent = expected && stage !== "ready" ? `~${Math.max(0, expected - elapsed).toFixed(0)} s left` : "";
}
function mswFail(msg) {
  const el = $("#mswitch");
  el.dataset.s = "error";
  const cur = $("#msw-steps li.cur");
  if (cur) { cur.classList.remove("cur"); cur.classList.add("fail"); }
  $("#msw-title").textContent = "Couldn't Switch the Model";
  $("#msw-err").textContent = msg;
  $("#msw-err").hidden = false;
  $("#msw-close").hidden = false;
  $("#msw-close").focus({ preventScroll: true });
}
function mswHide() {
  const el = $("#mswitch");
  if (el.hidden) return;
  el.classList.add("out");
  setTimeout(() => { el.hidden = true; el.classList.remove("out"); }, mq.still.matches ? 0 : 340);
}

async function switchModel(name) {
  if (!state.status || name === state.status.model || state.modelSwitching) return;
  if (cam.live) stopLive();
  const from = state.status.state === "ready" ? state.status.model : null;
  state.modelSwitching = name;
  $("#model-err").hidden = true;
  renderStatus(state.status);
  mswShow(from, name);
  // poll the stage while the switch request is open (it returns when the model is ready)
  let done = false;
  const poll = (async () => {
    let expected = null;
    while (!done) {
      const elapsed = (performance.now() - msw.t0) / 1000;
      try {
        const s = await fetch("/api/status").then((r) => r.json());
        const ld = s.load || {};
        if (ld.target === name && ld.expected_s) expected = ld.expected_s;
        if (!done && ld.stage && ld.target === name) mswStage(ld.stage, ld.elapsed_s || elapsed, expected);
        else if (!done) mswStage(msw.stage, elapsed, expected);
      } catch (e) { /* the switch request reports errors */ }
      await sleep(200);
    }
  })();
  let err = null;
  try {
    await postJSON("/api/model", { name });
  } catch (e) {
    err = e;
  }
  // a fast switch (or the stand-in backend) still walks through the stages
  const left = MSW_MIN_MS - (performance.now() - msw.t0);
  if (!err && left > 0) {
    for (const st of MSW_STEPS.slice(MSW_STEPS.indexOf(msw.stage) + 1, 3)) {
      mswStage(st, (performance.now() - msw.t0) / 1000, null);
      await sleep(left / 3);
    }
  }
  done = true;
  await poll;
  state.modelSwitching = null;
  modelsSig = "";
  if (err) {
    mswFail(err.message);
    $("#model-err").textContent = err.message;
    $("#model-err").hidden = false;
    await refreshStatus();
    return;
  }
  mswStage("ready", (performance.now() - msw.t0) / 1000, null);
  $("#msw-title").textContent = `${name} Is Ready on the MLA`;
  await refreshStatus();
  await sleep(mq.still.matches ? 400 : 1100);
  mswHide();
  if (state.current && state.status && state.status.state === "ready") requestRun();
}

/* ------------------------------------------------------------------ gallery */
async function loadPairs() {
  try {
    const r = await api("/api/pairs");
    state.pairs = r.pairs || [];
  } catch (e) {
    toast(`Could not load the test set: ${e.message}`, "error");
  }
  const n = state.pairs.length, ch = state.pairs.filter((p) => p.gt_change).length;
  $("#f-all").textContent = n; $("#f-change").textContent = ch; $("#f-none").textContent = state.pairs.filter((p) => p.gt_change === false).length;
  renderPairs();
  renderSuggest();
}

function shownPairs() {
  return state.pairs.filter((p) => state.filter === "all" || (state.filter === "change" ? p.gt_change === true : p.gt_change === false));
}

function renderPairs() {
  const list = $("#pairs");
  const shown = shownPairs();
  $("#gal-count").textContent = `${shown.length} Pairs`;
  list.innerHTML = "";
  if (!shown.length) { list.innerHTML = `<p class="empty-row">No pairs here.</p>`; return; }
  const frag = document.createDocumentFragment();
  for (const p of shown) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "pair"; b.dataset.id = p.id;
    b.setAttribute("role", "option"); b.tabIndex = -1;
    const truth = p.gt_change == null ? "" : `<span class="badge truth ${p.gt_change ? "" : "none"}"><i></i>${p.gt_change ? "Change" : "No Change"}</span>`;
    b.innerHTML = `<span class="ph"><img class="b" alt="" loading="lazy" decoding="async"><img class="a" alt="" loading="lazy" decoding="async">${truth}<span class="badge res" hidden></span></span>
      <span class="pi"><span class="pi-title">${esc(pairName(p.id))}</span><span class="pi-id">${esc(p.id)}</span></span>`;
    b.setAttribute("aria-label", `${pairName(p.id)} (${p.id})${p.gt_change == null ? "" : p.gt_change ? ", has a change" : ", no change"}`);
    const ph = b.querySelector(".ph");
    let loaded = 0;
    for (const [cls, which] of [["b", "before"], ["a", "after"]]) {
      const img = b.querySelector(`img.${cls}`);
      img.addEventListener("load", () => { img.classList.add("ok"); if (++loaded === 2) ph.classList.add("loaded"); }, { once: true });
      img.addEventListener("error", () => ph.classList.add("loaded"), { once: true });
      img.src = `/api/thumb/${encodeURIComponent(p.id)}/${which}`;
    }
    frag.appendChild(b);
  }
  list.appendChild(frag);
  updatePairTiles();
}

function updatePairTiles() {
  const sel = state.current && state.current.kind === "pair" ? state.current.id : null;
  const running = state.inflight && state.inflight.id;
  let focusable = null;
  for (const el of $$("#pairs .pair")) {
    const id = el.dataset.id;
    const on = id === sel;
    el.setAttribute("aria-selected", on ? "true" : "false");
    el.classList.toggle("running", id === running);
    const res = state.pairResults[id];
    const badge = el.querySelector(".badge.res");
    if (res) {
      badge.hidden = false;
      badge.textContent = res.boxes === 0 ? "0" : `${res.boxes}`;
      badge.title = `${res.boxes} box${res.boxes === 1 ? "" : "es"} found`;
      badge.classList.toggle("zero", res.boxes === 0);
    } else badge.hidden = true;
    if (on) focusable = el;
    el.tabIndex = -1;
  }
  (focusable || $("#pairs .pair") || {}).tabIndex = 0;
}

function renderSuggest() {
  const have = new Set(state.pairs.map((p) => p.id));
  const list = SUGGEST.filter(([id]) => !state.pairs.length || have.has(id));
  $("#suggest-row").innerHTML = list.map(([id, label]) =>
    `<button type="button" class="suggest-chip" data-id="${esc(id)}"><img src="/api/thumb/${encodeURIComponent(id)}/after" alt="" loading="lazy">${esc(label)}</button>`).join("");
}

function moveGalleryFocus(key, from) {
  const tiles = $$("#pairs .pair");
  if (!tiles.length) return;
  let i = tiles.indexOf(from);
  if (i < 0) i = tiles.findIndex((t) => t.getAttribute("aria-selected") === "true");
  const cols = getComputedStyle($("#pairs")).gridTemplateColumns.split(" ").length || 1;
  if (i < 0) i = key === "ArrowLeft" || key === "ArrowUp" ? tiles.length : -1;
  const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols, ArrowUp: -cols }[key];
  let j = key === "Home" ? 0 : key === "End" ? tiles.length - 1 : clamp(i + step, 0, tiles.length - 1);
  tiles.forEach((t) => { t.tabIndex = -1; });
  tiles[j].tabIndex = 0;
  tiles[j].focus({ preventScroll: true });
  tiles[j].scrollIntoView({ block: "nearest", behavior: mq.still.matches ? "auto" : "smooth" });
}

/* Open a test pair: fill the composer slots, select it, run it. */
function openPair(id, { run = true, fromRunAll = false } = {}) {
  if (cam.live) stopLive();
  if (!fromRunAll && state.runAll) state.runAll.stop = true;
  clearSlotFiles();
  state.slots.before = { kind: "pair", id, url: `/api/thumb/${encodeURIComponent(id)}/before` };
  state.slots.after = { kind: "pair", id, url: `/api/thumb/${encodeURIComponent(id)}/after` };
  state.current = { id, kind: "pair" };
  state.uploadId = null;
  renderSlots();
  updatePairTiles();
  if (!fromRunAll) {
    if (mq.phone.matches) setTab("detect");
    else if (mq.tablet.matches) setDrawer(false);
  }
  if (run) requestRun();
}

/* ------------------------------------------------------------------ composer slots / uploads */
function clearSlotFiles() {
  for (const r of ["before", "after"]) {
    const s = state.slots[r];
    if (s && s.kind === "file") URL.revokeObjectURL(s.url);
    state.slots[r] = null;
  }
}
function isImageFile(f) { return f && ((f.type || "").startsWith("image/") || /\.(jpe?g|png|bmp|webp|tiff?)$/i.test(f.name || "")); }

function setSlotFile(role, file) {
  if (!file) return;
  if (cam.live) stopLive();
  if (!isImageFile(file)) { toast(`${file.name || "That file"} is not an image`, "error"); return; }
  const other = role === "before" ? "after" : "before";
  if (state.slots[other] && state.slots[other].kind !== "file") state.slots[other] = null;
  const old = state.slots[role];
  if (old && old.kind === "file") URL.revokeObjectURL(old.url);
  state.slots[role] = { kind: "file", file, url: URL.createObjectURL(file) };
  state.uploadId = null;
  if (state.current && state.current.kind !== "file") { state.current = null; updatePairTiles(); }
  renderSlots();
}
function addFiles(files, role) {
  const imgs = [...files].filter(isImageFile);
  if (!imgs.length) { if (files.length) toast("Those files are not images", "error"); return; }
  if (imgs.length >= 2) {
    // Name hints win (before/after, ref/cap, t0/t1); otherwise the older shot is "before".
    const hint = (f) => { const n = (f.name || "").toLowerCase();
      return /(before|_ref|ref\.|reference|\bt0\b|_0\.)/.test(n) ? -1 : /(after|_cap|cap\.|capture|\bt1\b|_1\.)/.test(n) ? 1 : 0; };
    imgs.sort((a, b) => (hint(a) - hint(b)) || (a.lastModified - b.lastModified) || a.name.localeCompare(b.name, undefined, { numeric: true }));
    setSlotFile("before", imgs[0]); setSlotFile("after", imgs[1]);
  } else {
    const target = role || (!fileIn("before") ? "before" : !fileIn("after") ? "after" : "after");
    setSlotFile(target, imgs[0]);
  }
  if (bothFilled() && !mq.phone.matches) $("#detect-btn").focus({ preventScroll: true });
}
const fileIn = (r) => state.slots[r] && state.slots[r].kind === "file";
const bothFilled = () => !!(state.slots.before && state.slots.after);

function clearSlot(role) {
  const s = state.slots[role];
  if (!s) return;
  if (cam.live) stopLive();
  if (s.kind !== "file") { state.slots.before = state.slots.after = null; state.current = null; updatePairTiles(); }
  else { URL.revokeObjectURL(s.url); state.slots[role] = null; state.uploadId = null; }
  renderSlots();
}

function renderSlots() {
  for (const role of ["before", "after"]) {
    const s = state.slots[role];
    const sub = !s ? (mq.coarse.matches ? "Tap to Add" : "Drop or Pick") : s.kind === "pair" ? pairName(s.id)
      : s.kind === "upload" ? "Your Photo" : (s.file.name || "photo");
    for (const el of $$(`.slot[data-role="${role}"], .drop[data-role="${role}"]`)) {
      const img = el.querySelector("img");
      el.classList.toggle("filled", !!s);
      if (s) { if (img.getAttribute("src") !== s.url) img.src = s.url; img.hidden = false; }
      else { img.hidden = true; img.removeAttribute("src"); }
      const subEl = el.querySelector(".slot-sub");
      if (subEl) subEl.textContent = sub;
      const clr = el.querySelector(".slot-clear");
      if (clr) clr.hidden = !s;
      el.setAttribute("aria-label", `${role === "before" ? "Before" : "After"} photo: ${s ? sub + " — change" : "choose or drop a file"}`);
    }
  }
  updateComposer();
}

function updateComposer() {
  const ready = state.status && state.status.state === "ready" && !state.modelSwitching;
  const busy = !!state.inflight;
  const btn = $("#detect-btn");
  btn.disabled = !(bothFilled() && ready) || !!state.runAll;
  btn.classList.toggle("busy", busy);
  btn.setAttribute("aria-busy", busy ? "true" : "false");
  $("#up-go").disabled = !(fileIn("before") && fileIn("after") && ready) || busy;
  const info = $("#composer-info");
  const s = state.slots;
  if (state.runAll) info.innerHTML = `Running the <b>Whole Test Set</b>…`;
  else if (cam.live) info.innerHTML = `<b>Webcam · Live</b> — Comparing Each Frame with Before`;
  else if (s.before && s.before.kind === "pair") info.innerHTML = `Test Pair · <b>${esc(pairName(s.before.id))}</b>`;
  else if (bothFilled() || (s.before && s.before.kind === "upload")) info.innerHTML = `<b>Your Photos</b> — Ready to Detect`;
  else if (s.before || s.after) info.innerHTML = `Now Add the <b>${s.before ? "After" : "Before"}</b> Photo`;
  else info.textContent = "Pick a test pair, or add a before and an after photo.";
  btn.title = !ready ? "Waiting for the Model" : bothFilled() ? "Detect Changes (Enter)" : "Add Both Photos First";
}

const picker = { role: null };
function pickFiles(role, camera = false) {
  const input = camera ? $("#file-cam") : $("#file-pick");
  picker.role = role;
  input.multiple = !camera && !role;
  input.value = "";
  input.click();
}
function onPicked(e) {
  const files = e.target.files;
  if (files && files.length) addFiles(files, picker.role);
}

function slotActivate(role) {
  if (mq.coarse.matches) {
    const s = state.slots[role];
    const items = [
      { icon: "i-camera", label: "Take a Photo", run: () => pickFiles(role, true) },
      { icon: "i-image", label: "Choose from Library", run: () => pickFiles(role, false) },
      { icon: "i-webcam", label: "Live Camera", run: openCam },
    ];
    if (s) items.push({ icon: "i-trash", label: s.kind === "pair" ? "Clear the Test Pair" : s.kind === "upload" ? "Clear Your Photos" : "Remove This Photo", danger: true, run: () => clearSlot(role) });
    openSheet(role === "before" ? "Before Photo" : "After Photo", items);
  } else pickFiles(role, false);
}

async function detect() {
  if ($("#detect-btn").disabled || state.inflight) return;
  const s = state.slots;
  if (!bothFilled()) return;
  if (s.before.kind === "pair") { state.current = { id: s.before.id, kind: "pair" }; requestRun(); return; }
  if (s.before.kind === "upload") { state.current = { id: s.before.id, kind: "upload" }; requestRun(); return; }
  if (!state.uploadId) {
    const btn = $("#detect-btn");
    btn.classList.add("busy");
    const files = {};
    for (const role of ["before", "after"]) {
      files[role] = await shrinkForUpload(s[role].file);
      if (!files[role]) { btn.classList.remove("busy"); toast(`That ${role} photo isn't a readable image`, "error"); return; }
    }
    const fd = new FormData();
    fd.append("before", files.before, files.before.name || "before.jpg");
    fd.append("after", files.after, files.after.name || "after.jpg");
    try {
      const r = await api("/api/upload", { method: "POST", body: fd });
      state.uploadId = r.id;
    } catch (e) {
      btn.classList.remove("busy");
      toast(uploadError(e.message), "error");
      return;
    }
  }
  state.current = { id: state.uploadId, kind: "upload" };
  if (mq.tablet.matches && !mq.phone.matches) setDrawer(false);
  if (mq.phone.matches) setTab("detect");
  requestRun();
}

/* Phone photos (12 MP and up) are shrunk to a 2048 px long side in the browser before
   upload: the app works at about that size anyway (work_long 1920), so nothing is lost,
   and the upload is several times smaller and the board's JPEG decode several times
   faster. The browser applies the EXIF rotation, so the result is upright.
   Returns the file to upload, or null if the browser can't decode a format it should.
   Formats browsers don't decode (TIFF, HEIC) are passed through for the server. */
const UPLOAD_LONG = 2048;
async function shrinkForUpload(file) {
  if (!/^image\/(jpeg|png|webp|gif|bmp)$/i.test(file.type || "") || !window.createImageBitmap) return file;
  let bm;
  try { bm = await createImageBitmap(file, { imageOrientation: "from-image" }); } catch (e) { return null; }
  try {
    const long = Math.max(bm.width, bm.height);
    if (long <= UPLOAD_LONG * 1.25) return file;            // already small: keep the original bytes
    const k = UPLOAD_LONG / long;
    const w = Math.round(bm.width * k), h = Math.round(bm.height * k);
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    const ctx = cv.getContext("2d");
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bm, 0, 0, w, h);
    const blob = await new Promise(res => cv.toBlob(res, "image/jpeg", 0.92));
    if (!blob) return file;
    const name = (file.name || "photo").replace(/\.[^.]+$/, "") + ".jpg";
    return new File([blob], name, { type: "image/jpeg", lastModified: file.lastModified || Date.now() });
  } catch (e) {
    return file;                                              // any canvas trouble: upload the original
  } finally {
    if (bm && bm.close) bm.close();
  }
}
/* Server messages can carry file-system paths; show the gist instead. */
function uploadError(msg) {
  msg = String(msg || "");
  if (msg === NET_ERR) return msg;
  const role = /\bafter\b/i.test(msg) && !/\bbefore\b/i.test(msg) ? "after" : /\bbefore\b/i.test(msg) ? "before" : null;
  if (/cannot (read|identify)|readable image/i.test(msg)) return role ? `That ${role} photo isn't a readable image` : "One of the photos isn't a readable image";
  if (/unsupported file type/i.test(msg)) return `${role ? `The ${role} photo` : "That file"} is not a supported image type (JPEG, PNG, WebP, BMP, TIFF)`;
  const clean = msg.replace(/(?:[A-Za-z]:)?[\/\\][^\s'"]+/g, "").replace(/\s{2,}/g, " ").trim();
  return `Upload failed${clean ? `: ${clean.slice(0, 140)}` : ""}`;
}

/* ------------------------------------------------------------------ settings */
/* ------------------------------------------------------------------ webcam */
/* Capture Before, then After, from the browser's camera; or, with Before captured, watch
   live: every new frame is uploaded as the After photo and detected, and the viewer
   updates. Browsers only allow the camera on HTTPS pages (or localhost); the board also
   serves the page over HTTPS with a self-signed certificate (status.https_port). */
const CAM_LONG = 1920;                 // captured frames are at most this long (the app works at 1920)
const cam = {
  stream: null, devices: [], dev: 0, shots: { before: null, after: null }, urls: { before: null, after: null },
  ghost: true, timer: 0, countTok: null, live: false, liveSrc: null, liveIds: new Set(), liveUrl: null, frames: [], canvas: null, ret: null,
};
const CAM_TIMER_STEPS = [0, 3, 5, 10];   // seconds; the self-timer button cycles through these
cam.timer = CAM_TIMER_STEPS.includes(Number(store.get("cyws-cam-timer", 0))) ? Number(store.get("cyws-cam-timer", 0)) : 0;
const camSupported = () => !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
const slotUses = (url) => ["before", "after"].some((r) => state.slots[r] && state.slots[r].url === url);
function revokeIfUnused(url) { if (url && !slotUses(url)) URL.revokeObjectURL(url); }

async function openCam() {
  const el = $("#cam");
  if (cam.live) { $("#cam-stop").focus(); return; }
  cam.ret = document.activeElement;
  el.hidden = false;
  el.classList.remove("live");
  $(".cam-panel").setAttribute("aria-modal", "true");
  renderCam();
  if (!camSupported()) { camUnavailable(); return; }
  await startStream();
  $("#cam-shutter").focus({ preventScroll: true });
}

function camUnavailable() {
  const port = state.status && state.status.https_port;
  const msg = $("#cam-msg");
  if (port && location.protocol === "http:") {
    const url = `https://${location.hostname}:${port}${location.pathname}${location.hash}`;
    msg.innerHTML = `<b>Open the Secure Page</b><p>Browsers only allow the webcam on HTTPS pages. The board serves one with a self-signed certificate: if the browser warns, choose Advanced, then Continue.</p><a class="btn primary" href="${esc(url)}">Open Secure Page</a>`;
  } else {
    msg.innerHTML = `<b>Webcam Not Available</b><p>${window.isSecureContext ? "This browser doesn't offer camera access." : "Browsers only allow the webcam on HTTPS pages. Start the demo with HTTPS on (webapp/run.sh does by default), or open it on the board at localhost."}</p>`;
  }
  msg.hidden = false;
  $("#cam-step").textContent = "";
  renderCam();
}

async function startStream() {
  stopStream();
  const msg = $("#cam-msg");
  msg.hidden = true;
  $("#cam-step").textContent = "Starting the Camera…";
  const dev = cam.devices[cam.dev];
  const video = { width: { ideal: 1920 }, height: { ideal: 1080 } };
  if (dev && dev.deviceId) video.deviceId = { exact: dev.deviceId }; else video.facingMode = { ideal: "environment" };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  } catch (e) {
    const retry = `<button type="button" class="btn primary" data-cam-retry>Try Again</button>`;
    msg.innerHTML = e.name === "NotAllowedError" || e.name === "SecurityError"
      ? `<b>Camera Access Blocked</b><p>Allow the camera for this site (the camera icon in the address bar), then try again.</p>${retry}`
      : e.name === "NotFoundError" || e.name === "OverconstrainedError"
        ? `<b>No Camera Found</b><p>Connect a webcam, then try again.</p>${retry}`
        : e.name === "NotReadableError"
          ? `<b>Camera Busy</b><p>Another app is using the camera. Close it, then try again.</p>${retry}`
          : `<b>Camera Error</b><p>${esc(e.message || e.name)}</p>${retry}`;
    msg.hidden = false;
    $("#cam-step").textContent = "";
    renderCam();
    return;
  }
  if ($("#cam").hidden) { stream.getTracks().forEach((t) => t.stop()); return; }   // closed while asking
  cam.stream = stream;
  const track = stream.getVideoTracks()[0];
  const settings = track.getSettings ? track.getSettings() : {};
  if (!cam.devices.length) {               // labels and ids are only listed once permission is granted
    try {
      cam.devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      cam.dev = Math.max(0, cam.devices.findIndex((d) => d.deviceId === settings.deviceId));
    } catch (e) { cam.devices = []; }
  }
  $("#cam-flip").hidden = cam.devices.length < 2;
  track.addEventListener("ended", () => { if (cam.stream === stream) { if (cam.live) { toast("The camera stopped", "error"); stopLive(); } else { stopStream(); camUnavailable(); } } });
  const v = $("#cam-video");
  v.srcObject = stream;
  try { await v.play(); } catch (e) { /* autoplay with muted video is allowed */ }
  if (!v.videoWidth) await new Promise((res) => { v.addEventListener("loadedmetadata", res, { once: true }); setTimeout(res, 3000); });
  // a camera facing the user is shown mirrored, like a mirror; captures are never mirrored
  const facesUser = settings.facingMode ? settings.facingMode === "user" : !mq.coarse.matches;
  $("#cam-stage").classList.toggle("mirror", facesUser);
  if (v.videoWidth) $("#cam-stage").style.setProperty("--cam-ar", `${v.videoWidth} / ${v.videoHeight}`);
  renderCam();
}

function stopStream() {
  cancelCountdown();
  if (cam.stream) cam.stream.getTracks().forEach((t) => t.stop());
  cam.stream = null;
  const v = $("#cam-video");
  v.srcObject = null;
}

function closeCam() {
  if (cam.live) { stopLive(); return; }
  stopStream();
  $("#cam").hidden = true;
  if (cam.ret && cam.ret.focus && document.contains(cam.ret)) cam.ret.focus({ preventScroll: true });
}

function renderCam() {
  const { before, after } = cam.urls;
  const ready = !!cam.stream;
  for (const role of ["before", "after"]) {
    const b = $(`.cam-shot[data-role="${role}"]`), img = b.querySelector("img"), url = cam.urls[role];
    b.classList.toggle("filled", !!url);
    b.classList.toggle("next", ready && !url && (role === "before" || !!before));
    if (url) { if (img.getAttribute("src") !== url) img.src = url; img.hidden = false; } else { img.hidden = true; img.removeAttribute("src"); }
    b.disabled = !url;
    b.setAttribute("aria-label", url ? `Retake the ${role} photo` : `${role === "before" ? "Before" : "After"} photo: not captured yet`);
  }
  const shutter = $("#cam-shutter");
  const role = !before ? "before" : "after";
  const counting = !!cam.countTok;
  const Role = role === "before" ? "Before" : "After";
  shutter.disabled = !ready;
  shutter.classList.toggle("after", role === "after");
  shutter.classList.toggle("counting", counting);
  shutter.setAttribute("aria-label", counting ? "Cancel the timer" : `Capture the ${role} photo${cam.timer ? ` in ${cam.timer} seconds` : ""}`);
  shutter.title = counting ? "Cancel the Timer (Space)" : `Capture ${Role}${cam.timer ? ` in ${cam.timer} s` : ""} (Space)`;
  $("#cam-count-hint").textContent = `Capturing ${Role} · Press the Shutter to Cancel`;
  if (ready) $("#cam-step").textContent = counting ? ""
    : !before ? "Step 1 · Frame the Scene, Then Capture Before" : "Step 2 · Change Something, Then Capture After";
  $("#cam-timer-v").textContent = cam.timer ? `${cam.timer} s` : "Off";
  $("#cam-timer").setAttribute("aria-pressed", String(!!cam.timer));
  $("#cam-timer").setAttribute("aria-label", `Self-timer: ${cam.timer ? `${cam.timer} seconds` : "off"}`);
  const ghost = $("#cam-ghost");
  if (before) { if (ghost.getAttribute("src") !== before) ghost.src = before; } else ghost.removeAttribute("src");
  ghost.hidden = !(before && cam.ghost && !after);
  $("#cam-ghost-btn").setAttribute("aria-pressed", String(cam.ghost));
  $("#cam-ghost-btn").disabled = !before;
  const live = $("#cam-live");
  live.disabled = !(before && ready) || counting;
  live.title = !before ? "Capture Before first" : "Compare every new frame with Before";
}

async function grabFrame(role) {
  const v = $("#cam-video");
  const w0 = v.videoWidth, h0 = v.videoHeight;
  if (!w0 || !h0) throw new Error("The camera isn't ready yet");
  const k = Math.min(1, CAM_LONG / Math.max(w0, h0));
  const w = Math.round(w0 * k), h = Math.round(h0 * k);
  const cv = cam.canvas || (cam.canvas = document.createElement("canvas"));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  cv.getContext("2d").drawImage(v, 0, 0, w, h);
  const blob = await new Promise((res) => cv.toBlob(res, "image/jpeg", 0.9));
  if (!blob) throw new Error("Couldn't capture a frame");
  return new File([blob], `webcam-${role}.jpg`, { type: "image/jpeg", lastModified: Date.now() });
}

function setShot(role, file) {
  revokeIfUnused(cam.urls[role]);
  cam.shots[role] = file;
  cam.urls[role] = file ? URL.createObjectURL(file) : null;
  if (role === "before" && !file) setShot("after", null);   // a new Before needs a new After
  renderCam();
  if (file && !mq.still.matches) { const b = $(`.cam-shot[data-role="${role}"]`); b.classList.remove("pop"); void b.offsetWidth; b.classList.add("pop"); }
}

function cycleTimer() {
  cam.timer = CAM_TIMER_STEPS[(CAM_TIMER_STEPS.indexOf(cam.timer) + 1) % CAM_TIMER_STEPS.length];
  store.set("cyws-cam-timer", String(cam.timer));
  renderCam();
}

function cancelCountdown() {
  if (!cam.countTok) return;
  cam.countTok = null;
  $("#cam-count").hidden = true;
  renderCam();
}

/* Self-timer: a dial counts down over the camera; resolves true when it ran out,
   false when it was cancelled (shutter pressed again, camera closed). */
async function countdown(seconds) {
  const tok = cam.countTok = {};
  const box = $("#cam-count"), num = $("#cam-count-n"), ring = $("#cam-count-ring");
  box.hidden = false;
  renderCam();
  for (let n = seconds; n > 0; n--) {
    num.textContent = n;
    num.classList.remove("tick"); void num.offsetWidth; num.classList.add("tick");
    ring.classList.remove("run"); void ring.offsetWidth; ring.classList.add("run");
    box.classList.toggle("last", n === 1);
    await sleep(1000);
    if (cam.countTok !== tok) return false;
  }
  cam.countTok = null;
  box.hidden = true;
  return true;
}

async function camShutter() {
  if (cam.countTok) { cancelCountdown(); return; }
  if (!cam.stream || cam.busy) return;
  const role = !cam.shots.before ? "before" : "after";
  if (cam.timer && !(await countdown(cam.timer))) return;
  if (!cam.stream) return;
  cam.busy = true;
  try {
    const f = $("#cam-flash");
    f.classList.remove("go"); void f.offsetWidth; f.classList.add("go");
    setShot(role, await grabFrame(role));
    if (role === "after") { await sleep(mq.still.matches ? 0 : 420); useShots(); }
  } catch (e) {
    toast(e.message, "error");
  } finally {
    cam.busy = false;
  }
}

/* Both captured: hand them to the composer as two photos and detect. */
function useShots() {
  const { before, after } = cam.shots;
  if (!before || !after) return;
  closeCam();
  setSlotFile("before", before);
  setSlotFile("after", after);
  for (const r of ["before", "after"]) { revokeIfUnused(cam.urls[r]); cam.urls[r] = null; cam.shots[r] = null; }
  if (mq.tablet.matches && !mq.phone.matches) setDrawer(false);
  detect();
}

function startLive() {
  if (!cam.shots.before || !cam.stream || cam.live) return;
  if (state.runAll) state.runAll.stop = true;
  setShot("after", null);
  cam.live = true; cam.liveSrc = null; cam.frames = [];
  $("#cam-fps").textContent = "";
  $("#cam").classList.add("live");
  $(".cam-panel").setAttribute("aria-modal", "false");
  if (mq.phone.matches) setTab("detect");
  if (mq.tablet.matches && !mq.phone.matches) setDrawer(false);
  toast("Watching live. Each new frame is compared with Before.", "ok", 3000);
  liveLoop();
}

function stopLive() {
  if (!cam.live) return;
  cam.live = false;
  $("#cam").classList.remove("live");
  stopStream();
  $("#cam").hidden = true;
  // keep the last live pair in the composer; the next capture starts fresh
  for (const r of ["before", "after"]) { revokeIfUnused(cam.urls[r]); cam.urls[r] = null; cam.shots[r] = null; }
  updateComposer();
}

/* Wait until the run queue is idle (the job asked for and anything queued behind it). */
function runAndWait() {
  return new Promise((res) => { state.idleWaiters.push(res); requestRun(); });
}

async function liveLoop() {
  let fails = 0;
  while (cam.live) {
    try {
      const after = await grabFrame("after");
      const fd = new FormData();
      if (cam.liveSrc) fd.append("before_from", cam.liveSrc); else fd.append("before", cam.shots.before, "webcam-before.jpg");
      fd.append("after", after, "webcam-after.jpg");
      let up;
      try {
        up = await api("/api/upload", { method: "POST", body: fd });
      } catch (e) {
        if (cam.liveSrc && /expired/i.test(e.message)) { cam.liveSrc = null; continue; }   // resend Before
        throw e;
      }
      if (!cam.live) break;
      cam.liveSrc = up.id;
      cam.liveIds.add(up.id);
      const prev = cam.liveUrl;
      cam.liveUrl = URL.createObjectURL(after);
      state.slots.before = { kind: "upload", id: up.id, url: cam.urls.before };
      state.slots.after = { kind: "upload", id: up.id, url: cam.liveUrl };
      state.uploadId = null;
      state.current = { id: up.id, kind: "upload" };
      revokeIfUnused(prev);
      renderSlots();
      updatePairTiles();
      await runAndWait();
      fails = 0;
      const now = performance.now();
      cam.frames.push(now);
      if (cam.frames.length > 6) cam.frames.shift();
      if (cam.frames.length > 1) {
        const fps = (cam.frames.length - 1) / ((now - cam.frames[0]) / 1000);
        $("#cam-fps").textContent = `${fps < 10 ? fps.toFixed(1) : Math.round(fps)} fps`;
      }
    } catch (e) {
      if (!cam.live) break;
      if (++fails >= 3 || e.message === NET_ERR) { toast(`Live detection stopped: ${e.message}`, "error"); stopLive(); break; }
      await sleep(600);
    }
  }
}

function normalizeOpts(d) {
  const o = {};
  if (d.thr != null) o.thr = Number(d.thr);
  if (d.levels != null) o.levels = Array.isArray(d.levels) ? d.levels.join(",") : String(d.levels);
  if (d.register != null) o.register = !!d.register;
  if (d.tiles) o.tiles = d.tiles;
  if (d.mask_mode) o.mask_mode = d.mask_mode;
  return o;
}
/* A preset named in the link: its values apply to every option the link doesn't set explicitly. */
function applyHashPreset() {
  const name = state.opts.preset;
  if (!name || !state.presetsLoaded) return;
  if (!state.presets[name]) { state.opts.preset = null; return; }
  const p = normalizeOpts(state.presets[name]);
  for (const k of Object.keys(p)) if (!state.hashKeys.has(k)) state.opts[k] = p[k];
}
/* The preset whose values equal the current options (highlighted when none was picked). */
function matchingPreset() {
  const o = state.opts;
  return Object.keys(state.presets).find((n) => {
    const p = normalizeOpts(state.presets[n]);
    return Object.keys(p).every((k) => (k === "thr" ? Math.abs(p[k] - o[k]) < 1e-6 : p[k] === o[k]));
  }) || null;
}
function buildPresets() {
  const names = Object.keys(state.presets);
  $("#preset-wrap").hidden = !names.length;
  $("#ctl-preset").innerHTML = names.map((n) => `<button type="button" class="pill" role="radio" data-v="${esc(n)}" aria-checked="false">${esc(cap(n))}</button>`).join("");
}
function syncControls() {
  const o = state.opts;
  const shownPreset = o.preset || matchingPreset();
  $$("#ctl-preset button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.v === shownPreset)));
  $$("#ctl-levels button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.v === o.levels)));
  const thr = $("#ctl-thr");
  thr.value = o.thr;
  thr.style.setProperty("--fill", `${100 * (o.thr - thr.min) / (thr.max - thr.min)}%`);
  $("#thr-v").textContent = Number(o.thr).toFixed(2);
  $("#ctl-register").checked = o.register;
  $("#ctl-gated").checked = o.tiles === "gated";
  $("#ctl-diff").checked = o.mask_mode === "diff";
  $("#ctl-diff").disabled = !o.register;
  $("#set-meta").innerHTML = `${LEVEL_NAME[o.levels] || o.levels} · <b>${Number(o.thr).toFixed(2)}</b>${o.register ? " · Aligned" : ""}`;
}
let thrTimer = null;
function optionsChanged({ debounce = false } = {}) {
  syncControls();
  writeHash();
  clearTimeout(thrTimer);
  if (!state.current || state.runAll) return;
  if (debounce) thrTimer = setTimeout(() => requestRun(), 350);
  else requestRun();
}
function requestOptions() {
  const o = { ...state.opts };
  if (!o.preset) delete o.preset;
  return o;
}

/* ------------------------------------------------------------------ running */
function requestRun() {
  if (!state.current) return;
  state.want = { id: state.current.id, kind: state.current.kind, opts: requestOptions() };
  if (!state.inflight) pump();
}

async function pump() {
  while (state.want) {
    const job = state.want;
    state.want = null;
    state.inflight = job;
    setRunning(true, job);
    try {
      const r = await apiRun(job);
      if (!state.want) await showResult(r);
    } catch (e) {
      if (!state.want) { toast(e.message, "error"); recoverView(); }
    }
  }
  state.inflight = null;
  setRunning(false);
  for (const res of state.idleWaiters.splice(0)) res();
}

/* A run failed: drop the placeholder and fall back to the last result (or the start screen). */
function recoverView() {
  if (!$("#viewer").classList.contains("pending")) return;
  clearPlaceholder();
  const r = state.result;
  if (!r) { $("#viewer").hidden = true; $("#hero").hidden = false; return; }
  setAspect(r.view.w / r.view.h);
  const isPair = state.pairs.some((p) => p.id === r.pair);
  $("#vh-name").textContent = isPair ? pairName(r.pair) : "Your Photos";
  // the viewer fell back to the last result: the composer, gallery and link follow it
  if (!state.current || state.current.id !== r.pair) {
    clearSlotFiles();
    if (isPair) {
      state.slots.before = { kind: "pair", id: r.pair, url: `/api/thumb/${encodeURIComponent(r.pair)}/before` };
      state.slots.after = { kind: "pair", id: r.pair, url: `/api/thumb/${encodeURIComponent(r.pair)}/after` };
      state.current = { id: r.pair, kind: "pair" };
    } else {
      state.slots.before = { kind: "upload", id: r.pair, url: r.images.before };
      state.slots.after = { kind: "upload", id: r.pair, url: r.images.after };
      state.current = { id: r.pair, kind: "upload" };
    }
    state.uploadId = null;
    renderSlots();
    updatePairTiles();
    writeHash();
  }
}

async function apiRun(job) {
  const r = await postJSON("/api/run", { id: job.id, options: job.opts });
  state.pairResults[job.id] = { boxes: r.detections.length, f1: r.metrics ? r.metrics.f1 : null };
  state.winCount[job.opts.levels] = r.windows.length;
  if (r.timings_ms && r.timings_ms.total) state.lastTotal = r.timings_ms.total;
  return r;
}

let stepTimer = null;
function setRunning(on, job) {
  const viewer = $("#viewer");
  updateComposer();
  updatePairTiles();
  clearTimeout(stepTimer);
  if (!on) { viewer.classList.remove("running"); return; }
  if (cam.live && state.result) { showViewer(); return; }
  // Show the viewer straight away: a blurred thumbnail stands in until the result arrives.
  const samePic = state.result && state.result.pair === job.id;
  if (!samePic) {
    const sb = state.slots.before, sa = state.slots.after;
    const thumbA = sa ? sa.url : `/api/thumb/${encodeURIComponent(job.id)}/after`;
    const thumbB = sb ? sb.url : `/api/thumb/${encodeURIComponent(job.id)}/before`;
    showPlaceholder(thumbB, thumbA);
    $("#vh-name").textContent = job.kind === "pair" ? pairName(job.id) : cam.live ? "Webcam · Live" : "Your Photos";
    $("#vh-meta").textContent = job.kind === "pair" ? job.id : "";
    $("#m-summary").hidden = true;
  }
  showViewer();
  viewer.classList.add("running");
  // Indeterminate stage read-out, paced from the last run's wall-clock time.
  const n = state.winCount[job.opts.levels];
  const steps = ["Decoding", job.opts.register ? "Aligning" : null, n ? `MLA · ${n} Window${n === 1 ? "" : "s"}` : "MLA · Global + Tiles", "Merging"].filter(Boolean);
  const share = job.opts.register ? [0.25, 0.2, 0.45, 0.1] : [0.32, 0.55, 0.13];
  const total = clamp(state.lastTotal, 300, 20000);
  let cur = 0;
  const draw = () => {
    $("#run-status").innerHTML = `<span class="rs-spin"></span>` + steps.map((s, i) =>
      `${i ? '<span class="arr">→</span>' : ""}<span class="step ${i < cur ? "done" : i === cur ? "cur" : ""}">${esc(s)}</span>`).join("");
  };
  const next = () => { draw(); if (cur < steps.length - 1) stepTimer = setTimeout(() => { cur += 1; next(); }, total * share[cur]); };
  next();
}

function showPlaceholder(beforeUrl, afterUrl) {
  const viewer = $("#viewer");
  viewer.classList.add("pending");
  for (const [frame, url] of [[$("#frame-before"), beforeUrl], [$("#frame-after"), afterUrl]]) {
    let ph = frame.querySelector(".placeholder");
    if (!ph) { ph = document.createElement("img"); ph.className = "placeholder"; ph.alt = ""; frame.querySelector(".zoom").prepend(ph); }
    ph.onload = () => { if (viewer.classList.contains("pending") && frame.id === "frame-after" && ph.naturalWidth) setAspect(ph.naturalWidth / ph.naturalHeight); };
    ph.src = url;
  }
  resetZoom();
}
function clearPlaceholder() {
  $("#viewer").classList.remove("pending");
  $$(".frame .placeholder").forEach((p) => p.remove());
}
function setAspect(ar) { $$(".frame").forEach((f) => f.style.setProperty("--ar", ar.toFixed(4))); }

function showViewer() {
  $("#hero").hidden = true;
  $("#viewer").hidden = false;
}

function preload(url) {
  return new Promise((resolve) => {
    const im = new Image();
    im.onload = im.onerror = () => resolve();
    im.src = url;
    if (im.decode) im.decode().then(resolve, resolve);
  });
}

async function showResult(r) {
  await Promise.all([preload(r.images.after), preload(r.images.before)]);
  const liveFrame = cam.liveIds.has(r.pair);
  const newPic = (!state.result || state.result.pair !== r.pair) && !(liveFrame && state.result && cam.liveIds.has(state.result.pair));
  state.result = r;
  state.pinned = null; state.hl = null;
  setAspect(r.view.w / r.view.h);
  $("#img-after").src = r.images.after;
  $("#img-before-side").src = r.images.before;
  $("#img-before-swipe").src = r.images.before;
  $("#img-heat").src = r.images.heat;
  $("#img-mask").src = r.images.mask;
  if (r.images.gt) $("#img-gt").src = r.images.gt; else $("#img-gt").removeAttribute("src");
  clearPlaceholder();
  showViewer();
  if (newPic) resetZoom();
  const isPair = state.pairs.some((p) => p.id === r.pair);
  $("#vh-name").textContent = isPair ? pairName(r.pair) : liveFrame ? (cam.live ? "Webcam · Live" : "Webcam") : "Your Photos";
  const reg = r.registration;
  $("#vh-meta").textContent = [isPair ? r.pair : null, `${r.size.w}×${r.size.h}`,
    reg ? (reg.ok ? `Aligned · ${String(reg.method).toUpperCase()} · ${reg.inliers} Inliers` : `Align Failed${reg.reason ? ` (${reg.reason})` : ""}`) : null].filter(Boolean).join(" · ");
  const gtChip = $('.lchip[data-l="gt"]');
  gtChip.disabled = !r.images.gt;
  gtChip.title = r.images.gt ? "Ground-truth outline" : "No ground truth for this pair";
  $(".legend .lg-gt").hidden = !r.images.gt;
  drawOverlay(r, !(liveFrame && cam.live));
  applyLayers();
  renderChips(r);
  renderDetections(r);
  renderMetrics(r);
  renderTimings(r);
  renderSummary(r);
  updatePairTiles();
  writeHash();
}

/* ------------------------------------------------------------------ overlay: boxes, tiles, tags */
function drawOverlay(r, animate) {
  const svg = $("#svg");
  const W = r.size.w, H = r.size.h;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const tiles = r.windows.map((w) => `<rect class="tile l${w.level}" x="${w.x0}" y="${w.y0}" width="${w.w}" height="${w.h}"/>`).join("");
  const dets = r.detections.map((d, i) => {
    const [x1, y1, x2, y2] = d.box;
    const w = Math.max(1, x2 - x1), h = Math.max(1, y2 - y1);
    const lab = LABELS.includes(d.label) ? d.label : "changed";
    return `<g class="det ${lab}" data-i="${i}" style="--p:${(2 * (w + h)).toFixed(1)};--d:${(Math.min(i, 14) * 0.05).toFixed(2)}s">
      <rect class="halo" x="${x1}" y="${y1}" width="${w}" height="${h}"/><rect class="box" x="${x1}" y="${y1}" width="${w}" height="${h}"/></g>`;
  }).join("");
  svg.innerHTML = `<g data-l="tiles">${tiles}</g><g data-l="boxes">${dets}</g>`;
  svg.classList.remove("drawing", "has-hl");
  if (animate && !mq.still.matches) {
    void svg.getBoundingClientRect();
    svg.classList.add("drawing");
    clearTimeout(drawOverlay._t);
    drawOverlay._t = setTimeout(() => svg.classList.remove("drawing"), 1600);
  }
  $("#tags").innerHTML = r.detections.map((d, i) => {
    const lab = LABELS.includes(d.label) ? d.label : "changed";
    return `<div class="tag ${lab}" data-i="${i}" style="--d:${(0.35 + Math.min(i, 14) * 0.05).toFixed(2)}s"><span class="tl">${LABEL_TEXT[lab]}</span><b>${d.score.toFixed(2)}</b></div>`;
  }).join("");
  $("#tags").classList.remove("has-hl");
  layoutOverlay();
}

/* Place label tags in frame pixels: above the box, else inside its top, else below;
   tags on tiny boxes shrink to the score; tags that would collide hide until hovered. */
function layoutOverlay() {
  const r = state.result;
  const frame = $("#frame-after");
  if (!r || !frame.clientWidth) return;
  const fw = frame.clientWidth, fh = frame.clientHeight;
  const k = fw / r.size.w;
  $("#svg").style.setProperty("--u", (r.size.w / (fw * state.zoom.s)).toFixed(4));
  const tags = $$("#tags .tag");
  const order = r.detections.map((d, i) => i).sort((a, b) => r.detections[b].score - r.detections[a].score);
  for (const t of tags) t.classList.remove("mini", "off");
  // Fewer tags on smaller frames (about 5 on a phone, 14 on a desktop); on a crowded phone frame tiny boxes get no tag.
  const cap = clamp(Math.round(fw / 70), 3, 14);
  const minSide = (mq.phone.matches || fw < 520) && r.detections.length > cap ? 18 : 0;
  const rects = r.detections.map((d) => { const [x1, y1, x2, y2] = d.box; return { x: x1 * k, y: y1 * k, w: (x2 - x1) * k, h: (y2 - y1) * k }; });
  const placed = [];
  const over = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  const covered = (a, b) => (Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x))
    * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y))) / Math.max(1, b.w * b.h);
  // a tag may not cover another tag, nor hide much of another box (crossing a big box's edge is fine)
  const hit = (a, self) => placed.some((b) => over(a, b)) || rects.some((b, j) => j !== self && covered(a, b) > 0.35);
  let shown = 0;
  for (const i of order) {
    const t = tags[i];
    if (!t) continue;
    const { x: bx, y: by, w: bw, h: bh } = rects[i];
    if (shown >= cap || Math.min(bw, bh) < minSide) { t.classList.add("off"); placeOff(t, bx, by, fw, fh); continue; }
    if (bw < 40 || bh < 14) t.classList.add("mini");
    const tw = t.offsetWidth || 60, th = t.offsetHeight || 18;
    const x = clamp(bx, 0, Math.max(0, fw - tw));
    const cands = [by - th - 2, by + 2, by + bh + 2].filter((y) => y >= 0 && y + th <= fh);
    let spot = null;
    for (const y of cands) { const c = { x, y, w: tw, h: th }; if (!hit(c, i)) { spot = c; break; } }
    if (spot) shown += 1;
    if (!spot) { t.classList.add("off"); spot = { x, y: clamp(by - th - 2, 0, fh - th), w: tw, h: th }; }
    else placed.push(spot);
    t.style.left = `${(100 * spot.x / fw).toFixed(3)}%`;
    t.style.top = `${(100 * spot.y / fh).toFixed(3)}%`;
  }
}
/* A hidden tag still gets a position, so it appears next to its box when hovered, tapped or picked in the list. */
function placeOff(t, bx, by, fw, fh) {
  const tw = t.offsetWidth || 60, th = t.offsetHeight || 18;
  t.style.left = `${(100 * clamp(bx, 0, Math.max(0, fw - tw)) / fw).toFixed(3)}%`;
  t.style.top = `${(100 * clamp(by - th - 2, 0, Math.max(0, fh - th)) / fh).toFixed(3)}%`;
}

function setHL(i) {
  i = i == null || i === "" ? null : Number(i);
  if (i === state.hl) return;
  state.hl = i;
  const on = i != null;
  $("#svg").classList.toggle("has-hl", on);
  $("#tags").classList.toggle("has-hl", on);
  $$("#svg .det").forEach((g) => g.classList.toggle("hl", Number(g.dataset.i) === i));
  $$("#tags .tag").forEach((t) => t.classList.toggle("hl", Number(t.dataset.i) === i));
  $$(".det-row").forEach((li) => li.classList.toggle("hl", Number(li.dataset.i) === i));
}
function pin(i) {
  state.pinned = state.pinned === i ? null : i;
  setHL(state.pinned);
}

function applyLayers() {
  for (const [k, on] of Object.entries(state.layers)) {
    $$(`#frame-after [data-l="${k}"]`).forEach((el) => el.classList.toggle("hidden-layer", !on));
    const chip = $(`.lchip[data-l="${k}"]`);
    if (chip) chip.setAttribute("aria-pressed", String(on));
  }
  if (!state.result || !state.result.images.gt) $("#img-gt").classList.add("hidden-layer");
}

function setMode(m) {
  if (!["overlay", "side", "swipe"].includes(m)) return;
  state.mode = m;
  $("#stage").dataset.mode = m;
  $$("#mode button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.m === m)));
  resetZoom();
  requestAnimationFrame(() => { layoutOverlay(); applySwipe(); });
  writeHash();
}

/* ------------------------------------------------------------------ zoom / pan / swipe */
const zoomEls = () => $$(".frame .zoom");
function applyZoom() {
  const { s, x, y } = state.zoom;
  for (const z of zoomEls()) {
    z.style.transform = s === 1 ? "" : `translate(${x}px, ${y}px) scale(${s})`;
    z.style.setProperty("--z", s);
  }
  $$(".frame").forEach((f) => f.classList.toggle("zoomed", s > 1));
  $("#btn-zoom-reset").hidden = s === 1;
  if (state.result) $("#svg").style.setProperty("--u", (state.result.size.w / (($("#frame-after").clientWidth || 1) * s)).toFixed(4));
  applySwipe();
}
function clampZoom(frame, z) {
  const w = frame.clientWidth, h = frame.clientHeight;
  if (z.s <= 1.01) return { s: 1, x: 0, y: 0 };
  return { s: z.s, x: clamp(z.x, w - w * z.s, 0), y: clamp(z.y, h - h * z.s, 0) };
}
function zoomAt(frame, cx, cy, ns) {
  const { s, x, y } = state.zoom;
  ns = clamp(ns, 1, 6);
  const lx = (cx - x) / s, ly = (cy - y) / s;
  state.zoom = clampZoom(frame, { s: ns, x: cx - lx * ns, y: cy - ly * ns });
  applyZoom();
}
function resetZoom() { state.zoom = { s: 1, x: 0, y: 0 }; applyZoom(); }

function setSwipe(pctX) {
  state.swipe = clamp(pctX, 0, 100);
  applySwipe();
}
function applySwipe() {
  const frame = $("#frame-after");
  const w = frame.clientWidth || 1;
  const { s, x } = state.zoom;
  frame.style.setProperty("--sw", `${state.swipe}%`);
  const layerPct = clamp(((state.swipe / 100) * w - x) / s / w * 100, 0, 100);
  $("#img-before-swipe").style.setProperty("--clip", `${layerPct}%`);
  $("#swipe-knob").setAttribute("aria-valuenow", String(Math.round(state.swipe)));
}

function bindFrame(frame) {
  const ptrs = new Map();
  let g = null, lastTap = 0, lastTapXY = [0, 0], lastPtr = "mouse";
  const local = (e) => { const r = frame.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  frame.addEventListener("pointerdown", (e) => {
    lastPtr = e.pointerType;
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (e.target.closest(".swipe-knob")) return;
    ptrs.set(e.pointerId, local(e));
    const pts = [...ptrs.values()];
    g = { moved: false, start: local(e), z0: { ...state.zoom }, swipe: frame.id === "frame-after" && state.mode === "swipe" && state.zoom.s === 1 };
    if (pts.length === 2) {
      const [a, b] = pts;
      g.pinch = { d0: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, m0: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
      g.moved = true;
      try { frame.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    } else if (g.swipe && e.pointerType === "mouse") {
      setSwipe(100 * g.start[0] / frame.clientWidth);
    }
  });
  frame.addEventListener("pointermove", (e) => {
    if (!ptrs.has(e.pointerId) || !g) return;
    const p = local(e);
    ptrs.set(e.pointerId, p);
    const pts = [...ptrs.values()];
    if (pts.length >= 2 && g.pinch) {
      const [a, b] = pts;
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const ns = clamp(g.z0.s * d / g.pinch.d0, 1, 6);
      const lx = (g.pinch.m0[0] - g.z0.x) / g.z0.s, ly = (g.pinch.m0[1] - g.z0.y) / g.z0.s;
      state.zoom = clampZoom(frame, { s: ns, x: m[0] - lx * ns, y: m[1] - ly * ns });
      applyZoom();
      e.preventDefault();
      return;
    }
    const dx = p[0] - g.start[0], dy = p[1] - g.start[1];
    if (!g.moved) {
      if (Math.hypot(dx, dy) < 6) return;
      if (state.zoom.s === 1 && !g.swipe) return;                     // let the page scroll
      if (state.zoom.s === 1 && g.swipe && e.pointerType !== "mouse" && Math.abs(dy) > Math.abs(dx)) { g = null; return; }
      g.moved = true;
      frame.classList.add("dragging");
      try { frame.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    }
    if (state.zoom.s > 1) {
      state.zoom = clampZoom(frame, { s: g.z0.s, x: g.z0.x + dx, y: g.z0.y + dy });
      applyZoom();
    } else if (g.swipe) setSwipe(100 * p[0] / frame.clientWidth);
    e.preventDefault();
  });
  const end = (e) => {
    if (!ptrs.has(e.pointerId)) return;
    ptrs.delete(e.pointerId);
    frame.classList.remove("dragging");
    if (!g) return;
    state.suppressClick = g.moved;
    if (!g.moved && e.pointerType !== "mouse" && e.type === "pointerup") {
      const now = performance.now(), p = local(e);
      if (now - lastTap < 320 && Math.hypot(p[0] - lastTapXY[0], p[1] - lastTapXY[1]) < 30) {
        state.zoom.s > 1 ? resetZoom() : zoomAt(frame, p[0], p[1], 2.5);
        lastTap = 0;
        state.suppressClick = true;
      } else { lastTap = now; lastTapXY = p; }
    }
    if (ptrs.size === 0) g = null;
    else if (g.pinch) { g.pinch = null; g.z0 = { ...state.zoom }; g.start = [...ptrs.values()][0]; }
  };
  frame.addEventListener("pointerup", end);
  frame.addEventListener("pointercancel", end);
  frame.addEventListener("dblclick", (e) => {
    if (lastPtr !== "mouse") return;               // touch/pen double-taps are handled in pointerup
    const [x, y] = local(e);
    state.zoom.s > 1 ? resetZoom() : zoomAt(frame, x, y, 2.5);
  });
  frame.addEventListener("wheel", (e) => {
    if (!state.result || mq.phone.matches) return;
    e.preventDefault();
    const [x, y] = local(e);
    zoomAt(frame, x, y, state.zoom.s * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0022)));
  }, { passive: false });
}

function bindSwipeKnob() {
  const knob = $("#swipe-knob"), frame = $("#frame-after");
  knob.addEventListener("pointerdown", (e) => {
    e.preventDefault(); e.stopPropagation();
    try { knob.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    const move = (ev) => { const r = frame.getBoundingClientRect(); setSwipe(100 * (ev.clientX - r.left) / r.width); };
    const up = () => { knob.removeEventListener("pointermove", move); knob.removeEventListener("pointerup", up); knob.removeEventListener("pointercancel", up); };
    knob.addEventListener("pointermove", move);
    knob.addEventListener("pointerup", up);
    knob.addEventListener("pointercancel", up);
  });
  knob.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: -2, ArrowRight: 2, PageDown: -10, PageUp: 10 }[e.key];
    if (d) { setSwipe(state.swipe + (e.shiftKey ? d * 5 : d)); e.preventDefault(); e.stopPropagation(); }
    if (e.key === "Home") { setSwipe(0); e.preventDefault(); }
    if (e.key === "End") { setSwipe(100); e.preventDefault(); }
  });
}

/* fullscreen */
function toggleFullscreen(el, btn) {
  if (document.fullscreenElement) { document.exitFullscreen().catch(() => {}); return; }
  if (el.requestFullscreen && document.fullscreenEnabled) {
    el.requestFullscreen().catch(() => togglePseudoFs(el, btn));
  } else togglePseudoFs(el, btn);
}
function togglePseudoFs(el, btn) {
  const on = el.classList.toggle("pseudo-fs");
  btn.classList.toggle("is-fs", on);
  setTimeout(() => { layoutOverlay(); applySwipe(); syncStrip(); }, 50);
}
function exitViewerFs() {
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  const v = $("#viewer");
  if (v.classList.contains("pseudo-fs")) togglePseudoFs(v, $("#btn-vfs"));
}

/* Phones: full screen / zoom reset float on the image; elsewhere they end the toolbar, outside its scrolling strip. */
function placeViewerCtl() {
  const end = $("#tb-end"), target = mq.phone.matches ? $("#stage") : $("#toolbar");
  if (end.parentElement !== target) target.appendChild(end);
}
function syncStrip() {
  const sc = $("#tb-scroll");
  const max = sc.scrollWidth - sc.clientWidth;
  sc.classList.toggle("ovf-l", max > 2 && sc.scrollLeft > 2);
  sc.classList.toggle("ovf-r", max > 2 && sc.scrollLeft < max - 2);
}

/* ------------------------------------------------------------------ result panels */
function renderChips(r) {
  const t = r.timings_ms || {};
  countTo($("#m-windows"), r.windows.length, fmtInt);
  $("#m-windows-label").textContent = r.windows.length === 1 ? "Window" : "Windows";
  countTo($("#m-total"), t.total, fmtMs);
  countTo($("#m-decode"), t.load, fmtMs);
  if (r.config && r.config.register) countTo($("#m-align"), t.register || 0, fmtMs);
  else setText($("#m-align"), "Off");
  const c = r.counts || {};
  countTo($("#m-app"), c.appeared || 0, fmtInt);
  countTo($("#m-dis"), c.disappeared || 0, fmtInt);
  countTo($("#m-chg"), c.changed || 0, fmtInt);
  const m = r.metrics;
  if (m && m.f1 != null) countTo($("#m-f1"), 100 * m.f1, fmtPct); else setText($("#m-f1"), "—");
  // no annotated regions (a no-change pair) reads "—", like the whole-set table
  setText($("#m-regions"), m && m.gt_components ? `${m.gt_components_hit}<small>/</small>${m.gt_components}` : "—");
  const lat = state.status && state.status.latency && state.status.latency.mean_ms;
  if (lat) countTo($("#m-mla"), lat, fmtMs);
}

function renderDetections(r) {
  const n = r.detections.length;
  $("#det-count").innerHTML = `<b>${n}</b> Found`;
  const c = r.counts || {};
  $("#counts").innerHTML = LABELS.map((k) => `<span class="count-pill"><i class="sw ${k}"></i><b>${c[k] || 0}</b>${LABEL_TEXT[k]}</span>`).join("");
  $("#dets").innerHTML = n ? detRows(r, r.detections.map((d, i) => i))
    : `<li class="empty-row">No changes above the threshold (${Number(r.config ? r.config.thr : state.opts.thr).toFixed(2)}).</li>`;
  // phones show the first 6 rows; the rest sit behind "Show all"
  $("#dets").classList.toggle("capped", n > 6);
  setDetsExpanded(false);
  const more = $("#dets-more");
  more.hidden = n <= 6;
  more.dataset.n = n;
  const badge = $("#tab-badge");
  badge.hidden = n === 0;                          // a "0" would read as a notification
  badge.textContent = n;
}

function setDetsExpanded(on) {
  const more = $("#dets-more");
  $("#dets").classList.toggle("show-all", on);
  more.setAttribute("aria-expanded", String(on));
  more.querySelector("span").textContent = on ? "Show Fewer" : `Show All ${more.dataset.n || ""}`.trim();
}

function detRows(r, idx) {
  const f = Array.isArray(r.scale) ? r.scale[0] : (r.scale || 1);
  return idx.map((i) => {
    const d = r.detections[i];
    const [x1, y1, x2, y2] = d.box;
    const lab = LABELS.includes(d.label) ? d.label : "changed";
    const w = Math.round((x2 - x1) * f), h = Math.round((y2 - y1) * f);
    return `<li class="det-row" data-i="${i}" style="--c: var(--${lab})" tabindex="0" aria-label="${LABEL_TEXT[lab]}, score ${d.score.toFixed(2)}">
      <i class="sw ${lab}"></i>
      <span class="dr-main"><span class="dr-label">${LABEL_TEXT[lab]}</span><span class="dr-meta">${esc(cap(d.branch))} Branch · Level ${d.level} · ${w}×${h} px</span></span>
      <span class="dr-score"><b>${d.score.toFixed(2)}</b><span class="dr-bar"><i style="width:${Math.round(100 * d.score)}%"></i></span></span></li>`;
  }).join("");
}

function renderMetrics(r) {
  const m = r.metrics;
  $("#acc-meta").innerHTML = m && m.f1 != null ? `F1 <b>${pct(m.f1)}</b>` : "";
  if (!m) { $("#metrics").innerHTML = `<p class="empty-row">No ground truth for this pair — accuracy appears for annotated test pairs.</p>`; return; }
  const tile = (v, k, title = "") => `<div class="mtile" title="${esc(title)}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
  const p = (v) => (v == null ? "—" : `${Math.round(100 * v)}<small>%</small>`);
  $("#metrics").innerHTML = [
    tile(p(m.f1), "Pixel F1", "Harmonic mean of pixel precision and recall"),
    tile(p(m.iou), "Pixel IoU"),
    tile(p(m.precision), "Precision"),
    tile(p(m.recall), "Recall"),
    tile(m.gt_components ? `${m.gt_components_hit}<small>/${m.gt_components}</small>` : "<small>None</small>", "Regions", "Annotated regions touched by a detection"),
    tile(p(m.box_precision), "On Truth", "Share of boxes that overlap an annotated change"),
  ].join("");
}

function renderTimings(r) {
  const t = r.timings_ms || {};
  const total = t.total || Object.values(t).reduce((a, b) => a + b, 0);
  const order = ["load", "register", "stream", "decode", "postprocess"];
  const rows = order.filter((k) => t[k] > 0.05).map((k) => [k, t[k]]);
  const known = rows.reduce((a, [, v]) => a + v, 0);
  if (total - known > 1) rows.push(["other", total - known]);
  $("#t-total").innerHTML = `<b>${Math.round(total)}</b> ms · ${r.windows.length} Window${r.windows.length === 1 ? "" : "s"}`;
  $("#tbar").innerHTML = rows.map(([k, v]) => `<div title="${TIME_LABEL[k] || k} · ${v.toFixed(0)} ms" style="width:${(100 * v / total).toFixed(2)}%;background:${TIME_COLOR[k] || TIME_COLOR.other}"></div>`).join("");
  const sub = ["preprocess_cpu", "preprocess_wait", "infer_wait"].filter((k) => t[k] > 0.05);
  const extra = Object.entries(t).filter(([k, v]) => !order.includes(k) && !sub.includes(k) && k !== "total" && v > 0.05);
  const li = ([k, v], cls = "") => `<li class="${cls}"><i style="background:${TIME_COLOR[k] || TIME_COLOR.other}"></i><span>${esc(TIME_LABEL[k] || cap(k.replace(/_/g, " ")))}</span><b>${Number(v).toFixed(0)} ms</b></li>`;
  $("#tlist").innerHTML = rows.map((row) => li(row) + (row[0] === "stream" ? sub.map((k) => li([k, t[k]], "sub")).join("") : "")).join("")
    + extra.map((row) => li(row)).join("");
  const lat = state.status && state.status.latency && state.status.latency.mean_ms;
  $("#t-note").textContent = `Model ${r.model}${lat ? ` · MLA ≈ ${lat < 10 ? lat.toFixed(1) : Math.round(lat)} ms per 256×256 window` : ""}. Cropping the next window overlaps the MLA, so the indented rows overlap "Preprocess + MLA".`;
}

function renderSummary(r) {
  const c = r.counts || {};
  $("#m-summary").hidden = false;
  const m = r.metrics;
  // non-zero counts first; zero counts shrink to dot + 0 so the strip fits beside "Details"
  const labs = [...LABELS].sort((a, b) => (!(c[b] || 0) ? 0 : 1) - (!(c[a] || 0) ? 0 : 1));
  $("#ms-counts").innerHTML = labs.map((k) => { const v = c[k] || 0;
    return `<span class="count-pill${v ? "" : " zero"}" title="${v} ${k}" aria-label="${v} ${k}"><i class="sw ${k}"></i><b>${v}</b><span class="cp-word">${LABEL_TEXT[k]}</span></span>`; }).join("")
    + (m && m.f1 != null ? `<span class="count-pill">F1 <b>${pct(m.f1)}</b></span>` : "");
  const top = r.detections.map((d, i) => i).sort((a, b) => r.detections[b].score - r.detections[a].score).slice(0, 5);
  $("#ms-dets").innerHTML = detRows(r, top);
}

function resetChips() {
  for (const id of ["m-windows", "m-total", "m-decode", "m-align", "m-app", "m-dis", "m-chg", "m-f1", "m-regions"]) setText($(`#${id}`), "—");
  $("#m-windows-label").textContent = "Windows";
  $("#tab-badge").hidden = true;
}

/* ------------------------------------------------------------------ whole test set */
async function runAll() {
  if (state.runAll) { state.runAll.stop = true; return; }
  if (!state.pairs.length) { toast("The test set is empty", "error"); return; }
  if (!state.status || state.status.state !== "ready") { toast("The model is not ready yet", "error"); return; }
  if (cam.live) stopLive();
  state.want = null;
  while (state.inflight) await sleep(50);
  const ids = state.pairs.map((p) => p.id);
  const ra = state.runAll = { stop: false };
  const btn = $("#run-all");
  btn.className = "btn stop block";
  btn.innerHTML = `<svg><use href="#i-stop"/></svg><span>Stop</span>`;
  $("#all-progress").hidden = false; $("#all-sum").hidden = false; $("#all-wrap").hidden = false;
  $("#all-body").innerHTML = "";
  $("#all-prog").style.width = "0%";
  const noChange = state.pairs.filter((p) => p.gt_change === false).length;
  let hit = 0, comps = 0, fa = 0, ms = 0, n = 0, f1s = 0, f1n = 0;
  const summary = () => {
    const tile = (v, k, t = "") => `<div class="mtile" title="${t || k}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
    $("#all-sum").innerHTML = tile(`${hit}<small>/${comps}</small>`, "Regions Found") + tile(`${fa}<small>/${noChange}</small>`, "False Alarms", "No-change pairs with at least one box")
      + tile(n ? `${Math.round(ms / n)}<small>ms</small>` : "—", "Per Pair") + tile(f1n ? `${Math.round(100 * f1s / f1n)}<small>%</small>` : "—", "Mean F1", "Mean pixel F1 over the pairs with a change");
    $("#all-meta").innerHTML = `<b>${n}</b>/${ids.length}`;
  };
  summary();
  updateComposer();
  for (const id of ids) {
    if (ra.stop) break;
    openPair(id, { run: false, fromRunAll: true });
    const job = { id, kind: "pair", opts: requestOptions() };
    state.inflight = job;
    setRunning(true, job);
    let r;
    // inflight stays set until the result is on screen, so a click can't start a second run alongside the batch
    try {
      r = await apiRun(job);
      if (!state.want) await showResult(r);      // a pair opened meanwhile wins the viewer
    } catch (e) { toast(`${id}: ${e.message}`, "error"); continue; }
    finally { state.inflight = null; }
    const p = state.pairs.find((x) => x.id === id);
    const m = r.metrics || {};
    n += 1; ms += r.timings_ms.total || 0;
    if (m.gt_components != null) { hit += m.gt_components_hit; comps += m.gt_components; }
    if (p && p.gt_change && m.f1 != null) { f1s += m.f1; f1n += 1; }
    if (p && p.gt_change === false && r.detections.length) fa += 1;
    const tr = document.createElement("tr");
    tr.dataset.id = id;
    tr.tabIndex = 0;
    const f1 = m.f1 == null ? "—" : m.f1.toFixed(2);
    const cls = m.f1 == null ? "" : m.f1 >= 0.5 ? "good" : m.f1 < 0.2 && p && p.gt_change ? "bad" : "";
    const boxCls = p && p.gt_change === false ? (r.detections.length ? "bad" : "good") : "";
    tr.innerHTML = `<td>${esc(id.replace(/^2024/, ""))}</td><td><span class="tdot ${p && p.gt_change ? "" : "none"}" title="${p && p.gt_change ? "Change" : "No Change"}"></span></td>
      <td class="${boxCls}">${r.detections.length}</td><td class="${cls}">${f1}</td><td>${m.gt_components ? `${m.gt_components_hit}/${m.gt_components}` : "—"}</td><td>${Math.round(r.timings_ms.total || 0)}</td>`;
    tr.title = `${pairName(id)} (${id})`;
    $("#all-body").appendChild(tr);
    $("#all-prog").style.width = `${(100 * n / ids.length).toFixed(1)}%`;
    summary();
  }
  state.runAll = null;
  setRunning(false);
  btn.className = "btn spectrum block";
  btn.innerHTML = `<svg><use href="#i-play"/></svg><span>Run All Pairs Again</span>`;
  updateComposer();
  if (n) toast(`${ra.stop ? "Stopped" : "Done"} — ${n} pairs · regions found ${hit}/${comps} · false alarms ${fa}/${noChange} · ${Math.round(ms / n)} ms/pair`, "ok", 6000);
  // a pair opened during the batch stopped it and queued itself: run it now
  if (state.want && !state.inflight) pump();
}

/* ------------------------------------------------------------------ deep links */
const BASE_OPTS = { preset: null, levels: "0,1", thr: 0.3, register: false, tiles: "full", mask_mode: "box" };
const BASE_LAYERS = { boxes: true, heat: true, mask: false, gt: false, tiles: false };
let hashReading = false;
function readHash(opts) {
  hashReading = true;                              // setMode() must not rewrite the link while it is being read
  try { return readHashInner(opts); } finally { hashReading = false; }
}
function readHashInner({ reset = false } = {}) {
  const h = new URLSearchParams(location.hash.slice(1));
  const o = state.opts;
  const keys = state.hashKeys = new Set();
  if (reset) {
    // an edited link starts from the defaults, so keys it leaves out don't keep stale values
    Object.assign(o, BASE_OPTS, state.defaults || {}, { preset: null });
    if (h.get("layers") == null) Object.assign(state.layers, BASE_LAYERS);
    if (!h.get("mode")) setMode("overlay");
  }
  if (h.get("levels") && LEVEL_NAME[h.get("levels")]) { o.levels = h.get("levels"); keys.add("levels"); }
  if (h.get("thr") && isFinite(+h.get("thr"))) { o.thr = clamp(+h.get("thr"), 0.05, 0.9); keys.add("thr"); }
  if (h.has("register")) { o.register = ["1", "true", "yes"].includes(h.get("register")); keys.add("register"); }
  if (h.get("tiles")) { o.tiles = h.get("tiles") === "gated" ? "gated" : "full"; keys.add("tiles"); }
  if (h.get("masks") || h.get("mask_mode")) { o.mask_mode = (h.get("masks") || h.get("mask_mode")) === "diff" ? "diff" : "box"; keys.add("mask_mode"); }
  if (h.get("preset")) { o.preset = h.get("preset"); applyHashPreset(); }
  if (h.get("layers") != null) {
    const on = h.get("layers").split(",");
    for (const k of Object.keys(state.layers)) state.layers[k] = on.includes(k);
  }
  if (h.get("mode")) setMode(h.get("mode"));
  return h.get("pair");
}
let hashReady = false;
function writeHash() {
  if (!hashReady || hashReading) return;
  const o = state.opts, p = new URLSearchParams();
  if (state.current && state.current.kind === "pair") p.set("pair", state.current.id);
  p.set("levels", o.levels);
  p.set("thr", Number(o.thr).toFixed(2));
  if (o.register) p.set("register", "1");
  if (o.tiles === "gated") p.set("tiles", "gated");
  if (o.mask_mode === "diff") p.set("masks", "diff");
  if (o.preset) p.set("preset", o.preset);
  if (state.mode !== "overlay") p.set("mode", state.mode);
  p.set("layers", Object.keys(state.layers).filter((k) => state.layers[k]).join(","));
  const h = "#" + p.toString().replace(/%2C/gi, ",");
  if (h !== location.hash) history.replaceState(null, "", h);
}

/* ------------------------------------------------------------------ navigation: tabs, drawer, sheet */
function setTab(t) {
  document.body.dataset.tab = t;
  $$("#tabbar button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
  $("#rail").scrollTop = 0;
  if (t === "detect") requestAnimationFrame(() => { layoutOverlay(); applySwipe(); });
}
function setDrawer(open) {
  const ws = $("#workspace");
  ws.classList.toggle("rail-open", open);
  $("#scrim").hidden = !open;
  $("#btn-menu").setAttribute("aria-expanded", String(open));
  $("#btn-menu").setAttribute("aria-label", open ? "Close panel" : "Open panel");
  if (open) setTimeout(() => { const f = $("#pairs .pair[tabindex='0']") || $("#rail button"); if (f) f.focus({ preventScroll: true }); }, 320);
}

let sheetReturn = null;
function openSheet(title, items, aboutHtml = "") {
  const sheet = $("#sheet");
  sheetReturn = document.activeElement;
  $("#sheet-title").textContent = title;
  const body = $("#sheet-body");
  body.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `sheet-item${it.danger ? " danger" : ""}`;
    b.innerHTML = `<svg><use href="#${it.icon}"/></svg><span></span>`;
    b.querySelector("span").textContent = it.label;
    b.addEventListener("click", () => { closeSheet(); setTimeout(it.run, 30); });
    body.appendChild(b);
  }
  if (aboutHtml) body.insertAdjacentHTML("beforeend", `<div class="sheet-about">${aboutHtml}</div>`);
  sheet.hidden = false;
  const first = body.querySelector("button");
  if (first) first.focus({ preventScroll: true });
}
function closeSheet() {
  $("#sheet").hidden = true;
  if (sheetReturn && sheetReturn.focus) sheetReturn.focus({ preventScroll: true });
}

function goHome() {
  if (cam.live) stopLive();
  exitViewerFs();
  if (state.runAll) state.runAll.stop = true;
  state.want = null;
  state.result = null;
  state.current = null;
  clearSlotFiles();
  renderSlots();
  updatePairTiles();
  $("#viewer").hidden = true;
  $("#viewer").classList.remove("running", "pending");
  $("#hero").hidden = false;
  $("#m-summary").hidden = true;
  resetChips();
  if (mq.phone.matches) setTab("detect");
  writeHash();
}

async function copyLink() {
  writeHash();
  try {
    await navigator.clipboard.writeText(location.href);
    toast("Link copied", "ok", 2500);
  } catch (e) {
    toast(location.href, "info", 8000);
  }
}

/* ------------------------------------------------------------------ cards */
function initCards() {
  let saved = {};
  try { saved = JSON.parse(store.get("cyws-cards", "{}")) || {}; } catch (e) { saved = {}; }
  for (const card of $$(".rail .card")) {
    const key = card.dataset.card;
    const btn = card.querySelector(".card-toggle");
    if (key in saved) setCard(card, !saved[key]);
    btn.addEventListener("click", () => {
      if (mq.phone.matches) return;
      const open = card.classList.contains("collapsed");
      setCard(card, !open);
      saved[key] = open;
      store.set("cyws-cards", JSON.stringify(saved));
    });
  }
}
function setCard(card, collapsed) {
  card.classList.toggle("collapsed", collapsed);
  card.querySelector(".card-toggle").setAttribute("aria-expanded", String(!collapsed));
}

/* ------------------------------------------------------------------ events */
function bind() {
  $("#btn-theme").addEventListener("click", toggleTheme);
  mq.dark.addEventListener("change", syncThemeMeta);
  $("#btn-home").addEventListener("click", goHome);
  $("#brand").addEventListener("click", (e) => { e.preventDefault(); goHome(); });
  $("#btn-link").addEventListener("click", copyLink);
  $("#btn-fs").addEventListener("click", () => toggleFullscreen(document.documentElement, $("#btn-fs")));
  $("#btn-vfs").addEventListener("click", () => toggleFullscreen($("#viewer"), $("#btn-vfs")));
  $("#btn-fs-close").addEventListener("click", exitViewerFs);
  $("#tb-scroll").addEventListener("scroll", syncStrip, { passive: true });
  document.addEventListener("fullscreenchange", () => {
    const fs = document.fullscreenElement;
    $("#btn-fs").classList.toggle("is-fs", fs === document.documentElement);
    $("#btn-vfs").classList.toggle("is-fs", fs === $("#viewer"));
    setTimeout(() => { layoutOverlay(); applySwipe(); }, 60);
  });
  $("#status-pill").addEventListener("click", () => openSettings());
  $("#hero-model").addEventListener("click", () => openSettings());
  $("#btn-menu").addEventListener("click", () => {
    if (mq.phone.matches) {
      openSheet("Neat Change Studio", [
        { icon: "i-home", label: "Back to Start", run: goHome },
        { icon: "i-link", label: "Copy a Link to This View", run: copyLink },
        { icon: "i-play", label: "Run the Whole Test Set", run: () => { setTab("results"); runAll(); } },
      ], `CYWS (Sachdeva &amp; Zisserman, WACV 2023; Apache-2.0) · Powered by <strong>Neat</strong> · © 2026 SiMa.ai, Inc.`);
    } else setDrawer(!$("#workspace").classList.contains("rail-open"));
  });
  $("#scrim").addEventListener("click", () => setDrawer(false));
  $$("#sheet [data-close]").forEach((el) => el.addEventListener("click", closeSheet));
  mq.tablet.addEventListener("change", () => { setDrawer(false); requestAnimationFrame(layoutOverlay); });
  mq.phone.addEventListener("change", () => { if (!mq.phone.matches) document.body.dataset.tab = "detect"; placeViewerCtl(); renderSlots(); requestAnimationFrame(() => { layoutOverlay(); applySwipe(); syncStrip(); }); });

  // tabs
  $("#tabbar").addEventListener("click", (e) => { const b = e.target.closest("button[data-tab]"); if (b) setTab(b.dataset.tab); });
  $("#ms-more").addEventListener("click", () => setTab("results"));
  $("#dets-more").addEventListener("click", () => setDetsExpanded(!$("#dets").classList.contains("show-all")));

  // gallery
  $("#pairs").addEventListener("click", (e) => { const b = e.target.closest(".pair"); if (b) openPair(b.dataset.id); });
  $("#pairs").addEventListener("keydown", (e) => {
    if (["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) {
      e.preventDefault(); e.stopPropagation();
      moveGalleryFocus(e.key, e.target.closest(".pair"));
    }
  });
  $("#filters").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-f]");
    if (!b) return;
    state.filter = b.dataset.f;
    $$("#filters .pill").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-checked", String(x === b)); });
    renderPairs();
  });
  $("#suggest-row").addEventListener("click", (e) => { const b = e.target.closest(".suggest-chip"); if (b) openPair(b.dataset.id); });

  // composer + uploads card
  for (const el of $$(".slot, .drop")) {
    const role = el.dataset.role;
    el.addEventListener("click", (e) => { if (e.target.closest(".slot-clear")) return; slotActivate(role); });
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); slotActivate(role); } });
    el.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); el.classList.add("over"); });
    el.addEventListener("dragleave", () => el.classList.remove("over"));
    el.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); el.classList.remove("over"); $("#composer").classList.remove("over"); addFiles(e.dataTransfer.files, e.dataTransfer.files.length > 1 ? null : role); });
  }
  $$(".slot-clear").forEach((b) => b.addEventListener("click", (e) => { e.stopPropagation(); clearSlot(b.closest(".slot").dataset.role); }));
  const comp = $("#composer");
  comp.addEventListener("dragover", (e) => { e.preventDefault(); comp.classList.add("over"); });
  comp.addEventListener("dragleave", (e) => { if (!comp.contains(e.relatedTarget)) comp.classList.remove("over"); });
  comp.addEventListener("drop", (e) => { e.preventDefault(); comp.classList.remove("over"); addFiles(e.dataTransfer.files, null); });
  comp.addEventListener("submit", (e) => { e.preventDefault(); detect(); });
  $("#btn-gallery").addEventListener("click", () => pickFiles(null, false));
  $("#btn-camera").addEventListener("click", () => pickFiles(null, true));
  $("#file-pick").addEventListener("change", onPicked);
  $("#file-cam").addEventListener("change", onPicked);
  $("#up-go").addEventListener("click", detect);

  // webcam
  $("#btn-webcam").addEventListener("click", openCam);
  $("#up-webcam").addEventListener("click", openCam);
  $("#cam-close").addEventListener("click", closeCam);
  $$("#cam [data-cam-close]").forEach((el) => el.addEventListener("click", closeCam));
  $("#cam-shutter").addEventListener("click", camShutter);
  $("#cam-live").addEventListener("click", startLive);
  $("#cam-stop").addEventListener("click", stopLive);
  $("#cam-flip").addEventListener("click", () => { if (cam.devices.length > 1) { cam.dev = (cam.dev + 1) % cam.devices.length; startStream(); } });
  $("#cam-ghost-btn").addEventListener("click", () => { cam.ghost = !cam.ghost; renderCam(); });
  $("#cam-timer").addEventListener("click", cycleTimer);
  $$(".cam-shot").forEach((b) => b.addEventListener("click", () => { if (cam.shots[b.dataset.role]) setShot(b.dataset.role, null); }));
  $("#cam-msg").addEventListener("click", (e) => { if (e.target.closest("[data-cam-retry]")) startStream(); });
  $("#cam").addEventListener("keydown", (e) => {
    if (e.key === " " && !cam.live && !e.target.closest("button, a")) { e.preventDefault(); camShutter(); }
  });
  $("#msw-close").addEventListener("click", mswHide);

  // settings
  $("#ctl-preset").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-v]");
    if (!b) return;
    state.opts.preset = b.dataset.v;
    Object.assign(state.opts, normalizeOpts(state.presets[b.dataset.v] || {}));
    optionsChanged();
  });
  $("#ctl-levels").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-v]");
    if (!b || b.dataset.v === state.opts.levels) return;
    state.opts.levels = b.dataset.v; state.opts.preset = null; optionsChanged();
  });
  $("#ctl-levels").addEventListener("keydown", (e) => {
    const vals = ["0", "0,1", "0,1,2"], i = vals.indexOf(state.opts.levels);
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!d) return;
    e.preventDefault(); e.stopPropagation();
    state.opts.levels = vals[clamp(i + d, 0, 2)]; state.opts.preset = null; optionsChanged();
    $(`#ctl-levels button[data-v="${state.opts.levels}"]`).focus();
  });
  $("#ctl-thr").addEventListener("input", (e) => {
    state.opts.thr = Number(e.target.value); state.opts.preset = null;
    optionsChanged({ debounce: true });
  });
  $("#ctl-register").addEventListener("change", (e) => {
    state.opts.register = e.target.checked; state.opts.preset = null;
    if (!state.opts.register) state.opts.mask_mode = "box";
    optionsChanged();
  });
  $("#ctl-gated").addEventListener("change", (e) => { state.opts.tiles = e.target.checked ? "gated" : "full"; state.opts.preset = null; optionsChanged(); });
  $("#ctl-diff").addEventListener("change", (e) => { state.opts.mask_mode = e.target.checked ? "diff" : "box"; optionsChanged(); });
  $("#models").addEventListener("click", (e) => {
    const row = e.target.closest(".model-opt");
    if (!row) return;
    const act = e.target.closest("[data-act]");
    if (act) { hubAction(act.dataset.act, row.dataset.name); return; }
    if (row.dataset.onboard === "1") switchModel(row.dataset.name);
  });
  $("#models").addEventListener("keydown", (e) => {
    const row = e.target.closest(".model-opt");
    if (row && e.target === row && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault(); e.stopPropagation();
      if (row.dataset.onboard === "1") switchModel(row.dataset.name);
    }
  });
  $("#btn-settings").addEventListener("click", toggleSettings);
  $("#rail-toggle").addEventListener("click", () => setRailCollapsed(!railCollapsed()));

  // viewer
  $("#mode").addEventListener("click", (e) => { const b = e.target.closest("button[data-m]"); if (b) setMode(b.dataset.m); });
  $("#layers").addEventListener("click", (e) => {
    const b = e.target.closest(".lchip");
    if (!b || b.disabled) return;
    state.layers[b.dataset.l] = !state.layers[b.dataset.l];
    applyLayers();
    writeHash();
  });
  $("#btn-zoom-reset").addEventListener("click", resetZoom);
  bindFrame($("#frame-after"));
  bindFrame($("#frame-before"));
  bindSwipeKnob();
  const overlayHover = (e) => { const t = e.target.closest("[data-i]"); setHL(t ? t.dataset.i : state.pinned); };
  for (const el of [$("#svg"), $("#tags")]) {
    el.addEventListener("pointerover", (e) => { if (e.pointerType === "mouse") overlayHover(e); });
    el.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") setHL(state.pinned); });
    el.addEventListener("click", (e) => {
      if (state.suppressClick) { state.suppressClick = false; return; }
      const t = e.target.closest("[data-i]");
      if (t) { e.stopPropagation(); pin(Number(t.dataset.i)); }
    });
  }
  $("#frame-after").addEventListener("click", (e) => {
    if (state.suppressClick) { state.suppressClick = false; return; }
    if (!e.target.closest("[data-i]") && state.pinned != null) pin(state.pinned);
  });
  for (const dets of [$("#dets"), $("#ms-dets")]) {
    dets.addEventListener("pointerover", (e) => { if (e.pointerType === "mouse") { const li = e.target.closest(".det-row"); setHL(li ? li.dataset.i : state.pinned); } });
    dets.addEventListener("pointerleave", () => setHL(state.pinned));
    dets.addEventListener("focusin", (e) => { const li = e.target.closest(".det-row"); if (li) setHL(li.dataset.i); });
    dets.addEventListener("focusout", () => setHL(state.pinned));
    const openDet = (li) => {
      const i = Number(li.dataset.i);
      state.pinned = null;
      pin(i);
      if (mq.phone.matches) setTab("detect");
    };
    dets.addEventListener("click", (e) => { const li = e.target.closest(".det-row"); if (li) openDet(li); });
    dets.addEventListener("keydown", (e) => { const li = e.target.closest(".det-row"); if (li && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); e.stopPropagation(); openDet(li); } });
  }

  // whole set
  $("#run-all").addEventListener("click", runAll);
  $("#all-body").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) openPair(tr.dataset.id); });
  $("#all-body").addEventListener("keydown", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr && e.key === "Enter") { e.stopPropagation(); openPair(tr.dataset.id); } });

  // keyboard
  document.addEventListener("keydown", (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === "Escape") {
      if (!$("#mswitch").hidden && !$("#msw-close").hidden) { mswHide(); return; }
      if (!$("#cam").hidden && !cam.live) { closeCam(); return; }
      if (!$("#sheet").hidden) { closeSheet(); return; }
      if ($("#workspace").classList.contains("rail-open")) { setDrawer(false); return; }
      if ($("#viewer").classList.contains("pseudo-fs")) { togglePseudoFs($("#viewer"), $("#btn-vfs")); return; }
      if (state.zoom.s > 1) { resetZoom(); return; }
      if (state.pinned != null) { pin(state.pinned); return; }
      return;
    }
    const el = e.target;
    if (el.closest && el.closest("input, textarea, select, [contenteditable], .sheet, .cam, .mswitch")) return;
    if (["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"].includes(e.key)) {
      if (el !== document.body && el.closest && el.closest("button, [role=slider], [tabindex]") && !el.closest("#hero, .viewer")) return;
      if (!$("#pairs").offsetParent) return;
      e.preventDefault();
      moveGalleryFocus(e.key, null);
    } else if (e.key === "Enter" && (el === document.body || (el.closest && el.closest("#hero") && !el.closest("button")))) {
      e.preventDefault();
      detect();
    }
  });

  // layout changes → re-place tags
  const ro = new ResizeObserver(() => requestAnimationFrame(() => {
    if (state.zoom.s > 1) { state.zoom = clampZoom($("#frame-after"), state.zoom); applyZoom(); }
    layoutOverlay(); applySwipe();
  }));
  ro.observe($("#frame-after"));
  new ResizeObserver(() => requestAnimationFrame(syncStrip)).observe($("#tb-scroll"));
  addEventListener("hashchange", () => {
    const before = JSON.stringify(requestOptions());
    const pair = readHash({ reset: true });
    syncControls(); applyLayers();
    if (pair && (!state.current || state.current.id !== pair) && state.pairs.some((p) => p.id === pair)) openPair(pair);
    else if (state.current && !state.runAll && JSON.stringify(requestOptions()) !== before) requestRun();
    writeHash();
  });
}

function openSettings() {
  if (mq.phone.matches) { setTab("settings"); return; }
  if (mq.tablet.matches) setDrawer(true);
  else if (railCollapsed()) setRailCollapsed(false);
  refreshHub();
  const card = $('.card[data-card="settings"]');
  setCard(card, false);
  card.scrollIntoView({ block: "start", behavior: mq.still.matches ? "auto" : "smooth" });
}

/* ------------------------------------------------------------------ start */
async function waitReady() {
  for (let i = 0; i < 180; i++) {
    const s = state.status;
    if (s && (s.state === "ready" || s.state === "error")) return s;
    await sleep(1000);
    await refreshStatus();
  }
  return state.status;
}

(async function start() {
  syncThemeMeta();
  placeViewerCtl();
  initCards();
  bind();
  setRailCollapsed(store.get("cyws-rail") === "collapsed", { remember: false });
  mq.desktop.addEventListener("change", () => setRailCollapsed(railCollapsed(), { remember: false }));
  const pair = readHash();
  syncControls();
  applyLayers();
  renderSlots();
  renderSuggest();
  await refreshStatus();
  await loadPairs();
  refreshHub();
  hashReady = true;
  setInterval(() => { if (!document.hidden) refreshStatus(); }, 4000);
  if (pair && state.pairs.some((p) => p.id === pair)) {
    openPair(pair, { run: false });
    const s = await waitReady();
    if (s && s.state === "ready") requestRun();
  }
})();
