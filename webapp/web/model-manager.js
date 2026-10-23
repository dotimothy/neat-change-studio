// Models: the model manager, a window over the page (like Neat Decision Studio's settings).
//
// Models is what is on this board (load, unload, verify, re-download, delete, storage); Add Model
// is what Hugging Face lists that is not here yet (download, with progress, cancel and resume);
// About says what is running. One model is on the MLA at a time.
//
//   ModelManager.open()                 open it on Models
//   ModelManager.open("add", "bf16")    another tab ("models" | "add" | "about"), pointing at a model
//   ModelManager.configure({ switchModel, confirm, toast, getStatus })   hooks from the page
//
// #models or #models=add opens it as the page loads. Events on window: "models-closed", and
// "models-changed" (detail {active, family}) after a load, unload, delete or finished download.
// Reads GET /api/status and GET /api/hub; acts through POST /api/models/unload, /api/hub/<action>
// and the page's switchModel hook (which keeps the full-screen switch animation).
"use strict";
(() => {
  const SVGNS = "http://www.w3.org/2000/svg";
  const el = (tag, props = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "className" || k === "textContent" || k === "onclick" || k === "disabled" || k === "id" || k === "href" || k === "src") node[k] = v;
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const child of children) if (child != null && child !== false && child !== "") node.append(child);
    return node;
  };
  const icon = (id, cls = "") => {
    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("aria-hidden", "true");
    if (cls) svg.setAttribute("class", cls);
    const use = document.createElementNS(SVGNS, "use");
    use.setAttribute("href", "#" + id);
    svg.append(use);
    return svg;
  };
  const link = (href, text) => el("a", { href, target: "_blank", rel: "noopener" }, text, icon("i-external"));
  const mb = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${Math.round(b / 1e6)} MB`);
  const left = (s) => (s == null ? "" : s >= 90 ? `${Math.round(s / 60)} min left` : `${Math.max(1, Math.round(s))} s left`);
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

  // What is known about the built-in models without asking Hugging Face (offline, or not listed).
  const BUILTIN = {
    "bf16": {
      title: "CYWS bf16", family: "cyws", precision: "BF16", latency_ms: 35.2, repo: "dotimothy/sima-cyws",
      file: "cyws_bf16_mpk.tar.gz", about: "The most faithful build: bfloat16 weights and activations.",
      accuracy: { changes_found: "33 / 47", pixel_f1: 0.587, false_alarm_pairs: "2 / 15" },
      card: { architecture: "CYWS (The Change You Want to See, WACV 2023): Siamese ResNet-50 U-Net, co-attention, CenterNet head; 49.5M parameters",
              input: "Before + after, 256×256 RGB each, run on a global view and tiled windows",
              upstream: "ragavsachdeva/The-Change-You-Want-to-See", license: "Apache-2.0" },
    },
    "w8a16": {
      title: "CYWS w8a16", family: "cyws", precision: "A_BF16_W_INT8", latency_ms: 27.2, repo: "dotimothy/sima-cyws",
      file: "cyws_w8a16_mpk.tar.gz", about: "Faster: int8 weights, bfloat16 activations.",
      accuracy: { changes_found: "31 / 47", pixel_f1: 0.581, false_alarm_pairs: "3 / 15" },
      card: { architecture: "CYWS (The Change You Want to See, WACV 2023): Siamese ResNet-50 U-Net, co-attention, CenterNet head; 49.5M parameters",
              input: "Before + after, 256×256 RGB each, run on a global view and tiled windows",
              upstream: "ragavsachdeva/The-Change-You-Want-to-See", license: "Apache-2.0" },
    },
    "dinov2-448": {
      title: "DINOv2 448", family: "dinov2", precision: "BF16", latency_ms: 24.5, repo: "dotimothy/sima-dinov2",
      file: "dinov2_vits14_reg_448_bf16_mpk.tar.gz", bytes: 76918681,
      about: "Zero-shot: compares DINOv2 features of the two photos, with no change-detection training.",
      speed: "24.5 ms per image on the MLA, 50 images/s, 14.0 W",
      accuracy: { changes_found: "28 / 47", pixel_f1: 0.614, false_alarm_pairs: "2 / 15" },
      accuracy_note: "With 0.6 tiles. Whole photo: 27 / 47 found, F1 0.566, 1 / 15 false-alarm pairs. Settings were tuned on the same pairs, so expect less on new scenes.",
      card: { architecture: "DINOv2 ViT-S/14 with 4 registers, one MLA ELF; changes from patch features, zero-shot",
              input: "One 448×448 image → 32×32 patches; run on each photo and its tiles",
              upstream: "facebookresearch/dinov2", license: "Apache-2.0 (Meta)" },
    },
    "c3po-pcd": {
      title: "C-3PO PCD", family: "c3po", precision: "BF16", latency_ms: 46.7,
      file: "c3po_pcd_256_bf16_mpk.tar.gz", bytes: 166260609,
      about: "Trained change segmentation, on PCD street panoramas. Marks what changed, without telling appeared from gone.",
      speed: "46.7 ms per 256×256 pair on the MLA (Model.benchmark), 24.5 pairs/s, 12.8 W; a plain pyneat push → pull loop took about 447 ms per pair",
      accuracy: { changes_found: "28 / 47", pixel_f1: 0.518, false_alarm_pairs: "2 / 15" },
      accuracy_note: "Balanced preset (gated 0.6 tiles), FP32. Accurate (all tiles): 30 / 47 found, F1 0.554. Whole photo: 26 / 47 found, F1 0.499, 1 / 15 false-alarm pairs. Trained on street scenes, not photos like these; thresholds tuned on the same pairs.",
      card: { architecture: "C-3PO (How to Reduce Change Detection to Semantic Segmentation, Pattern Recognition 2023): ResNet-18 shared by both photos, MTF, MSF over 4 scales, DeepLabV3 head; 44.7M parameters; one MLA ELF",
              input: "Before + after, 256×256 RGB each → change probability at 64×64; run on the whole photo and its tiles",
              upstream: "DoctorKey/C-3PO", license: "MIT" },
    },
    "c3po-csb": {
      title: "C-3PO ChangeSim", family: "c3po", precision: "BF16", latency_ms: 46.7,
      file: "c3po_csb_256_bf16_mpk.tar.gz", bytes: 166573893,
      about: "Trained change segmentation, on ChangeSim (a simulated indoor factory). Marks what changed, without telling appeared from gone.",
      speed: "46.7 ms per 256×256 pair on the MLA (Model.benchmark), 24.5 pairs/s, 12.8 W; a plain pyneat push → pull loop took about 447 ms per pair",
      accuracy: { changes_found: "19 / 47", pixel_f1: 0.524, false_alarm_pairs: "2 / 15" },
      accuracy_note: "Balanced preset (gated 0.6 tiles), FP32. Accurate (all tiles): 22 / 47 found, F1 0.536. Whole photo: 19 / 47 found, F1 0.524, 2 / 15 false-alarm pairs. Trained on simulated indoor scenes, not photos like these; thresholds tuned on the same pairs.",
      card: { architecture: "C-3PO (How to Reduce Change Detection to Semantic Segmentation, Pattern Recognition 2023): ResNet-18 shared by both photos, MTF, MSF over 4 scales, DeepLabV3 head; 44.7M parameters; one MLA ELF",
              input: "Before + after, 256×256 RGB each → change probability at 64×64; run on the whole photo and its tiles",
              upstream: "DoctorKey/C-3PO", license: "MIT" },
    },
  };
  const FAMILY = { cyws: { pill: "Trained", per: "Window", title: "CYWS" }, dinov2: { pill: "Zero-Shot", per: "Image", title: "DINOv2" },
    c3po: { pill: "Trained", per: "Pair", title: "C-3PO" } };
  const PRECISION = { BF16: "bfloat16 weights and activations", A_BF16_W_INT8: "int8 weights, bfloat16 activations" };
  const PREC_CHIP = { BF16: "bf16", A_BF16_W_INT8: "w8a16" };
  const GITHUB = "https://github.com/dotimothy/neat-change-studio";

  // Hooks from the page; each has a plain fallback so the module also works alone.
  const hooks = {
    switchModel: (name) => post("/api/models/load", { name }),
    confirm: ({ title, body }) => Promise.resolve(window.confirm(`${title}\n\n${body}`)),
    toast: () => {},
    getStatus: () => null,
  };

  let status = null, hub = null;        // GET /api/status, GET /api/hub
  let busy = null;                      // {name, action}: the action this window is waiting on
  let switching = null;                 // the model a Load is switching to
  let said = "";                        // what the last refused action answered
  let wanted = null, pointed = false;   // a model to point at once drawn
  let downloading = null;               // the model being downloaded at the last read
  let timer = null, drawn = "", opener = null, tab = "models", opened = false;

  async function post(path, data) {
    let r;
    try { r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) }); }
    catch (e) { throw new Error("Can't reach the board — check the connection"); }
    let body = null;
    try { body = await r.json(); } catch (e) { /* not JSON */ }
    if (!r.ok) throw new Error((body && body.error) || `${r.status} ${r.statusText}`);
    return body;
  }
  const getJSON = (path) => fetch(path, { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); });

  // ---------------------------------------------------------------------------- the window
  const panels = {
    models: el("section", { className: "mmgr-panel", role: "tabpanel", id: "mmgr-p-models", "aria-labelledby": "mmgr-t-models", tabindex: "-1" }),
    add: el("section", { className: "mmgr-panel", role: "tabpanel", id: "mmgr-p-add", "aria-labelledby": "mmgr-t-add", tabindex: "-1" }),
    about: el("section", { className: "mmgr-panel mmgr-about", role: "tabpanel", id: "mmgr-p-about", "aria-labelledby": "mmgr-t-about", tabindex: "-1" }),
  };
  const tabs = el("nav", { className: "mmgr-tabs", role: "tablist", "aria-label": "Model sections", "aria-orientation": "vertical" });
  for (const [key, name, ic] of [["models", "Models", "i-chip"], ["add", "Add Model", "i-download"], ["about", "About", "i-info"]]) {
    const b = el("button", { type: "button", role: "tab", id: "mmgr-t-" + key, "aria-controls": "mmgr-p-" + key, onclick: () => show(key, true) },
      icon(ic), el("span", { textContent: name }));
    b.dataset.tab = key;
    tabs.append(b);
  }
  tabs.addEventListener("keydown", (e) => {
    const keys = [...tabs.children].map((b) => b.dataset.tab), i = keys.indexOf(tab);
    const d = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
    let next = null;
    if (d) next = keys[(i + d + keys.length) % keys.length];
    else if (e.key === "Home") next = keys[0];
    else if (e.key === "End") next = keys[keys.length - 1];
    if (!next) return;
    e.preventDefault(); e.stopPropagation();
    show(next, true);
  });
  const closeBtn = el("button", { type: "button", className: "mmgr-x", title: "Close", "aria-label": "Close models", onclick: () => close() }, icon("i-close"));
  const card = el("div", { className: "mmgr-card" },
    el("div", { className: "mmgr-head" },
      el("span", { className: "mmgr-head-ic", "aria-hidden": "true" }, icon("i-chip")),
      el("h2", { id: "mmgr-title", textContent: "Models" }), closeBtn),
    el("div", { className: "mmgr-body" }, tabs, el("div", { className: "mmgr-panels" }, ...Object.values(panels))));
  const root = el("div", { className: "mmgr", role: "dialog", "aria-modal": "true", "aria-labelledby": "mmgr-title", hidden: true }, card);
  root.addEventListener("mousedown", (e) => { if (e.target === root) close(); });
  // Keys inside the window stay there (the page's shortcuts must not act behind it).
  root.addEventListener("keydown", (e) => {
    if (e.key === "Tab") trap(e);
    if (e.key !== "Escape") e.stopPropagation();
  });
  // Escape: capture phase, so the page's Escape chain doesn't act too; a <dialog> or the
  // model-switch overlay above the window handles its own.
  addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !isOpen()) return;
    if (document.querySelector("dialog[open]")) return;
    const msw = document.getElementById("mswitch");
    if (msw && !msw.hidden) return;
    e.stopPropagation(); e.preventDefault();
    close();
  }, true);

  function trap(e) {
    const f = [...card.querySelectorAll("button:not(:disabled), a[href], summary, [tabindex]:not([tabindex='-1'])")].filter((x) => x.offsetParent !== null);
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1];
    if (e.shiftKey && (document.activeElement === first || !card.contains(document.activeElement))) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  function show(key, focusTab = false) {
    tab = panels[key] ? key : "models";
    for (const b of tabs.children) {
      const on = b.dataset.tab === tab;
      b.classList.toggle("here", on);
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
      if (on && focusTab) b.focus();
    }
    for (const [k, p] of Object.entries(panels)) p.hidden = k !== tab;
  }

  function open(which = "models", model = null) {
    if (!root.isConnected) document.body.append(root);
    const fresh = !isOpen();
    if (fresh) opener = document.activeElement;
    wanted = model; pointed = false;
    if (model && which === "models" && !onBoard(model)) which = "add";
    show(which);
    root.hidden = false;
    if (fresh) for (const p of Object.values(panels)) p.scrollTop = 0;   // a fresh open starts at the top (a targeted model scrolls itself into view)
    opened = true;
    document.documentElement.classList.add("mmgr-open");
    status = hooks.getStatus() || status;
    drawn = "";
    render();
    closeBtn.focus({ preventScroll: true });
    refresh();
  }

  function close() {
    if (!isOpen()) return;
    root.hidden = true;
    opened = false;
    document.documentElement.classList.remove("mmgr-open");
    if (location.hash.startsWith("#models")) history.replaceState(null, "", location.pathname + location.search);
    if (!downloading) clearTimeout(timer);          // a download keeps being watched
    const back = opener;
    opener = null;
    if (back && back.isConnected && back.focus) back.focus({ preventScroll: true });
    dispatchEvent(new Event("models-closed"));
  }
  const isOpen = () => opened;

  // ------------------------------------------------------------------------------- reading
  async function refresh() {
    clearTimeout(timer);
    try {
      const [s, h] = await Promise.all([getJSON("/api/status"), getJSON("/api/hub").catch((e) => hub || { error: e.message, models: {} })]);
      status = s; hub = h;
    } catch (e) {
      status = hooks.getStatus() || status;          // board unreachable: keep the last view
    }
    const models = (hub && hub.models) || {};
    const now = Object.keys(models).find((n) => models[n].state === "downloading") || null;
    if (downloading && downloading !== now) finished(downloading);
    downloading = now;
    if (isOpen()) render();
    const loading = status && status.state === "loading";
    if (isOpen() || downloading) timer = setTimeout(refresh, downloading || loading || busy || switching ? 800 : 4000);
  }

  // A download ended: say so, and on a board with nothing loaded, load what arrived.
  function finished(name) {
    const m = ((hub && hub.models) || {})[name];
    if (!m || m.state !== "on_board") return;
    hooks.toast(`${titleOf(name)} is on the board`, "ok", 3000);
    changed();
    const st = status && status.state;
    if (st !== "ready" && st !== "loading" && !switching) load(name);
  }

  function changed() {
    dispatchEvent(new CustomEvent("models-changed", { detail: { active: active(), family: (status && status.family) || null } }));
  }

  // ---------------------------------------------------------------------------- the models
  const localOf = (name) => ((status && status.models) || []).find((m) => m.name === name) || null;
  const remoteOf = (name) => ((hub && hub.models) || {})[name] || null;
  const active = () => (status && status.state === "ready" ? status.model : null);
  const anySwitching = () => switching || (status && status.state === "loading" ? (status.load && status.load.target) || status.model || "?" : null);
  function onBoard(name) {
    const local = localOf(name), remote = remoteOf(name);
    return !!(local && local.available) || !!(remote && remote.on_disk_bytes > 0);
  }
  function titleOf(name) {
    const local = localOf(name), remote = remoteOf(name), known = BUILTIN[name];
    return (remote && remote.title) || (local && local.title) || (known && known.title) || name;
  }
  function familyOf(name) {
    const local = localOf(name), remote = remoteOf(name), known = BUILTIN[name];
    return (local && local.family) || (remote && remote.family) || (known && known.family) || "cyws";
  }

  async function act(name, action) {
    const remote = remoteOf(name) || {}, title = titleOf(name);
    const size = remote.on_disk_bytes || remote.bytes;
    if (action === "delete") {
      const ok = await hooks.confirm({
        title: `Delete ${title}?`,
        body: `This removes ${remote.file || (BUILTIN[name] && BUILTIN[name].file) || "its file"}${size ? ` (${mb(size)})` : ""} from ${(hub && hub.directory) || "the models folder"}.`
          + (remote.url || remote.bytes ? " You can download it again from Hugging Face." : " It is not on Hugging Face, so it can't be downloaded again here."),
        ok: "Delete" });
      if (!ok) return;
    } else if (action === "redownload" && onBoard(name)) {
      const ok = await hooks.confirm({
        title: `Re-download ${title}?`,
        body: `This fetches a fresh copy${remote.bytes ? ` (${mb(remote.bytes)})` : ""} from Hugging Face. The file on the board stays in use until the new one has arrived and passed its checksum.`,
        ok: "Re-download" });
      if (!ok) return;
    }
    busy = { name, action }; said = "";
    render();
    try {
      if (action === "unload") {
        status = await post("/api/models/unload", {});
        hooks.toast(`${title} is off the MLA`, "ok", 2500);
        changed();
      } else {
        const out = await post(`/api/hub/${action}`, { name });
        if (out && out.models) hub = out;
        if (action === "download" || action === "redownload") hooks.toast(`Downloading ${title} from Hugging Face`, "info", 2500);
        else if (action === "delete") { hooks.toast(`Deleted ${title}`, "ok", 3000); changed(); }
        else if (action === "verify") {
          if (out && out.verified && out.verified.ok) hooks.toast(`${title} matches the published file`, "ok", 3000);
          else hooks.toast(`${title} doesn't match the published file. Re-download it.`, "error");
        }
      }
    } catch (e) {
      said = e.message;
      hooks.toast(e.message, "error");
    } finally {
      busy = null;
    }
    await refresh();
    if (!isOpen()) return;
    render();
  }

  async function load(name) {
    if (anySwitching()) return;
    switching = name; said = "";
    render();
    try {
      await hooks.switchModel(name);
    } catch (e) {
      said = e.message;
    } finally {
      switching = null;
    }
    await refresh();
    changed();
  }

  // --------------------------------------------------------------------------------- cards
  function btn(name, action, text, ic, cls = "", props = {}) {
    const doing = busy && busy.name === name && busy.action === action;
    const label = doing ? ({ verify: "Checking…", delete: "Deleting…", redownload: "Starting…", unload: "Unloading…", download: "Starting…", cancel: "Cancelling…", discard: "Discarding…" }[action] || text) : text;
    const b = el("button", { type: "button", className: `mmgr-btn ${cls}${doing ? " busy" : ""}`, ...props,
      onclick: () => (action === "load" ? load(name) : act(name, action)) }, icon(doing ? "i-refresh" : ic), el("span", { textContent: label }));
    b.dataset.key = `${name}:${action}`;
    if (busy || props.disabled) b.disabled = true;
    return b;
  }

  function facts(name, remote, known, fam, precision) {
    const card = Object.assign({}, known && known.card, remote && remote.card);
    const dl = el("dl", { className: "mmgr-facts" });
    const fact = (label, ...value) => { if (value.some((v) => v)) dl.append(el("dt", { textContent: label }), el("dd", {}, ...value)); };
    fact("Family", `${FAMILY[fam] ? FAMILY[fam].title : fam} · ${FAMILY[fam] ? FAMILY[fam].pill : ""}`);
    fact("Architecture", card.architecture || card.model);
    const input = remote && remote.input;
    fact("Input", card.input || (input && input.size ? `${input.images === 2 ? "Two images" : "One image"}, ${input.size}×${input.size}` : ""));
    fact("Precision", PRECISION[precision] || precision);
    const lat = (remote && remote.latency_ms) || (known && known.latency_ms);
    const per = FAMILY[fam] ? FAMILY[fam].per.toLowerCase() : "window";
    fact("Latency", known && known.speed && (!remote || remote.latency_ms === known.latency_ms) ? known.speed
      : lat ? `${lat} ms per ${per === "window" ? "window pair" : per} on the MLA` : "");
    const acc = (remote && remote.accuracy && Object.keys(remote.accuracy).length ? remote.accuracy : known && known.accuracy) || {};
    const accText = [acc.changes_found ? `${acc.changes_found} found` : "", acc.pixel_f1 != null ? `pixel F1 ${Number(acc.pixel_f1).toFixed(3)}` : "",
      acc.false_alarm_pairs ? `${acc.false_alarm_pairs} false-alarm pairs` : ""].filter(Boolean).join(" · ");
    fact("Accuracy on testCYWS", accText, accText && known && known.accuracy_note ? el("span", { className: "mmgr-sub", textContent: known.accuracy_note }) : "");
    const bytes = (remote && (remote.on_disk_bytes || remote.bytes)) || (known && known.bytes);
    fact("Size", bytes ? (onBoard(name) ? `${mb(bytes)} on this board` : `${mb(bytes)} to download`)
      + (remote && remote.partial_bytes ? `, ${mb(remote.partial_bytes)} of it already here` : "") : "");
    const repo = (remote && remote.repo) || (hub && hub.repo);
    const upstream = /^[\w.-]+\/[\w.-]+$/.test(card.upstream || "") ? card.upstream : null;
    const fromUp = upstream ? [", compiled from ", link(`https://github.com/${upstream}`, upstream)] : [];
    if (remote && repo) fact("Source", link(`https://huggingface.co/${repo}/tree/main/${name}`, `${repo}/${name}`), ...fromUp, card.license ? ` (${card.license})` : "");
    else fact("Source", "On this board; not listed on Hugging Face", ...fromUp, card.license ? ` (${card.license})` : "");
    const file = (remote && remote.file) || (known && known.file);
    fact("File", file ? el("code", { textContent: file }) : "");
    return dl;
  }

  function modelCard(name) {
    const local = localOf(name), remote = remoteOf(name), known = BUILTIN[name];
    const fam = familyOf(name), title = titleOf(name);
    const precision = (remote && remote.precision) || (known && known.precision);
    const here = onBoard(name), isActive = name === active();
    const sw = anySwitching(), loadingThis = sw === name;
    const damaged = remote && remote.state === "damaged";
    const dling = remote && remote.state === "downloading";
    const partial = remote && !dling && remote.partial_bytes > 0;

    const pills = [];
    const pill = (text, cls = "", t = null) => pills.push(el("span", { className: `mmgr-pill ${cls}`, textContent: text, title: t }));
    pill(FAMILY[fam] ? FAMILY[fam].pill : fam, fam === "dinov2" ? "zs" : "trained",
      fam === "dinov2" ? "No change-detection training: compares the photos' DINOv2 features"
        : fam === "c3po" ? "Trained for change segmentation: change or no change per pixel" : "Trained for change detection");
    if (loadingThis) pill("Loading…", "warn");
    else if (isActive) pill("Loaded", "ok");
    else if (damaged) pill("Damaged", "warn", "The file doesn't match the published one");
    else if (here && !dling) pill("On This Board", "board");
    if (dling) pill("Downloading…", "warn");
    else if (!here) pill("On Hugging Face", "muted");

    const about = (remote && remote.about) || (known && known.about) || ((local && local.label) || "").split("·").slice(1).join("·").trim() || "Compiled for the MLA";
    const acc = (remote && remote.accuracy && Object.keys(remote.accuracy).length ? remote.accuracy : known && known.accuracy) || {};
    const lat = (remote && remote.latency_ms) || (known && known.latency_ms);
    const stats = [lat ? `${lat} ms per ${FAMILY[fam] ? FAMILY[fam].per : "Window"}` : "", acc.changes_found ? `${acc.changes_found} Found` : "",
      acc.pixel_f1 != null ? `F1 ${Number(acc.pixel_f1).toFixed(3)}` : "", remote && remote.bytes ? mb(remote.bytes) : known && known.bytes ? mb(known.bytes) : "",
      remote && remote.verified === true && here ? "SHA-256 ✓" : ""].filter(Boolean);

    let progress = "";
    if (dling) {
      const p = remote.progress || {}, frac = p.fraction || 0, percent = Math.round(100 * frac);
      const fill = el("i"); fill.style.width = `${(100 * frac).toFixed(1)}%`;
      progress = el("div", { className: "mmgr-progress" },
        remote.replacing ? el("p", { className: "mmgr-sub", textContent: "Re-downloading · the copy on the board stays in use until this one is checked" }) : "",
        el("div", { className: "mmgr-bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(percent), "aria-label": `Downloading ${title}` }, fill),
        el("span", { textContent: `${percent}% · ${mb(p.done_bytes || 0)} of ${mb(remote.bytes)}${p.bytes_per_second ? ` · ${mb(p.bytes_per_second)}/s` : ""}${p.seconds_left != null ? ` · ${left(p.seconds_left)}` : ""}` }));
    } else if (loadingThis && status && status.load && status.load.stage) {
      const ld = status.load, stage = { release: "Releasing the MLA", load: "Loading the model", warm: "Warming up" }[ld.stage] || ld.stage;
      progress = el("div", { className: "mmgr-progress" }, el("span", { textContent: `${stage} · ${Number(ld.elapsed_s || 0).toFixed(1)} s${ld.expected_s ? ` of about ${Math.round(ld.expected_s)} s` : ""}` }));
    }
    const err = (remote && remote.error) || (damaged ? "The file on the board is not the published one. Re-download it." : "");

    // Actions by state. One model is on the MLA at a time.
    const actions = el("div", { className: "mmgr-actions" });
    const others = !!sw && !loadingThis;
    const lock = { disabled: others || undefined, title: others ? "Wait for the model switch" : undefined };
    if (loadingThis) {
      actions.append(el("button", { type: "button", className: "mmgr-btn primary busy", disabled: true }, icon("i-refresh"), el("span", { textContent: "Loading…" })));
    } else if (dling) {
      actions.append(btn(name, "cancel", "Cancel", "i-close", "ghost", { title: "Stop the download; what arrived is kept to continue later" }));
    } else if (isActive) {
      actions.append(btn(name, "unload", "Unload", "i-close", "ghost-primary", { ...lock, title: "Take it off the MLA, freeing it for other apps" }));
      if (remote) actions.append(btn(name, "verify", "Verify", "i-shield", "", { ...lock, title: "Check the file against its published SHA-256" }));
      if (remote) actions.append(btn(name, "redownload", "Re-download", "i-refresh", "", { ...lock, title: "Fetch a fresh copy from Hugging Face" }));
      actions.append(btn(name, "delete", "Delete", "i-trash", "danger", { disabled: true, title: "Unload it first" }));
    } else if (here && damaged) {
      if (remote) actions.append(btn(name, "redownload", "Re-download", "i-refresh", "primary", { ...lock, title: "Fetch a fresh copy from Hugging Face" }));
      actions.append(btn(name, "delete", "Delete", "i-trash", "danger", { ...lock, title: lock.title || `Delete ${title} from the board` }));
    } else if (here) {
      actions.append(btn(name, "load", active() ? "Switch to This" : "Load", "i-chip", "primary",
        { ...lock, title: lock.title || (active() ? `Unload ${titleOf(active())} and load ${title}` : `Load ${title} on the MLA`) }));
      if (remote) actions.append(btn(name, "verify", "Verify", "i-shield", "", { ...lock, title: lock.title || "Check the file against its published SHA-256" }));
      if (remote) actions.append(btn(name, "redownload", "Re-download", "i-refresh", "", { ...lock, title: lock.title || "Fetch a fresh copy from Hugging Face" }));
      actions.append(btn(name, "delete", "Delete", "i-trash", "danger", { ...lock, title: lock.title || `Delete ${title} from the board` }));
    } else if (remote) {
      const another = !!downloading && downloading !== name;
      const free = hub && hub.free_bytes, short = free != null && remote.bytes && free < remote.bytes - (remote.partial_bytes || 0);
      const why = another ? "Wait for the other download" : short ? "Not enough free space on the board" : undefined;
      actions.append(btn(name, "download", partial ? `Continue · ${mb(remote.partial_bytes)} of ${mb(remote.bytes)}` : `Download · ${mb(remote.bytes)}`,
        "i-download", "primary", { disabled: another || short || undefined, title: why }));
      if (partial) actions.append(btn(name, "discard", "Discard", "i-trash", "danger", { title: "Remove the part that was downloaded" }));
    }

    const cls = "mmgr-model" + (isActive ? " on" : "") + (loadingThis ? " loading" : "") + (!here ? " remote" : "") + (wanted === name ? " wanted" : "");
    return el("article", { className: cls, id: "mmgr-m-" + name, "aria-label": title },
      el("div", { className: "mmgr-mbody" },
        el("div", { className: "mmgr-mhead" }, el("i", { className: "mmgr-dot", "aria-hidden": "true" }), el("h3", { textContent: title }),
          precision ? el("span", { className: "mmgr-prec", textContent: PREC_CHIP[precision] || precision, title: PRECISION[precision] || precision }) : "", ...pills),
        el("p", { className: "mmgr-about-line", textContent: about }),
        stats.length ? el("p", { className: "mmgr-stats" }, ...stats.flatMap((s, i) => (i ? [" · ", el("span", { textContent: s })] : [el("span", { textContent: s })]))) : "",
        progress,
        err ? el("p", { className: "mmgr-err", textContent: err }) : "",
        el("details", { className: "mmgr-card-facts" }, el("summary", { textContent: "Model Card" }), facts(name, remote, known, fam, precision))),
      actions);
  }

  // ----------------------------------------------------------------------------- the blocks
  function storageBlock() {
    if (!hub || hub.used_bytes == null || hub.free_bytes == null) return "";
    const used = hub.used_bytes, total = used + hub.free_bytes;
    const fill = el("i"); fill.style.width = `${Math.max(used ? 1 : 0, total ? 100 * used / total : 0).toFixed(1)}%`;
    return el("div", { className: "mmgr-block" },
      el("div", { className: "mmgr-row" }, el("span", { className: "mmgr-label grow", textContent: "Storage" }),
        el("span", { className: "mmgr-note", textContent: `${mb(hub.free_bytes)} Free` })),
      el("p", { className: "mmgr-note", textContent: `${plural(hub.model_count || 0, "Model")} · ${mb(used)}${hub.partial_count ? ` + ${hub.partial_count} Partial` : ""}` }),
      el("div", { className: "mmgr-bar store", role: "img", "aria-label": `Models use ${mb(used)}, ${mb(hub.free_bytes)} free` }, fill),
      hub.directory ? el("p", { className: "mmgr-note" }, "Folder: ", el("code", { textContent: hub.directory })) : "");
  }

  function repos() {
    if (!hub) return [];
    if (Array.isArray(hub.repos) && hub.repos.length) return hub.repos;
    return hub.repo ? [{ repo: hub.repo, page: hub.page, listed: hub.listed, error: hub.error }] : [];
  }

  /* One plain line per repository that couldn't be read (the server's full text stays in the tooltip). */
  function repoProblem(r) {
    const e = String(r.error || "");
    const code = (e.match(/HTTP Error (\d{3})/) || [])[1];
    if (code === "401" || code === "404") return `${r.repo} isn't published yet (HTTP ${code}).`;
    if (code) return `Hugging Face answered HTTP ${code} for ${r.repo}. Try again later.`;
    if (/urlopen error|timed out|handshake|Name or service|Temporary failure|Connection|Network is unreachable/i.test(e)) return `Can't reach Hugging Face right now (${r.repo}).`;
    const rest = e.replace(/^Could not read the model list from Hugging Face \([^)]*\):\s*/, "");
    return `Can't read the model list from ${r.repo}: ${rest}`;
  }

  function render() {
    if (!isOpen()) return;
    const sig = JSON.stringify([status, hub, busy, switching, said, wanted]);
    if (sig === drawn) return;
    drawn = sig;
    // Keep what the user has open (a model card, the scroll position, the focus) across a redraw.
    const openCards = new Set([...root.querySelectorAll("details[open]")].map((d) => d.closest(".mmgr-model") && d.closest(".mmgr-model").id));
    const scroll = Object.fromEntries(Object.entries(panels).map(([k, p]) => [k, p.scrollTop]));
    const focusKey = document.activeElement && root.contains(document.activeElement) ? document.activeElement.dataset.key
      || (document.activeElement.tagName === "SUMMARY" && document.activeElement.closest(".mmgr-model") ? "sum:" + document.activeElement.closest(".mmgr-model").id : null) : null;

    const names = [...new Set([...((status && status.models) || []).map((m) => m.name), ...Object.keys((hub && hub.models) || {})])];
    const board = names.filter(onBoard), more = names.filter((n) => !onBoard(n) && remoteOf(n));   // a built-in not on the hub has nowhere to come from
    const act_ = active(), sw = anySwitching();
    board.sort((a, b) => (b === act_) - (a === act_) || (b === sw) - (a === sw));
    const fam = status && status.family;

    // Models
    const head = sw ? `Loading ${titleOf(sw)}…` : act_ ? `${titleOf(act_)} Is on the MLA` : status ? "No Model Is Loaded" : "Connecting…";
    const counts = `${board.length} on this board` + (more.length ? ` · ${more.length} more on Hugging Face` : "");
    const mlaRow = el("div", { className: "mmgr-row mmgr-mla" },
      el("i", { className: `mmgr-dot${act_ ? " on" : sw ? " loading" : ""}`, "aria-hidden": "true" }),
      el("span", { className: "mmgr-mla-name grow" }, el("b", { textContent: sw ? titleOf(sw) : act_ ? titleOf(act_) : "Nothing loaded" }),
        act_ && fam && FAMILY[fam] ? el("span", { className: `mmgr-pill ${fam === "dinov2" ? "zs" : "trained"}`, textContent: FAMILY[fam].pill }) : "",
        act_ && status.latency && status.latency.mean_ms ? el("span", { className: "mmgr-note inline", textContent: `${Number(status.latency.mean_ms).toFixed(1)} ms per ${FAMILY[fam] ? FAMILY[fam].per : "Window"}` }) : ""),
      act_ ? btn(act_, "unload", "Unload", "i-close", "ghost-primary", { disabled: !!sw || undefined, title: "Take it off the MLA, freeing it for other apps" }) : "");
    panels.models.replaceChildren(
      el("div", { className: "mmgr-block" },
        el("div", { className: "mmgr-label", textContent: "Status" }),
        el("p", { className: "mmgr-status" }, el("b", { textContent: head }), el("span", { className: "mmgr-note inline", textContent: counts })),
        status && status.state !== "ready" && status.state !== "loading" && status.error && !sw ? el("p", { className: "mmgr-err", textContent: status.error }) : "",
        said ? el("p", { className: "mmgr-err", role: "alert", textContent: said }) : ""),
      el("div", { className: "mmgr-block" },
        el("div", { className: "mmgr-label", textContent: "On the MLA" }), mlaRow,
        el("p", { className: "mmgr-note", textContent: "One model is on the MLA at a time: Load switches to it, Unload frees the MLA for other apps." })),
      storageBlock(),
      el("div", { className: "mmgr-row" }, el("span", { className: "mmgr-label grow", textContent: "On This Board" })),
      el("div", { className: "mmgr-list" }, ...(board.length ? board.map(modelCard)
        : [el("div", { className: "mmgr-empty" }, el("span", { textContent: "No model is on this board yet." }),
            el("button", { type: "button", className: "mmgr-btn primary", onclick: () => show("add", true) }, icon("i-download"), el("span", { textContent: "Add Model" })))])));

    // Add Model
    const rs = repos();
    const src = rs.length ? el("p", { className: "mmgr-note" }, "Models come from ",
      ...rs.flatMap((r, i) => [i ? (i === rs.length - 1 ? " and " : ", ") : "", link(r.page || `https://huggingface.co/${r.repo}`, r.repo)]),
      " on Hugging Face", hub.collection ? el("span", {}, " (", link(hub.collection, "Collection"), ")") : "",
      hub.directory ? el("span", {}, " into ", el("code", { textContent: hub.directory })) : "",
      hub.free_bytes != null ? `, which has ${mb(hub.free_bytes)} free.` : ".")
      : hub && hub.free_bytes != null ? el("p", { className: "mmgr-note" }, "The models folder",
        hub.directory ? el("span", {}, " (", el("code", { textContent: hub.directory }), ")") : "", ` has ${mb(hub.free_bytes)} free.`) : "";
    const errors = rs.filter((r) => r.error).map((r) => el("p", { className: "mmgr-err", title: r.error, textContent: repoProblem(r) }));
    if (!rs.length && hub && hub.error) errors.push(el("p", { className: "mmgr-err", textContent: hub.error }));
    const skipped = hub && Array.isArray(hub.skipped) && hub.skipped.length
      ? el("p", { className: "mmgr-note", textContent: `Skipped ${plural(hub.skipped.length, "entry").replace("entrys", "entries")} that this version can't use.` }) : "";
    const listing = rs.some((r) => !r.listed && !r.error);
    panels.add.replaceChildren(
      el("div", { className: "mmgr-block" }, el("div", { className: "mmgr-label", textContent: "Available to Download" }), src, ...errors, skipped,
        said ? el("p", { className: "mmgr-err", role: "alert", textContent: said }) : ""),
      el("div", { className: "mmgr-list" }, ...(more.length ? more.map(modelCard)
        : [el("div", { className: "mmgr-empty", textContent: !hub ? "Reading the list…"
            : !rs.length ? "This app was started without a Hugging Face repository (--hub-repo '')."
            : listing ? "Reading the list from Hugging Face…"
            : rs.every((r) => r.error) ? "Hugging Face can't be reached from the board right now."
            : rs.some((r) => r.error) ? "Every model from the repositories that could be read is already on this board."
            : "Every model published there is already on this board." })])));

    // About
    const dd = (...v) => el("dd", {}, ...v);
    panels.about.replaceChildren(
      el("div", { className: "mmgr-brand" }, el("img", { src: "/static/assets/neat-change.svg", alt: "", width: "54", height: "54" }),
        el("div", {}, el("b", { textContent: "Neat Change Studio" }),
          el("p", { className: "mmgr-note", textContent: "Before/after change detection on the SiMa.ai Modalix MLA" }))),
      el("dl", { className: "mmgr-facts about" },
        el("dt", { textContent: "Active Model" }), dd(act_ ? `${titleOf(act_)} (${act_})` : "None"),
        el("dt", { textContent: "Models" }), dd(`${board.length} on this board${more.length ? `, ${more.length} more on Hugging Face` : ""}`),
        el("dt", { textContent: "Backend" }), dd((status && status.backend) || "unknown"),
        el("dt", { textContent: "Models Folder" }), dd(hub && hub.directory ? el("code", { textContent: hub.directory }) : "unknown"),
        el("dt", { textContent: "Hugging Face" }), dd(...(rs.length ? rs.flatMap((r, i) => [i ? ", " : "", link(r.page || `https://huggingface.co/${r.repo}`, r.repo)]) : ["None"])),
        el("dt", { textContent: "Source" }), dd(link(GITHUB, "github.com/dotimothy/neat-change-studio")),
        el("dt", { textContent: "Models Used" }), dd("CYWS (Sachdeva & Zisserman, WACV 2023) and DINOv2 (Meta), both Apache-2.0; C-3PO (Wang et al., Pattern Recognition 2023), MIT")));

    for (const id of openCards) { const d = id && document.getElementById(id); if (d) { const det = d.querySelector("details"); if (det) det.open = true; } }
    for (const [k, p] of Object.entries(panels)) p.scrollTop = scroll[k];
    if (focusKey) {
      const f = focusKey.startsWith("sum:") ? document.getElementById(focusKey.slice(4)) && document.getElementById(focusKey.slice(4)).querySelector("summary")
        : root.querySelector(`[data-key="${CSS.escape(focusKey)}"]`);
      if (f && !f.disabled) f.focus({ preventScroll: true }); else if (!root.contains(document.activeElement)) closeBtn.focus({ preventScroll: true });
    }
    if (wanted && !pointed) {
      const w = document.getElementById("mmgr-m-" + wanted);
      if (w) {
        pointed = true;
        w.scrollIntoView({ block: "center", behavior: "auto" });
        setTimeout(() => { wanted = null; pointed = false; drawn = ""; render(); }, 2400);   // the outline fades after a moment
      } else if (status && hub) wanted = null;
    }
  }

  // --------------------------------------------------------------------------------- export
  window.ModelManager = {
    open, close, isOpen,
    refresh() { return refresh(); },
    configure(h = {}) { for (const k of Object.keys(hooks)) if (typeof h[k] === "function") hooks[k] = h[k]; },
  };

  // #models or #models=add opens the window, on load and when the link changes.
  const fromHash = () => {
    const h = location.hash;
    if (h === "#models" || h.startsWith("#models=")) open(h.split("=")[1] || "models");
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fromHash); else fromHash();
  addEventListener("hashchange", fromHash);
})();
