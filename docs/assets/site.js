/* Neat Change Studio — project site: theme toggle, phone menu, copy buttons,
 * section highlighting, animated screenshots and a small lightbox.
 * The page works without it. */
(function () {
  "use strict";
  var root = document.documentElement;
  var $ = function (s, el) { return (el || document).querySelector(s); };
  var $$ = function (s, el) { return Array.prototype.slice.call((el || document).querySelectorAll(s)); };
  var mm = function (q) { return window.matchMedia ? window.matchMedia(q) : null; };
  var onChange = function (mq, fn) {
    if (!mq) return;
    if (mq.addEventListener) mq.addEventListener("change", fn); else if (mq.addListener) mq.addListener(fn);
  };

  /* ---- Theme (same key as the web demo) ---- */
  var mqLight = mm("(prefers-color-scheme: light)");
  function isDark() {
    var t = root.getAttribute("data-theme");
    if (t === "dark") return true;
    if (t === "light") return false;
    return !(mqLight && mqLight.matches);
  }
  function syncThemeUi() {
    var dark = isDark();
    var btn = $("#btn-theme");
    if (btn) {
      var label = dark ? "Switch to Light Theme" : "Switch to Dark Theme";
      btn.setAttribute("aria-label", label);
      btn.setAttribute("title", label);
    }
    var meta = $("#meta-theme");
    if (meta) meta.setAttribute("content", dark ? "#0c0f14" : "#f4f6f8");
  }
  function toggleTheme() {
    var next = isDark() ? "light" : "dark";
    root.classList.add("theme-anim");
    root.setAttribute("data-theme", next);
    clearTimeout(toggleTheme._t);
    toggleTheme._t = setTimeout(function () { root.classList.remove("theme-anim"); }, 420);
    try { localStorage.setItem("cyws-theme", next); } catch (e) { /* storage blocked */ }
    syncThemeUi();
  }
  var themeBtn = $("#btn-theme");
  if (themeBtn) themeBtn.addEventListener("click", toggleTheme);
  onChange(mqLight, syncThemeUi);
  syncThemeUi();

  /* ---- Phone menu ---- */
  var menuBtn = $("#btn-menu"), nav = $("#nav");
  function setMenu(open) {
    if (!menuBtn || !nav) return;
    menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    menuBtn.setAttribute("aria-label", open ? "Close Menu" : "Open Menu");
    nav.classList.toggle("open", open);
  }
  if (menuBtn && nav) {
    menuBtn.addEventListener("click", function () { setMenu(menuBtn.getAttribute("aria-expanded") !== "true"); });
    $$("a", nav).forEach(function (a) { a.addEventListener("click", function () { setMenu(false); }); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") setMenu(false); });
    document.addEventListener("click", function (e) {
      if (nav.classList.contains("open") && !nav.contains(e.target) && !menuBtn.contains(e.target)) setMenu(false);
    });
  }

  /* ---- Active section in the nav (the hero clears it) ---- */
  var links = nav ? $$("a[href^='#']", nav) : [];
  if ("IntersectionObserver" in window && links.length) {
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var setActive = function (a) {
      links.forEach(function (l) { l.classList.remove("active"); l.removeAttribute("aria-current"); });
      if (a) { a.classList.add("active"); a.setAttribute("aria-current", "true"); }
    };
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) setActive(byId[en.target.id] || null); });
    }, { rootMargin: "-45% 0px -50% 0px" });
    ["top"].concat(Object.keys(byId)).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }

  /* ---- Copy buttons ---- */
  var status = $("#copy-status");
  function announce(msg) {
    if (!status) return;
    status.textContent = "";
    setTimeout(function () { status.textContent = msg; }, 50);
  }
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      var ta = document.createElement("textarea");
      ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      try { if (document.execCommand("copy")) resolve(); else reject(new Error("copy")); } catch (e) { reject(e); }
      document.body.removeChild(ta);
    });
  }
  $$(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy"); // a block may copy just its first command
      if (!text) {
        var pre = $("pre", btn.closest(".code"));
        if (!pre) return;
        var clone = pre.cloneNode(true);
        $$(".p, .c", clone).forEach(function (n) { n.remove(); }); // skip prompt markers and comments
        text = clone.textContent.split("\n").map(function (l) { return l.replace(/\s+$/, ""); })
          .filter(function (l) { return l; }).join("\n");
      }
      text = text.replace(/\s+$/, "") + "\n";
      var label = $(".lbl", btn);
      copyText(text).then(function () {
        btn.classList.add("done"); if (label) label.textContent = "Copied";
        announce("Copied");
        clearTimeout(btn._t);
        btn._t = setTimeout(function () { btn.classList.remove("done"); if (label) label.textContent = "Copy"; }, 1600);
      }, function () {
        if (label) label.textContent = "Select and Copy";
        announce("Copy Failed: Select the Text and Copy It");
      });
    });
  });

  /* ---- Animated screenshots (MP4 copies of the GIFs) ----
   * Without JS the videos show their still poster and native controls.
   * With JS they play while on screen, unless the visitor prefers reduced
   * motion; the Pause/Play button overrides both. */
  var mqReduce = mm("(prefers-reduced-motion: reduce)");
  var reduce = function () { return !!(mqReduce && mqReduce.matches); };
  var vids = $$("[data-vid]").map(function (box) {
    var v = $("video", box), btn = $(".vid-toggle", box);
    if (!v) return null;
    v.removeAttribute("controls");
    v.muted = true;
    // Start on a detected frame (the poster) instead of the GIF's first frame.
    var start = parseFloat(box.getAttribute("data-start"));
    if (start > 0) {
      var seek = function () { try { if (v.currentTime < start) v.currentTime = start; } catch (e) { /* not seekable yet */ } };
      if (v.readyState >= 1) seek(); else v.addEventListener("loadedmetadata", seek, { once: true });
    }
    var st = { box: box, v: v, btn: btn, user: null, inView: false };
    var lbl = btn && $(".lbl", btn);
    st.sync = function () {
      var want = st.user === null ? !reduce() : st.user === "play";
      var paused = !(want && st.inView);
      if (paused) { if (!v.paused) v.pause(); }
      else if (v.paused) { var pr = v.play(); if (pr && pr.catch) pr.catch(function () { box.classList.add("paused"); if (lbl) lbl.textContent = "Play"; }); }
      var showPaused = !want;
      box.classList.toggle("paused", showPaused);
      if (lbl) lbl.textContent = showPaused ? "Play" : "Pause";
    };
    if (btn) btn.addEventListener("click", function (e) {
      e.preventDefault(); e.stopPropagation();
      st.user = box.classList.contains("paused") ? "play" : "pause";
      st.sync();
    });
    return st;
  }).filter(Boolean);
  if (vids.length) {
    if ("IntersectionObserver" in window) {
      var vio = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          vids.forEach(function (st) { if (st.box === en.target) { st.inView = en.isIntersecting; st.sync(); } });
        });
      }, { rootMargin: "120px 0px" });
      vids.forEach(function (st) { vio.observe(st.box); });
    } else {
      vids.forEach(function (st) { st.inView = true; st.sync(); });
    }
    onChange(mqReduce, function () { vids.forEach(function (st) { st.sync(); }); });
  }

  /* ---- Lightbox (native <dialog>; links open the file without it) ---- */
  var lb = $("#lightbox");
  if (lb && typeof lb.showModal === "function") {
    var media = $(".lb-media", lb), lbCap = $("#lb-cap", lb), lbClose = $(".lb-close", lb);
    var lastFocus = null;
    var close = function () { if (lb.open) lb.close(); };
    lb.addEventListener("close", function () {
      media.textContent = ""; // drops the image or video
      root.classList.remove("lb-lock");
      if (lastFocus) lastFocus.focus();
    });
    $$("a.shot, a.lb-open").forEach(function (a) {
      a.setAttribute("aria-haspopup", "dialog");
      a.addEventListener("click", function (e) {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;
        e.preventDefault(); e.stopPropagation();
        var img = $("img", a), vsrc = a.getAttribute("data-video"), el;
        var cap = a.getAttribute("data-caption") || (img ? img.alt : "Screenshot");
        if (vsrc) {
          el = document.createElement("video");
          el.muted = true; el.loop = true; el.playsInline = true; el.controls = true;
          el.setAttribute("playsinline", ""); el.setAttribute("aria-label", cap);
          el.poster = a.getAttribute("data-poster") || "";
          el.src = vsrc;
          if (!reduce()) el.autoplay = true;
        } else {
          el = document.createElement("img");
          el.src = a.getAttribute("href");
          el.alt = img ? img.alt : cap;
        }
        media.textContent = "";
        media.appendChild(el);
        lbCap.textContent = cap;
        lastFocus = a;
        root.classList.add("lb-lock");
        lb.showModal();
        lbClose.focus();
      });
    });
    lbClose.addEventListener("click", close);
    lb.addEventListener("click", function (e) { if (e.target === lb || e.target.tagName === "FIGURE") close(); });
  }
})();
