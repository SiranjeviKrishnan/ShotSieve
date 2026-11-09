/* T23g: one dispatcher for the native macOS menu bar and the browser's hamburger menu.
   window.shotsieveMenu(id) maps a FIXED allow-list of ids onto the existing handlers (the old toolbar
   buttons live on, hidden, in #menu-actions). Python (desktop.py) calls it through evaluate_js with an id
   it checked against the same list; the id is never built from user data. Page -> app: the page reports
   its toggle states and recent folders to window.pywebview.api.menu_state so the native menu can show a
   check mark and only the items that apply. */
"use strict";

(function () {
  const click = (id) => { const b = $(id); if (b) b.click(); return !!b; };
  const key = (k, extra) => document.dispatchEvent(new KeyboardEvent("keydown", Object.assign({ key: k, bubbles: true, cancelable: true }, extra)));
  const barOn = () => { try { return localStorage.getItem("shotsieve.filterbar") !== "0"; } catch (e) { return true; } };
  const unavailable = (what) => { toast(what + " is not available for this shoot yet."); return false; };

  // id -> handler. Everything here already existed as a button or a key; nothing is new behaviour.
  const HANDLERS = {
    choose: () => chooseFolder(),
    crops: () => click("chk-crops"),
    video: () => click("chk-video"),
    autoxmp: () => setAutoXmp(!xmp.auto),
    lropen: () => key("l", { metaKey: true }),
    trash: () => (window.ShotSelect ? window.ShotSelect.trash() : false),
    undotrash: () => (window.ShotSelect && window.ShotSelect.canUndo() ? window.ShotSelect.undo() : unavailable("Undo Move to Trash")),
    subject: () => click("mi-subject"),
    chips: () => click("mi-chips"),
    filterbar: () => key("F", { shiftKey: true }),
    cameras: () => click("btn-cameras"),
    people: () => click("btn-people"),
    shotlist: () => click("btn-shotlist"),
    edit: () => click("btn-edit"),
    matchref: () => (S.sel === null ? unavailable("Colour match (select a photo first)") : (matchSetRef(), true)),
    jersey: () => ($("mi-jersey").hidden ? unavailable("Jersey numbers") : click("mi-jersey")),
    lrsync: () => ($("mi-lr").hidden ? unavailable("Lightroom Classic sync") : click("mi-lr")),
    auditdone: () => ($("btn-audit-done").hidden ? unavailable("Finish audit") : click("btn-audit-done")),
    denoise: () => {
      for (const [open, sel] of [["overlay-compare", "#cmp-denoise .dn-btn"], ["overlay-loupe", "#lz-denoise .dn-btn"]]) {
        if (!$(open).hidden) { const b = document.querySelector(sel); if (b) b.click(); return true; }
      }
      return toggleMainDenoise();
    },
    compare: () => toggleCompare(),
    help: () => click("btn-help"),
    setup: () => click("btn-setup"),
    lrcard: () => click("btn-lrcard"),
  };
  const RECENT = /^recent:[0-9]$/;
  window.shotsieveMenuIds = Object.keys(HANDLERS);

  let xmp = { auto: true, written: 0, skipped: 0, failed: 0, warnings: [] };
  async function loadXmp() {
    if (!S.folder) return;
    try { xmp = await api("/api/xmp/status" + q({ folder: S.folder })); } catch (e) { return; }
    drawXmpNote();
  }
  async function setAutoXmp(on) {
    if (!S.folder) return;
    try { xmp = await api("/api/xmp/auto", { method: "POST", body: { folder: S.folder, on: !!on } }); } catch (e) { toast(e.message); return; }
    drawXmpNote();
    report();
  }
  function drawXmpNote() {   // unobtrusive: a status-bar note, details on click, never a modal
    const el = $("sb-xmp");
    if (!el) return;
    const n = (xmp.skipped || 0) + (xmp.failed || 0);
    el.hidden = !n || !xmp.auto;
    el.textContent = n ? `XMP: ${xmp.skipped ? xmp.skipped + " skipped (your Lightroom edits kept)" : ""}${xmp.skipped && xmp.failed ? ", " : ""}${xmp.failed ? xmp.failed + " failed" : ""}` : "";
  }
  if ($("sb-xmp")) $("sb-xmp").addEventListener("click", () => toast((xmp.warnings || []).join(" ") || "Nothing more to report.", 12000));
  try { $("chk-video").checked = localStorage.getItem("shotsieve.includeVideo") !== "0"; } catch (e) { /* default on */ }
  $("chk-video").addEventListener("change", () => { try { localStorage.setItem("shotsieve.includeVideo", $("chk-video").checked ? "1" : "0"); } catch (e) { /* ignore */ } });

  let recent = [];
  async function loadRecent() {
    try { recent = (await api("/api/recent")).folders || []; } catch (e) { recent = []; }
  }
  window.shotsieveMenu = function (id) {
    if (typeof id !== "string") return false;
    if (RECENT.test(id)) {
      const f = recent[+id.slice(7)];
      if (!f) return false;
      setFolderField(f.path); startCull();
      return true;
    }
    if (!Object.prototype.hasOwnProperty.call(HANDLERS, id)) return false;
    try { HANDLERS[id](); } catch (e) { console.error("menu", id, e); return false; }
    setTimeout(report, 150);
    return true;
  };

  function state() {
    return {
      subject: !!(window.ShotSubject && window.ShotSubject.isOn()),
      chips: $("mi-chips").getAttribute("aria-checked") === "true",
      filterbar: barOn(),
      crops: $("chk-crops").checked,
      video: $("chk-video").checked,
      autoxmp: !!xmp.auto,
      denoise: !!S.dnMain || ["#cmp-denoise", "#lz-denoise"].some((s) => { const e = document.querySelector(s); return e && e.dataset.on === "1"; }),
      jersey: !$("mi-jersey").hidden,
      lrsync: !$("mi-lr").hidden,
      auditdone: !$("btn-audit-done").hidden,
      undotrash: !!(window.ShotSelect && window.ShotSelect.canUndo()),
      recent: recent.slice(0, 10).map((f) => String(f.name || f.path)),
    };
  }
  let last = "";
  function report() {
    const s = state(), j = JSON.stringify(s);
    if (j === last) return;
    last = j;
    renderHamburger(s);
    const b = window.pywebview && window.pywebview.api;
    if (b && b.menu_state) { try { b.menu_state(s); } catch (e) { /* menu keeps its last state */ } }
  }

  /* ---------- browser: the compact hamburger menu (hidden in the app) ---------- */
  const LAYOUT = [
    ["File", [["choose", "Choose Folder…", "native"], ["lropen", "Open in Lightroom"], ["trash", "Move to Trash"], ["undotrash", "Undo Move to Trash", "", "", "undotrash"]]],
    ["View", [["subject", "Subject mode", "", "subject"], ["crops", "Show crop suggestions", "", "crops"], ["chips", "Name chips on grid cells", "", "chips"], ["filterbar", "Filter bar", "", "filterbar"], ["denoise", "Denoise", "", "denoise"], ["compare", "Compare with the group"]]],
    ["Tools", [["video", "Include videos", "", "video"], ["autoxmp", "Write XMP sidecars automatically", "", "autoxmp"], ["cameras", "Cameras…"], ["people", "People…"], ["shotlist", "Shot list…"], ["edit", "Edit profile…"], ["matchref", "Colour match: set this photo as reference"],
      ["jersey", "Jersey numbers…", "", "", "jersey"], ["lrsync", "Sync with Lightroom Classic catalog…", "", "", "lrsync"],
      ["auditdone", "Finish audit", "", "", "auditdone"]]],
    ["Help", [["help", "Keyboard shortcuts"], ["setup", "Setup assistant"], ["lrcard", "What to do in Lightroom"]]],
  ];
  function renderHamburger(s) {
    const m = $("hamburger-menu");
    if (!m) return;
    let h = "";
    for (const [group, items] of LAYOUT) {
      h += `<div class="mh">${group}</div>`;
      for (const [id, title, only, check, needs] of items) {
        if (only === "native") continue;
        if (needs && !s[needs]) continue;
        const on = check ? s[check] : null;
        h += `<button role="${check ? "menuitemcheckbox" : "menuitem"}" data-menu="${id}"` + (check ? ` aria-checked="${on}"` : "") + `>${check ? (on ? "✓ " : "") : ""}${title}</button>`;
      }
    }
    m.innerHTML = h;
  }
  const hb = $("btn-hamburger");
  if (hb) {
    const close = () => { $("hamburger-menu").hidden = true; hb.setAttribute("aria-expanded", "false"); };
    hb.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = $("hamburger-menu").hidden;
      if (open) { last = ""; report(); }
      $("hamburger-menu").hidden = !open;
      hb.setAttribute("aria-expanded", String(open));
    });
    $("hamburger-menu").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-menu]");
      if (!b) return;
      close();
      window.shotsieveMenu(b.dataset.menu);
    });
    document.addEventListener("click", close);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); }, true);
  }

  let folderSeen = null, xmpTick = 0;
  function tick() {
    if (folderSeen !== S.folder) { folderSeen = S.folder; loadRecent().then(report); loadXmp(); }
    else { report(); if (++xmpTick % 2 === 0) loadXmp(); }
  }
  setInterval(tick, 2000);
  window.addEventListener("pywebviewready", () => { last = ""; tick(); });
  tick();
})();
