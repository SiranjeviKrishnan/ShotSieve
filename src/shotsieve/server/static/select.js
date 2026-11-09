/* ShotSieve multi-select + Move to Trash (T23i).
   Selection state lives in app.js (S.multi / S.sel / S.anchor, selIds(), toggleSelect(), rangeSelect()); this file adds
   the keyboard (Cmd+A, Shift+arrows, 0 and 6-9 on the whole selection, Cmd+Delete, Cmd+Z after a trash), the grid
   context menu (right-click and, on a Mac, Ctrl-click: Ctrl-click is a right-click there, so it is never a toggle),
   the "N photos selected" batch panel, and Move to Trash with its confirm sheet and Undo.
   Move to Trash is recoverable: the server moves files to the macOS Trash (Finder's "Put Back" works), never deletes. */
"use strict";

(function () {
  const LABELS = [["red", "Red", "#e5484d"], ["yellow", "Yellow", "#f2c230"], ["green", "Green", "#30a46c"],
    ["blue", "Blue", "#3e8ef7"], ["purple", "Purple", "#9b6bff"]];
  const LABEL_KEYS = { 6: "red", 7: "yellow", 8: "green", 9: "blue" };
  const OVERLAYS = ["overlay-compare", "overlay-people", "overlay-cameras", "overlay-shotlist", "overlay-edit", "overlay-lr", "overlay-jersey", "overlay-help", "overlay-trash", "overlay-onboarding"];
  const isTyping = (e) => !!(e.target && e.target.closest && e.target.closest("input, textarea, select"));
  const noOverlay = () => OVERLAYS.every((id) => !$(id) || $(id).hidden) && $("crop-ui").hidden;
  const mainView = () => noOverlay() && $("overlay-loupe").hidden;
  const plural = (n, w = "photo") => `${n} ${w}${n === 1 ? "" : "s"}`;
  const undoStack = [];   // [{token, folder, n, ids}] until the shoot is closed
  let pendingJob = null;  // the incremental cull that follows a trash / undo
  let askAgain = true, rememberedPairs = true;   // "Don't ask again this session"

  /* ---------- keyboard (window, capture: runs before filters.js / app.js) ---------- */

  function neighbour(from, dx, dy) {   // the photo a Shift+arrow moves the cursor to (left / right: one photo; up / down: the next / previous group)
    const i = S.flatIdx.get(from);
    if (i === undefined) return null;
    if (dx) return S.flat[Math.max(0, Math.min(S.flat.length - 1, i + dx))];
    const g = (id) => S.byId.get(id).group_id;
    let j = i;
    if (dy > 0) {
      while (j < S.flat.length && g(S.flat[j]) === g(from)) j++;
      return j < S.flat.length ? S.flat[j] : null;
    }
    while (j >= 0 && g(S.flat[j]) === g(from)) j--;
    if (j < 0) return null;
    const pg = g(S.flat[j]);
    while (j > 0 && g(S.flat[j - 1]) === pg) j--;
    return S.flat[j];
  }

  function allLabel(ids, label) {   // toggle: every photo already has it -> remove, else set on all
    const ims = ids.map((id) => S.byId.get(id)).filter(Boolean);
    return ims.length && ims.every((im) => (im.label || "") === label) ? "none" : label;
  }
  async function applyLabel(ids, label) {
    if (!window.ShotFilters || !ids.length) return;
    S.holdMulti = ids.length > 1;   // filters.js re-selects the primary photo when it is done: keep the selection
    try { await window.ShotFilters.setLabel(ids, label); } finally { S.holdMulti = false; }
    if (ids.length > 1 && S.multi.size < 2) selectMany(ids, S.sel, false);
  }
  const setLabels = (ids, label) => applyLabel(ids, label === "none" ? "none" : allLabel(ids, label));   // 6-9 / dots: toggle on the whole selection
  const setLabelsRaw = applyLabel;   // an explicit choice (no toggling)

  window.addEventListener("keydown", (e) => {
    if (isTyping(e) || e.altKey) return;
    const k = e.key;
    if (e.metaKey && !e.shiftKey && !e.ctrlKey && k.toLowerCase() === "a") {   // Cmd+A: every visible (filtered) photo
      if (!mainView() || !S.flat.length) return;
      e.preventDefault(); e.stopPropagation();
      const primary = S.sel !== null && S.flatIdx.has(S.sel) ? S.sel : S.flat[0];
      S.anchor = S.flat[0];
      selectMany(S.flat.slice(), primary, false);
      return;
    }
    if (e.metaKey && !e.shiftKey && !e.ctrlKey && (k === "Backspace" || k === "Delete")) {   // Cmd+Delete: the Finder shortcut
      if (!noOverlay()) return;
      e.preventDefault(); e.stopPropagation();
      trash();
      return;
    }
    if (e.metaKey && !e.shiftKey && !e.ctrlKey && k.toLowerCase() === "z" && canUndo() && noOverlay()) {
      e.preventDefault(); e.stopPropagation();
      undo();
      return;
    }
    if (e.metaKey || e.ctrlKey) return;
    if (e.shiftKey && (k === "ArrowLeft" || k === "ArrowRight" || k === "ArrowUp" || k === "ArrowDown")) {   // Shift+arrow extends the range
      if (!mainView() || !S.flat.length) return;
      e.preventDefault(); e.stopPropagation();
      if (S.sel === null || !S.flatIdx.has(S.sel)) return;
      const to = neighbour(S.sel, k === "ArrowLeft" ? -1 : k === "ArrowRight" ? 1 : 0, k === "ArrowUp" ? -1 : k === "ArrowDown" ? 1 : 0);
      if (to !== null && to !== undefined) rangeSelect(to, S.anchor);
      return;
    }
    if (e.shiftKey || S.multi.size <= 1 || !noOverlay()) return;
    if (LABEL_KEYS[k]) {   // 6-9 on the whole selection (filters.js would only label one)
      e.preventDefault(); e.stopPropagation();
      setLabels(selIds(), LABEL_KEYS[k]);
    } else if (k === "0") {
      e.preventDefault(); e.stopPropagation();
      decide(null, 0);
    }
  }, true);

  /* ---------- context menu ---------- */

  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || "");
  const dot = (k, n, c, on) => `<button class="lp-dot${on ? " on" : ""}" data-label="${k}" style="--c:${c}" title="${n}" aria-label="${n}" aria-pressed="${!!on}"></button>`;

  function menuHtml(ids) {
    const ims = ids.map((id) => S.byId.get(id)).filter(Boolean);
    const n = ims.length;
    const info = window.ShotLightroom ? window.ShotLightroom.info() : null;
    const apps = info ? info.apps : [];
    const anyAi = ims.some((im) => im.overridden);
    const labelOn = (k) => ims.every((im) => (im.label || "") === k);
    let h = "";
    for (const a of apps) h += `<button role="menuitem" data-app="${esc(a.id)}">${n > 1 ? `Open ${n} Photos in ${esc(a.name)}` : `Open in ${esc(a.name)}`}</button>`;
    if (anyAi) h += `<button role="menuitem" data-ai="1" title="Undo your keep/reject for ${n > 1 ? "these photos" : "this photo"} and use ShotSieve's decision">Use AI decision</button>`;
    h += `<div class="ctx-sep"></div><div class="ctx-label"><span class="muted">Label</span>${LABELS.map(([k, nm, c]) => dot(k, nm, c, labelOn(k))).join("")}<button class="lp-none" data-label="none" title="No label">none</button></div>`;
    h += `<div class="ctx-stars">${[1, 2, 3, 4, 5].map((s) => `<button data-stars="${s}" class="${ims.every((im) => im.rating === s) ? "on" : ""}" title="${s} star${s === 1 ? "" : "s"}">${s}★</button>`).join("")}<button data-stars="0" title="Clear the rating">0</button></div>`;
    h += `<div class="ctx-sep"></div><button role="menuitem" class="ctx-danger" data-trash="1" title="Moves the files to the Trash; Put Back in Finder restores them">${n > 1 ? `Move ${n} Photos to Trash` : "Move to Trash"} <kbd>⌘⌫</kbd></button>`;
    return h;
  }

  function openMenu(e, cell) {
    const id = Number(cell.dataset.id);
    if (!S.multi.has(id) || S.multi.size < 2) select(id, false);   // Finder: right-click outside the selection selects that photo alone; inside keeps it
    const ids = selIds();
    const m = $("ctx-menu");
    m.innerHTML = menuHtml(ids);
    m.hidden = false;
    const r = m.getBoundingClientRect();
    m.style.left = Math.max(4, Math.min(e.clientX, innerWidth - r.width - 6)) + "px";
    m.style.top = Math.max(4, Math.min(e.clientY, innerHeight - r.height - 6)) + "px";
    m.onclick = (ev) => {
      const b = ev.target.closest("button");
      if (!b) return;
      ev.stopPropagation();
      m.hidden = true;
      const cur = selIds();
      if (b.dataset.trash) trash(cur);
      else if (b.dataset.ai) decide("model");
      else if (b.dataset.app) { if (window.ShotLightroom) window.ShotLightroom.open(cur, b.dataset.app); }
      else if (b.dataset.label) setLabels(cur, b.dataset.label);
      else if (b.dataset.stars !== undefined) { const s = Number(b.dataset.stars); if (s || cur.some((i) => S.byId.get(i).rating)) decide(null, s); }
    };
  }
  const gridCell = (e) => (e.target.closest ? e.target.closest("#grid .cell") : null);
  window.addEventListener("contextmenu", (e) => {
    const cell = gridCell(e);
    if (!cell) return;
    e.preventDefault(); e.stopPropagation();   // filters.js / lightroom.js no longer build this menu; this one covers them
    openMenu(e, cell);
  }, true);
  window.addEventListener("mousedown", (e) => {   // WebKit / Chromium on a Mac: Ctrl-click = right-click (some builds send no contextmenu)
    if (!(isMac && e.button === 0 && e.ctrlKey && !e.metaKey && !e.shiftKey)) return;
    const cell = gridCell(e);
    if (!cell) return;
    e.preventDefault(); e.stopPropagation();
    openMenu(e, cell);
  }, true);

  /* ---------- the batch panel ---------- */

  function renderPanel() {
    const box = $("multi-info"), n = S.multi.size;
    if (!box) return;
    const many = n > 1;
    box.hidden = !many;
    if (!many) return;
    $("info").hidden = true;
    const ids = selIds(), ims = ids.map((id) => S.byId.get(id)).filter(Boolean);
    const c = { keep: 0, review: 0, reject: 0 };
    for (const im of ims) c[im.decision] = (c[im.decision] || 0) + 1;
    const allOn = (d) => ims.every((im) => im.overridden && im.decision === d);
    const apps = window.ShotLightroom && window.ShotLightroom.info() ? window.ShotLightroom.info().apps : [];
    box.innerHTML =
      `<h2 id="m-title">${plural(ims.length)} selected</h2>` +
      `<div class="verdict m-sum"><span class="pill keep">${c.keep} keep</span><span class="pill review">${c.review} review</span><span class="pill reject">${c.reject} reject</span></div>` +
      `<div class="labelrow"><span class="muted">Label</span>${LABELS.map(([k, nm, col]) => dot(k, nm, col, ims.every((im) => (im.label || "") === k))).join("")}<button class="lp-none" data-label="none" title="No label">none</button></div>` +
      `<div class="m-stars" aria-label="Stars">${[1, 2, 3, 4, 5].map((s) => `<button data-stars="${s}" class="${ims.every((im) => im.rating === s) ? "on" : ""}" title="${s} star${s === 1 ? "" : "s"} (${s})">${s}★</button>`).join("")}<button data-stars="0" title="Clear the rating (0)">0</button></div>` +
      `<p class="muted m-hint">Keep, Reject, 0-5 and 6-9 apply to all ${ims.length}. <kbd>Esc</kbd> clears the selection.</p>` +
      `<div class="actionbar"><div class="actions">` +
      `<button data-m="keep" class="keep" aria-pressed="${allOn("keep")}" title="Keep all (P). Again: back to ShotSieve's decision">Keep <kbd>P</kbd></button>` +
      `<button data-m="reject" class="reject" aria-pressed="${allOn("reject")}" title="Reject all (X). Again: back to ShotSieve's decision">Reject <kbd>X</kbd></button>` +
      `<button data-m="model" title="Use ShotSieve's decision for all">AI decision</button></div>` +
      `<div class="actions"><button data-m="compare" title="Compare the selected photos side by side (C)">Compare <kbd>C</kbd></button></div>` +
      `<div class="actions">${apps.length ? `<button data-m="lr" title="Open all in Lightroom (at most 500)">Lightroom <kbd>⌘L</kbd></button>` : ""}` +
      `<button data-m="trash" class="danger" title="Move the selected photos to the Trash (recoverable)">Move to Trash <kbd>⌘⌫</kbd></button></div>` +
      `<div class="actions"><button data-m="clear" title="Clear the selection (Esc)">Clear selection <kbd>Esc</kbd></button></div></div>`;
    const sbi = $("sb-info");
    if (sbi) sbi.textContent = `${ims.length} selected · ${S.flat.length} shown`;
  }
  $("multi-info").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    const ids = selIds();
    if (b.dataset.label) setLabels(ids, b.dataset.label);
    else if (b.dataset.stars !== undefined) decide(null, Number(b.dataset.stars));
    else if (b.dataset.m === "keep" || b.dataset.m === "reject" || b.dataset.m === "model") decide(b.dataset.m);
    else if (b.dataset.m === "lr") window.ShotLightroom.trigger(ids, b);
    else if (b.dataset.m === "trash") trash(ids);
    else if (b.dataset.m === "compare") window.ShotCompare.open();
    else if (b.dataset.m === "clear") clearMulti();
  });

  /* ---------- Move to Trash ---------- */

  const sheet = document.createElement("div");
  sheet.id = "overlay-trash"; sheet.className = "overlay"; sheet.hidden = true;
  sheet.setAttribute("role", "alertdialog"); sheet.setAttribute("aria-modal", "true"); sheet.setAttribute("aria-labelledby", "tr-title");
  sheet.innerHTML = '<div class="sheet" id="trash-sheet"></div>';
  document.body.appendChild(sheet);

  async function trash(ids) {
    if (!Array.isArray(ids)) ids = selIds();
    if (!S.folder || !ids.length) { toast("Select a photo first."); return false; }
    if (!askAgain) return doTrash(ids, rememberedPairs, false);
    let pv;
    try { pv = await api("/api/trash/preview", { method: "POST", body: { folder: S.folder, ids } }); } catch (e) { toast(e.message); return false; }
    showSheet(ids, pv);
    return true;
  }

  function showSheet(ids, pv) {
    const n = pv.n, names = pv.names, more = n - names.length;
    const onlyVideo = pv.n_video_frames === n;
    const box = $("trash-sheet");
    box.innerHTML =
      `<h2 id="tr-title">Move ${plural(n)} to the Trash?</h2>` +
      `<p class="muted">You can put ${n === 1 ? "it" : "them"} back from the Trash in Finder.</p>` +
      `<ul class="tr-names">${names.map((x) => `<li>${esc(x)}</li>`).join("")}${more > 0 ? `<li class="muted">and ${more} more</li>` : ""}</ul>` +
      `<label class="tr-opt"><input type="checkbox" id="tr-pairs" ${rememberedPairs ? "checked" : ""} ${pv.n_pairs ? "" : "disabled"}> Also move the paired JPEG/RAW <span class="muted">(${pv.n_pairs ? pv.n_pairs + " found" : "none found"})</span></label>` +
      (pv.n_video_frames ? `<p class="muted tr-note">Video frames can't be trashed individually. ${onlyVideo ? "" : `${plural(pv.n_video_frames, "frame")} will stay.`}</p>` +
        `<label class="tr-opt"><input type="checkbox" id="tr-video"> Also move the video file itself <span class="muted">(every frame of it leaves the shoot)</span></label>` : "") +
      (pv.n_unsafe ? `<p class="tr-note tr-warn">${plural(pv.n_unsafe)} cannot be moved (outside the shoot folder or a symbolic link); they will be skipped.</p>` : "") +
      `<label class="tr-opt"><input type="checkbox" id="tr-noask"> Don't ask again this session</label>` +
      `<div class="tr-btns"><button id="tr-cancel" type="button">Cancel <kbd>Esc</kbd></button><button id="tr-go" type="button" class="danger">Move to Trash</button></div>`;
    sheet.hidden = false;
    const go = $("tr-go"), vid = $("tr-video");
    const sync = () => { go.disabled = onlyVideo && !(vid && vid.checked); };
    if (vid) vid.addEventListener("change", sync);
    sync();
    $("tr-cancel").addEventListener("click", () => { sheet.hidden = true; });
    go.addEventListener("click", () => {
      const pairs = $("tr-pairs").checked, video = !!(vid && vid.checked);
      if ($("tr-noask").checked) { askAgain = false; rememberedPairs = pairs; }
      sheet.hidden = true;
      doTrash(ids, pairs, video);
    });
    $("tr-cancel").focus();   // Cancel is the default
  }

  function removeLocal(gone, fallback) {
    S.images = S.images.filter((im) => !gone.has(im.id));
    S.byId = new Map(S.images.map((im) => [im.id, im]));
    S.groups = S.groups.map((g) => {
      const members = g.members.filter((id) => !gone.has(id));
      return { ...g, members, best: gone.has(g.best) ? members[0] : g.best };
    }).filter((g) => g.members.length);
    S.multi = new Set();
    S.sel = null;
    relayout();
    if (fallback !== null && S.rowOf.has(fallback)) select(fallback);
    else if (S.flat.length) select(S.flat[0]);
    else { S.sel = null; renderDetail(); }
  }

  function toastAction(msg, label, fn, ms = 20000) {
    const t = $("toast");
    t.textContent = msg + " ";
    const b = document.createElement("button");
    b.type = "button"; b.className = "small toast-act"; b.textContent = label;
    b.addEventListener("click", () => { t.hidden = true; fn(); });
    t.appendChild(b);
    t.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { t.hidden = true; }, ms);
  }

  function afterJob(jobId) {
    pendingJob = jobId
      ? waitJob(jobId).then(() => loadShoot(true)).then(() => window.dispatchEvent(new CustomEvent("people-changed"))).catch(() => {})
      : loadShoot(true);
    return pendingJob;
  }

  async function doTrash(ids, pairs, video) {
    if (pendingJob) await pendingJob;
    const idx = ids.map((id) => S.flatIdx.get(id)).filter((i) => i !== undefined);
    const drop = new Set(ids);
    let fallback = idx.length ? (S.flat.find((id, i) => i > Math.max(...idx) && !drop.has(id)) ?? null) : null;
    if (fallback === null && idx.length) for (let i = Math.min(...idx) - 1; i >= 0; i--) if (!drop.has(S.flat[i])) { fallback = S.flat[i]; break; }
    // stop every request for the photos that are about to leave (a late preview / face crop would be a 404)
    clearTimeout(S.loupeTimer); clearTimeout(preloadNeighbours.t);
    $("loupe").removeAttribute("src");
    document.querySelectorAll("#d-faces-list img").forEach((i) => i.removeAttribute("src"));
    let r;
    try {
      r = await api("/api/trash", { method: "POST", body: { folder: S.folder, ids, include_pairs: pairs, include_videos: video } });
    } catch (e) { toast("Nothing was moved: " + e.message, 7000); return false; }
    const n = r.trashed.length;
    if (!n) {
      const why = r.failed.length ? r.failed[0].error : r.skipped.length ? r.skipped[0].reason : "nothing to move";
      toast("Nothing was moved to the Trash: " + why + ".", 8000);
      return false;
    }
    removeLocal(new Set(r.trashed), fallback);
    let msg = `Moved ${plural(n)} to Trash.`;
    if (r.failed.length) msg += ` ${r.failed.length} could not be moved (${r.failed[0].error}).`;
    if (r.skipped.length) msg += ` ${plural(r.skipped.length, "video frame")} stayed: video frames can't be trashed individually.`;
    undoStack.push({ token: r.token, folder: S.folder, n, ids: r.trashed });
    toastAction(msg, "Undo", undo);
    afterJob(r.job_id);
    return true;
  }

  const canUndo = () => undoStack.some((u) => u.folder === S.folder);
  async function undo() {
    let rec = null;
    for (let i = undoStack.length - 1; i >= 0; i--) if (undoStack[i].folder === S.folder) { rec = undoStack[i]; break; }
    if (!rec) { toast("Nothing to undo."); return false; }
    if (pendingJob) await pendingJob;
    let r;
    try { r = await api("/api/trash/undo", { method: "POST", body: { folder: S.folder, token: rec.token } }); } catch (e) {
      if (e.status === 404) undoStack.splice(undoStack.indexOf(rec), 1);
      toast("Could not undo: " + e.message, 7000);
      return false;
    }
    if (!r.failed.length) undoStack.splice(undoStack.indexOf(rec), 1);
    if (r.job_id) await afterJob(r.job_id); else await loadShoot(true);
    const back = r.ids.filter((id) => S.byId.has(id));
    if (back.length > 1) selectMany(back, back[0]); else if (back.length) select(back[0]);
    window.dispatchEvent(new CustomEvent("people-changed"));
    toast(r.failed.length ? `Put ${plural(r.restored, "file")} back; ${r.failed.length} could not be restored (${r.failed[0].error}).` : `Put ${plural(back.length || r.restored)} back.`, 7000);
    return true;
  }

  window.ShotSelect = { renderPanel, trash, undo, canUndo, setLabels: setLabelsRaw, ids: () => selIds() };
})();
