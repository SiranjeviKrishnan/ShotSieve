/* ShotSieve review UI: vanilla JS, no build step. */
"use strict";

const CELL_W = 156 + 6;   // cell width + gap
const ROW_H = 140;
const LABEL_W = 64;
const BUFFER_ROWS = 3;
const FILTERS = ["all", "keep", "review", "reject", "audit"];

const S = {
  folder: "",
  dropped: new Set(), // image ids just added by drag and drop (highlighted)
  focusNames: null,   // file names of the pending drop
  images: [],
  byId: new Map(),
  groups: [],
  stats: null,
  filter: "all",
  collapsed: false,
  sel: null,          // selected image id (the one shown in the detail panel; the "cursor")
  multi: new Set(),   // T23i: every selected image id (always contains `sel`; a single selection is a set of one)
  anchor: null,       // image id a Shift-click / Shift+arrow range starts from
  rows: [],           // layout rows
  flat: [],           // visible image ids in navigation order
  flatIdx: new Map(), // image id -> index in `flat` (no indexOf scans at 10k)
  rowOf: new Map(),   // image id -> row index
  rendered: { from: -1, to: -1 },
  job: null,
  loupeTimer: null,
  jerseys: {},
  editTrained: false, // an editing profile exists (T15)
  editPred: new Map(), // image id -> predicted edits (cached)
  matchRef: null,     // image id of the colour-match reference (T16)
  matchMsg: "",
  search: null,       // Map id -> score while a search is shown
  searchQ: "",
};

const REASON_LABELS = {
  blur: "Blurry", face_out_of_focus: "Missed focus on face", af_missed: "Missed AF point", banding: "Light-flicker banding",
  eyes_closed: "Eyes closed", exposure_bad: "Poor exposure", flash_misfire: "Flash misfire", accidental: "Accidental shot",
  duplicate: "Duplicate of a better shot", no_subject: "No clear subject", low_quality: "Low overall quality",
  unreadable: "Could not be read", duplicate_tie: "Near-identical to a better frame", uncertain: "Borderline score",
  best_available: "Best available of its moment", audit: "Spot-check",
};
const SOFT_REASONS = new Set(["duplicate_tie", "uncertain", "best_available", "audit"]);

const $ = (id) => document.getElementById(id);
const grid = $("grid");
const spacer = $("spacer");

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const isNative = () => !!(window.shotsieveDesktop && window.shotsieveDesktop.available());
const nameOf = (p) => String(p).replace(/\/+$/, "").split("/").pop() || String(p);
function setFolderField(p) {
  $("folder").value = p;
  const chip = $("folder-chip");
  chip.hidden = !p;
  chip.textContent = p ? nameOf(p) : "";
  chip.title = p || "";
}
async function chooseFolder() {
  if (!isNative()) return;
  let p = null;
  try { p = await window.shotsieveDesktop.pick("folder"); } catch (e) { toast(e.message); }
  if (p) { setFolderField(p); startCull(); }
}
async function toggleRecent(force) {
  const m = $("recent-menu");
  const open = force === undefined ? m.hidden : force;
  $("btn-recent").setAttribute("aria-expanded", String(open));
  if (!open) { m.hidden = true; return; }
  let folders = [];
  try { folders = (await api("/api/recent")).folders; } catch (e) { /* none */ }
  m.innerHTML = folders.length
    ? folders.map((f, n) => `<button role="menuitem" data-n="${n}" title="${esc(f.path)}">${esc(f.name)}</button>`).join("")
    : `<div class="none">No recent folders</div>`;
  m.querySelectorAll("button").forEach((b) => (b.onclick = () => { toggleRecent(false); setFolderField(folders[+b.dataset.n].path); startCull(); }));
  m.hidden = false;
}
$("btn-choose").addEventListener("click", chooseFolder);
$("btn-recent").addEventListener("click", (e) => { e.stopPropagation(); toggleRecent(); });
document.addEventListener("click", () => { if (!$("recent-menu").hidden) toggleRecent(false); });
function q(params) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== "") u.set(k, v);
  const s = u.toString();
  return s ? "?" + s : "";
}
function toast(msg, ms = 3500) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, ms);
}
async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: {} };
  if (opts.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const r = await fetch(path, init);
  let data = null;
  try { data = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : r.statusText;
    const err = new Error(detail);
    err.status = r.status;
    throw err;
  }
  return data;
}
// grid thumbnails are rendered for the screen they are on: x2 on a Retina display (T23f)
const thumbUrl = (id, size = 400) => `/api/thumb/${id}${q({ size: Math.min(1600, size * ((window.devicePixelRatio || 1) >= 1.5 ? 2 : 1)), folder: S.folder })}`;
// the picture the loupe needs (T23f): the box it is shown in, in device pixels (rounded up to 128).
// The server knows the image's aspect, works out the long edge it needs at Fit and snaps it up to
// 1280 / 2048 / 2880 / 3840, so the cache stays effective and nothing is upscaled or over-fetched.
function viewBox(el) {
  const dpr = window.devicePixelRatio || 1;
  el = el || (!$("overlay-loupe").hidden ? $("lz-view") : $("loupe-wrap"));
  let W = el ? el.clientWidth : 0, H = el ? el.clientHeight : 0;
  if (W < 40 || H < 40) { W = window.innerWidth; H = window.innerHeight; }
  const up = (v) => Math.max(128, Math.ceil((v * dpr) / 128) * 128);
  return { bw: up(W), bh: up(H) };
}
const previewUrl = (id, el) => `/api/preview/${id}${q({ ...viewBox(el), folder: S.folder })}`;
const fullUrl = (id) => `/api/full/${id}${q({ folder: S.folder })}`;
const fullSizeUrl = (id) => `/api/full/${id}/size${q({ folder: S.folder })}`;
const denoiseUrl = (id, el, level) => `/api/denoise/${id}${q({ level, ...viewBox(el), folder: S.folder })}`;
const denoiseFullUrl = (id, level) => `/api/denoise/${id}/full${q({ level, folder: S.folder })}`;
const denoiseRegionUrl = (id, level, b) => `/api/denoise/${id}/region${q({ level, x0: b[0].toFixed(5), y0: b[1].toFixed(5), x1: b[2].toFixed(5), y1: b[3].toFixed(5), folder: S.folder })}`;
const zoomUrls = { preview: previewUrl, full: fullUrl, size: fullSizeUrl, denoise: denoiseUrl, denoiseFull: denoiseFullUrl, denoiseRegion: denoiseRegionUrl };

/* ---------- data ---------- */

async function loadShoot(silent = false) {
  if (!S.folder) return;
  if (!S.images.length) { const em = $("empty"); em.hidden = false; em.classList.add("loading"); em.textContent = "Loading…"; }
  try {
    const d = await api("/api/shoot" + q({ folder: S.folder }));
    S.images = d.images;
    S.byId = new Map(d.images.map((im) => [im.id, im]));
    S.groups = d.groups;
    S.stats = d.stats;
    if (S.sel !== null && !S.byId.has(S.sel)) S.sel = null;
    if (window.ShotFilters) window.ShotFilters.onShoot();   // T23e: restore the filters, load camera / people data
    relayout(true);
    if (S.sel === null && S.flat.length) select(S.flat[0]);
  } catch (e) {
    S.images = []; S.byId = new Map(); S.groups = []; S.stats = null;
    relayout(true);
    $("empty").hidden = false;
    $("empty").classList.remove("loading");
    $("empty").textContent = e.status === 404 ? "This folder has not been culled yet. Press Cull." : e.message;
    if (!silent && e.status !== 404) toast(e.message);
  }
}

function counts() {
  const c = { keep: 0, review: 0, reject: 0, audit: 0, auditPending: 0 };
  for (const im of S.images) {
    c[im.decision] = (c[im.decision] || 0) + 1;
    if (im.audit) {
      c.audit++;
      if (!im.overridden || im.decision === "review") c.auditPending++;
    }
  }
  return c;
}

function updateStats() {
  const c = counts();
  $("c-all").textContent = S.images.length;
  $("c-keep").textContent = c.keep;
  $("c-review").textContent = c.review;
  $("c-reject").textContent = c.reject;
  $("c-audit").textContent = c.auditPending + "/" + c.audit;
  $("btn-audit-done").hidden = c.audit === 0;
  const needs = S.images.filter((im) => im.edit && im.edit.needs_edit).length;
  const any = S.images.some((im) => im.edit);
  $("tab-needs-edit").hidden = !any;
  $("c-needs-edit").textContent = needs;
  if (!any && S.filter === "needs_edit") S.filter = "all";
}

/* ---------- layout (rows of groups; virtualised) ---------- */

function matches(im) {
  if (S.search && !S.search.has(im.id)) return false;
  if (window.ShotFilters && !window.ShotFilters.pass(im)) return false;   // T23e filter bar
  if (S.filter === "all") return true;
  if (S.filter === "audit") return im.audit;
  if (S.filter === "needs_edit") return !!(im.edit && im.edit.needs_edit);
  return im.decision === S.filter;
}

function relayout(resetScroll = false) {
  const t0 = performance.now();
  $("empty").classList.remove("loading");
  const width = Math.max(grid.clientWidth, 320);
  const perGroup = Math.max(1, Math.floor((width - LABEL_W - 24) / CELL_W));
  const perSingles = Math.max(1, Math.floor((width - 24) / CELL_W));
  const rows = [];
  let singles = [];
  const flushSingles = () => {
    for (let i = 0; i < singles.length; i += perSingles) rows.push({ kind: "singles", ids: singles.slice(i, i + perSingles) });
    singles = [];
  };
  if (S.search) {  // search active: rank by score (best first), ignoring groups
    const ids = [...S.search.keys()].filter((id) => S.byId.has(id) && matches(S.byId.get(id)));
    ids.sort((a, b) => S.search.get(b) - S.search.get(a));  // stable: ties keep server order
    singles = ids;
    flushSingles();
  } else for (const g of S.groups) {
    let members = g.members.filter((id) => matches(S.byId.get(id)));
    if (!members.length) continue;
    if (g.members.length === 1) { singles.push(members[0]); continue; }
    flushSingles();
    const total = members.length;
    if (S.collapsed) members = [members.includes(g.best) ? g.best : members[0]];
    for (let i = 0; i < members.length; i += perGroup) {
      rows.push({ kind: "group", gid: g.id, ids: members.slice(i, i + perGroup), first: i === 0, total, size: g.members.length });
    }
  }
  flushSingles();
  S.rows = rows;
  S.flat = rows.flatMap((r) => r.ids);
  S.flatIdx = new Map(S.flat.map((id, i) => [id, i]));
  S.rowOf = new Map();
  rows.forEach((r, i) => r.ids.forEach((id) => S.rowOf.set(id, i)));
  spacer.style.height = rows.length * ROW_H + 8 + "px";
  if (resetScroll) grid.scrollTop = 0;
  $("empty").hidden = rows.length > 0 || S.images.length === 0 ? true : false;
  if (!rows.length && S.images.length) { $("empty").hidden = false; $("empty").textContent = "Nothing in this filter."; }
  S.rendered = { from: -1, to: -1 };
  renderRows();
  updateStats();
  if (S.multi.size > 1) {   // photos that left the grid (filter / decision) leave the selection
    S.multi = new Set([...S.multi].filter((id) => S.rowOf.has(id)));
    if (S.multi.size && !S.multi.has(S.sel)) S.sel = selIds().pop();
  }
  if (S.sel !== null && !S.rowOf.has(S.sel) && S.flat.length) select(S.flat[0], false);
  $("btn-collapse").textContent = "Groups: " + (S.collapsed ? "collapsed" : "expanded");
  window.__layoutMs = performance.now() - t0;  // T21 budget: <= 100 ms at 10k images
  if (window.ShotFilters) window.ShotFilters.afterLayout();
}

/* Thumbnails load through a small queue: the visible rows first (in viewport order), then one
   screen further, then the buffer rows. Requests for cells that scrolled away are aborted. */
const THUMB_PAR = 8;
const thumbQ = [];
const inflight = new Set();
function pumpThumbs() {
  while (inflight.size < THUMB_PAR && thumbQ.length) {
    const img = thumbQ.shift();
    if (!img.isConnected || !img.dataset.src) continue;
    inflight.add(img);
    const fin = () => { inflight.delete(img); pumpThumbs(); };
    img.addEventListener("load", fin, { once: true });
    img.addEventListener("error", fin, { once: true });
    img.src = img.dataset.src;
    delete img.dataset.src;
  }
}
function queueThumbs() {
  const first = Math.floor(grid.scrollTop / ROW_H), last = Math.ceil((grid.scrollTop + grid.clientHeight) / ROW_H);
  const screen = Math.max(1, last - first);
  thumbQ.length = 0;
  const waiting = [];
  for (const el of spacer.querySelectorAll(".row")) {
    const i = Number(el.dataset.row);
    const pri = i >= first && i < last ? 0 : i >= last && i < last + screen ? 1 : 2;
    for (const img of el.querySelectorAll("img[data-src]")) waiting.push([pri, i, img]);
  }
  waiting.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  for (const w of waiting) thumbQ.push(w[2]);
  pumpThumbs();
}
function abortThumbs(rowEl) {
  for (const img of rowEl.querySelectorAll("img")) {
    if (inflight.delete(img)) img.removeAttribute("src");  // cancels the request
  }
}

function cellHtml(id, row) {
  const im = S.byId.get(id);
  const star = im.best && im.group_id !== undefined && row.size > 1 ? '<span class="star" title="Best of its group">★</span>' : "";
  const audit = im.audit ? '<span class="badge" title="Spot-check of a reject">AUDIT</span>' : "";
  const more = S.collapsed && row.kind === "group" && row.total > 1 ? `<span class="badge more">+${row.total - 1}</span>` : "";
  const vid = im.video_t != null ? `<span class="badge" title="Frame at ${im.video_t.toFixed(2)} s of a video (G exports it)">VIDEO</span>` : "";
  const lr = im.lr ? '<span class="badge lr" title="Your choice, synced from Lightroom">LR</span>' : "";
  const ovr = im.overridden ? '<span class="badge ovr" title="You changed this">edited</span>' : "";
  const rate = im.rating ? `<span class="rate">${"★".repeat(im.rating)}</span>` : "";
  const crop = window.ShotCrop ? window.ShotCrop.badge(im) : "";
  const pic = window.ShotCrop ? window.ShotCrop.cellImg(im, thumbUrl(id)) : `<img alt="" draggable="false" data-src="${thumbUrl(id)}">`;
  const chips = window.ShotFaces ? window.ShotFaces.chipHtml(id) : "";
  const many = S.multi.size > 1 && S.multi.has(id);
  return `<div class="cell dec-${im.decision}${id === S.sel ? " sel" : ""}${many ? " msel" : ""}${S.dropped.has(id) ? " dropped" : ""}" data-id="${id}" title="${esc(im.name)}"${many ? ' aria-selected="true"' : ""}>` +
    `${many ? '<span class="chk" aria-hidden="true">✓</span>' : ""}${pic}${star}${audit}${vid}${more}${ovr}${lr}${crop}${window.ShotFilters ? window.ShotFilters.cellBadges(im) : rate}${chips}` +
    `<div class="cap">${esc(im.name)}</div></div>`;
}

function renderRows(force = false) {
  const top = grid.scrollTop, h = grid.clientHeight;
  const from = Math.max(0, Math.floor(top / ROW_H) - BUFFER_ROWS);
  const to = Math.min(S.rows.length, Math.ceil((top + h) / ROW_H) + BUFFER_ROWS);
  if (!force && from === S.rendered.from && to === S.rendered.to) return;
  S.rendered = { from, to };
  for (const el of spacer.querySelectorAll(".row")) { abortThumbs(el); el.remove(); }
  const frag = document.createDocumentFragment();
  for (let i = from; i < to; i++) {
    const r = S.rows[i];
    const el = document.createElement("div");
    el.className = "row " + r.kind;
    el.dataset.row = i;
    el.style.top = i * ROW_H + 4 + "px";
    const label = r.kind === "group" ? `<div class="glabel">${r.first ? "Group<br>×" + r.size : ""}</div>` : "";
    el.innerHTML = label + r.ids.map((id) => cellHtml(id, r)).join("");
    frag.appendChild(el);
  }
  spacer.appendChild(frag);
  queueThumbs();
}

grid.addEventListener("scroll", () => renderRows(), { passive: true });
window.addEventListener("resize", () => relayout());
grid.addEventListener("click", (e) => {
  const cell = e.target.closest(".cell");
  if (!cell) return;
  const id = Number(cell.dataset.id);
  if (e.ctrlKey && !e.metaKey) return;   // Mac: Ctrl-click is a right-click (the context menu), never a selection change
  if (e.shiftKey) return rangeSelect(id);
  if (e.metaKey) return toggleSelect(id);
  select(id);
});
grid.addEventListener("dblclick", (e) => {
  if (e.target.closest(".cell") && !e.shiftKey && !e.metaKey && !e.ctrlKey) toggleLoupe();
});
spacer.addEventListener("error", (e) => {
  if (e.target.tagName === "IMG") e.target.style.visibility = "hidden";
}, true);

/* ---------- selection + detail ---------- */

/* T23i multi-select. S.multi holds every selected id (a plain selection is a set of one), S.sel is the
   photo shown in the detail panel. Cmd-click toggles, Shift-click selects the range from the anchor in the
   current grid order; Ctrl-click stays the Mac right-click. */
const selIds = () => (S.multi.size > 1 ? S.flat.filter((id) => S.multi.has(id)) : S.sel !== null && S.sel !== undefined && S.byId.has(S.sel) ? [S.sel] : []);
function paintSel() {
  const many = S.multi.size > 1;
  spacer.querySelectorAll(".cell").forEach((el) => {
    const id = Number(el.dataset.id), on = many && S.multi.has(id);
    el.classList.toggle("sel", id === S.sel);
    el.classList.toggle("msel", on);
    if (on) el.setAttribute("aria-selected", "true"); else el.removeAttribute("aria-selected");
    const chk = el.querySelector(":scope > .chk");
    if (on && !chk) el.insertAdjacentHTML("afterbegin", '<span class="chk" aria-hidden="true">✓</span>');
    else if (!on && chk) chk.remove();
  });
}
function selectMany(ids, primary, scroll = true) {   // ids: image ids (any order); primary: the one shown in the panel
  const keep = ids.filter((id) => S.rowOf.has(id));
  if (!keep.length) return;
  S.multi = new Set(keep);
  if (!S.multi.has(primary)) primary = keep[keep.length - 1];
  S.sel = primary;
  if (S.anchor === null || !S.multi.has(S.anchor)) S.anchor = keep[0];
  afterSelect(primary, scroll);
}
function toggleSelect(id) {
  if (S.multi.size <= 1 && S.sel !== null) S.multi = new Set([S.sel]);
  if (S.multi.has(id) && S.multi.size > 1) {
    S.multi.delete(id);
    if (S.sel === id) S.sel = selIds().pop() ?? [...S.multi][0];
  } else if (!S.multi.has(id)) {
    S.multi.add(id); S.sel = id;
  }
  S.anchor = id;
  afterSelect(S.sel, false);
}
function rangeSelect(id, from = S.anchor) {
  if (from === null || !S.flatIdx.has(from)) from = S.sel !== null && S.flatIdx.has(S.sel) ? S.sel : id;
  const a = S.flatIdx.get(from), b = S.flatIdx.get(id);
  if (a === undefined || b === undefined) return select(id);
  const [lo, hi] = a < b ? [a, b] : [b, a];
  S.multi = new Set(S.flat.slice(lo, hi + 1));
  S.anchor = from; S.sel = id;
  afterSelect(id, true);
}
function afterSelect(id, scroll) {
  paintSel();
  if (scroll && S.rowOf.has(id)) scrollToCell(id);
  renderDetail();
  if (!$("overlay-compare").hidden && window.ShotCompare) window.ShotCompare.refresh();
  if (!$("overlay-loupe").hidden) showLoupe(id);
}
function scrollToCell(id) {
  const top = S.rowOf.get(id) * ROW_H;
  if (top < grid.scrollTop) grid.scrollTop = top - 4;
  else if (top + ROW_H > grid.scrollTop + grid.clientHeight) grid.scrollTop = top + ROW_H - grid.clientHeight + 8;
  renderRows();
}
function clearMulti() {   // Esc: back to the one photo in the panel
  if (S.multi.size <= 1) return false;
  S.multi = new Set([S.sel]); S.anchor = S.sel;
  afterSelect(S.sel, false);
  return true;
}

function select(id, scroll = true) {
  S.sel = id;
  if (!(S.holdMulti && S.multi.size > 1 && S.multi.has(id))) { S.multi = new Set([id]); S.anchor = id; }   // holdMulti: a batch action (select.js) re-selects its primary photo without dropping the selection
  paintSel();
  if (scroll && S.rowOf.has(id)) {
    const top = S.rowOf.get(id) * ROW_H;
    if (top < grid.scrollTop) grid.scrollTop = top - 4;
    else if (top + ROW_H > grid.scrollTop + grid.clientHeight) grid.scrollTop = top + ROW_H - grid.clientHeight + 8;
    renderRows();
  }
  renderDetail();
  if (!$("overlay-compare").hidden && window.ShotCompare) window.ShotCompare.refresh();
  if (!$("overlay-loupe").hidden) showLoupe(id);
}

function preloadNeighbours(id) {   // the next and previous previews, so arrow-key browsing is instant
  clearTimeout(preloadNeighbours.t);
  preloadNeighbours.t = setTimeout(() => {
    const i = S.flatIdx.get(id);
    if (i === undefined) return;
    for (const j of [i + 1, i - 1]) {
      const nid = S.flat[j];
      if (nid !== undefined && !S.byId.get(nid).video_t) { const im = new Image(); im.src = previewUrl(nid); }
    }
  }, 250);
}
function renderChips(im) {
  const box = $("d-chips");
  box.innerHTML = "";
  const codes = im.reasons || [];
  if (!codes.length) {
    const c = document.createElement("span");
    c.className = "chip " + (im.decision === "keep" ? "good" : "");
    c.textContent = im.decision === "keep" ? "No problems found" : "No reason recorded";
    box.appendChild(c);
    return;
  }
  for (const code of codes) {
    const c = document.createElement("span");
    c.className = "chip " + (SOFT_REASONS.has(code) ? "warn" : "bad");
    c.textContent = REASON_LABELS[code] || code;
    box.appendChild(c);
  }
}
function renderDetail() {
  renderDetailOne();
  if (window.ShotSelect) window.ShotSelect.renderPanel();   // T23i: "N photos selected" + batch actions replace the one-photo panel
}
function renderDetailOne() {
  if (window.ShotFilters) window.ShotFilters.onSelect();
  const im = S.byId.get(S.sel);
  $("info").hidden = !im;
  $("dn-main").hidden = !im;
  $("loupe-empty").hidden = !!im;
  if (!im) { $("loupe").removeAttribute("src"); if (window.ShotFaces) window.ShotFaces.panel(null); if (window.ShotCrop) { window.ShotCrop.syncButton(null); window.ShotCrop.loupe(null); } return; }
  clearTimeout(S.loupeTimer);
  S.loupeTimer = setTimeout(() => {
    const lp = $("loupe"), url = S.dnMain ? denoiseUrl(im.id, $("loupe-wrap"), ShotZoom.denoiseLevel()) : previewUrl(im.id, $("loupe-wrap"));
    if (lp.getAttribute("src") !== url) {
      $("loupe-wrap").classList.add("loading");
      if (S.dnMain) drawMainDenoise(true);
      const done = () => $("loupe-wrap").classList.remove("loading");
      lp.onload = done; lp.onerror = done;
      lp.src = url;
    }
    if (window.ShotCrop) window.ShotCrop.loupe(im);
    preloadNeighbours(im.id);
  }, 60);
  if (window.ShotCrop) window.ShotCrop.syncButton(im);
  const grp = S.groups.find((g) => g.id === im.group_id);
  document.querySelectorAll('.actions button[data-act="keep"], .actions button[data-act="reject"]').forEach((b) => {
    const on = im.overridden && im.decision === b.dataset.act;
    b.setAttribute("aria-pressed", String(on));
    b.classList.toggle("on", on);
  });
  $("d-name").textContent = im.rel;
  const dec = $("d-decision");
  dec.textContent = im.decision + (im.overridden ? " (yours)" : "");
  dec.className = "pill " + im.decision;
  $("d-audit").hidden = !im.audit;
  $("d-rating").textContent = im.rating ? "★".repeat(im.rating) : "";
  const pct = Math.round(Math.max(0, Math.min(1, im.score)) * 100);
  $("d-score-fill").style.width = pct + "%";
  $("d-score").textContent = "score " + pct + "%";
  renderChips(im);
  if (window.ShotFaces) window.ShotFaces.panel(im);
  if (window.ShotLightroom) window.ShotLightroom.sync();
  const sbi = $("sb-info");
  if (sbi) sbi.textContent = `${S.flat.length} shown · ${im.rel}`;
  const ul = $("d-reasons");
  ul.innerHTML = "";
  const texts = im.reason_texts.length ? im.reason_texts : [im.decision === "keep" ? "No problems found: sharp, well exposed, eyes open." : "No specific reason recorded."];
  texts.forEach((t, i) => {
    const li = document.createElement("li");
    const code = im.reasons[i] || "";
    const dup = im.duplicate_of;
    const at = dup && (code === "duplicate" || code === "duplicate_tie") ? t.indexOf(dup.name) : -1;
    if (at < 0) { li.textContent = t; ul.appendChild(li); return; }
    // the better frame's file name is a link (built with DOM nodes: nothing is parsed as HTML)
    const a = document.createElement("a");
    a.href = "#";
    a.className = "dup-link";
    a.dataset.id = dup.image_id;
    a.textContent = dup.name;
    a.title = "Open " + dup.name + " in the loupe";
    li.append(t.slice(0, at), a, t.slice(at + dup.name.length));
    const cmp = document.createElement("button");
    cmp.type = "button";
    cmp.className = "small dup-compare";
    cmp.dataset.id = dup.image_id;
    cmp.textContent = "Compare side by side";
    li.append(" ", cmp);
    ul.appendChild(li);
  });
  renderJerseyNote();
  renderEditBadge(im);
  const note = $("d-note");
  note.hidden = !im.audit;
  if (im.audit) {
    const p = im.audit_prob ? ` (picked with probability ${(im.audit_prob * 100).toFixed(0)}%)` : "";
    note.textContent = "Spot-check: the model wanted to reject this photo" + p + ". Press P if you would have kept it, X to confirm the reject. Your answers tune how strict the model is.";
  }
}

/* ---------- editing profile (T15) ---------- */

const EDIT_LABELS = {
  Exposure2012: "Exposure", Contrast2012: "Contrast", Highlights2012: "Highlights", Shadows2012: "Shadows",
  Whites2012: "Whites", Blacks2012: "Blacks", Vibrance: "Vibrance", Saturation: "Saturation",
  Temperature: "Temperature", Tint: "Tint", IncrementalTemperature: "Temp (rel.)", IncrementalTint: "Tint (rel.)",
};
function fmtEdit(k, v) {
  if (k === "Exposure2012") return (v >= 0 ? "+" : "") + v.toFixed(2);
  if (k === "Temperature") return Math.round(v) + " K";
  return (v >= 0 ? "+" : "") + Math.round(v);
}
async function renderEditBadge(im) {
  const box = $("d-edit");
  if (!S.editTrained && !im.edit) { box.hidden = true; return; }
  let p = S.editPred.get(im.id);
  if (!p) {
    try {
      const d = await api("/api/edit/predictions" + q({ ids: im.id, folder: S.folder }));
      p = d.predictions[String(im.id)] || null;
      S.editPred.set(im.id, p);
    } catch (e) { p = null; }
  }
  if (S.sel !== im.id) return;   // the selection moved while we waited
  if (!p) { box.hidden = true; return; }
  const rows = Object.entries(p.values || {}).map(([k, v]) => {
    const c = p.confidence && p.confidence[k] !== undefined ? Math.round(p.confidence[k] * 100) + "%" : "";
    return `<tr><td>${esc(EDIT_LABELS[k] || k)}</td><td class="v">${esc(fmtEdit(k, v))}</td><td class="c">${c}</td></tr>`;
  }).join("");
  const state = p.needs_edit ? `<span class="pill needs">needs your edit</span>` : `<span class="pill keep">predicted</span>`;
  const written = p.applied ? ` · ${esc(p.written_mode || "")}` : " · not applied yet";
  box.innerHTML = `<b>Edit profile</b> ${state}<span class="note-inline">${written}</span>` +
    (p.omitted && p.omitted.length ? `<div>Withheld (low confidence): ${p.omitted.map((k) => esc(EDIT_LABELS[k] || k)).join(", ")}</div>` : "") +
    (rows ? `<table>${rows}</table>` : "<div>No slider is confident enough for this image.</div>");
  box.hidden = false;
}
/* ---------- shot list coverage (T17) ---------- */
const SL = { data: null, text: "", list: "", tag: false };
const SL_MARK = { covered: "✓", uncertain: "?", weak: "~", missing: "✗", pending: "…" };
function renderShotBanner(d) {
  const b = $("shotlist-banner");
  const miss = (d && d.must_missing) || [];
  const unc = (d && d.must_uncertain) || [];
  if (!miss.length && !unc.length) { b.hidden = true; return; }
  b.hidden = false;
  b.innerHTML = (miss.length ? `<b>Missing must-have shots:</b> <span id="shotlist-missing">${miss.map(esc).join(", ")}</span> ` : "") +
    (unc.length ? `<b>Review must-have shots (uncertain match):</b> <span id="shotlist-uncertain">${unc.map(esc).join(", ")}</span> ` : "") +
    `<button id="btn-shotlist-open">Open shot list</button>`;
  $("btn-shotlist-open").onclick = () => showShotlist();
}
async function checkShotlist() {   // after a cull or an index run finishes
  if (!S.folder || !S.stats) return;
  try {
    const d = await api("/api/shotlist/coverage" + q({ folder: S.folder }));
    SL.data = d;
    renderShotBanner(d);
    if (!$("overlay-shotlist").hidden) renderShotlist();
  } catch (e) { /* no models, no list or not culled yet */ }
}
async function loadShotlistStored() {   // page load: what was last computed (no model work)
  if (!S.folder || !S.stats) return;
  try { SL.data = await api("/api/shotlist" + q({ folder: S.folder })); renderShotBanner(SL.data); } catch (e) { /* not culled yet */ }
}
function showShotFrames(it) {
  const ids = (it.best_ids || []).filter((id) => S.byId.has(id));
  if (!ids.length) { toast("No frames for this shot."); return; }
  S.search = new Map(ids.map((id, i) => [id, 1 - i * 0.01]));
  S.searchQ = it.text;
  $("search").value = "";
  closeOverlays();
  relayout(true);
  if (S.flat.length) select(S.flat[0]);
  toast(`${ids.length} frame(s) for "${it.text}" (Esc shows all)`);
}
function renderShotlist(msg = "") {
  const d = SL.data || { items: [], templates: [], counts: {} };
  const box = $("shotlist-sheet");
  const c = d.counts || {};
  let h = "<h2>Shot list</h2><p class='note'>Checks the shots you need against the frames you kept or are reviewing (rejects do not count). One shot per line: <code>!must-have shot</code>, <code>group photo x3</code>, <code>speech person:\"Ana\"</code>.</p>";
  h += `<div class="btns"><select id="sl-template" aria-label="Template"><option value="">Template…</option>` +
    (d.templates || []).map((t) => `<option value="${esc(t)}"${SL.list === t ? " selected" : ""}>${esc(t)}</option>`).join("") +
    `</select><button id="sl-load" class="primary">Load list</button><button id="sl-check">Re-check</button>` +
    `<button id="sl-clear">Clear</button><label class="chk"><input id="sl-tag" type="checkbox"${SL.tag ? " checked" : ""}> tag matched keeps in XMP</label>` +
    `<span id="sl-msg" class="note">${esc(msg)}</span></div>`;
  h += `<textarea id="sl-text" spellcheck="false" placeholder="or paste your own list here">${esc(SL.text)}</textarea>`;
  if ((d.items || []).length) {
    h += `<p id="sl-summary">${c.covered || 0} covered, ${c.uncertain || 0} uncertain, ${c.weak || 0} weak, ${c.missing || 0} missing, ${c.pending || 0} pending` +
      (d.indexed_pct !== undefined && d.indexed_pct < 100 ? ` (index ${Math.round(d.indexed_pct)} %)` : "") + "</p>";
    h += `<table id="sl-items">` + d.items.map((it) => {
      const label = (it.must ? '<span class="must">!</span> ' : "") + esc(it.text) + (it.min_count > 1 ? ` <span class="note">x${it.min_count}</span>` : "") +
        (it.person ? ` <span class="note">${esc(it.person)}</span>` : "");
      const note = it.status === "pending" ? (it.detail || "pending") : it.status === "uncertain" ? (it.detail || "uncertain: review") : it.status === "weak" ? `weak: ${it.n_hits}/${it.min_count}` : "";
      return `<tr class="sl-item sl-${esc(it.status)}" data-id="${esc(it.item_id)}"><td class="mark" title="${esc(it.status)}">${SL_MARK[it.status] || "?"}</td>` +
        `<td>${label}<div class="note">${esc(note)}</div></td><td class="thumbs">` +
        (it.best_ids || []).slice(0, 4).map((id) => `<img alt="" loading="lazy" src="${thumbUrl(id, 96)}">`).join("") + "</td></tr>";
    }).join("") + "</table>";
  } else h += "<p class='note' id='sl-empty'>No shot list loaded yet.</p>";
  box.innerHTML = h;
  box.querySelectorAll("tr.sl-item").forEach((tr) => tr.addEventListener("click", () => {
    const it = (SL.data.items || []).find((x) => x.item_id === tr.dataset.id);
    if (it) showShotFrames(it);
  }));
}
async function showShotlist(msg = "") {
  $("overlay-shotlist").hidden = false;
  try { SL.data = await api("/api/shotlist" + q({ folder: S.folder })); } catch (e) { msg = e.message; SL.data = null; }
  renderShotlist(msg);
}
async function shotlistRun(body) {
  $("sl-msg").textContent = "Working…";
  try {
    SL.data = await api("/api/shotlist", { method: "POST", body: { folder: S.folder, ...body, tag_xmp: SL.tag } });
    renderShotBanner(SL.data);
    renderShotlist(SL.tag ? "Matched keeps were tagged." : "");
  } catch (e) { $("sl-msg").textContent = e.message; }
}
$("shotlist-sheet").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  SL.text = $("sl-text").value;
  SL.tag = $("sl-tag").checked;
  SL.list = $("sl-template").value;
  if (b.id === "sl-load") {
    if (SL.text.trim()) await shotlistRun({ text: SL.text });
    else if (SL.list) await shotlistRun({ list: SL.list });
    else $("sl-msg").textContent = "Pick a template or paste a list.";
  } else if (b.id === "sl-check") {
    $("sl-msg").textContent = "Working…";
    try {
      SL.data = SL.tag
        ? await api("/api/shotlist/coverage", { method: "POST", body: { folder: S.folder, tag_xmp: true } })
        : await api("/api/shotlist/coverage" + q({ folder: S.folder }));
      renderShotBanner(SL.data);
      renderShotlist("Checked.");
    } catch (err) { $("sl-msg").textContent = err.message; }
  } else if (b.id === "sl-clear") {
    try { await api("/api/shotlist" + q({ folder: S.folder }), { method: "DELETE" }); SL.data = null; renderShotBanner(null); showShotlist("Shot list cleared."); }
    catch (err) { $("sl-msg").textContent = err.message; }
  }
});
$("btn-shotlist").addEventListener("click", () => showShotlist());

/* ---------- reference-frame colour match (T16) ---------- */
async function matchSetRef() {   // Tools menu: Colour match: set this photo as reference (API/CLI do the matching)
  const im = S.byId.get(S.sel);
  if (!im) return;
  if (im.video_t !== null && im.video_t !== undefined) { toast("Colour match needs a photo, not a video frame."); return; }
  try {
    const r = await api("/api/edit/reference" + q({ id: im.id, folder: S.folder }));
    if (!r.ok) { S.matchMsg = r.message; }
    else { S.matchRef = im.id; S.matchMsg = `Reference set (${r.n_settings} develop settings from ${r.source}).`; }
  } catch (e) { S.matchMsg = e.message; }
  toast(S.matchMsg);
}
async function refreshEditState() {
  try { S.editTrained = (await api("/api/edit/profile")).trained; } catch (e) { S.editTrained = false; }
}

async function waitJob(id, onProgress) {
  for (;;) {
    const j = await api("/api/jobs/" + id);
    if (onProgress) onProgress(j);
    if (j.state === "done" || j.state === "error") return j;
    await new Promise((r) => setTimeout(r, 300));
  }
}
const EDIT = { dry: null, xmp: [] };
function xmpChipsHtml() {
  return EDIT.xmp.map((p, n) => `<span class="xmp-chip" title="${esc(p)}"><span>${esc(nameOf(p))}</span><button data-xmp-rm="${n}" aria-label="Remove ${esc(nameOf(p))}">×</button></span>`).join("");
}
function sliderTable(rows) {
  const f = (x) => (x === null || x === undefined ? "–" : x);
  return `<table id="edit-sliders"><tr><th>Slider</th><th>Mode</th><th>n</th><th>CV error</th><th>Median baseline</th></tr>` +
    rows.map((r) => `<tr data-slider="${esc(r.slider)}"><td>${esc(r.slider)}</td><td class="mode-${esc(r.mode)}">${esc(r.mode)}</td>` +
      `<td class="num">${r.n}</td><td class="num">${f(r.mae_cv)}</td><td class="num">${f(r.mae_median)}</td></tr>`).join("") + "</table>";
}
async function showEdit(msg = "") {
  const box = $("edit-sheet");
  $("overlay-edit").hidden = false;
  let st = { trained: false };
  try { st = await api("/api/edit/profile"); } catch (e) { msg = e.message; }
  S.editTrained = !!st.trained;
  let html = "<h2>Edit profile</h2>";
  html += "<p class='note'>Learns your develop settings from a Lightroom catalog (or XMP folders) and predicts them for new shoots. Needs at least 200 edited frames. Originals are never modified.</p>";
  EDIT.xmp = [];
  html += `<div class="form"><label for="edit-cat">Catalog (.lrcat)</label><div class="pick-row"><input id="edit-cat" class="browser-only" type="text" spellcheck="false" placeholder="/absolute/path/catalog.lrcat">` +
    `<button id="edit-cat-pick" class="native-only">Choose catalog…</button><span id="edit-cat-name" class="path-name native-only"></span></div>` +
    `<label for="edit-xmp">XMP folders</label><div class="pick-row"><input id="edit-xmp" class="browser-only" type="text" spellcheck="false" placeholder="/folder/a, /folder/b (optional)">` +
    `<span id="edit-xmp-chips" class="native-only pick-row"></span><button id="edit-xmp-add" class="native-only">Add folder…</button></div>` +
    `<span></span><label class="chk"><input id="edit-render" type="checkbox"> render DNGs whose preview carries an edit</label></div>`;
  html += `<div class="btns"><button id="edit-train" class="primary">Train</button>` +
    (st.trained ? `<button id="edit-delete">Delete profile</button>` : "") + `<span id="edit-msg">${esc(msg)}</span></div>`;
  if (st.trained) {
    html += `<h3 id="edit-trained">Trained on ${st.n} edits (profile ${esc(st.profile_hash)})</h3>` + sliderTable(st.sliders);
    html += `<h3>Apply to this shoot</h3><div class="btns"><label class="chk"><input id="edit-embed" type="checkbox"> write embedded copies for JPEG/DNG</label>` +
      `<button id="edit-dry">Preview (dry run)</button><button id="edit-apply" disabled>Write for keeps</button></div><div id="edit-diff"></div>`;
  }
  box.innerHTML = html;
  EDIT.dry = null;
}
function diffHtml(r) {
  const sl = Object.entries(r.sliders || {}).map(([k, n]) => `${esc(EDIT_LABELS[k] || k)} ${n}`).join(" · ");
  let h = `<p id="edit-summary">${r.dry_run ? "Would write" : "Wrote"} ${r.written_sidecar} sidecar(s) and ${r.written_copies} embedded copy(ies) ` +
    `of ${r.n} image(s); ${r.needs_edit} need your edit.` + (r.report_only ? ` ${r.report_only} JPEG/DNG file(s) are report-only.` : "") + "</p>";
  if (sl) h += `<p class="note">${sl}</p>`;
  for (const w of r.warnings || []) h += `<p class="note">${esc(w)}</p>`;
  for (const [why, n] of Object.entries(r.skipped || {})) h += `<p class="note">Skipped ${n}: ${esc(why)}</p>`;
  const rows = (r.diff || []).slice(0, 40).map((d) => `<div>${esc(d.name)} · ${esc(d.status)} · ` +
    Object.entries(d.values).map(([k, v]) => `${esc(EDIT_LABELS[k] || k)} ${esc(fmtEdit(k, v))}`).join(", ") +
    (d.needs_edit ? " · needs your edit" : "") + "</div>").join("");
  return h + (rows ? `<div class="diff" id="edit-diff-rows">${rows}</div>` : "");
}
async function editTrain() {
  const cat = $("edit-cat").value.trim();
  const xmp = isNative() ? EDIT.xmp.slice() : $("edit-xmp").value.split(",").map((x) => x.trim()).filter(Boolean);
  if (!cat && !xmp.length) { $("edit-msg").textContent = "Give a catalog and/or XMP folders."; return; }
  $("edit-train").disabled = true;
  try {
    const r = await api("/api/edit/train", { method: "POST", body: { catalog: cat || null, xmp, dng_render: $("edit-render").checked } });
    const j = await waitJob(r.job_id, (jj) => { $("edit-msg").textContent = `${jj.stage || "starting"} ${jj.done}/${jj.total}`; });
    if (j.state === "error") { await showEdit("Training failed: " + j.error); return; }
    S.editPred.clear();
    await showEdit(`Trained on ${j.result.n} edits in ${j.result.seconds} s.`);
  } catch (e) { $("edit-msg").textContent = e.message; $("edit-train").disabled = false; }
}
async function editApply(dry) {
  if (!S.folder) { toast("Cull a folder first."); return; }
  const embed = $("edit-embed").checked;
  $("edit-dry").disabled = $("edit-apply").disabled = true;
  $("edit-diff").innerHTML = "<p class='note'>Working…</p>";
  try {
    const r = await api("/api/edit/apply", { method: "POST", body: { folder: S.folder, scope: "keeps", dry_run: dry, embed_copies: embed } });
    const j = await waitJob(r.job_id, (jj) => { $("edit-msg").textContent = `${jj.stage || "starting"} ${jj.done}/${jj.total}`; });
    if (j.state === "error") { $("edit-diff").innerHTML = `<p class="note">${esc(j.error)}</p>`; $("edit-dry").disabled = false; return; }
    $("edit-diff").innerHTML = diffHtml(j.result);
    $("edit-dry").disabled = false;
    if (dry) { EDIT.dry = j.result; $("edit-apply").disabled = false; }
    else { S.editPred.clear(); await loadShoot(true); }
  } catch (e) { $("edit-diff").innerHTML = `<p class="note">${esc(e.message)}</p>`; $("edit-dry").disabled = false; }
}
$("edit-sheet").addEventListener("click", async (e) => {
  e.stopPropagation();
  const b = e.target.closest("button");
  if (!b) return;
  if (b.id === "edit-train") editTrain();
  else if (b.id === "edit-cat-pick") {
    const p = await window.shotsieveDesktop.pick("lrcat");
    if (p) { $("edit-cat").value = p; $("edit-cat-name").textContent = nameOf(p); $("edit-cat-name").title = p; }
  } else if (b.id === "edit-xmp-add") {
    const p = await window.shotsieveDesktop.pick("folder");
    if (p && !EDIT.xmp.includes(p)) { EDIT.xmp.push(p); $("edit-xmp-chips").innerHTML = xmpChipsHtml(); }
  } else if (b.dataset.xmpRm !== undefined) {
    EDIT.xmp.splice(+b.dataset.xmpRm, 1); $("edit-xmp-chips").innerHTML = xmpChipsHtml();
  }
  if (b.id === "edit-dry") editApply(true);
  else if (b.id === "edit-apply") editApply(false);
  else if (b.id === "edit-delete") {
    try { await api("/api/edit/profile", { method: "DELETE" }); S.editTrained = false; S.editPred.clear(); showEdit("Profile deleted."); }
    catch (err) { toast(err.message); }
  }
});
$("btn-edit").addEventListener("click", () => showEdit());

/* ---------- navigation ---------- */

function move(delta) {
  if (!S.flat.length) return;
  const i = S.flatIdx.has(S.sel) ? S.flatIdx.get(S.sel) : -1;
  const j = Math.max(0, Math.min(S.flat.length - 1, (i < 0 ? 0 : i + delta)));
  select(S.flat[j]);
}
function groupOf(id) { return S.byId.get(id).group_id; }
function moveGroup(delta) {
  if (!S.flat.length) return;
  let i = S.flatIdx.has(S.sel) ? S.flatIdx.get(S.sel) : -1;
  if (i < 0) return select(S.flat[0]);
  const g = groupOf(S.flat[i]);
  if (delta > 0) {
    while (i < S.flat.length && groupOf(S.flat[i]) === g) i++;
    if (i < S.flat.length) select(S.flat[i]);
  } else {
    while (i >= 0 && groupOf(S.flat[i]) === g) i--;
    if (i < 0) return;
    const pg = groupOf(S.flat[i]);
    while (i > 0 && groupOf(S.flat[i - 1]) === pg) i--;
    select(S.flat[i]);
  }
}
function setFilter(f) {
  S.filter = f;
  document.querySelectorAll(".tab[data-filter]").forEach((b) => b.classList.toggle("active", b.dataset.filter === f));
  relayout(true);
  if (S.sel !== null && S.rowOf.has(S.sel)) select(S.sel);
  else if (S.flat.length) select(S.flat[0]);
  else { S.sel = null; renderDetail(); }
}
function cycleFilter(dir) {
  const i = FILTERS.indexOf(S.filter);
  setFilter(FILTERS[(i + dir + FILTERS.length) % FILTERS.length]);
}
function nextAudit() {
  const pend = (im) => im.audit && (!im.overridden || im.decision === "review");
  const order = S.groups.flatMap((g) => g.members);
  const start = Math.max(0, order.indexOf(S.sel));
  for (let k = 1; k <= order.length; k++) {
    const id = order[(start + k) % order.length];
    if (pend(S.byId.get(id))) {
      if (!S.rowOf.has(id)) setFilter("audit");
      return select(id);
    }
  }
  toast(S.images.some((im) => im.audit) ? "Audit complete: press Finish audit to recalibrate." : "No audit items in this shoot.");
}

/* ---------- decisions ---------- */

async function decideMany(ids, kind, rating) {   // T23i: one request for the whole selection
  const ims = ids.map((id) => S.byId.get(id)).filter(Boolean);
  if (!ims.length) return;
  if ((kind === "keep" || kind === "reject") && rating === null && ims.every((im) => im.overridden && im.decision === kind)) kind = "model";
  const body = { ids, folder: S.folder };
  if (kind) body.decision = kind;
  if (rating !== null) body.rating = rating;
  const prev = new Map(ims.map((im) => [im.id, { decision: im.decision, overridden: im.overridden, rating: im.rating }]));
  const idx = ids.map((id) => S.flatIdx.get(id)).filter((i) => i !== undefined);
  const fallback = S.flat[Math.max(...idx) + 1] ?? S.flat[Math.min(...idx) - 1] ?? null;
  for (const im of ims) {
    if (kind === "keep" || kind === "reject") { im.decision = kind; im.overridden = true; }
    if (rating !== null) { im.rating = rating || null; if (!im.overridden) { im.decision = "keep"; im.overridden = true; } }
  }
  try {
    const r = await api("/api/override", { method: "POST", body });
    for (const x of r.results) {
      const im = S.byId.get(x.id);
      if (!im) continue;
      im.decision = x.decision; im.overridden = x.overridden; im.rating = x.rating;
      if (x.overridden === false) im.rating = null;
    }
  } catch (e) {
    for (const im of ims) Object.assign(im, prev.get(im.id));
    toast("Could not save: " + e.message);
  }
  const primary = S.sel;
  relayout();
  const left = ids.filter((id) => S.rowOf.has(id));
  if (left.length) selectMany(left, left.includes(primary) ? primary : left[left.length - 1], false);
  else if (fallback !== null && S.rowOf.has(fallback)) select(fallback);
  else if (S.flat.length) select(S.flat[0]);
  else { S.sel = null; S.multi = new Set(); renderDetail(); }
}

async function decide(kind, rating = null) {
  const many = selIds();
  if (many.length > 1) return decideMany(many, kind, rating);
  const im = S.byId.get(S.sel);
  if (!im) return;
  if ((kind === "keep" || kind === "reject") && rating === null && im.overridden && im.decision === kind) kind = "model";   // pressing your own choice again: back to ShotSieve's decision
  const idx = S.flatIdx.has(im.id) ? S.flatIdx.get(im.id) : -1;
  const fallback = S.flat[idx + 1] ?? S.flat[idx - 1] ?? null;
  const body = { image_id: im.id, folder: S.folder };
  if (kind) body.decision = kind;
  if (rating !== null) body.rating = rating;
  const prev = { decision: im.decision, overridden: im.overridden, rating: im.rating };
  if (kind === "keep" || kind === "reject") { im.decision = kind; im.overridden = true; }
  if (rating !== null) { im.rating = rating || null; if (!im.overridden) { im.decision = "keep"; im.overridden = true; } }
  try {
    const r = await api("/api/override", { method: "POST", body });
    im.decision = r.decision; im.overridden = r.overridden; im.rating = r.rating;
    if (r.overridden === false) im.rating = null;
  } catch (e) {
    Object.assign(im, prev);
    toast("Could not save: " + e.message);
  }
  const stillVisible = matches(im);
  relayout();
  if (stillVisible) select(im.id, false);
  else if (fallback !== null && S.rowOf.has(fallback)) select(fallback);
  else if (S.flat.length) select(S.flat[0]);
  else { S.sel = null; renderDetail(); }
}

/* ---------- overlays ---------- */

function toggleLoupe() {
  const o = $("overlay-loupe");
  if (!S.sel && o.hidden) return;
  o.hidden = !o.hidden;
  if (!o.hidden) { showLoupe(S.sel); }   // the photo opens at its own remembered zoom (Fit if never zoomed)
}

/* ---------- loupe zoom (shared module static/zoom.js; crop mode turns zoom off) ---------- */

const loupeGroup = ShotZoom.group({ linked: false });
const cropOn = () => !$("crop-ui").hidden;
const loupeView = ShotZoom.view(loupeGroup, {
  view: $("lz-view"), img: $("big"), urls: zoomUrls, enabled: () => !cropOn(),
});
ShotZoom.toolbar($("lz-zoom"), loupeGroup);
if (window.ShotDenoise) window.ShotDenoise.attach(loupeGroup, { host: $("lz-denoise"), enabled: () => !cropOn() });
function showLoupe(id) {
  const im = S.byId.get(id);
  $("lz-name").textContent = im ? im.rel : "";
  loupeView.setImage(id);
}
$("lz-close").addEventListener("click", () => { $("overlay-loupe").hidden = true; });
document.addEventListener("keydown", (e) => {   // zoom keys while the loupe is open
  if ($("overlay-loupe").hidden || cropOn() || e.target.closest("input[type=text], input[type=search], textarea")) return;
  if (e.key === "Escape") return;
  if (ShotZoom.handleKey(e, loupeGroup, { view: loupeView })) { e.preventDefault(); e.stopPropagation(); }
}, true);
/* main view denoise (D): the right-panel preview shows /api/denoise (same strength as the loupe) */
S.dnMain = false;
let dnStatus = null;
async function toggleMainDenoise() {
  if (cropOn() || S.sel === null) return;
  S.dnMain = !S.dnMain;
  if (S.dnMain && !dnStatus) {
    try { dnStatus = await api("/api/denoise/status"); } catch (e) { dnStatus = { backend: "basic" }; }
  }
  drawMainDenoise(true);
  renderDetail();
}
function drawMainDenoise(busy) {
  const b = $("btn-dn-main");
  b.setAttribute("aria-pressed", String(S.dnMain));
  b.classList.toggle("on", S.dnMain);
  b.querySelector(".dn-badge").textContent = !S.dnMain ? "Off" : busy ? "…" : (dnStatus && dnStatus.backend === "ai" ? "AI" : "Basic");
  b.querySelector(".dn-spin").hidden = !(S.dnMain && busy);
  $("loupe-wrap").classList.toggle("dn-on", S.dnMain);
}
$("btn-dn-main").addEventListener("click", toggleMainDenoise);
$("loupe").addEventListener("load", () => { if (S.dnMain) drawMainDenoise(false); });
$("loupe").addEventListener("error", () => { if (S.dnMain) drawMainDenoise(false); });
ShotZoom.onDenoiseLevel(() => { if (S.dnMain) { drawMainDenoise(true); renderDetail(); } });
function toggleCompare() { if (window.ShotCompare) window.ShotCompare.toggle(); }
$("d-reasons").addEventListener("click", (e) => {
  const link = e.target.closest(".dup-link"), btn = e.target.closest(".dup-compare");
  if (!link && !btn) return;
  e.preventDefault();
  const id = Number((link || btn).dataset.id);
  if (!S.byId.has(id)) return;
  if (btn) {   // current frame vs the better one
    if (window.ShotCompare && S.sel !== null) window.ShotCompare.open({ a: id, b: S.sel });
    return;
  }
  if (!S.rowOf.has(id)) { clearSearch(); setFilter("all"); }   // hidden by the filter or a search: show everything
  if (!S.rowOf.has(id)) return;
  select(id);                       // the link opens that frame in the loupe
  const o = $("overlay-loupe");
  if (o.hidden) toggleLoupe();
});
function toggleHelp() { const o = $("overlay-help"); o.hidden = !o.hidden; }
const MENUS = [["btn-view", "view-menu"], ["btn-more", "more-menu"]];
function closeMenus() {
  let closed = false;
  for (const [b, m] of MENUS) {
    if (!$(m).hidden) { $(m).hidden = true; $(b).setAttribute("aria-expanded", "false"); closed = true; }
  }
  const cm = $("ctx-menu");
  if (!cm.hidden) { cm.hidden = true; closed = true; }
  return closed;
}
for (const [b, m] of MENUS) {
  $(b).addEventListener("click", (e) => {
    e.stopPropagation();
    const open = $(m).hidden;
    closeMenus();
    $(m).hidden = !open;
    $(b).setAttribute("aria-expanded", String(open));
  });
  $(m).addEventListener("click", (e) => { if (e.target.closest("button")) setTimeout(closeMenus, 0); });
}
document.addEventListener("click", () => closeMenus());
function closeOverlays() {
  let closed = false;
  document.querySelectorAll(".overlay").forEach((o) => { if (!o.hidden) { o.hidden = true; closed = true; } });
  return closed;
}

/* ---------- server actions ---------- */

async function finishAudit() {
  try {
    const r = await api("/api/audit/complete", { method: "POST", body: { folder: S.folder } });
    const pct = (r.missed_keeper_rate * 100).toFixed(1);
    let msg = `Audit: ${r.n_kept} of ${r.n_audited} spot-checked rejects were keepers; estimated missed-keeper rate ${pct}%.`;
    msg += r.lowered ? ` Thresholds relaxed (reject < ${r.thresholds.t_reject.toFixed(2)}, review margins +${(r.thresholds.hard_margin || 0).toFixed(2)} / +${(r.thresholds.dup_margin || 0).toFixed(2)}).` : " Thresholds unchanged.";
    if (r.n_pending) msg += ` ${r.n_pending} item(s) still undecided.`;
    toast(msg, 9000);
    await loadShoot(true);
  } catch (e) { toast("Audit failed: " + e.message); }
}
async function grabFrame(im) {
  try {
    const r = await api("/api/video/grab", { method: "POST", body: { image_id: im.id, folder: S.folder } });
    toast("Frame exported: " + r.rel + " (cull again to include it as a photo).", 6000);
  } catch (e) { toast(e.message); }
}

async function startCull(files) {
  if (!Array.isArray(files)) files = null; // (click handlers pass an Event)
  S.focusNames = files ? files.map((f) => f.split("/").pop()) : null;
  if (!files) S.dropped = new Set();
  const folder = $("folder").value.trim();
  if (!folder) return toast(isNative() ? "Choose a folder first." : "Enter a folder path first.");
  S.folder = folder;
  try { localStorage.setItem("shotsieve.folder", folder); } catch (e) { /* private mode */ }
  try {
    const r = await api("/api/cull", { method: "POST", body: { folder, write_xmp: false, video: $("chk-video").checked, ...(files ? { files } : {}) } });
    pollJob(r.job_id);
  } catch (e) { toast(e.message); }
}
async function pollJob(id) {
  $("progress").hidden = false;
  $("btn-cull").disabled = true;
  for (;;) {
    let j;
    try { j = await api("/api/jobs/" + id); } catch (e) { toast(e.message); break; }
    const frac = j.total ? j.done / j.total : 0;
    $("progress-fill").style.width = Math.round(frac * 100) + "%";
    $("progress-text").textContent = j.state === "running" ? `${j.stage || "starting"} ${j.done}/${j.total}` : j.state;
    if (j.state === "done") {
      await loadShoot();
      if (S.focusNames) { // add-only drop: the whole folder is shown, the dropped photos highlighted
        const hit = S.images.filter((im) => S.focusNames.includes(im.name));
        S.dropped = new Set(hit.map((im) => im.id)); S.focusNames = null;
        if (hit.length) { select(hit[0].id); renderRows(true); }
      }
      const r = j.result || {};
      toast(r.redecide === "full" && r.n_cached > 0
        ? "Cull finished. Many files changed, so the whole shoot was re-decided (" + r.redecide_reason + ")."
        : r.redecide === "incremental"
          ? `Cull finished: ${r.n_new} new, ${r.n_removed} removed, ${r.n_affected} frames re-decided; the rest kept.`
          : "Cull finished.", r.redecide === "full" ? 9000 : 3500);
      checkShotlist();
      break;
    }
    if (j.state === "error") { toast("Cull failed: " + j.error, 9000); break; }
    await new Promise((r) => setTimeout(r, 350));
  }
  $("btn-cull").disabled = false;
  setTimeout(() => { $("progress").hidden = true; }, 1200);
}

/* ---------- cameras (clock offsets between bodies) ---------- */

function fmtOffset(s) {
  const sign = s < 0 ? "-" : "+";
  const a = Math.abs(Math.round(s));
  const hh = Math.floor(a / 3600), mm = Math.floor((a % 3600) / 60), ss = a % 60;
  return `${sign}${hh}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}
function parseOffset(t) {
  const m = String(t).trim().match(/^([+-]?)(\d+)(?::(\d{1,2}))?(?::(\d{1,2}(?:\.\d+)?))?$/);
  if (!m) return null;
  const sign = m[1] === "-" ? -1 : 1;
  if (m[3] === undefined) return sign * Number(m[2]);          // plain seconds
  return sign * (Number(m[2]) * 3600 + Number(m[3]) * 60 + Number(m[4] || 0));  // h:mm[:ss]
}
function pairHtml(p, serial) {
  return `<div class="pair"><figure><img alt="" src="${thumbUrl(p.ref_id, 300)}"><figcaption>${esc(p.ref_name)}</figcaption></figure>` +
    `<figure><img alt="" src="${thumbUrl(p.id, 300)}"><figcaption>${esc(p.name)} (${esc(serial)})</figcaption></figure></div>`;
}
let camSeq = 0;
async function showCameras(keep) {   // keep=true: refresh in place, no loading flash
  const box = $("cameras-sheet");
  const seq = ++camSeq;
  $("overlay-cameras").hidden = false;
  if (!keep || !box.querySelector(".cam")) box.innerHTML = "<h2>Cameras</h2><p class='note'>Looking for matching frames…</p>";
  let d;
  try { d = await api("/api/clock" + q({ folder: S.folder })); }
  catch (e) { if (seq !== camSeq) return; box.innerHTML = `<h2>Cameras</h2><p class="note">${esc(e.message)}</p>`; return; }
  if (seq !== camSeq) return;   // a newer refresh superseded this one
  const sug = new Map(d.suggestions.map((s) => [s.serial, s]));
  let html = "<h2>Cameras</h2>";
  if (d.cameras.length < 2) html += "<p class='note'>Only one camera body found, nothing to align.</p>";
  for (const c of d.cameras) {
    const off = d.offsets[c.serial], s = sug.get(c.serial);
    html += `<div class="cam" data-serial="${esc(c.serial)}"><b>${esc(c.serial)}</b> · ${c.n} frames · ` +
      (off ? `offset ${fmtOffset(off.offset_s)} (${esc(off.source)})` : "no offset");
    if (s && s.offset_s !== null) {
      html += `<div>Suggested ${fmtOffset(s.offset_s)} · confidence ${(s.confidence * 100).toFixed(0)}% (${s.support}/${s.n_candidates} frames)</div>`;
      if (s.pair) html += pairHtml(s.pair, c.serial);
      html += `<div class="row"><button data-act="accept" data-off="${s.offset_s}">Accept</button></div>`;
    } else if (s) {
      html += "<div>No consistent offset found. Pick a pair that shows the same moment:</div>";
      html += s.alternatives.map((p, i) => pairHtml(p, c.serial) +
        `<div class="row"><button data-act="pair" data-a="${p.ref_id}" data-b="${p.id}">Use this pair (${fmtOffset(p.delta_s)})</button></div>`).join("");
    }
    html += `<div class="row"><input type="text" placeholder="±h:mm:ss" aria-label="offset"><button data-act="manual">Set</button>` +
      (off ? `<button data-act="clear">Clear</button>` : "") + "</div></div>";
  }
  box.innerHTML = html;
}
$("cameras-sheet").addEventListener("click", async (e) => {
  e.stopPropagation();
  const b = e.target.closest("button");
  if (!b) return;
  const cam = b.closest(".cam");
  const serial = cam.dataset.serial;
  const body = { folder: S.folder, serial };
  if (b.dataset.act === "accept") body.offset_s = Number(b.dataset.off);
  else if (b.dataset.act === "pair") body.pair = [Number(b.dataset.a), Number(b.dataset.b)];
  else if (b.dataset.act === "clear") body.clear = true;
  else if (b.dataset.act === "manual") {
    const v = parseOffset(cam.querySelector("input").value);
    if (v === null) { toast("Type an offset like +1:02:05 or -90"); return; }
    body.offset_s = v;
  }
  try {
    await api("/api/clock", { method: "POST", body });
    toast("Clock offset saved");
    await loadShoot(true);
    showCameras(true);
  } catch (err) { toast(err.message); }
});
$("btn-cameras").addEventListener("click", () => showCameras());

/* ---------- Lightroom sync-back (T20): the catalog is only ever read ---------- */

const LR = { dry: null };
const pct = (x) => (x === null || x === undefined ? "–" : (100 * x).toFixed(1) + " %");
function lrReportHtml(r) {
  if (!r) return "";
  const rows = [
    ["Frames compared", r.n_evaluated],
    ["Missed keepers", `${r.missed_keepers} of ${r.n_kept_truth} (${pct(r.missed_keeper_rate)})`],
    ["Moment miss rate (review counts as saved)", pct(r.group_gate_rate)],
    ["Sent to review", pct(r.review_fraction)],
    ["Keepers in review", r.keepers_in_review],
    ["Reject precision", pct(r.reject_precision)],
  ];
  const missed = (r.missed_files || []).slice(0, 12).map(esc).join(", ");
  return `<div class="lr-card" id="lr-report"><b>This shoot</b><table>` +
    rows.map((x) => `<tr><td>${x[0]}</td><td>${x[1]}</td></tr>`).join("") + `</table>` +
    (missed ? `<p class="note">Missed: ${missed}</p>` : "") + `<p class="note">${esc(r.note || "")}</p></div>`;
}
function lrSummary(r) {
  if (r.summary) return r.summary; // built by the API (SyncResult.describe), shared with the wizard
  const n = r.n_new + r.n_changed + r.n_cleared;
  let t = `${n} change${n === 1 ? "" : "s"}: ${r.n_keeps} keeps, ${r.n_rejects} rejects`;
  if (r.n_conflicts_ui_won || r.n_conflicts_lr_won) {
    t += `; ${r.n_conflicts_ui_won + r.n_conflicts_lr_won} conflicts` +
      (r.n_conflicts_ui_won ? `, your ShotSieve choice kept on ${r.n_conflicts_ui_won}` : "") +
      (r.n_conflicts_lr_won ? `, Lightroom won on ${r.n_conflicts_lr_won}` : "");
  }
  t += `. Matched ${r.n_matched} of ${r.n_matched + r.n_unmatched} frames.`;
  return t;
}
async function showLr(msg = "") {
  if (!S.folder) return toast("Cull a folder first.");
  const box = $("lr-sheet");
  $("overlay-lr").hidden = false;
  let cats = { catalogs: [], remembered: null }, rep = null;
  try { cats = await api("/api/lr/catalogs" + q({ folder: S.folder })); } catch (e) { /* listing only */ }
  try { rep = (await api("/api/lr/report" + q({ folder: S.folder }))).report; } catch (e) { /* not culled */ }
  const cur = LR.path || cats.remembered || cats.catalogs[0] || "";
  box.innerHTML = `<h2>Sync from Lightroom</h2>` +
    `<p class="note">Quit Lightroom before syncing (recommended, not required). ShotSieve only reads the catalog; it never writes to it.</p>` +
    (cats.catalogs.length ? `<select id="lr-cat" aria-label="Detected catalogs"><option value="">Detected catalogs…</option>` +
      cats.catalogs.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("") + `</select>` : "") +
    `<div class="pick-row"><input id="lr-path" class="browser-only" type="text" spellcheck="false" placeholder="/absolute/path/to/catalog.lrcat" value="${esc(cur)}" aria-label="Catalog path">` +
    `<button id="lr-pick" class="native-only">Choose catalog…</button>` +
    `<span id="lr-name" class="path-name native-only" title="${esc(cur)}">${esc(cur ? nameOf(cur) : "none chosen")}</span></div>` +
    `<div class="btns-row"><button id="lr-dry" class="primary">Preview changes</button>` +
    `<button id="lr-apply" hidden>Apply</button></div>` +
    `<p class="note" id="lr-msg">${esc(msg)}</p><div id="lr-summary"></div>${lrReportHtml(rep)}`;
}
async function lrRun(dry) {
  const path = $("lr-path").value.trim();
  if (!path) { $("lr-msg").textContent = isNative() ? "Choose the Lightroom catalog first." : "Enter the catalog path (…/Lightroom Catalog.lrcat)."; return; }
  LR.path = path;
  $("lr-dry").disabled = true; $("lr-apply").hidden = true;
  $("lr-msg").textContent = "Reading the catalog…";
  try {
    const r = await api("/api/lr/sync", { method: "POST", body: { folder: S.folder, catalog: path, dry_run: dry } });
    const j = await waitJob(r.job_id, (jj) => { $("lr-msg").textContent = `${jj.stage || "starting"} ${jj.done}/${jj.total}`; });
    if (j.state === "error") { $("lr-msg").textContent = j.error; return; }
    const res = j.result;
    if (dry) {
      LR.dry = res;
      $("lr-msg").textContent = "Preview (nothing changed yet): " + lrSummary(res);
      $("lr-summary").innerHTML = lrReportHtml(res.report);
      $("lr-apply").hidden = res.n_writes === 0;
    } else {
      LR.dry = null;
      await loadShoot(true);
      await showLr("Applied: " + lrSummary(res));
    }
  } catch (e) { $("lr-msg").textContent = e.message; }
  finally { if ($("lr-dry")) $("lr-dry").disabled = false; }
}
function setLrPath(p) {
  $("lr-path").value = p;
  if ($("lr-name")) { $("lr-name").textContent = nameOf(p); $("lr-name").title = p; }
}
$("lr-sheet").addEventListener("click", async (e) => {
  e.stopPropagation();
  const b = e.target.closest("button");
  if (b && b.id === "lr-pick") {
    const p = await window.shotsieveDesktop.pick("lrcat");
    if (p) setLrPath(p);
  } else if (b && b.id === "lr-dry") lrRun(true);
  else if (b && b.id === "lr-apply") lrRun(false);
});
$("lr-sheet").addEventListener("change", (e) => {
  if (e.target.id === "lr-cat" && e.target.value) setLrPath(e.target.value);
});
$("mi-lr").addEventListener("click", () => showLr());

/* ---------- search, indexer status, people ---------- */

async function runSearch() {
  const text = $("search").value.trim();
  if (!text) { clearSearch(); return; }
  if (!S.folder) return;
  try {
    const r = await api("/api/search" + q({ q: text, k: 200, folder: S.folder }));
    S.search = new Map(r.hits.map((h) => [h.id, h.score || 0]));
    S.searchQ = text;
    relayout(true);
    if (S.flat.length) select(S.flat[0]);
    else { S.sel = null; renderDetail(); }
    toast(`${r.hits.length} result(s)` + (r.indexing ? " (still indexing)" : "")
      + ((r.warnings || []).length ? " | " + r.warnings.join("; ") : ""), r.warnings && r.warnings.length ? 6000 : undefined);
  } catch (e) { toast("Search: " + e.message, 6000); }
}
function clearSearch() {
  if (!S.search) return false;
  S.search = null;
  S.searchQ = "";
  $("search").value = "";
  relayout(true);
  if (S.sel !== null && S.rowOf.has(S.sel)) select(S.sel);
  return true;
}
async function pollIndex() {
  if (!S.folder) return;
  try {
    const s = await api("/api/index/status" + q({ folder: S.folder }));
    if (S.indexWasRunning && !s.running) checkShotlist();   // the index just finished: re-check the shot list
    S.indexWasRunning = !!s.running;
    const pill = $("index-pill");
    if (s.running) {
      pill.hidden = false;
      pill.textContent = `${s.task} ${s.done}/${s.total}` + (s.paused ? " (paused)" : ` ${s.img_s} img/s`);
    } else if (s.models_ready && s.indexed) {
      pill.hidden = false;
      pill.textContent = `indexed ${s.indexed}`;
    } else { pill.hidden = true; }
    await renderPeople(s.people_consent, s.models_ready);
    await renderJerseys(s);
  } catch (e) { /* server busy or no folder yet */ }
}
const peopleChanged = () => window.dispatchEvent(new CustomEvent("people-changed"));
window.addEventListener("people-changed", () => { S.peopleSig = null; pollIndex(); });
function initPeopleScroller(box, left) {   // one row; wheel / trackpad / arrows scroll it; the buttons at the right stay put
  const sc = $("people-scroll"), l = $("ps-left"), r = $("ps-right");
  const upd = () => {
    const over = sc.scrollWidth > sc.clientWidth + 1;
    l.hidden = !over || sc.scrollLeft <= 0;
    r.hidden = !over || sc.scrollLeft + sc.clientWidth >= sc.scrollWidth - 1;
  };
  sc.addEventListener("scroll", upd, { passive: true });
  sc.addEventListener("wheel", (e) => {   // a plain mouse wheel is vertical: map it to horizontal
    if (sc.scrollWidth <= sc.clientWidth || Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
    e.preventDefault();
    sc.scrollBy({ left: e.deltaY, behavior: "auto" });
  }, { passive: false });
  const step = (d) => sc.scrollBy({ left: d * Math.max(120, sc.clientWidth * 0.8), behavior: "smooth" });
  l.onclick = () => step(-1);
  r.onclick = () => step(1);
  if (window.ResizeObserver) new ResizeObserver(upd).observe(sc);
  sc.scrollLeft = left;
  upd();
}
function syncPeopleStrip() {   // mark the faces in the filter, show the Any/All switch for 2+
  const box = $("people-strip"), F = window.ShotFilters && window.ShotFilters.state();
  if (!F) return;
  box.querySelectorAll("img[data-id]").forEach((im) => im.classList.toggle("on", F.people.includes(Number(im.dataset.id))));
  const pm = $("people-pm");
  if (pm) {
    pm.hidden = F.people.length < 2;
    pm.querySelectorAll("button").forEach((b) => { b.classList.toggle("on", b.dataset.pm === F.pmode); b.setAttribute("aria-pressed", String(b.dataset.pm === F.pmode)); });
  }
}
window.addEventListener("filters-changed", syncPeopleStrip);
async function renderPeople(consent, ready) {
  const box = $("people-strip");
  if (!ready) { box.hidden = true; S.peopleOn = false; return; }
  box.hidden = false;
  S.peopleOn = !!consent;
  if (!consent && window.ShotFilters) window.ShotFilters.peopleState(false, null);
  if (!consent) {
    S.peopleSig = null;
    box.innerHTML = '<span>People search is off.</span> <button id="btn-consent">Turn on face grouping</button>' +
      ' <span class="note">Faces are compared on this computer only; you can delete them at any time.</span>';
    $("btn-consent").onclick = async () => {
      await api("/api/people/consent", { method: "POST", body: { folder: S.folder, on: true } });
      pollIndex();
    };
    return;
  }
  let data = { people: [], unsorted: 0 };
  try { data = await api("/api/people" + q({ folder: S.folder })); } catch (e) { /* 403 */ }
  S.people = data;
  const sig = JSON.stringify(data);
  if (window.ShotFilters) window.ShotFilters.peopleState(true, sig);   // T23e: reload the people map when it changed
  if (sig === S.peopleSig && box.querySelector("#btn-people-del")) return;   // unchanged: keep what the user is typing in
  if (box.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;
  S.peopleSig = sig;
  const keepLeft = box.querySelector("#people-scroll") ? box.querySelector("#people-scroll").scrollLeft : 0;
  box.innerHTML = `<button id="ps-left" class="ps-arrow" aria-label="Scroll people left" hidden>‹</button><div id="people-scroll">` +
    data.people.map((p) =>
      `<div class="person"><img src="${esc(window.ShotPeople ? window.ShotPeople.faceUrl(p, 96) : "")}" loading="lazy" decoding="async" data-id="${p.id}" alt="">` +
      `<input data-id="${p.id}" value="${esc(p.name || "")}" placeholder="${p.n === 1 ? "Unknown" : "name"} (${p.n})"></div>`).join("") +
    `</div><button id="ps-right" class="ps-arrow" aria-label="Scroll people right" hidden>›</button>` +
    `<div id="people-fixed"><span id="people-pm" class="fb-seg" role="group" aria-label="Match" hidden><button data-pm="any">Any of them</button><button data-pm="all">All of them</button></span>` +
    `<button id="btn-people-all" title="Every recognised person (Shift+P)">All people (${data.people.length})…</button>` +
    `<button id="btn-people-del">Delete people data</button></div>`;
  initPeopleScroller(box, keepLeft);
  $("btn-people-all").onclick = () => window.ShotPeople && window.ShotPeople.open();
  box.querySelectorAll("input").forEach((inp) => inp.addEventListener("change", async () => {
    await api("/api/people/" + inp.dataset.id, { method: "POST", body: { folder: S.folder, name: inp.value } });
    peopleChanged();
    if (inp.value.trim()) { $("search").value = `person:"${inp.value.trim()}"`; runSearch(); }
  }));
  box.querySelectorAll("img").forEach((im) => im.addEventListener("click", () => {
    if (window.ShotFilters) { if (S.search) clearSearch(); window.ShotFilters.togglePerson(Number(im.dataset.id)); }   // one filter state + removable chip
  }));
  $("people-pm").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b && window.ShotFilters) window.ShotFilters.setPmode(b.dataset.pm);
  });
  syncPeopleStrip();
  $("btn-people-del").onclick = async () => {
    await api("/api/people" + q({ folder: S.folder }), { method: "DELETE" });
    toast("People data deleted; face grouping is off.");
    peopleChanged();
  };
}
/* jersey OCR (opt-in) + roster */
async function refreshJerseys() {
  try { S.jerseys = (await api("/api/jerseys" + q({ folder: S.folder }))).jerseys; } catch (e) { S.jerseys = {}; }
  if (S.sel !== null) renderJerseyNote();
}
function renderJerseyNote() {
  const el = $("d-jersey");
  const list = (S.jerseys && S.jerseys[S.sel]) || [];
  el.hidden = !list.length;
  el.textContent = list.map((j) => "#" + j.number + (j.name ? " " + j.name : j.status === "unmatched" ? " (not on roster)" : "") +
    (j.source === "burst" ? " (from burst)" : "")).join(" · ");
}
async function renderJerseys(s) {
  // the controls live in a sheet opened from the More menu (no bar in the main window)
  S.jerseyReady = !!(s.ocr_models_ready && S.stats);
  $("mi-jersey").hidden = !S.jerseyReady;
  syncMoreMenu();
  const box = $("jersey-sheet");
  if (!S.jerseyReady) { if (!$("overlay-jersey").hidden) $("overlay-jersey").hidden = true; return; }
  if ($("overlay-jersey").hidden) { if (S.sel !== null) await refreshJerseys(); return; }
  await drawJerseySheet(s, box);
}
function syncMoreMenu() {
  $("more-none").hidden = !$("mi-lr").hidden || !$("mi-jersey").hidden;
}
async function showJersey() {
  $("overlay-jersey").hidden = false;
  try { await drawJerseySheet(await api("/api/index/status" + q({ folder: S.folder })), $("jersey-sheet")); }
  catch (e) { $("jersey-sheet").textContent = e.message; }
}
async function drawJerseySheet(s, box) {
  let r = { entries: 0, images_with_numbers: 0, images_matched: 0 };
  try { r = await api("/api/roster" + q({ folder: S.folder })); } catch (e) { /* not culled */ }
  const busy = s.running && s.task === "ocr";
  const sig = JSON.stringify([busy, s.done, s.total, r]);
  if (box.dataset.sig === sig && box.firstChild) return;
  box.dataset.sig = sig;
  box.innerHTML = `<h2>Jersey numbers</h2><div class="jersey-bar"><button id="btn-ocr">${busy ? "Reading numbers " + s.done + "/" + s.total : "Read jersey numbers"}</button>` +
    ` <button id="btn-roster">Load roster CSV</button><input id="roster-file" type="file" accept=".csv,text/csv,text/plain" hidden>` +
    ` <button id="btn-jersey-xmp">Write number tags</button></div>` +
    `<p id="jersey-count">${r.images_with_numbers} image(s) with numbers, ${r.entries} on roster (${r.images_matched} matched)</p>` +
    `<p class="note">Opt-in. Numbers are read on this computer; search with <code>jersey:23</code>.</p>` +
    `<div class="menu-actions"><button id="jersey-close">Close <kbd>Esc</kbd></button></div>`;
  $("jersey-close").onclick = () => { $("overlay-jersey").hidden = true; };
  $("btn-ocr").onclick = async () => {
    try { await api("/api/index/start", { method: "POST", body: { folder: S.folder, ocr: true, people: !!S.peopleOn } }); pollIndex(); }
    catch (e) { toast(e.message, 6000); }
  };
  $("btn-roster").onclick = () => $("roster-file").click();
  $("roster-file").onchange = async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    try {
      const out = await api("/api/roster", { method: "POST", body: { folder: S.folder, csv: await f.text() } });
      toast(`Roster: ${out.entries} players, ${out.images_matched} image(s) matched`);
      await refreshJerseys(); box.dataset.sig = ""; pollIndex();
    } catch (e) { toast("Roster: " + e.message, 6000); }
  };
  $("btn-jersey-xmp").onclick = async () => {
    try {
      const out = await api("/api/jerseys/xmp", { method: "POST", body: { folder: S.folder } });
      toast(`Number tags: ${out.written} sidecar(s) updated`);
    } catch (e) { toast(e.message, 6000); }
  };
  if (!busy) await refreshJerseys();
}
$("mi-jersey").addEventListener("click", showJersey);
setInterval(pollIndex, 3000);
$("search").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.stopPropagation(); runSearch(); } });

/* ---------- keyboard ---------- */

document.addEventListener("keydown", (e) => {
  const typing = e.target && (e.target.tagName === "TEXTAREA" ||
    (e.target.tagName === "INPUT" && !["checkbox", "radio", "button"].includes(e.target.type)));
  if (e.key === "Escape") {
    e.preventDefault();   // never let WebKit pass an Esc up to the window (it would leave full screen)
    if (typing) e.target.blur();
    if (!closeMenus() && !closeOverlays() && !clearMulti()) clearSearch();   // T23i: Esc also drops a multi-selection
    return;
  }
  if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === "o" && isNative()) { e.preventDefault(); chooseFolder(); return; }
  if (typing || e.ctrlKey || e.metaKey || e.altKey) {
    if (typing && e.key === "Enter" && e.target.id === "folder") startCull();
    return;
  }
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  let handled = true;
  switch (key) {
    case "ArrowLeft": move(-1); break;
    case "ArrowRight": move(1); break;
    case "ArrowUp": moveGroup(-1); break;
    case "ArrowDown": moveGroup(1); break;
    case "p": decide("keep"); break;
    case "x": decide("reject"); break;
    case "d": if ($("overlay-loupe").hidden && $("overlay-compare").hidden) toggleMainDenoise(); break;
    case "1": case "2": case "3": case "4": case "5": decide(null, Number(key)); break;
    case " ": toggleLoupe(); break;
    case "c": toggleCompare(); break;
    case "g": {
      const cur = S.byId.get(S.sel);
      if (cur && cur.video_t != null) { grabFrame(cur); break; }  // a video frame: export it
      S.collapsed = !S.collapsed; relayout(); if (S.sel !== null && S.rowOf.has(S.sel)) select(S.sel); break;
    }
    case "Tab": cycleFilter(e.shiftKey ? -1 : 1); break;
    case "a": nextAudit(); break;
    case "/": $("search").focus(); break;
    case "?": toggleHelp(); break;
    default: handled = false;
  }
  if (handled) e.preventDefault();
});

/* ---------- wiring ---------- */

document.querySelectorAll(".tab[data-filter]").forEach((b) => b.addEventListener("click", () => setFilter(b.dataset.filter)));
$("btn-collapse").addEventListener("click", () => { S.collapsed = !S.collapsed; relayout(); });
$("btn-cull").addEventListener("click", () => startCull());

/* ---------- drag and drop (real paths only inside the Mac app) ---------- */

(function () {
  const ov = $("drop-overlay");
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  const hide = () => { depth = 0; ov.hidden = true; };
  window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { e.preventDefault(); depth++; ov.hidden = false; } }, true);
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); }, true);
  window.addEventListener("dragleave", (e) => { if (hasFiles(e) && --depth <= 0) hide(); }, true);
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); hide();
    if (!isNative()) toast("Drop works in the ShotSieve app; use Choose Folder.", 5000);
  }, true);
  window.addEventListener("shotsieve-drop", async (e) => {
    hide();
    const plan = e.detail || {};
    if ($("btn-cull").disabled) return toast("A cull is running.");
    if (plan.error) return toast(plan.error, 5000);
    setFolderField(plan.folder);
    const note = (plan.notes || []).join(" ");
    if (note) toast(note, 7000);
    await startCull(plan.files || null);
  });
})();
$("btn-help").addEventListener("click", toggleHelp);
$("btn-audit-done").addEventListener("click", finishAudit);
document.querySelectorAll(".actions button[data-act]").forEach((b) => b.addEventListener("click", () => {
  const act = b.dataset.act;
  decide(act === "model" ? "model" : act);
}));
document.querySelectorAll(".overlay").forEach((o) => o.addEventListener("click", (e) => {
  if (o.id === "overlay-loupe") {   // the picture itself is for zoom / pan: only the backdrop closes
    if (e.target === o || e.target.id === "lz" || e.target.classList.contains("lz-bar")) o.hidden = true;
    return;
  }
  if (e.target === o || (e.target.tagName === "IMG" && o.id !== "overlay-compare")) o.hidden = true;
}));

async function init() {
  let saved = "";
  try { saved = localStorage.getItem("shotsieve.folder") || ""; } catch (e) { /* ignore */ }
  let initial = new URLSearchParams(location.search).get("folder") || "";
  try {
    const h = await api("/api/health");
    if (!initial && h.default_folder) initial = h.default_folder;
    if (!h.models_ready) toast("Model files are missing: run `shotsieve models download`.", 8000);
  } catch (e) { toast("Server unreachable: " + e.message); }
  S.folder = initial || saved;
  setFolderField(S.folder);
  refreshEditState().then(() => { if (S.sel !== null) renderDetail(); });
  if (S.folder) { await loadShoot(true); pollIndex(); loadShotlistStored(); }
  else { $("empty").hidden = false; $("empty").textContent = isNative() ? "Choose a folder (Cmd+O) to start." : "Enter a folder and press Cull."; }
}
init();
