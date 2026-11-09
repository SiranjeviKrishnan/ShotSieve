/* ShotSieve crop mode (T22): subject-aware crop suggestions. Vanilla JS, no build step.
   Opens inside the full-screen loupe (R). The whole frame is always shown; the crop is an overlay.
   Everything the user sees is client-side except suggestions (/api/crop/*). */
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const PRESETS = ["original", "1:1", "4:5", "5:4", "2:3", "3:2", "5:7", "7:5", "16:9", "9:16", "free"];
  const CHIP_LABEL = { original: "Original", free: "Free" };
  const MIN_N = 0.04;  // smallest crop side while dragging (normalised)

  let boxSeq = 0;
  const MAX_BOXES = 6, MIN_BOX = 0.012;   // a drawn box needs at least 1.2 % of the frame per side
  const C = {
    active: false, id: null, info: null, subj: null,
    sel: [],            // [{sid, mode, point?: {x, y}, box?}]  selection, in click order
    boxes: [],          // [{bid, box:[x0,y0,x1,y1]}] user-drawn must-keep subject boxes (persisted)
    activeBox: null,    // bid of the box Delete removes (the last one drawn / touched)
    ratio: "3:2", flip: false,
    cands: [], alt: 0, resolved: "", rect: null, source: "ai", score: null,
    req: 0, timer: null, drag: null, hover: null, esc: 0, burst: null, msg: "", infeasible: false,
  };
  window.ShotCrop = { state: C };

  /* ---------- show crops in the grid and the detail loupe ---------- */

  function showCrops() {
    const el = $("chk-crops");
    return !el || el.checked;
  }
  function cropRectOf(im) { return im && im.crop && im.crop.length === 4 ? im.crop : null; }

  /* A wrapper with overflow:hidden and a percentage-positioned <img> (no object-view-box: Safari). */
  function placeCropImg(wrap, rect, nw, nh, slotW, slotH) {
    const cw = rect[2] - rect[0], ch = rect[3] - rect[1];
    const ca = (cw * nw) / (ch * nh);
    let w = slotW, h = slotW / ca;
    if (h > slotH) { h = slotH; w = slotH * ca; }
    wrap.style.width = w + "px";
    wrap.style.height = h + "px";
    const img = wrap.querySelector("img");
    img.style.width = (100 / cw) + "%";
    img.style.height = (100 / ch) + "%";
    img.style.left = (-rect[0] / cw * 100) + "%";
    img.style.top = (-rect[1] / ch * 100) + "%";
    wrap.style.visibility = "visible";
  }
  function fitCell(img) {
    const wrap = img.closest(".cropwrap");
    if (!wrap || !wrap.dataset.crop || !img.naturalWidth) return;
    const slot = wrap.parentElement;
    const rect = wrap.dataset.crop.split(",").map(Number);
    placeCropImg(wrap, rect, img.naturalWidth, img.naturalHeight, slot.clientWidth, slot.clientHeight);
  }
  document.addEventListener("load", (e) => {
    if (e.target && e.target.tagName === "IMG" && e.target.closest && e.target.closest(".cropwrap")) fitCell(e.target);
  }, true);

  window.ShotCrop.cellImg = function (im, thumb) {   // markup for a grid cell's image
    const r = showCrops() ? cropRectOf(im) : null;
    if (!r) return `<img alt="" draggable="false" data-src="${thumb}">`;
    return `<div class="cropslot"><div class="cropwrap" data-crop="${r.map((v) => v.toFixed(5)).join(",")}">` +
           `<img alt="" draggable="false" data-src="${thumb}"></div></div>`;
  };
  window.ShotCrop.badge = (im) => (cropRectOf(im) ? '<span class="badge crop" title="Cropped in ShotSieve (the original is untouched)">⧉</span>' : "");

  window.ShotCrop.loupe = function (im) {   // the detail panel's preview
    const wrapEl = $("loupe-crop"), plain = $("loupe");
    const r = showCrops() ? cropRectOf(im) : null;
    if (!wrapEl) return;
    if (!r) { wrapEl.hidden = true; plain.hidden = false; return; }
    const img = wrapEl.querySelector("img");
    wrapEl.dataset.crop = r.join(",");
    img.onload = () => {
      const box = $("loupe-wrap");
      placeCropImg(wrapEl.querySelector(".cropwrap"), r, img.naturalWidth, img.naturalHeight, box.clientWidth, box.clientHeight);
    };
    img.src = previewUrl(im.id);
    plain.hidden = true;
    wrapEl.hidden = false;
  };

  function refreshCrops() {
    if (typeof renderRows === "function") renderRows(true);
    const im = S.byId.get(S.sel);
    if (im) window.ShotCrop.loupe(im);
  }
  window.ShotCrop.refresh = refreshCrops;

  /* ---------- ratio helpers (port of pipeline/crop_geom.py) ---------- */

  function parseRatio(text) {
    const t = String(text).trim().toLowerCase();
    if (t === "original" || t === "free") return { kind: t, a: 0, b: 0 };
    const p = t.split(":");
    if (p.length !== 2) return null;
    const re = /^\d+(\.\d{1,2})?$/;
    if (!re.test(p[0].trim()) || !re.test(p[1].trim())) return null;
    const a = parseFloat(p[0]), b = parseFloat(p[1]);
    if (!(a > 0) || !(b > 0) || Math.max(a, b) / Math.min(a, b) > 4 + 1e-9) return null;
    return { kind: "fixed", a, b };
  }
  function fmtNum(v) { return String(+v.toFixed(2)); }
  function ratioText(r) { return r.kind === "fixed" ? `${fmtNum(r.a)}:${fmtNum(r.b)}` : r.kind; }
  function ab(r) { return r.kind === "original" ? [C.info.full.w, C.info.full.h] : [r.a, r.b]; }

  function snapPx(rect, r, fw, fh) {
    if (r.kind === "original") return [0, 0, fw, fh];
    const cx = ((rect[0] + rect[2]) / 2) * fw, cy = ((rect[1] + rect[3]) / 2) * fh;
    let w, h;
    if (r.kind === "free") {
      w = Math.min(fw, Math.max(1, Math.round((rect[2] - rect[0]) * fw)));
      h = Math.min(fh, Math.max(1, Math.round((rect[3] - rect[1]) * fh)));
    } else {
      h = Math.min(fh, Math.max(1, Math.round((rect[3] - rect[1]) * fh)));
      w = Math.round((h * r.a) / r.b);
      if (w > fw) {
        h = Math.max(1, Math.floor((fw * r.b) / r.a + 1e-9));
        w = Math.round((h * r.a) / r.b);
        while (w > fw && h > 1) { h -= 1; w = Math.round((h * r.a) / r.b); }
      }
      w = Math.max(1, w);
    }
    let x0 = Math.round(cx - w / 2), y0 = Math.round(cy - h / 2);
    x0 = Math.min(Math.max(x0, 0), fw - w);
    y0 = Math.min(Math.max(y0, 0), fh - h);
    return [x0, y0, x0 + w, y0 + h];
  }
  function snapRect(rect, r) {
    const { w: fw, h: fh } = C.info.full;
    const p = snapPx(rect, r, fw, fh);
    return [p[0] / fw, p[1] / fh, p[2] / fw, p[3] / fh];
  }
  function lockedRn() {   // width/height of a locked crop in normalised units
    const r = parseRatio(C.resolved);
    if (!r || r.kind === "free") return null;
    const [a, b] = ab(r);
    return (a / b) * (C.info.full.h / C.info.full.w);
  }

  /* ---------- opening / closing ---------- */

  function typing(e) {
    const t = e.target;
    return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
  }
  async function open() {
    const im = S.byId.get(S.sel);
    if (!im) return;
    if (im.video_t != null) { toast("Crop is not for video frames."); return; }
    C.id = im.id;
    C.sel = []; C.boxes = []; C.activeBox = null; C.cands = []; C.alt = 0; C.rect = null; C.drag = null; C.esc = 0; C.burst = null; C.msg = "";
    C.source = "ai"; C.infeasible = false;
    closeOtherOverlays();
    $("overlay-loupe").hidden = false;
    $("lz").hidden = true;          // zoom is off in crop mode (the crop overlay is not zoomable)
    if (typeof loupeView !== "undefined") loupeView.reset();
    document.querySelector("#overlay-loupe > .hint").hidden = true;
    $("crop-ui").hidden = false;
    C.active = true;
    $("crop-name").textContent = im.name;
    $("crop-img").removeAttribute("src");
    setMsg("Loading…");
    try {
      C.info = await api(`/api/crop/${im.id}${q({ folder: S.folder })}`);
      C.subj = await api(`/api/crop/${im.id}/subjects${q({ folder: S.folder })}`);
    } catch (e) {
      setMsg(e.message);
      toast("Crop: " + e.message, 6000);
      close();
      return;
    }
    C.boxes = ((C.info && C.info.user_boxes) || []).map((b) => ({ bid: ++boxSeq, box: b.slice() }));
    $("crop-img").onload = () => { sizeStage(); draw(); };
    $("crop-img").src = previewUrl(im.id);
    sizeStage();
    buildChips();
    renderHistory();
    renderRotated();
    renderBurstBtn(im);
    $("crop-detector").hidden = !!C.subj.detector;
    setMsg(C.subj.detector ? "" : "Only faces and saliency are available without the object detector.");
    draw();
    schedule(0);
  }
  function closeOtherOverlays() {
    document.querySelectorAll(".overlay").forEach((o) => { if (o.id !== "overlay-loupe") o.hidden = true; });
  }
  function close() {
    C.active = false;
    C.req++;
    clearTimeout(C.timer);
    $("crop-ui").hidden = true;
    $("lz").hidden = false;
    const h = document.querySelector("#overlay-loupe > .hint");
    if (h) h.hidden = false;
    $("overlay-loupe").hidden = true;
  }
  window.ShotCrop.open = open;
  window.ShotCrop.close = close;
  new MutationObserver(() => { if (C.active && $("overlay-loupe").hidden) close(); })
    .observe($("overlay-loupe"), { attributes: true, attributeFilter: ["hidden"] });

  /* ---------- stage layout and drawing ---------- */

  function sizeStage() {
    if (!C.info) return;
    const { w, h } = C.info.full;
    const top = $("crop-main").getBoundingClientRect().top;
    const availW = Math.max(300, window.innerWidth - 360), availH = Math.max(240, window.innerHeight - top - 16);
    const s = Math.min(availW / w, availH / h);
    const st = $("crop-stage");
    st.style.width = Math.floor(w * s) + "px";
    st.style.height = Math.floor(h * s) + "px";
  }
  window.addEventListener("resize", () => { if (C.active) sizeStage(); });

  const pct = (v) => (v * 100).toFixed(3) + "%";
  function placeBox(el, b) {
    el.style.left = pct(b[0]); el.style.top = pct(b[1]);
    el.style.width = pct(b[2] - b[0]); el.style.height = pct(b[3] - b[1]);
  }
  function selected(sid) { return C.sel.findIndex((s) => s.sid === sid); }
  function allSubjects() {
    const pts = C.sel.filter((s) => s.point).map((s) => ({ sid: s.sid, kind: "point", box: s.box, label: "point" }));
    return [...C.subj.subjects, ...pts];
  }
  function draw() {
    if (!C.active || !C.info) return;
    const layer = $("crop-subjects");
    const subs = allSubjects();
    layer.innerHTML = "";
    for (const s of subs) {
      const d = document.createElement("div");
      let idx = selected(s.sid);
      if (idx < 0 && s.kind === "face" && s.linked) idx = selected(s.linked);   // the core of a selected person
      d.className = `subj k-${s.kind}${idx >= 0 ? " sel" : ""}${C.hover === s.sid ? " hover" : ""}`;
      d.dataset.sid = s.sid;
      d.dataset.kind = s.kind;
      placeBox(d, s.box);
      d.innerHTML = `<span class="subj-label">${esc(s.label || s.kind)}</span>` + (idx >= 0 ? `<span class="subj-badge">${idx + 1}</span>` : "");
      layer.appendChild(d);
    }
    drawBoxes();
    drawCrop();
  }
  /* user-drawn boxes: solid teal, corner handles, an x button; the layer itself ignores the mouse */
  function drawBoxes() {
    const layer = $("crop-uboxes");
    layer.innerHTML = "";
    C.boxes.forEach((b, i) => {
      const d = document.createElement("div");
      d.className = "ubox" + (b.bid === C.activeBox ? " active" : "");
      d.dataset.bid = b.bid;
      placeBox(d, b.box);
      d.innerHTML = `<span class="subj-label">your box ${i + 1}</span>` +
        `<button class="ubox-x" type="button" title="Remove this box (Delete)" aria-label="Remove box ${i + 1}">\u00d7</button>` +
        ["nw", "ne", "sw", "se"].map((h) => `<i class="ubh" data-ubh="${h}"></i>`).join("");
      layer.appendChild(d);
    });
    $("crop-hint").classList.toggle("has-boxes", C.boxes.length > 0);
  }
  const boxById = (bid) => C.boxes.find((b) => b.bid === Number(bid));
  async function commitBoxes() {
    draw();
    try {
      await api(`/api/crop/${C.id}/boxes`, { method: "POST", body: { folder: S.folder, boxes: C.boxes.map(boxJson) } });
    } catch (e) { toast("Could not save the box: " + e.message, 5000); }
    schedule(0);
  }
  const boxJson = (b) => ({ x0: b.box[0], y0: b.box[1], x1: b.box[2], y1: b.box[3] });
  function removeBox(bid) {
    const i = C.boxes.findIndex((b) => b.bid === Number(bid));
    if (i < 0) return;
    C.boxes.splice(i, 1);
    C.activeBox = C.boxes.length ? C.boxes[C.boxes.length - 1].bid : null;
    commitBoxes();
  }
  function drawCrop() {
    const box = $("crop-box"), cur = $("crop-cur");
    const curRect = C.info && C.info.current && C.info.current.rect;
    cur.hidden = !curRect || (C.rect && sameRect(curRect, C.rect));
    if (curRect) placeBox(cur, curRect);
    for (const id of ["crop-dim-t", "crop-dim-b", "crop-dim-l", "crop-dim-r"]) $(id).hidden = !C.rect;
    box.hidden = !C.rect;
    if (!C.rect) { box.removeAttribute("data-rect"); return; }
    const r = C.rect;
    placeBox(box, r);
    box.dataset.rect = r.map((v) => v.toFixed(6)).join(",");
    box.dataset.ratio = C.resolved;
    box.dataset.source = C.source;
    box.dataset.infeasible = C.infeasible ? "1" : "0";
    $("crop-dim-t").style.cssText = `left:0;top:0;width:100%;height:${pct(r[1])}`;
    $("crop-dim-b").style.cssText = `left:0;top:${pct(r[3])};width:100%;height:${pct(1 - r[3])}`;
    $("crop-dim-l").style.cssText = `left:0;top:${pct(r[1])};width:${pct(r[0])};height:${pct(r[3] - r[1])}`;
    $("crop-dim-r").style.cssText = `left:${pct(r[2])};top:${pct(r[1])};width:${pct(1 - r[2])};height:${pct(r[3] - r[1])}`;
    const out = leavingSubjects(r);
    box.classList.toggle("warn-red", out.length > 0);
    const cut = !!(C.source === "ai" && C.cands[C.alt] && C.cands[C.alt].warnings.includes("subject_may_be_cut"));
    box.dataset.mayBeCut = cut ? "1" : "0";
    box.classList.toggle("amber", (C.infeasible || cut) && out.length === 0);
    renderWarn(out);
    $("crop-accept").disabled = false;
  }
  function sameRect(a, b) { return a.every((v, i) => Math.abs(v - b[i]) < 1e-5); }
  function leavingSubjects(r) {
    const out = [];
    const items = [...C.sel, ...C.boxes.map((u) => ({ sid: "box" + u.bid, box: u.box }))];
    for (const s of items) {
      const b = s.box || (C.subj.subjects.find((x) => x.sid === s.sid) || {}).box;
      if (!b) continue;
      const iw = Math.min(r[2], b[2]) - Math.max(r[0], b[0]), ih = Math.min(r[3], b[3]) - Math.max(r[1], b[1]);
      const inside = iw > 0 && ih > 0 ? (iw * ih) / ((b[2] - b[0]) * (b[3] - b[1])) : 0;
      if (inside < 0.98) out.push(s.sid);
    }
    return out;
  }
  function renderWarn(out) {
    const el = $("crop-warn");
    const c = C.cands[C.alt];
    const parts = [];
    if (out.length) parts.push("A selected subject leaves the crop.");
    if (C.infeasible) parts.push("Subjects don't fit at this ratio: the widest possible crop is shown.");
    if (c && C.source === "ai") {
      for (const w of c.warnings) {
        if (w === "subjects_do_not_fit") continue;
        parts.push(w === "subject_may_be_cut" ? "An edge may cut through part of a selected subject." : w === "unselected_person_cut" ? "Cuts through a person you did not select." : w === "limb_cut" ? "Crop edge falls on a joint." : w.startsWith("horizon_tilted") ? "Horizon tilted " + w.replace("horizon_tilted_", "").replace("deg", "°") + " (not corrected)." : w.startsWith("auto_dropped") ? "Not everyone fits: the most prominent subjects are kept." : w);
      }
    }
    el.textContent = parts.join(" ");
    el.hidden = parts.length === 0;
  }
  function setMsg(t) { C.msg = t; const el = $("crop-msg"); el.textContent = t; el.hidden = !t; }

  /* ---------- chips, custom ratio, flip ---------- */

  function buildChips() {
    const box = $("crop-ratios");
    box.innerHTML = "";
    for (const p of PRESETS) {
      const b = document.createElement("button");
      b.dataset.ratio = p;
      b.textContent = CHIP_LABEL[p] || p;
      b.setAttribute("aria-pressed", String(C.ratio === p));
      b.onclick = () => { C.ratio = p; C.flip = false; $("crop-ratio-err").textContent = ""; buildChips(); schedule(0); };
      box.appendChild(b);
    }
  }
  function applyCustom() {
    const a = $("crop-w").value.trim(), b = $("crop-h").value.trim();
    const r = parseRatio(`${a}:${b}`);
    const err = $("crop-ratio-err");
    if (!r || r.kind !== "fixed") { err.textContent = "Enter numbers like 1.91 and 1 (up to 2 decimals, between 1:4 and 4:1)."; return; }
    err.textContent = "";
    C.ratio = ratioText(r); C.flip = false;
    buildChips();
    schedule(0);
  }
  function flip() {
    C.flip = !C.flip;
    schedule(0);
  }
  $("crop-custom-apply").onclick = applyCustom;
  for (const id of ["crop-w", "crop-h"]) $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") { e.stopPropagation(); applyCustom(); } });
  $("crop-flip").onclick = flip;
  $("crop-clear").onclick = clearSelection;

  /* ---------- suggestions ---------- */

  function schedule(delay = 150) {
    clearTimeout(C.timer);
    $("crop-ui").dataset.busy = "1";   // tests (and the UI) can tell when a suggestion is pending
    C.timer = setTimeout(suggest, delay);
  }
  async function suggest() {
    if (!C.active) return;
    const my = ++C.req;
    try { await suggestInner(my); } finally { if (my === C.req) $("crop-ui").dataset.busy = "0"; }
  }
  async function suggestInner(my) {
    const body = {
      folder: S.folder, ratio: C.ratio, flip: C.flip, n: 3,
      subjects: C.sel.filter((s) => !s.point).map((s) => s.sid),
      points: C.sel.filter((s) => s.point).map((s) => s.point),
      boxes: C.boxes.map(boxJson),
      select_modes: Object.fromEntries(C.sel.filter((s) => s.mode).map((s) => [s.sid, s.mode])),
    };
    let r;
    try { r = await api(`/api/crop/${C.id}/suggest`, { method: "POST", body }); }
    catch (e) { if (my === C.req) { setMsg("Suggestion failed: " + e.message); } return; }
    if (my !== C.req || !C.active) return;   // stale response
    C.cands = r.candidates; C.alt = 0; C.resolved = r.ratio; C.infeasible = r.infeasible;
    C.auto = r.auto;
    if (r.auto && C.sel.length === 0) C.autoSel = r.selected;
    setAlt(0);
    setMsg(r.auto && C.sel.length === 0 && !C.boxes.length ? "Auto: click a subject, or drag a box around it, to choose what to keep." : "");
    markChips();
  }
  function markChips() {
    document.querySelectorAll("#crop-ratios button").forEach((b) => {
      const on = b.dataset.ratio === C.ratio || (C.resolved === "free" && b.dataset.ratio === "free" && C.source === "ai_adjusted");
      b.setAttribute("aria-pressed", String(on));
    });
  }
  function setAlt(i) {
    if (!C.cands.length) { C.rect = null; renderCands(); drawCrop(); return; }
    C.alt = (i + C.cands.length) % C.cands.length;
    const c = C.cands[C.alt];
    C.rect = c.rect.slice(); C.score = c.score; C.source = "ai";
    renderCands();
    drawCrop();
  }
  function renderCands() {
    const box = $("crop-cands");
    box.innerHTML = "";
    const { w: fw, h: fh } = C.info.full;
    C.cands.forEach((c, i) => {
      const b = document.createElement("button");
      b.className = "cand";
      b.dataset.alt = i;
      b.setAttribute("aria-pressed", String(i === C.alt));
      const cw = c.rect[2] - c.rect[0], ch = c.rect[3] - c.rect[1];
      const ca = (cw * fw) / (ch * fh);
      let w = 108, h = 108 / ca;
      if (h > 84) { h = 84; w = 84 * ca; }
      b.innerHTML = `<span class="cropslot mini"><span class="cropwrap" style="width:${w}px;height:${h}px;visibility:visible">` +
        `<img alt="" src="${thumbUrl(C.id, 400)}" style="width:${100 / cw}%;height:${100 / ch}%;left:${-c.rect[0] / cw * 100}%;top:${-c.rect[1] / ch * 100}%"></span></span>` +
        `<span class="cand-n">${i + 1}</span>` + (c.warnings.length ? '<span class="cand-w" title="' + esc(c.warnings.join(", ")) + '">!</span>' : "");
      b.onclick = () => setAlt(i);
      box.appendChild(b);
    });
  }

  /* ---------- selection: click / shift-click / Option-click / point prompt ---------- */

  function stagePoint(e) {
    const r = $("crop-stage").getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
  }
  function hit(x, y) {   // the smallest detected box containing the point; a face wins over its person
    let best = null, bestA = 9;
    for (const s of C.subj.subjects) {
      const b = s.box;
      if (x < b[0] || x > b[2] || y < b[1] || y > b[3]) continue;
      const a = (b[2] - b[0]) * (b[3] - b[1]);
      if (a < bestA) { best = s; bestA = a; }
    }
    return best;
  }
  async function pick(p, e) {
    const s = hit(p.x, p.y);
    const add = e.shiftKey;
    let entry;
    if (s) {
      if (s.kind === "face") {
        // a face linked to a person selects the PERSON with the face as must-keep core;
        // Option/Alt-click (or an unlinked face) selects the face only
        entry = !e.altKey && s.linked ? { sid: s.linked, mode: "person" } : { sid: s.sid, mode: "face" };
      } else if (s.kind === "person") entry = { sid: s.sid, mode: "person" };
      else entry = { sid: s.sid };
    } else {
      try {
        const r = await api(`/api/crop/${C.id}/point`, { method: "POST", body: { folder: S.folder, x: p.x, y: p.y } });
        entry = { sid: r.subject.sid, point: { x: p.x, y: p.y }, box: r.subject.box };
      } catch (err) { setMsg(err.message); return; }
    }
    const i = selected(entry.sid);
    if (add) { if (i >= 0) C.sel.splice(i, 1); else C.sel.push(entry); }
    else if (i >= 0) C.sel.splice(i, 1);   // clicking a selected subject again deselects it
    else C.sel = [entry];
    C.esc = 0;
    draw();
    schedule();
  }
  function clearSelection() {
    const hadBoxes = C.boxes.length > 0;
    C.sel = []; C.boxes = []; C.activeBox = null;
    if (hadBoxes) { commitBoxes(); return; }
    draw();
    schedule(0);
  }

  /* ---------- pointer handling: select, move, resize ---------- */

  const stage = $("crop-stage");
  stage.addEventListener("pointermove", (e) => {
    if (!C.active || C.drag || !C.subj) return;
    const p = stagePoint(e);
    const s = hit(p.x, p.y);
    const sid = s ? s.sid : null;
    if (sid !== C.hover) {
      C.hover = sid;
      document.querySelectorAll("#crop-subjects .subj").forEach((d) => d.classList.toggle("hover", d.dataset.sid === sid));
    }
  });
  const CROP_MODES = (m) => m !== "draw" && m !== "umove" && m !== "ures";
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const normBox = (a, b) => [Math.min(a.x, b.x), Math.min(a.y, b.y), Math.max(a.x, b.x), Math.max(a.y, b.y)];
  stage.addEventListener("pointerdown", (e) => {
    if (!C.active || !C.info || e.button !== 0) return;
    const t = e.target && e.target.closest ? e.target : null;
    if (t && t.closest(".ubox-x")) return;   // the x button handles its own click
    const p = stagePoint(e);
    const handle = t && t.closest(".handle"), grip = t && t.closest(".grip");
    const ubh = t && t.closest(".ubh"), ub = t && t.closest(".ubox");
    const inCrop = C.rect && p.x >= C.rect[0] && p.x <= C.rect[2] && p.y >= C.rect[1] && p.y <= C.rect[3];
    let mode = "draw", ref = null;
    if (handle) mode = handle.dataset.h;
    else if (grip || ((e.metaKey || e.ctrlKey) && inCrop)) mode = "move";   // plain drags draw a subject box
    else if (ubh && ub) { mode = "ures"; ref = { bid: ub.dataset.bid, h: ubh.dataset.ubh }; }
    else if (ub) { mode = "umove"; ref = { bid: ub.dataset.bid }; }
    const ubx = ref && boxById(ref.bid);
    C.drag = { mode, ref, start: p, rect0: C.rect ? C.rect.slice() : null, box0: ubx ? ubx.box.slice() : null, moved: false, e0: { shiftKey: e.shiftKey, altKey: e.altKey }, id: e.pointerId };
    stage.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  stage.addEventListener("pointermove", (e) => {
    const d = C.drag;
    if (!d) return;
    const p = stagePoint(e);
    if (!d.moved && Math.hypot(p.x - d.start.x, p.y - d.start.y) * stage.clientWidth < 4) return;
    d.moved = true;
    if (d.mode === "draw") {
      const dr = $("crop-draft");
      d.draft = normBox(d.start, p);
      placeBox(dr, d.draft);
      dr.hidden = false;
      return;
    }
    if (d.mode === "umove" || d.mode === "ures") {
      const ub = boxById(d.ref.bid);
      if (!ub) return;
      const b0 = d.box0;
      if (d.mode === "umove") {
        const w = b0[2] - b0[0], h = b0[3] - b0[1];
        const x0 = Math.min(1 - w, Math.max(0, b0[0] + p.x - d.start.x)), y0 = Math.min(1 - h, Math.max(0, b0[1] + p.y - d.start.y));
        ub.box = [x0, y0, x0 + w, y0 + h];
      } else {
        const h = d.ref.h;
        let [x0, y0, x1, y1] = b0;
        if (h.includes("w")) x0 = Math.min(clamp01(p.x), x1 - MIN_BOX);
        if (h.includes("e")) x1 = Math.max(clamp01(p.x), x0 + MIN_BOX);
        if (h.includes("n")) y0 = Math.min(clamp01(p.y), y1 - MIN_BOX);
        if (h.includes("s")) y1 = Math.max(clamp01(p.y), y0 + MIN_BOX);
        ub.box = [x0, y0, x1, y1];
      }
      const el = document.querySelector(`#crop-uboxes .ubox[data-bid="${ub.bid}"]`);
      if (el) placeBox(el, ub.box);
      return;
    }
    if (!CROP_MODES(d.mode)) return;
    if (e.shiftKey && d.mode !== "move" && C.resolved !== "free") {   // Shift unlocks the ratio
      C.resolved = "free";
      markChips();
    }
    C.rect = dragRect(d, p);
    C.source = C.cands.length ? "ai_adjusted" : "manual";
    drawCrop();
  });
  stage.addEventListener("pointerup", (e) => {
    const d = C.drag;
    C.drag = null;
    try { stage.releasePointerCapture(e.pointerId); } catch (err) { /* already released */ }
    $("crop-draft").hidden = true;
    if (!d) return;
    if (d.mode === "draw") {
      const b = d.draft;
      if (d.moved && b && b[2] - b[0] >= MIN_BOX && b[3] - b[1] >= MIN_BOX) {
        if (C.boxes.length >= MAX_BOXES) { toast(`Up to ${MAX_BOXES} boxes: remove one first.`); return; }
        const nb = { bid: ++boxSeq, box: b.map(clamp01) };
        C.boxes.push(nb);
        C.activeBox = nb.bid;
        C.esc = 0;
        commitBoxes();
        return;
      }
      pick(d.start, e);   // a click (or a drag too small to be a box) selects / points
      return;
    }
    if (d.mode === "umove" || d.mode === "ures") {
      if (d.moved) commitBoxes();
      else { C.activeBox = Number(d.ref.bid); drawBoxes(); }
      return;
    }
    if (d.moved && C.rect) {
      const r = parseRatio(C.resolved) || { kind: "free", a: 0, b: 0 };
      C.rect = snapRect(C.rect, r);
      C.source = C.cands.length ? "ai_adjusted" : "manual";
      drawCrop();
      return;
    }
    if (d.mode === "move") return;   // a click on the move grip does nothing
    pick(d.start, e);   // a click without movement selects (also inside the crop box)
  });
  stage.addEventListener("pointercancel", () => { C.drag = null; $("crop-draft").hidden = true; });
  stage.addEventListener("click", (e) => {
    const x = e.target.closest && e.target.closest(".ubox-x");
    if (x) { e.stopPropagation(); removeBox(x.closest(".ubox").dataset.bid); }
  });

  function dragRect(d, p) {
    const r0 = d.rect0;
    const dx = p.x - d.start.x, dy = p.y - d.start.y;
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    if (d.mode === "move") {
      const w = r0[2] - r0[0], h = r0[3] - r0[1];
      const x0 = clamp(r0[0] + dx, 0, 1 - w), y0 = clamp(r0[1] + dy, 0, 1 - h);
      return [x0, y0, x0 + w, y0 + h];
    }
    const rn = lockedRn();
    const m = d.mode;
    let x0 = r0[0], y0 = r0[1], x1 = r0[2], y1 = r0[3];
    const west = m.includes("w"), east = m.includes("e"), north = m.includes("n"), south = m.includes("s");
    if (rn === null) {   // free resize (Free ratio, or Shift held: the ratio switched to Free)
      if (west) x0 = clamp(p.x, 0, x1 - MIN_N);
      if (east) x1 = clamp(p.x, x0 + MIN_N, 1);
      if (north) y0 = clamp(p.y, 0, y1 - MIN_N);
      if (south) y1 = clamp(p.y, y0 + MIN_N, 1);
      return [x0, y0, x1, y1];
    }
    const corner = (west || east) && (north || south);
    const ax = west ? r0[2] : r0[0], ay = north ? r0[3] : r0[1];   // fixed (opposite) corner / edge
    if (corner) {
      const sx = east ? 1 : -1, sy = south ? 1 : -1;
      let w = Math.max(sx * (p.x - ax), (sy * (p.y - ay)) * rn);
      const maxW = Math.min(sx > 0 ? 1 - ax : ax, (sy > 0 ? 1 - ay : ay) * rn);
      w = clamp(w, MIN_N, Math.max(MIN_N, maxW));
      const h = w / rn;
      const nx = sx > 0 ? [ax, ax + w] : [ax - w, ax], ny = sy > 0 ? [ay, ay + h] : [ay - h, ay];
      return [nx[0], ny[0], nx[1], ny[1]];
    }
    // an edge handle with the ratio locked: scale about the opposite edge, centred on the other axis
    if (west || east) {
      const sx = east ? 1 : -1;
      let w = sx * (p.x - ax);
      const cy = (r0[1] + r0[3]) / 2;
      const maxW = Math.min(sx > 0 ? 1 - ax : ax, 2 * Math.min(cy, 1 - cy) * rn);
      w = clamp(w, MIN_N, Math.max(MIN_N, maxW));
      const h = w / rn;
      return [sx > 0 ? ax : ax - w, cy - h / 2, sx > 0 ? ax + w : ax, cy + h / 2];
    }
    const sy = south ? 1 : -1;
    let h = sy * (p.y - ay);
    const cx = (r0[0] + r0[2]) / 2;
    const maxH = Math.min(sy > 0 ? 1 - ay : ay, (2 * Math.min(cx, 1 - cx)) / rn);
    h = clamp(h, MIN_N / rn, Math.max(MIN_N / rn, maxH));
    const w = h * rn;
    return [cx - w / 2, sy > 0 ? ay : ay - h, cx + w / 2, sy > 0 ? ay + h : ay];
  }

  /* ---------- accept / reject / reset / restore / history ---------- */

  async function accept() {
    if (!C.rect || !C.active) return;
    const body = {
      folder: S.folder, rect: C.rect, ratio: C.resolved || C.ratio, source: C.source,
      subjects: [...C.sel.map((s) => s.sid), ...C.boxes.map((b) => "box:" + b.box.map((v) => v.toFixed(4)).join(","))], alt_rank: C.source === "ai" ? C.alt : null, score: C.source === "ai" ? C.score : null,
      write: "auto", catalog: (window.ShotCrop.catalog && window.ShotCrop.catalog()) || null,
    };
    try {
      const r = await api(`/api/crop/${C.id}/accept`, { method: "POST", body });
      applyVersion(r);
      toast(writeText(r), 6000);
    } catch (e) { toast("Crop not saved: " + e.message, 7000); }
  }
  function writeText(r) {
    const w = r.write || {};
    const base = `Crop v${r.version} saved`;
    if (w.status === "written" || w.status === "replaced_own") return `${base}; sidecar ${w.mode === "embed" ? "copy " : ""}written.`;
    if (w.status === "skipped_user_crop") return `${base} in ShotSieve only: your Lightroom crop is kept.`;
    if (w.status === "skipped_rotated") return `${base} in ShotSieve only. ${w.reason || ""}`;
    if (w.status === "skipped_catalog") return `${base} in ShotSieve only. ${w.reason || ""}`;
    if (w.status === "db_only") return `${base} in ShotSieve only. ${w.reason || ""}`;
    return `${base}${w.status && w.status !== "none" ? " (" + w.status + ")" : " in ShotSieve"}.`;
  }
  async function refreshInfo() {
    C.info = await api(`/api/crop/${C.id}${q({ folder: S.folder })}`);
    renderRotated();
    const im = S.byId.get(C.id);
    const cur = C.info.current;
    if (im) im.crop = cur && cur.rect && !cur.readonly ? cur.rect : (cur && cur.rect ? cur.rect : null);
    renderHistory();
    refreshCrops();
  }
  async function applyVersion(r) {
    C.sel = C.sel.slice();
    await refreshInfo();
    const cur = C.info.current;
    C.rect = cur && cur.rect ? cur.rect.slice() : null;
    C.cands = []; renderCands();
    C.source = cur ? cur.source : "ai";
    draw();
  }
  async function rejectSuggestion() {
    if (!C.active) return;
    try { await api(`/api/crop/${C.id}/reject`, { method: "POST", body: { folder: S.folder, alt_rank: C.rect && C.source === "ai" ? C.alt : null } }); } catch (e) { /* logged only */ }
    C.cands = []; renderCands();
    const cur = C.info && C.info.current;
    C.rect = cur && cur.rect ? cur.rect.slice() : null;
    setMsg("Suggestion dismissed: the crop is unchanged.");
    drawCrop();
  }
  async function reset() {
    if (!C.active) return;
    try {
      const r = await api(`/api/crop/${C.id}/reset`, { method: "POST", body: { folder: S.folder, write: "auto", catalog: (window.ShotCrop.catalog && window.ShotCrop.catalog()) || null } });
      await applyVersion(r);
      C.rect = null; drawCrop();
      toast(`Crop reset (v${r.version}): the full frame is back.`, 5000);
    } catch (e) { toast("Reset failed: " + e.message, 6000); }
  }
  async function restore() {
    const v = Number($("crop-history").value);
    if (!v) return;
    try {
      const r = await api(`/api/crop/${C.id}/restore`, { method: "POST", body: { folder: S.folder, version: v, write: "auto", catalog: (window.ShotCrop.catalog && window.ShotCrop.catalog()) || null } });
      await applyVersion(r);
      toast(`Restored v${v} as v${r.version}.`, 5000);
    } catch (e) { toast("Restore failed: " + e.message, 6000); }
  }
  function renderRotated() {
    const el = $("crop-rotated");
    el.textContent = (C.info && C.info.rotated_notice) || "";
    el.hidden = !(C.info && C.info.rotated_in_lightroom);
  }
  function renderHistory() {
    const sel = $("crop-history");
    const h = (C.info && C.info.history) || [];
    sel.innerHTML = h.length ? "" : '<option value="">no versions yet</option>';
    for (const v of [...h].reverse()) {
      const o = document.createElement("option");
      o.value = v.version;
      const t = new Date(v.created * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      o.textContent = `v${v.version} · ${v.rect ? v.ratio || "free" : "full frame"} · ${v.source} · ${t}`;
      sel.appendChild(o);
    }
    $("crop-restore").disabled = !h.length;
  }
  $("crop-accept").onclick = accept;
  $("crop-reject").onclick = rejectSuggestion;
  $("crop-reset").onclick = reset;
  $("crop-restore").onclick = restore;

  /* ---------- apply to burst ---------- */

  function burstOthers(im) {
    if (!im || im.burst_id === null || im.burst_id === undefined) return [];
    return S.images.filter((x) => x.burst_id === im.burst_id && x.id !== im.id && x.video_t == null);
  }
  function renderBurstBtn(im) {
    const n = burstOthers(im).length;
    const b = $("crop-burst-btn");
    b.hidden = n < 1;
    b.textContent = `Apply to burst (${n})`;
    $("crop-burst").innerHTML = "";
  }
  async function burstApply() {
    if (!C.rect) { toast("Accept or draw a crop first."); return; }
    const im = S.byId.get(C.id);
    const ids = burstOthers(im).map((x) => x.id);
    $("crop-burst-btn").disabled = true;
    setMsg("Finding the subject in the other frames…");
    try {
      const body = {
        folder: S.folder, id: C.id, ids, rect: C.rect, ratio: C.resolved || C.ratio,
        subjects: C.sel.filter((s) => !s.point).map((s) => s.sid),
        points: C.sel.filter((s) => s.point).map((s) => s.point),
        select_modes: Object.fromEntries(C.sel.filter((s) => s.mode).map((s) => [s.sid, s.mode])),
      };
      const { job_id } = await api("/api/crop/burst", { method: "POST", body });
      const j = await waitJob(job_id, () => {});
      C.burst = { ratio: j.result.ratio, items: j.result.results.map((x) => ({ ...x, state: x.status === "suggested" ? "pending" : "needs" })) };
      renderBurst();
      setMsg("Proposals only: nothing is saved until you accept.");
    } catch (e) { setMsg(e.message); toast("Burst: " + e.message, 6000); }
    $("crop-burst-btn").disabled = false;
  }
  function renderBurst() {
    const box = $("crop-burst");
    box.innerHTML = "";
    if (!C.burst) return;
    const pending = C.burst.items.filter((x) => x.state === "pending");
    const head = document.createElement("div");
    head.className = "burst-head";
    head.innerHTML = `<b>Burst proposals</b> <button id="crop-burst-all" class="small" ${pending.length ? "" : "disabled"}>Accept all (${pending.length})</button>`;
    box.appendChild(head);
    for (const it of C.burst.items) {
      const f = document.createElement("div");
      f.className = "burst-item " + it.state;
      f.dataset.id = it.id;
      const im = S.byId.get(it.id);
      const rect = it.rect;
      let thumb = `<img alt="" src="${thumbUrl(it.id, 400)}">`;
      if (rect) {
        const cw = rect[2] - rect[0], ch = rect[3] - rect[1];
        thumb = `<span class="cropwrap" style="width:72px;height:72px;visibility:visible"><img alt="" src="${thumbUrl(it.id, 400)}" style="width:${100 / cw}%;height:${100 / ch}%;left:${-rect[0] / cw * 100}%;top:${-rect[1] / ch * 100}%"></span>`;
      }
      f.innerHTML = `<span class="cropslot mini">${thumb}</span><span class="cap">${esc(im ? im.name : it.id)}</span>` +
        (it.state === "pending" ? `<span class="btns"><button data-do="acc" class="small">Accept</button><button data-do="rej" class="small">Reject</button></span>`
          : `<span class="muted">${it.state === "needs" ? "needs selection" : it.state}</span>`);
      f.querySelectorAll("button").forEach((b) => { b.onclick = () => burstItem(it, b.dataset.do); });
      box.appendChild(f);
    }
    const all = $("crop-burst-all");
    if (all) all.onclick = () => burstAccept(C.burst.items.filter((x) => x.state === "pending"));
  }
  async function burstItem(it, what) {
    if (what === "rej") { it.state = "rejected"; renderBurst(); return; }
    await burstAccept([it]);
  }
  async function burstAccept(items) {
    if (!items.length) return;
    try {
      const r = await api("/api/crop/accept_many", { method: "POST", body: { folder: S.folder, write: "auto", catalog: (window.ShotCrop.catalog && window.ShotCrop.catalog()) || null, items: items.map((x) => ({ id: x.id, rect: x.rect, ratio: C.burst.ratio })) } });
      let ok = 0;
      for (const res of r.results) {
        const it = items.find((x) => x.id === res.id);
        if (res.ok) { ok++; it.state = "accepted"; const im = S.byId.get(res.id); if (im) im.crop = res.rect; }
        else { it.state = "failed"; }
      }
      renderBurst();
      refreshCrops();
      toast(`${ok} burst crop(s) saved.`, 4000);
    } catch (e) { toast("Burst accept failed: " + e.message, 6000); }
  }
  $("crop-burst-btn").onclick = burstApply;

  /* ---------- keyboard (capture phase: crop mode owns the keys while it is open) ---------- */

  document.addEventListener("keydown", (e) => {
    if (typing(e)) {
      if (C.active && e.key === "Escape") { e.target.blur(); e.stopPropagation(); e.preventDefault(); }
      return;
    }
    if (e.ctrlKey || e.metaKey) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (!C.active) {
      if (key === "r" && !e.altKey && S.sel !== null && !document.querySelector(".overlay:not([hidden]):not(#overlay-loupe)") && $("overlay-onboarding").hidden) {
        e.preventDefault(); e.stopPropagation();
        open();
      }
      return;
    }
    e.stopPropagation();   // none of the global shortcuts (X rejects, 1-5 rate, arrows move) while cropping
    e.preventDefault();
    switch (key) {
      case "Escape":
        if ((C.sel.length || C.boxes.length) && C.esc === 0) { clearSelection(); C.esc = 1; }
        else close();
        break;
      case "Enter": accept(); break;
      case "Backspace": rejectSuggestion(); break;
      case "Delete": if (C.activeBox !== null) removeBox(C.activeBox); break;
      case "x": flip(); break;
      case "[": if (C.cands.length) setAlt(C.alt - 1); break;
      case "]": if (C.cands.length) setAlt(C.alt + 1); break;
      case "r": close(); break;
      default: break;
    }
  }, true);

  /* ---------- wiring in the main UI ---------- */

  const btn = $("btn-crop");
  if (btn) btn.onclick = () => open();
  const chk = $("chk-crops");
  if (chk) {
    try { chk.checked = localStorage.getItem("shotsieve.showCrops") !== "0"; } catch (e) { /* ignore */ }
    chk.onchange = () => {
      try { localStorage.setItem("shotsieve.showCrops", chk.checked ? "1" : "0"); } catch (e) { /* ignore */ }
      refreshCrops();
    };
  }
  window.ShotCrop.catalog = () => (typeof LR !== "undefined" && LR.path) || null;
  $("crop-detector-btn").onclick = async () => {
    const b = $("crop-detector-btn");
    b.disabled = true;
    try {
      const r = await api("/api/models/download", { method: "POST", body: { groups: ["crop"] } });
      const j = await waitJob(r.job_id, (jj) => { b.textContent = `Downloading… ${jj.total ? Math.round((100 * jj.done) / jj.total) : 0}%`; });
      if (j.state === "error") throw new Error(j.error);
      toast("Object detector ready.", 3000);
      if (C.active) { C.subj = await api(`/api/crop/${C.id}/subjects${q({ folder: S.folder })}`); $("crop-detector").hidden = !!C.subj.detector; draw(); schedule(0); }
    } catch (e) { toast("Download failed: " + e.message, 6000); b.textContent = "Download (36 MB)"; b.disabled = false; }
  };
  window.ShotCrop.syncButton = function (im) {   // the detail panel's Crop button
    const b = $("btn-crop");
    if (!b) return;
    const video = !!im && im.video_t != null;
    b.disabled = !im || video;
    b.title = video ? "Not for video frames" : "Crop with AI suggestions (R)";
  };
})();
