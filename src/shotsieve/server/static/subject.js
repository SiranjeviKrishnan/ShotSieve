/* ShotSieve subject mode (T23h): S in the loupe. The cursor becomes a crosshair; click-and-drag draws a
   box around the subject, boxes move / resize by their corners / delete (X, Delete, or the x button), up
   to 6. The boxes ARE crop mode's user boxes (/api/crop/{id}/boxes: display-normalised x0,y0,x1,y1,
   EXIF orientation respected), so they persist, survive a reload and steer the crop suggestions; there is
   no second storage. Positions come from zoom.js's own frame (loupeView.frame()), exactly like the crop
   overlay, so the boxes stay glued to the picture while zooming and panning. Each box shows its
   sharpness (POST /api/subject/{id}/score: the cull's face sharpness measure on the full-resolution
   region). Esc (preventDefault: never leaves macOS full screen) or S leaves the mode. */
"use strict";

(function () {
  const MAX_BOXES = 6, MIN_BOX = 0.012;
  const host = $("lz-view");
  const layer = document.createElement("div");
  layer.id = "subj-layer";
  layer.dataset.mode = "0";
  host.appendChild(layer);
  const draft = document.createElement("div");
  draft.className = "sbox-draft";
  draft.hidden = true;
  layer.appendChild(draft);

  let on = false;
  let boxes = [];            // [{bid, box:[x0,y0,x1,y1], score}]
  let seq = 0, active = null, loadedFor = null, drag = null, scoreTok = 0;
  const scoreCache = new Map();   // JSON(box) + image id -> score payload
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const boxJson = (b) => ({ x0: b.box[0], y0: b.box[1], x1: b.box[2], y1: b.box[3] });
  const curId = () => (typeof loupeView !== "undefined" ? loupeView.id : null);
  const cropOpen = () => !$("crop-ui").hidden;
  const loupeOpen = () => !$("overlay-loupe").hidden;
  const visible = () => loupeOpen() && !cropOpen() && boxes.length > 0;

  function scoreText(b, i) {
    const s = b.score;
    if (!s) return `Subject ${i + 1}`;
    if (s.sharp === null || s.sharp === undefined) return `Subject ${i + 1}: sharpness n/a`;
    return `Subject ${i + 1}: ${s.in_focus ? "sharp" : "soft"} ${s.sharp.toFixed(2)} ${s.in_focus ? "✓ in focus" : "✗ soft"}`;
  }

  /* ---------- rendering ---------- */
  function build() {
    [...layer.querySelectorAll(".sbox")].forEach((e) => e.remove());
    boxes.forEach((b, i) => {
      const d = document.createElement("div");
      d.className = "sbox" + (b.bid === active ? " active" : "");
      d.dataset.bid = String(b.bid);
      d.innerHTML = `<span class="sbox-label"></span>` +
        `<button class="sbox-x" type="button" title="Remove this box (X or Delete)" aria-label="Remove subject box ${i + 1}">×</button>` +
        ["nw", "ne", "sw", "se"].map((h) => `<i class="sbh" data-sbh="${h}"></i>`).join("");
      layer.insertBefore(d, draft);
    });
    place();
  }
  function frameBox(b, fr) {
    return { x: fr.x + b[0] * fr.w, y: fr.y + b[1] * fr.h, w: (b[2] - b[0]) * fr.w, h: (b[3] - b[1]) * fr.h };
  }
  function place() {
    const show = loupeOpen() && !cropOpen();
    layer.hidden = !show;
    layer.dataset.mode = on && show ? "1" : "0";
    if (!show) return;
    const fr = loupeView.frame();
    const els = layer.querySelectorAll(".sbox");
    boxes.forEach((b, i) => {
      const el = els[i];
      if (!el) return;
      const r = frameBox(b.box, fr);
      el.style.left = r.x + "px"; el.style.top = r.y + "px"; el.style.width = r.w + "px"; el.style.height = r.h + "px";
      el.classList.toggle("active", b.bid === active);
      el.classList.toggle("low", r.y < 24);
      el.classList.toggle("passive", !on);
      el.dataset.box = b.box.map((v) => v.toFixed(5)).join(",");
      el.querySelector(".sbox-label").textContent = on ? scoreText(b, i) : `Subject ${i + 1}`;
    });
  }
  loupeView.onRender(place);
  $("big").addEventListener("load", place);
  window.addEventListener("resize", place);

  /* ---------- storage: crop mode's user boxes ---------- */
  async function load(id) {
    loadedFor = id;
    boxes = []; active = null; build();
    if (id === null || id === undefined || !S.folder) return;
    let d;
    try { d = await api(`/api/crop/${id}/boxes${q({ folder: S.folder })}`); } catch (e) { return; }
    if (curId() !== id) return;
    boxes = (d.boxes || []).map((b) => ({ bid: ++seq, box: b.slice(), score: null }));
    build();
    if (on) rescore();
  }
  async function save() {
    const id = curId();
    if (id === null || id === undefined) return;
    try {
      await api(`/api/crop/${id}/boxes`, { method: "POST", body: { folder: S.folder, boxes: boxes.map(boxJson) } });
    } catch (e) { toast(e.message, 5000); return; }
    rescore();
    window.dispatchEvent(new CustomEvent("subject-boxes-changed", { detail: { id } }));
  }
  async function rescore() {
    const id = curId(), tok = ++scoreTok;
    if (!boxes.length || id === null || id === undefined) { place(); return; }
    const key = (b) => id + ":" + b.box.map((v) => v.toFixed(4)).join(",");
    const need = boxes.filter((b) => !scoreCache.has(key(b)));
    if (need.length) {
      try {
        const r = await api(`/api/subject/${id}/score`, { method: "POST", body: { folder: S.folder, boxes: need.map(boxJson) } });
        need.forEach((b, i) => scoreCache.set(key(b), r.scores[i]));
      } catch (e) { /* the box works without a score */ }
    }
    if (tok !== scoreTok || curId() !== id) return;
    boxes.forEach((b) => { b.score = scoreCache.get(key(b)) || null; });
    place();
  }

  /* ---------- mouse ---------- */
  function norm(e) {   // pointer -> image-normalised (0..1, unclamped), through zoom.js's frame
    const r = layer.getBoundingClientRect(), fr = loupeView.frame();
    return { x: (e.clientX - r.left - fr.x) / fr.w, y: (e.clientY - r.top - fr.y) / fr.h };
  }
  layer.addEventListener("pointerdown", (e) => {
    if (!on || e.button !== 0) return;
    const t = e.target;
    if (t.closest(".sbox-x")) { e.stopPropagation(); return; }   // its own click handler
    e.stopPropagation(); e.preventDefault();                       // never a pan or a click-through
    const p = norm(e);
    const hb = t.closest(".sbh"), bx = t.closest(".sbox");
    let mode = "draw", ref = null;
    if (hb && bx) { mode = "res"; ref = { bid: Number(bx.dataset.bid), h: hb.dataset.sbh }; }
    else if (bx) { mode = "move"; ref = { bid: Number(bx.dataset.bid) }; }
    const b = ref && boxes.find((x) => x.bid === ref.bid);
    if (b) { active = b.bid; place(); }
    drag = { mode, ref, start: p, box0: b ? b.box.slice() : null, moved: false, px: e.clientX, py: e.clientY };
    try { layer.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  });
  layer.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const p = norm(e);
    if (!drag.moved && Math.hypot(e.clientX - drag.px, e.clientY - drag.py) < 4) return;
    drag.moved = true;
    if (drag.mode === "draw") {
      const fr = loupeView.frame();
      const a = { x: clamp01(drag.start.x), y: clamp01(drag.start.y) }, c = { x: clamp01(p.x), y: clamp01(p.y) };
      drag.draft = [Math.min(a.x, c.x), Math.min(a.y, c.y), Math.max(a.x, c.x), Math.max(a.y, c.y)];
      const r = frameBox(drag.draft, fr);
      Object.assign(draft.style, { left: r.x + "px", top: r.y + "px", width: r.w + "px", height: r.h + "px" });
      draft.hidden = false;
      return;
    }
    const b = boxes.find((x) => x.bid === drag.ref.bid);
    if (!b) return;
    const b0 = drag.box0;
    if (drag.mode === "move") {
      const w = b0[2] - b0[0], h = b0[3] - b0[1];
      const x0 = Math.min(1 - w, Math.max(0, b0[0] + p.x - drag.start.x)), y0 = Math.min(1 - h, Math.max(0, b0[1] + p.y - drag.start.y));
      b.box = [x0, y0, x0 + w, y0 + h];
    } else {
      const h = drag.ref.h;
      let [x0, y0, x1, y1] = b0;
      if (h.includes("w")) x0 = Math.min(clamp01(p.x), x1 - MIN_BOX);
      if (h.includes("e")) x1 = Math.max(clamp01(p.x), x0 + MIN_BOX);
      if (h.includes("n")) y0 = Math.min(clamp01(p.y), y1 - MIN_BOX);
      if (h.includes("s")) y1 = Math.max(clamp01(p.y), y0 + MIN_BOX);
      b.box = [x0, y0, x1, y1];
    }
    b.score = null;
    place();
  });
  layer.addEventListener("pointerup", (e) => {
    const d = drag;
    drag = null;
    try { layer.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    draft.hidden = true;
    if (!d) return;
    if (d.mode === "draw") {
      const b = d.draft;
      if (d.moved && b && b[2] - b[0] >= MIN_BOX && b[3] - b[1] >= MIN_BOX) {
        if (boxes.length >= MAX_BOXES) { toast(`Up to ${MAX_BOXES} subject boxes: remove one first.`); return; }
        const nb = { bid: ++seq, box: b, score: null };
        boxes.push(nb); active = nb.bid;
        build(); save();
      } else { active = null; place(); }
      return;
    }
    if (d.moved) save();
  });
  layer.addEventListener("pointercancel", () => { drag = null; draft.hidden = true; });
  layer.addEventListener("dblclick", (e) => { if (on) { e.stopPropagation(); e.preventDefault(); } });
  layer.addEventListener("click", (e) => {
    const x = e.target.closest && e.target.closest(".sbox-x");
    if (!x) return;
    e.stopPropagation();
    remove(Number(x.closest(".sbox").dataset.bid));
  });
  function remove(bid) {
    const i = boxes.findIndex((b) => b.bid === bid);
    if (i < 0) return;
    boxes.splice(i, 1);
    active = boxes.length ? boxes[boxes.length - 1].bid : null;
    build(); save();
  }

  /* ---------- mode ---------- */
  function setOn(v) {
    v = !!v;
    if (v && (cropOpen() || (!loupeOpen() && S.sel === null))) return false;
    if (v && !loupeOpen()) { toggleLoupe(); }                     // S in the main view opens the loupe
    on = v;
    if (!on) { drag = null; draft.hidden = true; }
    $("lz-subject").setAttribute("aria-pressed", String(on));
    $("mi-subject").setAttribute("aria-checked", String(on));
    layer.classList.toggle("subject-on", on);
    if (on) { const id = curId(); if (loadedFor !== id) load(id); else rescore(); }
    place();
    return true;
  }
  $("lz-subject").addEventListener("click", () => setOn(!on));
  $("mi-subject").addEventListener("click", () => setOn(!on));
  new MutationObserver(() => { if (!loupeOpen() || cropOpen()) { if (on) setOn(false); } place(); })
    .observe($("overlay-loupe"), { attributes: true, attributeFilter: ["hidden"] });
  new MutationObserver(() => { if (cropOpen()) { if (on) setOn(false); } else if (loupeOpen()) load(curId()); place(); })
    .observe($("crop-ui"), { attributes: true, attributeFilter: ["hidden"] });
  // a new photo in the loupe: its own boxes
  let lastId = null;
  loupeView.onRender(() => {
    const id = curId();
    if (id !== lastId) { lastId = id; if (loupeOpen()) load(id).then(() => { if (on) rescore(); }); }
  });

  /* ---------- keyboard (capture: runs before the global handlers) ---------- */
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const typing = e.target.closest && e.target.closest("input, textarea, select");
    if (typing) return;
    const k = e.key;
    if (on && k === "Escape") { e.preventDefault(); e.stopPropagation(); setOn(false); return; }
    if (k.toLowerCase() === "s" && !e.shiftKey) {
      if (cropOpen() || !$("overlay-compare").hidden || !$("overlay-help").hidden) return;
      e.preventDefault(); e.stopPropagation();
      setOn(!on);
      return;
    }
    if (on && (k === "Delete" || k === "Backspace" || k.toLowerCase() === "x")) {
      e.preventDefault(); e.stopPropagation();   // X must not reject the photo while a box is selected
      if (active !== null) remove(active);
    }
  }, true);

  window.ShotSubject = { isOn: () => on, setOn, boxes: () => boxes.map((b) => b.box.slice()), reload: () => load(curId()) };
})();
