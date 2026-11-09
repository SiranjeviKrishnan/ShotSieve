/* ShotSieve compare view (T23b): current frame vs the best frame of its group, side by side.
   Synced zoom/pan, sharpness / eyes-open / face numbers under each image, a filmstrip of the whole
   near-duplicate group, and "Keep this one instead" (written as ordinary user overrides through
   /api/override, so it counts like any other override).
   Keys while open: 1 / Left = left frame, 2 / Right = right frame, Enter or K = keep the chosen one,
   Up / Down = next / previous frame of the group on the right, + - 0 Z = zoom, L = link / unlink the two sides, C or Esc = close. */
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const M = { a: null, b: null, choice: "a", busy: false, set: null };   // set: the selected photos when comparing a multi-selection

  const isOpen = () => !$("overlay-compare").hidden;
  const imgOf = (id) => S.byId.get(id);
  function groupOf(id) {
    const im = imgOf(id);
    return im ? S.groups.find((g) => g.id === im.group_id) || null : null;
  }

  /* ---------- opening ---------- */

  // a = the group's best, b = the current frame (or the first other member when the current one is the best)
  const members = (g) => (M.set ? M.set : g ? g.members : []);
  function defaultPair() {
    const ids = typeof selIds === "function" ? selIds() : [];
    if (ids.length > 1) return { a: ids[0], b: ids[1], set: ids };   // a multi-selection compares the selected photos
    const im = imgOf(S.sel);
    if (!im) return null;
    const g = groupOf(im.id);
    if (!g || g.members.length < 2) return null;
    if (g.best !== im.id) return { a: g.best, b: im.id };
    return { a: im.id, b: g.members.find((id) => id !== g.best) };
  }
  function open(pair) {
    const p = pair || defaultPair();
    if (!p) { toast("This image is not part of a group."); return false; }
    M.a = p.a; M.b = p.b; M.set = p.set || null;
    const A = imgOf(M.a), B = imgOf(M.b);
    M.choice = A.decision !== "keep" && B.decision === "keep" ? "a" : B.decision !== "keep" && A.decision === "keep" ? "b" : "a";
    GZ.each((v) => v.reset());
    document.querySelectorAll(".overlay").forEach((o) => { if (o.id !== "overlay-compare") o.hidden = true; });
    $("overlay-compare").hidden = false;
    render(true);
    return true;
  }
  function close() { $("overlay-compare").hidden = true; }
  function toggle() { if (isOpen()) close(); else open(); }
  function refresh() { if (isOpen()) render(false); }

  /* ---------- rendering ---------- */

  const fmt = {
    sharp: (v) => (v === null || v === undefined ? "n/a" : v.toFixed(2)),
    eyes: (v, m) => (v === null || v === undefined ? (m.n_faces ? "n/a" : "no faces") : Math.round(v * 100) + " %"),
    face: (v, m) => (v === null || v === undefined ? (m.n_faces ? "n/a" : "no faces") : v.toFixed(2)),
  };
  function metricsHtml(im, other) {
    const m = im.metrics || {}, o = other.metrics || {};
    const better = (k) => (m[k] !== null && m[k] !== undefined && o[k] !== null && o[k] !== undefined && m[k] > o[k] + 1e-9 ? " better" : "");
    const cell = (label, k, f) => `<div class="m${better(k)}"><span>${label}</span><b>${esc(f(m[k], m))}</b></div>`;
    const sc = Math.round(Math.max(0, Math.min(1, im.score)) * 100);
    return cell("Sharpness", "sharp", fmt.sharp) + cell("Eyes open", "eyes_open", fmt.eyes) + cell("Face sharpness", "face", fmt.face) +
      `<div class="m"><span>Score</span><b>${sc} %</b></div>`;
  }
  function side(which, im, other, caption) {
    $(`cmp-${which}-cap`).textContent = caption;
    VIEW[which].setImage(im.id);
    $(`cmp-${which}-m`).innerHTML = metricsHtml(im, other);
    const fig = $(`cmp-fig-${which}`);
    fig.classList.toggle("chosen", M.choice === which);
    fig.setAttribute("aria-current", String(M.choice === which));
    const btn = $(`cmp-keep-${which}`);
    const kept = im.decision === "keep";
    btn.textContent = kept ? "Keep this one" : "Keep this one instead";
    btn.classList.toggle("primary", M.choice === which);
    btn.disabled = M.busy;
    fig.dataset.id = im.id;
    fig.dataset.decision = im.decision;
  }
  function render(resetStrip) {
    const A = imgOf(M.a), B = imgOf(M.b);
    if (!A || !B) { close(); return; }
    const g = groupOf(A.id);
    side("a", A, B, `${A.id === (g && g.best) ? "★ best" : "left"}: ${A.name} (${A.decision}${A.overridden ? ", yours" : ""})`);
    side("b", B, A, `${B.id === S.sel ? "current" : "other"}: ${B.name} (${B.decision}${B.overridden ? ", yours" : ""})`);
    $("cmp-sub").textContent = M.set ? `${M.set.length} selected photos` : g ? `${g.members.length} near-identical frames` : "";
    renderStrip(g);
  }
  function renderStrip(g) {
    const box = $("cmp-strip");
    box.innerHTML = "";
    if (!g && !M.set) return;
    for (const id of members(g)) {
      const im = imgOf(id);
      if (!im) continue;
      const b = document.createElement("button");
      b.type = "button";
      b.className = `cmp-thumb dec-${im.decision}${id === M.a ? " is-a" : ""}${id === M.b ? " is-b" : ""}`;
      b.dataset.id = id;
      b.title = `${im.name} (${im.decision}). Click to compare it with the left frame; Shift-click puts it on the left.`;
      b.innerHTML = `<img alt="" src="${thumbUrl(id, 200)}"><span>${esc(im.name)}</span>` +
        (g && id === g.best ? '<i class="star">★</i>' : "") + (id === M.a ? "<em>left</em>" : id === M.b ? "<em>right</em>" : "");
      b.onclick = (e) => {
        if (e.shiftKey) { if (id !== M.b) M.a = id; }
        else if (id !== M.a) M.b = id;
        render();
      };
      box.appendChild(b);
    }
  }

  /* ---------- the decision ---------- */

  async function override(im, decision) {
    const r = await api("/api/override", { method: "POST", body: { image_id: im.id, folder: S.folder, decision } });
    im.decision = r.decision; im.overridden = r.overridden;
  }
  async function keep(which) {
    if (M.busy) return;
    const win = imgOf(which === "a" ? M.a : M.b), lose = imgOf(which === "a" ? M.b : M.a);
    if (!win || !lose) return;
    M.busy = true;
    render();
    try {
      await override(win, "keep");
      await override(lose, "reject");
      toast(`Kept ${win.name}, rejected ${lose.name}.`);
    } catch (e) { toast("Could not save: " + e.message, 6000); }
    M.busy = false;
    relayout();
    if (S.sel !== null && S.rowOf.has(S.sel)) select(S.sel, false);
    if (isOpen()) render();
  }

  /* ---------- zoom / pan (shared module zoom.js; linked by default) ---------- */

  const GZ = ShotZoom.group({ linked: true });
  const VIEW = {
    a: ShotZoom.view(GZ, { view: $("cmp-fig-a").querySelector(".cmp-view"), img: $("cmp-a"), urls: zoomUrls }),
    b: ShotZoom.view(GZ, { view: $("cmp-fig-b").querySelector(".cmp-view"), img: $("cmp-b"), urls: zoomUrls }),
  };
  ShotZoom.toolbar($("cmp-zoombar"), GZ);
  if (window.ShotDenoise) window.ShotDenoise.attach(GZ, { host: $("cmp-denoise"), enabled: isOpen });
  function syncLink() {
    $("cmp-link").setAttribute("aria-pressed", String(GZ.linked));
    $("cmp-link").classList.toggle("on", GZ.linked);
    $("overlay-compare").dataset.linked = GZ.linked ? "1" : "0";
  }
  function toggleLink() { GZ.setLinked(!GZ.linked); syncLink(); toast(GZ.linked ? "Zoom linked" : "Zoom unlinked: each side on its own", 1800); }
  syncLink();

  /* ---------- wiring ---------- */

  $("cmp-keep-a").onclick = () => { M.choice = "a"; keep("a"); };
  $("cmp-keep-b").onclick = () => { M.choice = "b"; keep("b"); };
  $("cmp-close").onclick = close;
  $("cmp-link").onclick = toggleLink;
  for (const w of ["a", "b"]) $(`cmp-fig-${w}`).addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    M.choice = w; render();
  });
  function cycle(delta) {
    const ids = members(groupOf(M.a)).filter((id) => id !== M.a);
    if (!ids.length) return;
    const i = ids.indexOf(M.b);
    M.b = ids[(i + delta + ids.length) % ids.length];
    render();
  }
  document.addEventListener("keydown", (e) => {
    if (!isOpen() || e.altKey) return;
    const t = e.target;
    if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type !== "checkbox"))) return;
    if (ShotZoom.handleKey(e, GZ)) { e.stopPropagation(); e.preventDefault(); return; }
    if (e.ctrlKey || e.metaKey) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (key === "Escape" || key === "c" || key === "?") return;   // the global handlers close / show help
    e.stopPropagation();
    e.preventDefault();
    switch (key) {
      case "ArrowLeft": case "1": M.choice = "a"; render(); break;
      case "ArrowRight": case "2": M.choice = "b"; render(); break;
      case "ArrowDown": cycle(1); break;
      case "ArrowUp": cycle(-1); break;
      case "Enter": case "k": keep(M.choice); break;
      case "l": toggleLink(); break;
      default: break;
    }
  }, true);

  $("btn-cmp-detail").onclick = () => open();   // the detail panel button: same as the C key
  window.ShotCompare = { open, close, toggle, refresh, state: M };
})();
