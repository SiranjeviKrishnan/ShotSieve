/* ShotSieve "Open in Lightroom" (T23d): loupe bar, detail panel, grid context menu, Cmd+L.
   Adobe Lightroom (cloud) reads XMP sidecars only at import, so the server first writes
   ShotSieve's state into the sidecar (after a one-time question) and then `open -b <app> <file>`.
   The button is hidden when neither Lightroom is installed. */
"use strict";

(function () {
  let info = null, forFolder = null;

  async function sync(force) {
    if (!S.folder) return;
    if (!force && info && forFolder === S.folder) { show(); return; }
    forFolder = S.folder;
    try { info = await api("/api/lightroom/apps" + q({ folder: S.folder })); } catch (e) { info = null; }
    show();
  }
  function show() {
    const any = !!(info && info.apps.length);
    document.querySelectorAll(".lr-open").forEach((b) => { b.hidden = !any; });
    $("mi-lr").hidden = !(info && info.classic_sync);
    if (typeof syncMoreMenu === "function") syncMoreMenu();
  }
  const nameOfApp = (id) => (info && info.apps.find((a) => a.id === id) || { name: id }).name;

  async function call(ids, body = {}) {
    return api("/api/lightroom/open", { method: "POST", body: { folder: S.folder, ids, ...body } });
  }
  function summary(r) {
    const bits = [`Opened ${r.n} file${r.n === 1 ? "" : "s"} in ${r.name}`];
    if (r.xmp) {
      const c = r.xmp.crops || {};
      bits.push(`ratings written for ${r.xmp.decisions}` + (c.written ? `, crops for ${c.written}` : ""));
      const skipped = Object.values(c.skipped || {}).reduce((a, b) => a + b, 0);
      if (skipped) bits.push(`${skipped} crop(s) not written (the photographer's own edits win)`);
    }
    if (r.missing && r.missing.length) bits.push(`${r.missing.length} file(s) missing`);
    return bits.join(". ") + ".";
  }

  async function open(ids, app) {
    if (!ids || !ids.length) return;
    if (ids.length > 500) { toast(`Lightroom opens at most 500 photos at once: opening the first 500 of ${ids.length}.`, 6000); ids = ids.slice(0, 500); }
    if (!info) await sync(true);
    if (!info || !info.apps.length) { toast("Adobe Lightroom is not installed."); return; }
    try {
      const r = await call(ids, app ? { app } : {});
      info.preferred = r.app;
      toast(summary(r), r.xmp && r.xmp.warnings && r.xmp.warnings.length ? 9000 : 5000);
    } catch (e) { toast(e.message, 7000); }
  }

  // a click: the preferred app, or (when both are installed) a small chooser
  function trigger(ids, anchor) {
    if (!info || !info.apps.length) return;
    if (info.apps.length === 1 || !anchor) return open(ids, info.preferred || info.apps[0].id);
    const m = $("ctx-menu");
    const r = anchor.getBoundingClientRect();
    const order = [...info.apps].sort((a, b) => (b.id === info.preferred) - (a.id === info.preferred));
    m.innerHTML = order.map((a) => `<button role="menuitem" data-app="${esc(a.id)}">${esc(a.name)}${a.id === info.preferred ? ' <span class="muted">last used</span>' : ""}</button>`).join("");
    m.style.left = Math.min(r.left, innerWidth - 230) + "px";
    m.style.top = Math.min(r.bottom + 4, innerHeight - 100) + "px";
    m.hidden = false;
    m.onclick = (e) => { const b = e.target.closest("button"); if (!b) return; m.hidden = true; open(ids, b.dataset.app); };
  }
  const cur = () => selIds();   // T23i: the whole selection
  $("lz-lr").addEventListener("click", (e) => { e.stopPropagation(); trigger(cur(), $("lz-lr")); });
  $("btn-lrop").addEventListener("click", (e) => { e.stopPropagation(); trigger(cur(), $("btn-lrop")); });

  document.addEventListener("keydown", (e) => {      // capture: Cmd+L
    if (!(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey || e.key.toLowerCase() !== "l") return;
    e.preventDefault(); e.stopPropagation();
    trigger(cur(), null);
  }, true);

  // the grid context menu (Open in <app>, Use AI decision, labels, stars, Move to Trash) lives in select.js (T23i)

  window.ShotLightroom = { sync, open, trigger, info: () => info };
  setInterval(() => { if (S.folder && forFolder !== S.folder) sync(true); }, 2000);
  sync(true);
})();
