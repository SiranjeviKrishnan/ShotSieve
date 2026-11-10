/* ShotSieve denoise toggle (T23f): D in the loupe and in compare, view only (originals are never touched).

   ShotDenoise.attach(group, {host, enabled})
     - builds the toolbar control in `host`: [Denoise <badge>] [v strength menu]. Badge: "Off", a spinner
       while working, "AI" (NAFNet) or "Basic" (OpenCV fallback when the model is not installed).
     - D toggles every view of the group (compare: both sides), Shift+D only the side under the
       pointer. The state follows across frames.
     - zoomed in, the denoised pixels of the VISIBLE region come first (/api/denoise/{id}/region, the full
       resolution image, a small centre window first, then the whole viewport) and are laid over the
       picture; the whole denoised full-resolution image loads in the background and replaces them.
   The shared zoom views (zoom.js) own the picture swapping; this file owns the control and the overlay. */
"use strict";

(function () {
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const LABEL = { low: "Low", medium: "Medium", high: "High" };
  const CENTRE_PX = 1.5e6;     // the first region: a window of this many pixels around the view centre
  const REGION_DELAY = 220;    // ms of calm before a region is requested
  let status = null;           // /api/denoise/status

  async function loadStatus(force) {
    if (status && !force) return status;
    try {
      const r = await fetch("/api/denoise/status");
      status = r.ok ? await r.json() : { backend: "basic", model: { installed: false, bytes: 0 } };
    } catch (e) { status = { backend: "basic", model: { installed: false, bytes: 0 } }; }
    return status;
  }

  /* The first press of D waits seconds for the model to compile: opening the loupe / compare asks
     the server to warm it in the background (it ignores that while a cull runs). */
  let warmedAt = 0;
  function warm() {
    if (Date.now() - warmedAt < 20000) return;
    warmedAt = Date.now();
    loadStatus().then((st) => {
      if (st && st.model && st.model.installed) fetch("/api/denoise/warm", { method: "POST" }).catch(() => {});
    });
  }

  /* ---------- the region overlay of one view ---------- */

  function regionLayer(v, urls) {
    const ov = document.createElement("img");
    ov.className = "dn-region"; ov.alt = ""; ov.hidden = true; ov.draggable = false;
    v.el.appendChild(ov);
    let box = null, objUrl = null, timer = null, seq = 0, ready = false, stale = false;   // `ready`: the image of `box` has loaded

    const active = () => v.dn.on && v.id !== null && v.st.z > 1.0001 && (v.fullFor !== v.id || v.fullStale);
    const hide = () => { ov.hidden = true; ov.classList.remove("shown"); };
    function place() {
      if (!box || !active() || !ready) { hide(); return; }
      const f = v.frame();
      ov.style.left = f.x + box[0] * f.w + "px";
      ov.style.top = f.y + box[1] * f.h + "px";
      ov.style.width = (box[2] - box[0]) * f.w + "px";
      ov.style.height = (box[3] - box[1]) * f.h + "px";
      if (ov.hidden) {   // appears with a short fade: never a hard patchwork edge
        ov.hidden = false;
        void ov.offsetWidth;
      }
      ov.classList.add("shown");
    }
    function visible() {   // the visible part of the image, as fractions 0..1
      const G = v.geo();
      const hw = G.vw / (2 * v.st.z * G.rw), hh = G.vh / (2 * v.st.z * G.rh);
      return [clamp(v.st.ux - hw, 0, 1), clamp(v.st.uy - hh, 0, 1), clamp(v.st.ux + hw, 0, 1), clamp(v.st.uy + hh, 0, 1)];
    }
    const covers = (a, b) => a && b && a[0] <= b[0] + 1e-3 && a[1] <= b[1] + 1e-3 && a[2] >= b[2] - 1e-3 && a[3] >= b[3] - 1e-3;
    function centreWindow(w) {   // at most CENTRE_PX pixels around the middle of `w`
      if (!v.dims) return w;
      const px = (w[2] - w[0]) * v.dims.w * (w[3] - w[1]) * v.dims.h;
      if (px <= CENTRE_PX) return w;
      const k = Math.sqrt(CENTRE_PX / px), cx = (w[0] + w[2]) / 2, cy = (w[1] + w[3]) / 2;
      const hw = (w[2] - w[0]) * k / 2, hh = (w[3] - w[1]) * k / 2;
      return [cx - hw, cy - hh, cx + hw, cy + hh];
    }
    async function fetchBox(b, mine) {
      const id = v.id, gen = v.dn.gen;
      v.dnBusy(1);
      try {
        const r = await fetch(urls.denoiseRegion(id, ZL(), b));
        if (!r.ok) return false;
        const hdr = r.headers.get("X-Region"), sz = r.headers.get("X-Image-Size");
        const blob = await r.blob();
        if (seq !== mine || v.id !== id || v.dn.gen !== gen || !v.dn.on) return false;
        let nb = b;
        if (hdr && sz) {
          const [x0, y0, x1, y1] = hdr.split(",").map(Number), [iw, ih] = sz.split(",").map(Number);
          nb = [x0 / iw, y0 / ih, x1 / iw, y1 / ih];
        }
        const be = r.headers.get("X-Denoise-Backend");
        if (be) v.dn.backend = be;
        const u = URL.createObjectURL(blob);
        await new Promise((res) => {
          ov.onload = ov.onerror = res;
          ov.src = u;
        });
        if (seq !== mine || v.id !== id || v.dn.gen !== gen || !v.dn.on) { URL.revokeObjectURL(u); return false; }
        if (objUrl && objUrl !== u) URL.revokeObjectURL(objUrl);
        objUrl = u; box = nb; ready = true; stale = false;
        place();
        return true;
      } catch (e) { return false; } finally { if (v.dn.gen === gen) v.dnBusy(-1); }
    }
    async function update() {
      if (!active()) { hide(); return; }
      if (!v.dims) { await v.ensureDims(); if (!active()) return; }
      const w = visible();
      if (!stale && covers(box, w)) { place(); return; }
      const mine = ++seq;
      const c = centreWindow(w);
      if ((stale || !covers(box, c)) && c !== w) { if (!(await fetchBox(c, mine)) || seq !== mine) return; }
      if (seq === mine) await fetchBox(w, mine);
    }
    const ZL = () => ShotZoom.denoiseLevel();
    v.onRender(() => {
      place();
      clearTimeout(timer);
      if (active()) timer = setTimeout(update, REGION_DELAY); else hide();
    });
    v.onDenoise(() => {
      if (!v.dn.on) { box = null; ready = false; hide(); seq++; }
      else if (!active()) hide();
    });
    return {
      clear() { box = null; ready = false; stale = false; seq++; hide(); },
      restale() { stale = true; seq++; },   // the strength changed: the old overlay stays until the new one is loaded
      refresh() { if (active()) update(); },
    };
  }

  /* ---------- the toolbar control ---------- */

  function attach(group, opts) {
    const host = opts.host;
    if (!host) return null;
    const enabled = opts.enabled || (() => true);
    host.classList.add("dnbar");
    host.innerHTML =
      '<button type="button" class="dn-btn" aria-pressed="false" title="Denoise for viewing (D). Shift+D: only the side under the pointer. Your files are never changed.">' +
      '<span class="dn-text">Denoise</span><i class="dn-badge">Off</i><i class="dn-spin" hidden></i><kbd>D</kbd></button>' +
      '<span class="dn-wrap"><button type="button" class="dn-more" aria-haspopup="menu" aria-expanded="false" title="Denoise strength" aria-label="Denoise strength">▾</button>' +
      '<div class="dn-menu" role="menu" hidden></div></span>';
    const q = (s) => host.querySelector(s);
    const btn = q(".dn-btn"), badge = q(".dn-badge"), spin = q(".dn-spin"), more = q(".dn-more"), menu = q(".dn-menu");
    const layers = new Map();
    for (const v of group.views) layers.set(v, regionLayer(v, v.urls));

    const views = () => group.views;
    const anyOn = () => views().some((v) => v.dn.on);

    function refresh() {
      const on = anyOn();
      const busy = views().some((v) => v.dn.on && v.dn.state === "loading");
      const be = (views().find((v) => v.dn.on) || { dn: { backend: status ? status.backend : "ai" } }).dn.backend;
      btn.setAttribute("aria-pressed", String(on));
      btn.classList.toggle("on", on);
      spin.hidden = !busy;
      const per = views().length > 1 && on && !views().every((v) => v.dn.on) ? " (one side)" : "";
      badge.textContent = !on ? "Off" : busy ? "…" : (be === "basic" ? "Basic" : "AI") + per;
      host.dataset.state = !on ? "off" : busy ? "loading" : be;
      host.dataset.on = on ? "1" : "0";
      host.dataset.sides = views().map((v) => (v.dn.on ? "1" : "0")).join("");
      host.dataset.level = ShotZoom.denoiseLevel();
    }
    for (const v of views()) v.onDenoise(refresh);

    async function toggle(single) {
      if (!enabled()) return;
      const st = await loadStatus();
      const act = group.current();
      if (!act) return;
      const targets = single ? [act] : views();
      const on = !act.dn.on;
      for (const v of views()) v.dn.backend = st.backend;
      for (const v of targets) {
        if (v.dn.on !== on) v.setDenoise(on);
      }
      refresh();
    }
    group.denoiseKey = (shift) => { toggle(!!shift); };
    btn.addEventListener("click", (e) => { e.preventDefault(); toggle(false); });

    /* strength menu + model state */
    const closeMenu = () => { menu.hidden = true; more.setAttribute("aria-expanded", "false"); };
    async function buildMenu() {
      const st = await loadStatus(true);
      menu.innerHTML = "";
      for (const l of ShotZoom.DN_LEVELS) {
        const b = document.createElement("button");
        b.type = "button"; b.role = "menuitemradio"; b.dataset.level = l; b.textContent = LABEL[l];
        b.setAttribute("aria-checked", String(ShotZoom.denoiseLevel() === l));
        b.addEventListener("click", () => { ShotZoom.setDenoiseLevel(l); closeMenu(); });
        menu.appendChild(b);
      }
      const info = document.createElement("div");
      info.className = "dn-info";
      if (st.model.installed) info.textContent = "AI denoiser (NAFNet, MIT)";
      else {
        info.textContent = "Basic denoiser (OpenCV). ";
        const get = document.createElement("button");
        get.type = "button"; get.className = "dn-get";
        get.textContent = `Get the AI denoiser (${Math.round(st.model.bytes / 1e6)} MB)`;
        get.addEventListener("click", () => download(get));
        info.appendChild(get);
      }
      menu.appendChild(info);
    }
    async function download(button) {
      button.disabled = true;
      try {
        await fetch("/api/models/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ groups: ["denoise"] }) });
        for (let i = 0; i < 600; i++) {
          await new Promise((r) => setTimeout(r, 1000));
          const m = await (await fetch("/api/models")).json();
          const d = m.download;
          if (d && d.total) button.textContent = `Downloading… ${Math.round(100 * d.done / d.total)} %`;
          if (!d || d.state === "done" || d.state === "error" || (m.groups.denoise && m.groups.denoise.installed)) break;
        }
      } catch (e) { /* offline: stays Basic */ }
      await loadStatus(true);
      await buildMenu();
      views().forEach((v) => v.reloadDenoise(false));
    }
    more.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!menu.hidden) { closeMenu(); return; }
      await buildMenu();
      menu.hidden = false;
      more.setAttribute("aria-expanded", "true");
    });
    document.addEventListener("click", (e) => { if (!host.contains(e.target)) closeMenu(); });
    host.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { closeMenu(); e.stopPropagation(); e.preventDefault(); } }, true);

    ShotZoom.onDenoiseLevel(() => {
      for (const v of views()) { const l = layers.get(v); if (l) l.restale(); v.reloadDenoise(true); if (l) l.refresh(); }
      refresh();
    });
    const overlay = host.closest("[id^=overlay-]");
    if (overlay && window.MutationObserver) {
      new MutationObserver(() => { if (!overlay.hidden) warm(); }).observe(overlay, { attributes: true, attributeFilter: ["hidden"] });
    }
    loadStatus().then(refresh);
    refresh();
    return { toggle, refresh };
  }

  window.ShotDenoise = { attach, warm };
})();
