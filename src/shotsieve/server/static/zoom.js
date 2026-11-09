/* ShotSieve shared zoom / pan (T23c). Used by the loupe (app.js) and the compare view (compare.js)
   so both behave identically.

   Model: every view is a box holding one <img> (object-fit: contain). The zoom state is
   {z, ux, uy}: z = scale relative to "Fit" (>= 1), (ux, uy) = the point of the IMAGE (0..1) that
   sits at the centre of the view. Keeping it image-relative lets the zoom level and the relative
   position survive a frame change. "100 %" = one image pixel per device pixel (devicePixelRatio).
   The preview is shown at once; when z > 1 the full-resolution image (/api/full/{id}) is loaded in
   the background and swapped in (progressive).

   ShotZoom.group({linked})            -> a set of views that can share their zoom
   ShotZoom.view(group, opts)          -> one zoomable view (see below)
   ShotZoom.toolbar(host, group)       -> "−  [Fit ▾]  +" control for a group
   ShotZoom.handleKey(e, group, ctx)   -> true when the key was a zoom key (z + - 0, Cmd =/-/0, l) */
"use strict";

(function () {
  const STEP = 1.5;
  const MAX_Z = 16;
  const MAX_PCT = 800;
  const PRESETS = [["fit", "Fit"], [50, "50 %"], [100, "100 %"], [200, "200 %"], [400, "400 %"]];
  const FULL_DELAY = 180;     // ms of calm before the full-resolution image is requested
  const NAV_W = 150, NAV_H = 110;

  const dpr = () => window.devicePixelRatio || 1;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  /* ---------- per-photo zoom memory (in memory, per view, LRU) ---------- */
  const MEMO_MAX = 300;
  try { localStorage.removeItem("shotsieve.keepZoom"); localStorage.removeItem("shotsieve.kf.size"); } catch (e) { /* ignore */ }

  /* ---------- the denoise strength preference (T23f; the toggle itself lives in denoise.js) ---------- */
  const DN_LEVEL_KEY = "shotsieve.denoiseLevel";
  const DN_LEVELS = ["low", "medium", "high"];
  let dnLevel = "medium";
  try { const sv = localStorage.getItem(DN_LEVEL_KEY); if (DN_LEVELS.includes(sv)) dnLevel = sv; } catch (e) { /* ignore */ }
  const dnLevelListeners = [];
  const denoiseLevel = () => dnLevel;
  function setDenoiseLevel(l) {
    if (!DN_LEVELS.includes(l) || l === dnLevel) return;
    dnLevel = l;
    try { localStorage.setItem(DN_LEVEL_KEY, l); } catch (e) { /* ignore */ }
    dnLevelListeners.forEach((f) => f(l));
  }

  /* ---------- groups ---------- */

  function group(opts) {
    const g = { views: [], linked: !opts || opts.linked !== false, active: null, bars: [] };
    g.each = (fn) => g.views.forEach(fn);
    // apply a new state to `v` (and to every view when linked)
    g.set = (v, st) => {
      const targets = g.linked ? g.views : [v];
      for (const t of targets) { t.st = { z: st.z, ux: st.ux, uy: st.uy }; t.render(); }
      if (g.linked) g.views.forEach((t) => t.wantFull());
      else v.wantFull();
    };
    g.setLinked = (on) => {
      g.linked = !!on;
      if (g.linked && g.active) g.set(g.active, g.active.st);
      g.bars.forEach((f) => f());
    };
    g.refreshBars = () => g.bars.forEach((f) => f());
    g.current = () => g.active || g.views[0];
    return g;
  }

  /* ---------- one view ----------
     opts: {view, img, nav (bool), urls: {preview(id), full(id), size(id)}, enabled(): bool,
            onChange(v)} */

  function view(g, opts) {
    const el = opts.view, img = opts.img;
    const v = {
      group: g, el, img, id: null, st: { z: 1, ux: 0.5, uy: 0.5 }, dims: null, fullFor: null,
      loading: false, pt: null, timer: null, loader: null, gen: 0,
      // denoise (T23f): on = show denoised pixels; state = off | loading | ai | basic
      dn: { on: false, state: "off", busy: 0, gen: 0, backend: "ai" }, dnLoader: null, dnListeners: [],
    };
    const dimCache = (view.dimCache = view.dimCache || new Map());
    const dk = (id) => opts.urls.size(id);
    v.urls = opts.urls;
    g.views.push(v);
    el.classList.add("zview");
    el.dataset.z = "1";

    let nav = null, navImg = null, navRect = null;
    if (opts.nav !== false) {
      nav = document.createElement("div");
      nav.className = "znav";
      nav.hidden = true;
      nav.title = "Drag to move the visible area";
      navImg = document.createElement("img");
      navImg.alt = ""; navImg.draggable = false;
      navRect = document.createElement("i");
      nav.append(navImg, navRect);
      el.appendChild(nav);
    }

    const enabled = () => (opts.enabled ? opts.enabled() : true);

    /* geometry of the contain-fitted image inside the view box (z = 1) */
    function geo() {
      const vw = el.clientWidth || 1, vh = el.clientHeight || 1;
      let iw = 0, ih = 0;
      if (v.dims) { iw = v.dims.w; ih = v.dims.h; }
      else if (img.naturalWidth) { iw = img.naturalWidth; ih = img.naturalHeight; }
      if (!iw || !ih) { iw = vw; ih = vh; }
      const f = Math.min(vw / iw, vh / ih);
      const rw = iw * f, rh = ih * f;
      return { vw, vh, iw, ih, rw, rh, ox: (vw - rw) / 2, oy: (vh - rh) / 2 };
    }
    // translation that puts image point (ux, uy) at the view centre, clamped to the image rect
    function place(st, G) {
      const axis = (u, o, r, vs) => {
        const span = st.z * r;
        if (span <= vs) return (vs - span) / 2 - st.z * o;   // smaller than the view: centred
        let t = vs / 2 - st.z * (o + u * r);
        const left = t + st.z * o;
        if (left > 0) t -= left;
        else if (left + span < vs) t += vs - (left + span);
        return t;
      };
      return { tx: axis(st.ux, G.ox, G.rw, G.vw), ty: axis(st.uy, G.oy, G.rh, G.vh) };
    }
    const toU = (t, vs, o, r, z) => ((vs / 2 - t) / z - o) / r;
    const maxZ = (G) => Math.max(MAX_Z, zForPct(MAX_PCT, G));
    function zForPct(pct, G) {   // css px per image px = pct/100/dpr; fit = rw/iw
      return (pct / 100 / dpr()) / (G.rw / G.iw);
    }
    const pctOf = (st, G) => (st.z * G.rw * dpr()) / G.iw * 100;

    v.geo = geo;
    // where the whole image sits inside the view box right now, in view-local CSS px (the same
    // transform that `render` applies to the <img>): overlays use this, not a transform of their own
    v.frame = function () {
      const G = geo();
      const p = place(v.st, G);
      return { x: p.tx + v.st.z * G.ox, y: p.ty + v.st.z * G.oy, w: v.st.z * G.rw, h: v.st.z * G.rh, z: v.st.z };
    };
    v.listeners = [];
    v.onRender = (fn) => { v.listeners.push(fn); };
    v.maxZ = () => maxZ(geo());

    v.render = function () {
      const G = geo();
      v.st.z = clamp(v.st.z, 1, maxZ(G));
      const p = place(v.st, G);
      v.st.ux = toU(p.tx, G.vw, G.ox, G.rw, v.st.z);   // canonical (clamped) centre
      v.st.uy = toU(p.ty, G.vh, G.oy, G.rh, v.st.z);
      img.style.transformOrigin = "0 0";
      img.style.transform = `translate(${p.tx}px, ${p.ty}px) scale(${v.st.z})`;
      const zoomed = v.st.z > 1.0001;
      el.classList.toggle("zoomed", zoomed);
      el.dataset.z = v.st.z.toFixed(3);
      el.dataset.pct = String(Math.round(pctOf(v.st, G)));
      el.dataset.full = v.fullFor === v.id && v.id !== null ? "1" : "0";
      if (nav) renderNav(G, zoomed);
      g.refreshBars();
      if (opts.onChange) opts.onChange(v);
      for (const fn of v.listeners) fn(v);
    };

    function renderNav(G, zoomed) {
      nav.hidden = !zoomed;
      if (!zoomed) return;
      const k = Math.min(NAV_W / G.iw, NAV_H / G.ih);
      const nw = Math.max(20, Math.round(G.iw * k)), nh = Math.max(20, Math.round(G.ih * k));
      nav.style.width = nw + "px"; nav.style.height = nh + "px";
      const src = v.id !== null ? opts.urls.preview(v.id, el) : "";
      if (v.id !== null && navImg.getAttribute("src") !== src) navImg.src = src;
      const hw = G.vw / (2 * v.st.z * G.rw), hh = G.vh / (2 * v.st.z * G.rh);   // half visible span, 0..1
      const x0 = clamp(v.st.ux - hw, 0, 1), x1 = clamp(v.st.ux + hw, 0, 1);
      const y0 = clamp(v.st.uy - hh, 0, 1), y1 = clamp(v.st.uy + hh, 0, 1);
      navRect.style.left = x0 * nw + "px"; navRect.style.top = y0 * nh + "px";
      navRect.style.width = Math.max(4, (x1 - x0) * nw) + "px"; navRect.style.height = Math.max(4, (y1 - y0) * nh) + "px";
    }

    /* ----- operations (all go through the group so linked views follow) ----- */

    v.zoomAround = function (px, py, nz) {   // keep the image point under (px, py) fixed
      const G = geo();
      nz = clamp(nz, 1, maxZ(G));
      const p = place(v.st, G);
      const pu = ((px - p.tx) / v.st.z - G.ox) / G.rw, pv = ((py - p.ty) / v.st.z - G.oy) / G.rh;
      const tx = px - nz * (G.ox + pu * G.rw), ty = py - nz * (G.oy + pv * G.rh);
      g.set(v, { z: nz, ux: toU(tx, G.vw, G.ox, G.rw, nz), uy: toU(ty, G.vh, G.oy, G.rh, nz) });
    };
    v.zoomBy = function (f, px, py) {
      const G = geo();
      v.zoomAround(px === undefined ? G.vw / 2 : px, py === undefined ? G.vh / 2 : py, v.st.z * f);
    };
    v.panBy = function (dx, dy) {
      if (v.st.z <= 1.0001) return;
      const G = geo();
      g.set(v, { z: v.st.z, ux: v.st.ux - dx / (v.st.z * G.rw), uy: v.st.uy - dy / (v.st.z * G.rh) });
    };
    v.panTo = function (ux, uy) { g.set(v, { z: v.st.z, ux: clamp(ux, 0, 1), uy: clamp(uy, 0, 1) }); };
    v.fit = function () { g.set(v, { z: 1, ux: 0.5, uy: 0.5 }); };
    // zoom to pct % of the actual pixels, around (px, py) (default: the view centre)
    v.setPct = async function (pct, px, py) {
      if (pct === "fit") return v.fit();
      await v.ensureDims();
      const G = geo();
      const nz = zForPct(pct, G);
      if (nz <= 1.0001) return v.fit();
      v.zoomAround(px === undefined ? G.vw / 2 : px, py === undefined ? G.vh / 2 : py, nz);
    };
    v.toggle100 = async function (px, py) {
      if (v.st.z > 1.0001) return v.fit();
      await v.ensureDims();
      const G = geo();
      const z100 = zForPct(100, G);
      await v.setPct(z100 > 1.05 ? 100 : 200, px, py);
    };
    v.pct = () => pctOf(v.st, geo());
    v.reset = () => { g.set(v, { z: 1, ux: 0.5, uy: 0.5 }); };

    /* ----- image + full resolution ----- */

    v.ensureDims = async function () {
      const id = v.id;
      if (id === null) return;
      if (dimCache.has(dk(id))) { v.dims = dimCache.get(dk(id)); return; }
      try {
        const r = await fetch(opts.urls.size(id));
        if (!r.ok) return;
        const d = await r.json();
        dimCache.set(dk(id), { w: d.w, h: d.h });
        if (v.id === id) { v.dims = dimCache.get(dk(id)); v.render(); }
      } catch (e) { /* offline: stay on estimates */ }
    };

    v.memo = new Map();   // image id -> {z, ux, uy}: where this photo was left in THIS view
    v.setImage = function (id) {
      const same = v.id === id;
      if (!same && v.id !== null) {
        v.memo.delete(v.id);
        v.memo.set(v.id, { z: v.st.z, ux: v.st.ux, uy: v.st.uy });
        if (v.memo.size > MEMO_MAX) v.memo.delete(v.memo.keys().next().value);
      }
      v.id = id;
      v.dims = id !== null && dimCache.has(dk(id)) ? dimCache.get(dk(id)) : null;
      v.gen++;
      clearTimeout(v.timer);
      if (v.loader) { v.loader.onload = v.loader.onerror = null; v.loader.src = ""; v.loader = null; }
      setLoading(false);
      if (id === null) { img.removeAttribute("src"); v.fullFor = null; v.render(); return; }
      const pv = opts.urls.preview(id, el), cur = img.getAttribute("src");
      const pvd = v.dn.on && opts.urls.denoise ? opts.urls.denoise(id, el, dnLevel) : null;
      const shown = same && (cur === pvd || v.fullFor === id);   // already showing the right picture
      if (!same || (cur !== pv && !shown)) {
        img.src = pv;
        v.fullFor = null;
      }
      if (!same) {   // this photo's own zoom and position (Fit when never zoomed); linked: the first view leads
        const m = g.linked && g.views[0] !== v && g.views[0].id !== null ? g.views[0].st : v.memo.get(id);
        v.st = m ? { z: m.z, ux: m.ux, uy: m.uy } : { z: 1, ux: 0.5, uy: 0.5 };
        if (g.linked && g.views[0] === v) {
          for (const t of g.views) if (t !== v && t.id !== null) { t.st = { ...v.st }; t.render(); t.wantFull(); }
        }
      }
      if (!shown) v.dnReset();
      v.render();
      if (v.dn.on && !shown) v.dnLoadPreview();
      if (v.st.z > 1.0001) { v.ensureDims(); v.wantFull(); }
    };

    /* ----- denoise (T23f): the denoised Fit picture is swapped in once it has loaded (the first
       one takes seconds), then the whole denoised full-resolution image like /api/full; viewport
       regions are drawn by denoise.js on top meanwhile ----- */
    const dnEmit = () => {
      v.dn.state = v.dn.on ? (v.dn.busy > 0 ? "loading" : v.dn.backend) : "off";
      g.refreshBars();
      v.dnListeners.forEach((f) => f(v));
    };
    v.onDenoise = (fn) => { v.dnListeners.push(fn); };
    v.dnBusy = (d) => { v.dn.busy = Math.max(0, v.dn.busy + d); dnEmit(); };
    v.dnReset = function () {   // a new image / strength / switch: forget what was loading
      v.dn.gen++;
      if (v.dnLoader) { v.dnLoader.onload = v.dnLoader.onerror = null; v.dnLoader.src = ""; v.dnLoader = null; }
      v.dn.busy = 0;
      dnEmit();
    };
    v.dnLoadPreview = function () {
      if (!v.dn.on || v.id === null || !opts.urls.denoise || v.fullFor === v.id) return;
      const id = v.id, gen = v.dn.gen, ld = new Image();
      v.dnLoader = ld;
      v.dnBusy(1);
      const done = () => { if (v.dnLoader === ld) v.dnLoader = null; if (v.dn.gen === gen) v.dnBusy(-1); };
      ld.onload = () => {
        if (v.dn.gen === gen && v.id === id && v.dn.on && v.fullFor !== id) { img.src = ld.src; v.render(); }
        done();
      };
      ld.onerror = () => { done(); };
      ld.src = opts.urls.denoise(id, el, dnLevel);
    };
    const dnRestart = function () {
      clearTimeout(v.timer);
      if (v.loader) { v.loader.onload = v.loader.onerror = null; v.loader.src = ""; v.loader = null; }
      setLoading(false);
      v.dnReset();
      v.fullFor = null;
      if (v.id !== null) {
        img.src = opts.urls.preview(v.id, el);   // the plain picture at once; the denoised one swaps in
        if (v.dn.on) v.dnLoadPreview();
        v.render();
        if (v.st.z > 1.0001) v.wantFull();
      }
    };
    v.setDenoise = function (on) {
      on = !!on && !!opts.urls.denoise;
      if (v.dn.on === on) return;
      v.dn.on = on;
      dnRestart();
    };
    v.reloadDenoise = function () { if (v.dn.on) dnRestart(); };   // the strength changed

    function setLoading(on) {
      v.loading = on;
      el.dataset.loading = on ? "1" : "0";
      g.refreshBars();
    }
    v.wantFull = function () {
      if (v.id === null || v.st.z <= 1.0001 || v.fullFor === v.id) return;
      if (!v.dims) v.ensureDims();
      clearTimeout(v.timer);
      const id = v.id, gen = v.gen;
      v.timer = setTimeout(() => {
        if (v.id !== id || v.gen !== gen || v.st.z <= 1.0001 || v.fullFor === id) return;
        if (v.loader) return;
        setLoading(true);
        const ld = new Image();
        v.loader = ld;
        ld.onload = () => {
          if (v.loader !== ld) return;
          v.loader = null;
          setLoading(false);
          if (v.id !== id) return;
          dimCache.set(dk(id), { w: ld.naturalWidth, h: ld.naturalHeight });
          v.dims = dimCache.get(dk(id));
          img.src = ld.src;
          v.fullFor = id;
          v.render();
        };
        ld.onerror = () => { if (v.loader === ld) { v.loader = null; setLoading(false); } };
        v.fullDn = v.dn.on;
        const dgen = v.dn.gen, dnOn = v.dn.on;
        if (dnOn) v.dnBusy(1);
        const fin = () => { if (dnOn && v.dn.gen === dgen) v.dnBusy(-1); };
        const okLoad = ld.onload, okErr = ld.onerror;
        ld.onload = () => { okLoad(); fin(); };
        ld.onerror = () => { okErr(); fin(); };
        ld.src = dnOn ? opts.urls.denoiseFull(id, dnLevel) : opts.urls.full(id);
      }, FULL_DELAY);
    };

    /* ----- input ----- */

    const lineK = (e) => (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? el.clientHeight : 1);
    function local(e) { const r = el.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }

    el.addEventListener("pointerenter", () => { g.active = v; g.refreshBars(); });
    el.addEventListener("pointermove", (e) => { v.pt = local(e); g.active = v; });
    el.addEventListener("wheel", (e) => {
      if (!enabled()) return;
      e.preventDefault();
      g.active = v;
      const p = local(e);
      v.pt = p;
      if (e.ctrlKey || e.metaKey || e.altKey) {   // pinch (macOS sends ctrl+wheel) or modifier-wheel
        const k = e.ctrlKey ? 0.01 : 0.0025;
        v.zoomAround(p.x, p.y, v.st.z * Math.exp(-e.deltaY * lineK(e) * k));
      } else if (v.st.z > 1.0001) {               // two-finger scroll pans
        v.panBy(-e.deltaX * lineK(e), -e.deltaY * lineK(e));
      }
    }, { passive: false });

    // WebKit (WKWebView, Safari) pinch
    let gz = null;
    el.addEventListener("gesturestart", (e) => {
      if (!enabled()) return;
      e.preventDefault();
      g.active = v;
      gz = { z: v.st.z, p: local(e) };
    });
    el.addEventListener("gesturechange", (e) => {
      if (!enabled() || !gz) return;
      e.preventDefault();
      v.zoomAround(gz.p.x, gz.p.y, gz.z * e.scale);
    });
    el.addEventListener("gestureend", (e) => { if (gz) e.preventDefault(); gz = null; });

    let drag = null, moved = false;
    el.addEventListener("pointerdown", (e) => {
      if (!enabled() || e.button !== 0 || v.st.z <= 1.0001 || e.target.closest(".znav")) return;
      g.active = v;
      drag = { x: e.clientX, y: e.clientY, st: { ...v.st }, id: e.pointerId };
      moved = false;
      try { el.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      el.classList.add("panning");
    });
    el.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      const G = geo();
      g.set(v, { z: drag.st.z, ux: drag.st.ux - dx / (drag.st.z * G.rw), uy: drag.st.uy - dy / (drag.st.z * G.rh) });
    });
    const endDrag = () => { drag = null; el.classList.remove("panning"); };
    el.addEventListener("pointerup", endDrag);
    el.addEventListener("pointercancel", endDrag);
    el.addEventListener("click", (e) => { if (moved) { e.stopPropagation(); e.preventDefault(); moved = false; } }, true);
    el.addEventListener("dblclick", (e) => {
      if (!enabled() || e.target.closest(".znav")) return;
      const p = local(e);
      v.toggle100(p.x, p.y);
    });

    if (nav) {   // navigator: click / drag moves the visible area
      const moveTo = (e) => {
        const r = nav.getBoundingClientRect();
        v.panTo((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
      };
      let nd = false;
      nav.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.stopPropagation(); nd = true;
        try { nav.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
        moveTo(e);
      });
      nav.addEventListener("pointermove", (e) => { if (nd) { e.stopPropagation(); moveTo(e); } });
      nav.addEventListener("pointerup", () => { nd = false; });
      nav.addEventListener("pointercancel", () => { nd = false; });
      nav.addEventListener("dblclick", (e) => e.stopPropagation());
    }

    // the picture is requested at the size the screen needs: when the window (or the display's
    // pixel ratio) changes and a bigger bucket is now needed, swap to it (never to a smaller one)
    const edgeOf = (u) => Math.max(Number((/[?&]bw=(\d+)/.exec(u || "") || [])[1]) || 0, Number((/[?&]bh=(\d+)/.exec(u || "") || [])[1]) || 0);
    v.refreshPreview = function () {
      if (v.id === null || v.fullFor === v.id) return;
      const url = opts.urls.preview(v.id, el), cur = img.getAttribute("src") || "";
      if (cur.includes("/api/preview/") && url !== cur && edgeOf(url) > edgeOf(cur)) img.src = url;
    };
    let rt = null;
    const onResize = () => { clearTimeout(rt); rt = setTimeout(() => { if (el.offsetParent !== null) v.refreshPreview(); }, 250); };
    window.addEventListener("resize", () => { if (el.offsetParent !== null) v.render(); onResize(); });
    try { window.matchMedia(`(resolution: ${dpr()}dppx)`).addEventListener("change", onResize); } catch (e) { /* old WebKit */ }
    return v;
  }

  /* ---------- toolbar ---------- */

  function toolbar(host, g) {
    host.classList.add("zbar");
    host.innerHTML =
      '<button type="button" class="zb-minus" title="Zoom out (-)" aria-label="Zoom out">−</button>' +
      '<span class="zb-wrap"><button type="button" class="zb-badge" aria-haspopup="menu" aria-expanded="false" title="Zoom level">' +
      '<span class="zb-text">Fit</span><i class="zb-spin" hidden></i></button>' +
      '<div class="zb-menu" role="menu" hidden></div></span>' +
      '<button type="button" class="zb-plus" title="Zoom in (+)" aria-label="Zoom in">+</button>';
    const q = (s) => host.querySelector(s);
    const menu = q(".zb-menu"), badge = q(".zb-badge");
    for (const [val, label] of PRESETS) {
      const b = document.createElement("button");
      b.type = "button"; b.role = "menuitem"; b.dataset.pct = String(val); b.textContent = label;
      b.addEventListener("click", () => {
        closeMenu();
        g.each((v) => v.setPct(val === "fit" ? "fit" : val));
      });
      menu.appendChild(b);
    }
    const closeMenu = () => { menu.hidden = true; badge.setAttribute("aria-expanded", "false"); };
    badge.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = menu.hidden;
      menu.hidden = !open;
      badge.setAttribute("aria-expanded", String(open));
      if (open) g.each((v) => v.ensureDims());
    });
    document.addEventListener("click", (e) => { if (!host.contains(e.target)) closeMenu(); });
    host.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { closeMenu(); e.stopPropagation(); e.preventDefault(); } }, true);
    q(".zb-minus").addEventListener("click", () => g.each((v) => v.zoomBy(1 / STEP)));
    q(".zb-plus").addEventListener("click", () => g.each((v) => v.zoomBy(STEP)));
    const refresh = () => {
      const v = g.current();
      if (!v) return;
      const z = v.st.z;
      let text = "Fit";
      if (z > 1.0001) text = v.dims || v.img.naturalWidth ? Math.round(v.pct()) + " %" : "…";
      q(".zb-text").textContent = text;
      const busy = g.views.some((t) => t.loading);
      q(".zb-spin").hidden = !busy;
      host.dataset.loading = busy ? "1" : "0";
      host.dataset.pct = text;
      q(".zb-minus").disabled = z <= 1.0001;
      q(".zb-plus").disabled = z >= v.maxZ() - 1e-6;
    };
    g.bars.push(refresh);
    refresh();
    return { refresh, closeMenu };
  }

  /* ---------- keys ---------- */

  // ctx.view (optional) forces the target view; else the group's active one.
  function handleKey(e, g, ctx) {
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const mod = e.metaKey || e.ctrlKey;
    if (e.altKey) return false;
    const v = (ctx && ctx.view) || g.current();
    if (!v) return false;
    const at = v.pt || (() => { const G = v.geo(); return { x: G.vw / 2, y: G.vh / 2 }; })();
    if (!mod && key === "z") { v.toggle100(at.x, at.y); return true; }
    if (!mod && key === "d" && g.denoiseKey) { g.denoiseKey(e.shiftKey); return true; }   // denoise.js
    if (key === "+" || key === "=") { g.each((t) => t.zoomBy(STEP, t === v ? at.x : undefined, t === v ? at.y : undefined)); return true; }
    if (key === "-" || key === "_") { g.each((t) => t.zoomBy(1 / STEP, t === v ? at.x : undefined, t === v ? at.y : undefined)); return true; }
    if (key === "0") { g.each((t) => t.fit()); return true; }
    return false;
  }

  window.ShotZoom = {
    group, view, toolbar, handleKey, STEP,
    denoiseLevel, setDenoiseLevel, onDenoiseLevel: (f) => dnLevelListeners.push(f), DN_LEVELS,
  };
})();
