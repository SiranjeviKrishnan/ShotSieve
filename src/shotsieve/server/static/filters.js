/* ShotSieve filter bar (T23e): people, flags, colour labels, stars and the More panel (file type,
   orientation, warnings, faces, near-duplicates, camera, lens), the Key Faces panel, the star /
   label / pick badges on grid cells and the colour-label keys. One filter state `F` (remembered per
   shoot) is evaluated in the browser over the /api/shoot payload plus /api/shoot/facets and
   /api/people/map: AND between groups, OR within a group (people: any / all). app.js calls
   `ShotFilters.pass(im)` from matches(), `.onShoot()` after a load and `.afterLayout()` after each
   relayout. Needs app.js globals: S, $, api, q, esc, toast, relayout, select, renderDetail, decide. */
"use strict";

(function () {
  const LABELS = [["red", "Red", "#e5484d"], ["yellow", "Yellow", "#f2c230"], ["green", "Green", "#30a46c"],
    ["blue", "Blue", "#3e8ef7"], ["purple", "Purple", "#9b6bff"]];
  const LABEL_NAME = Object.fromEntries(LABELS.map((l) => [l[0], l[1]]));
  const LABEL_COLOR = Object.fromEntries(LABELS.map((l) => [l[0], l[2]]));
  const LABEL_KEYS = { 6: "red", 7: "yellow", 8: "green", 9: "blue" };   // Lightroom's keys
  const RAW_EXT = new Set("cr2 cr3 nef nrw arw srf sr2 raf orf rw2 pef dng srw 3fr iiq erf".split(" "));
  const NEUTRAL = new Set(["audit", "best_available"]);   // reasons that are not a warning
  const FACETS = ["make", "model", "lens", "serial"];
  const FACET_TITLE = { make: "Camera make", model: "Camera model", lens: "Lens model", serial: "Lens serial number" };
  const OTHERS = "__others";
  const emptyF = () => ({ people: [], pmode: "any", flags: [], labels: [], stars: [], ftype: [], orient: [],
    warn: [], faces: [], dups: [], make: [], model: [], lens: [], serial: [] });
  const LISTS = ["flags", "labels", "stars", "ftype", "orient", "warn", "faces", "dups", ...FACETS];

  let F = emptyF();
  let C = null;                       // compiled sets
  let forFolder = null;
  let D = new Map();                  // image id -> derived fields
  let facets = null;                  // /api/shoot/facets (null until loaded)
  let facetLists = {};                // field -> [{key,label,n,raw}] (with Others)
  let pmap = null;                    // person id -> Set(image ids) (null: people off / not loaded)
  let people = [];                    // /api/people list
  let barOn = true;
  let moreOpen = false;
  let visibleIds = new Set();         // images that pass filters + tab + search
  let scanTimer = null;

  const bar = $("filterbar");
  const lsGet = (k, d) => { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } };
  const key = () => "shotsieve.filters:" + S.folder;
  const anyActive = () => LISTS.some((k) => F[k].length) || F.people.length > 0;

  function compile() {
    C = {};
    for (const k of LISTS) C[k] = F[k].length ? new Set(F[k]) : null;
    C.people = F.people.length ? F.people : null;
  }
  function save() { lsSet(key(), JSON.stringify(F)); }
  function restore() {
    F = emptyF();
    try {
      const o = JSON.parse(lsGet(key(), "null"));
      if (o && typeof o === "object") for (const k of Object.keys(F)) if (k in o && (Array.isArray(F[k]) ? Array.isArray(o[k]) : typeof o[k] === typeof F[k])) F[k] = o[k];
    } catch (e) { /* ignore */ }
    compile();
  }

  /* ---------- derived data ---------- */

  const extOf = (n) => { const i = n.lastIndexOf("."); return i < 0 ? "" : n.slice(i + 1).toLowerCase(); };
  function fileType(im) {
    if (im.video_t != null) return "video";
    const e = extOf(im.name);
    if (RAW_EXT.has(e)) return "raw";
    if (e === "jpg" || e === "jpeg") return "jpeg";
    if (e === "heic" || e === "heif") return "heic";
    return "other";
  }
  function flagOf(im) { return im.overridden && im.decision === "keep" ? "picked" : im.overridden && im.decision === "reject" ? "rejected" : "unflagged"; }

  function derive() {
    const gsize = new Map();
    for (const g of S.groups) for (const id of g.members) gsize.set(id, g.members.length);
    D = new Map();
    // which facet values get their own entry: seen at least twice, or all of them when there are few
    const listed = {};
    const names = { make: "make", model: "model", lens: "lens", serial: "lens_serial" };
    for (const f of FACETS) {
      const vals = facets ? facets.vals[names[f]] : [];
      const few = vals.length <= 6;
      listed[f] = vals.map((v) => few || v.n >= 2);
    }
    facetLists = {};
    const counts = {};
    for (const f of FACETS) counts[f] = new Map();
    for (const im of S.images) {
      const d = { ft: fileType(im), dup: (gsize.get(im.id) || 1) > 1, faces: (im.n_faces || 0) > 0,
        warn: (im.reasons || []).some((c) => !NEUTRAL.has(c)), or: -1 };
      if (facets) {
        const r = facets.rows[String(im.id)];
        if (r) {
          d.or = r[4];
          FACETS.forEach((f, i) => {
            const idx = r[i], v = facets.vals[names[f]][idx];
            d[f] = idx >= 0 && listed[f][idx] ? v.label : OTHERS;
          });
        } else for (const f of FACETS) d[f] = OTHERS;
        const lab = facets.labels[String(im.id)];
        im.label = lab || "";
        for (const f of FACETS) counts[f].set(d[f], (counts[f].get(d[f]) || 0) + 1);
      }
      D.set(im.id, d);
    }
    if (facets) {
      for (const f of FACETS) {
        const vals = facets.vals[names[f]];
        const out = [];
        vals.forEach((v, i) => { if (listed[f][i]) out.push({ key: v.label, label: v.label, n: counts[f].get(v.label) || 0, raw: v.raw }); });
        const oth = counts[f].get(OTHERS) || 0;
        if (oth) out.push({ key: OTHERS, label: "Others", n: oth, raw: [] });
        facetLists[f] = out;
      }
    }
  }

  /* ---------- the predicate ---------- */

  function pass(im) {
    if (!C) return true;
    const d = D.get(im.id);
    if (C.flags && !C.flags.has(flagOf(im))) return false;
    if (C.labels && !C.labels.has(im.label || "none")) return false;
    if (C.stars && !C.stars.has(im.rating || 0)) return false;
    if (!d) return true;   // not derived yet (a load is in flight)
    if (C.ftype && !C.ftype.has(d.ft)) return false;
    if (C.warn && !C.warn.has(d.warn ? "with" : "without")) return false;
    if (C.faces && !C.faces.has(d.faces ? "with" : "without")) return false;
    if (C.dups && !C.dups.has(d.dup ? "with" : "without")) return false;
    if (C.orient && facets) {
      if (!C.orient.has(["landscape", "portrait", "square"][d.or] || "")) return false;
    }
    if (facets) for (const f of FACETS) if (C[f] && !C[f].has(d[f])) return false;
    if (C.people && pmap) {
      const has = (pid) => { const s = pmap.get(pid); return !!s && s.has(im.id); };
      if (F.pmode === "all" ? !C.people.every(has) : !C.people.some(has)) return false;
    }
    return true;
  }

  /* ---------- state changes ---------- */

  function changed(resetScroll = true) {
    compile(); save();
    relayout(resetScroll);
    window.dispatchEvent(new CustomEvent("filters-changed"));
    if (S.sel !== null && S.rowOf.has(S.sel)) select(S.sel);
    else if (S.flat.length) select(S.flat[0]);
    else { S.sel = null; renderDetail(); }
    render();
  }
  function toggle(list, v) {
    const a = F[list], i = a.indexOf(v);
    if (i < 0) a.push(v); else a.splice(i, 1);
    changed();
  }
  function clearAll() { F = emptyF(); changed(); }
  function setPeople(ids, mode) {
    F.people = ids.slice();
    if (mode) F.pmode = mode;
    changed();
  }
  function setPmode(m) { F.pmode = m; changed(); }
  function togglePerson(pid) {
    const i = F.people.indexOf(pid);
    if (i < 0) F.people.push(pid); else F.people.splice(i, 1);
    changed();
  }

  /* ---------- colour labels ---------- */

  async function setLabel(ids, label) {
    ids = ids.filter((id) => S.byId.has(id));
    if (!ids.length) return;
    const prev = ids.map((id) => [id, S.byId.get(id).label || ""]);
    const val = label === "none" ? "" : label;
    for (const id of ids) { S.byId.get(id).label = val; if (facets) facets.labels[String(id)] = val; }
    try {
      await api("/api/label", { method: "POST", body: { folder: S.folder, image_ids: ids, label } });
    } catch (e) {
      for (const [id, l] of prev) { S.byId.get(id).label = l; if (facets) facets.labels[String(id)] = l; }
      toast("Could not save the label: " + e.message);
    }
    const sel = S.sel;
    relayout();
    if (sel !== null && S.rowOf.has(sel)) select(sel, false); else renderDetail();
  }
  function labelKey(k) {
    const im = S.byId.get(S.sel);
    if (!im) return;
    const want = LABEL_KEYS[k];
    setLabel([im.id], im.label === want ? "none" : want);
  }

  /* ---------- grid cell badges ---------- */

  function cellBadges(im) {
    let h = "";
    const fl = flagOf(im);
    const bits = [];
    if (fl === "picked") bits.push('<span class="cf-pick" title="Picked">⚑</span>');
    else if (fl === "rejected") bits.push('<span class="cf-rej" title="Rejected">✕</span>');
    if (im.label) bits.push(`<span class="cf-dot" style="background:${LABEL_COLOR[im.label] || "#999"}" title="${esc(LABEL_NAME[im.label] || im.label)} label"></span>`);
    if (bits.length) h += `<span class="cell-flags">${bits.join("")}</span>`;
    if (im.rating) h += `<span class="rate rate-pill" title="${im.rating} star${im.rating === 1 ? "" : "s"}" data-stars="${im.rating}">${"★".repeat(im.rating)}</span>`;
    return h;
  }

  /* ---------- the bar ---------- */

  const dotBtn = (k) => `<button class="fb-dot${F.labels.includes(k) ? " on" : ""}" data-list="labels" data-v="${k}" style="--c:${k === "none" ? "transparent" : LABEL_COLOR[k]}" title="${k === "none" ? "No label" : LABEL_NAME[k]}" aria-pressed="${F.labels.includes(k)}" aria-label="${k === "none" ? "No label" : LABEL_NAME[k] + " label"}"></button>`;
  const tog = (list, v, text, title) => `<button class="fb-tog${F[list].includes(v) ? " on" : ""}" data-list="${list}" data-v="${v}" aria-pressed="${F[list].includes(v)}"${title ? ` title="${esc(title)}"` : ""}>${text}</button>`;

  function personById(pid) { return people.find((p) => p.id === pid); }
  function avatar(p, size) { return window.ShotPeople ? window.ShotPeople.faceUrl(p, size) : ""; }

  function chipsList() {
    const out = [];
    const add = (list, v, text) => out.push({ list, v, text });
    if (F.people.length) out.push({ list: "people", v: null, text: "People (" + F.pmode + "): " + F.people.map((id) => (personById(id) || {}).name || "Person " + id).join(", ") });
    for (const v of F.flags) add("flags", v, "Flag: " + v);
    for (const v of F.labels) add("labels", v, "Label: " + (v === "none" ? "none" : LABEL_NAME[v]));
    for (const v of F.stars) add("stars", v, v === 0 ? "Unrated" : v + " star" + (v === 1 ? "" : "s"));
    const ft = { raw: "RAW", jpeg: "JPEG", heic: "HEIC", video: "Video" };
    for (const v of F.ftype) add("ftype", v, "Type: " + ft[v]);
    for (const v of F.orient) add("orient", v, "Orientation: " + v);
    for (const [l, t] of [["warn", "Warnings"], ["faces", "Faces"], ["dups", "Near-duplicates"]]) for (const v of F[l]) add(l, v, t + ": " + v);
    for (const f of FACETS) for (const v of F[f]) add(f, v, FACET_TITLE[f] + ": " + (v === OTHERS ? "Others" : v));
    return out;
  }

  function render() {
    bar.hidden = !barOn || !S.images.length;
    if (bar.hidden) return;
    const stars = [0, 1, 2, 3, 4, 5].map((n) => `<button class="fb-star${F.stars.includes(n) ? " on" : ""}" data-list="stars" data-v="${n}" aria-pressed="${F.stars.includes(n)}" title="${n === 0 ? "Unrated" : n + " star" + (n === 1 ? "" : "s")}">${n === 0 ? "0" : n + "★"}</button>`).join("");
    const nact = chipsList().length;
    bar.innerHTML =
      `<div class="fb-row">` +
      `<span class="fb-grp" aria-label="Flags">${tog("flags", "picked", "Picked", "Photos you picked (kept by you)")}${tog("flags", "unflagged", "Unflagged", "Photos you have not flagged")}${tog("flags", "rejected", "Rejected", "Photos you rejected")}</span>` +
      `<span class="fb-grp" aria-label="Colour labels">${LABELS.map((l) => dotBtn(l[0])).join("")}${dotBtn("none")}</span>` +
      `<span class="fb-grp" aria-label="Stars">${stars}</span>` +
      `<button id="fb-more" class="fb-btn${moreCount() ? " on" : ""}" aria-haspopup="dialog" aria-expanded="${moreOpen}">More${moreCount() ? ` <b>${moreCount()}</b>` : ""} <span class="fb-chev">▾</span></button>` +
      `<span class="spacer"></span><span id="fb-count" class="fb-count"></span>` +
      `<button id="fb-clear" class="small"${nact ? "" : " hidden"}>Clear all filters</button></div>` +
      (nact ? `<div class="fb-chips">${chipsList().map((c, i) => `<span class="fb-chip" data-i="${i}">${esc(c.text)}<button aria-label="Remove ${esc(c.text)}" data-chip="${i}">×</button></span>`).join("")}</div>` : "");
    updateCount();
    if (moreOpen) drawMore();
  }
  const moreCount = () => ["ftype", "orient", "warn", "faces", "dups", ...FACETS].reduce((a, k) => a + F[k].length, 0);

  function updateCount() {
    const el = $("fb-count");
    if (el) el.textContent = `Showing ${visibleIds.size.toLocaleString("en-US")} of ${S.images.length.toLocaleString("en-US")}`;
  }

  bar.addEventListener("click", (e) => {
    const t = e.target.closest("button");
    if (!t) return;
    if (t.dataset.list) { const v = t.dataset.v; toggle(t.dataset.list, t.dataset.list === "stars" ? Number(v) : v); return; }
    if (t.id === "fb-clear") { clearAll(); return; }
    if (t.dataset.chip !== undefined) {
      const c = chipsList()[Number(t.dataset.chip)];
      if (!c) return;
      if (c.list === "people") { F.people = []; changed(); } else toggle(c.list, c.v);
      return;
    }
    if (t.id === "fb-more") { e.stopPropagation(); toggleMore(); return; }
  });

  /* ---------- people popover ---------- */

  const pop = document.createElement("div");
  pop.id = "fb-pop"; pop.className = "fb-pop"; pop.hidden = true; pop.setAttribute("role", "dialog");
  document.body.appendChild(pop);

  async function loadPeople() {
    if (!S.peopleOn) { people = []; pmap = null; return; }   // no consent (or not known yet): no request, no 403
    try {
      const [pl, pm] = await Promise.all([api("/api/people" + q({ folder: S.folder })), api("/api/people/map" + q({ folder: S.folder }))]);
      people = (pl.people || []).filter((p) => !p.ignored);
      pmap = new Map(Object.entries(pm.people).map(([k, v]) => [Number(k), new Set(v)]));
      const known = new Set(people.map((p) => p.id));
      if (F.people.some((id) => !known.has(id))) { F.people = F.people.filter((id) => known.has(id)); compile(); save(); }
    } catch (e) { people = []; pmap = null; }
  }
  const pName = (p) => p.name || (p.n === 1 ? "Unknown" : "Person " + p.id);
  function placePop(el, anchor) {
    const keep = el.scrollTop;
    el.hidden = false;
    el.scrollTop = keep;
    if (!anchor) return;
    const r = anchor.getBoundingClientRect();
    el.style.left = Math.max(8, Math.min(r.left, innerWidth - el.offsetWidth - 8)) + "px";
    el.style.top = r.bottom + 4 + "px";
    el.style.maxHeight = Math.max(160, innerHeight - r.bottom - 16) + "px";
  }
  function closePops() {
    const was = moreOpen;
    moreOpen = false;
    pop.hidden = true;
    { const b = $("fb-more"); if (b) b.setAttribute("aria-expanded", "false"); }
    return was;
  }
  function toggleMore() {
    const open = !moreOpen;
    closePops();
    if (!open) return;
    moreOpen = true;
    $("fb-more").setAttribute("aria-expanded", "true");
    ensureScan(true);
    drawMore();
  }

  /* ---------- More panel ---------- */

  function chk(list, v, text, n, tip) {
    const on = F[list].includes(v);
    return `<button class="fb-opt${on ? " on" : ""}" data-list="${list}" data-v="${esc(v)}" role="checkbox" aria-checked="${on}"${tip ? ` title="${esc(tip)}"` : ""}>` +
      `<span class="fb-box"></span><span class="fb-ot">${esc(text)}</span>${n === null || n === undefined ? "" : `<span class="fb-n">${n}</span>`}</button>`;
  }
  function count(fn) { let n = 0; for (const im of S.images) if (fn(im, D.get(im.id) || {})) n++; return n; }
  function drawMore() {
    const wl = (list, title, a, b) => `<section><h4>${title}</h4>${chk(list, "with", a, count((_, d) => d[{ warn: "warn", faces: "faces", dups: "dup" }[list]]))}${chk(list, "without", b, count((_, d) => !d[{ warn: "warn", faces: "faces", dups: "dup" }[list]]))}</section>`;
    const left =
      `<section><h4>File type</h4>${[["raw", "RAW"], ["jpeg", "JPEG"], ["heic", "HEIC"], ["video", "Video"]].map(([v, t]) => chk("ftype", v, t, count((_, d) => d.ft === v))).join("")}</section>` +
      `<section><h4>Orientation</h4>${[["landscape", "Landscape", 0], ["portrait", "Portrait", 1], ["square", "Square", 2]].map(([v, t, i]) => chk("orient", v, t, count((_, d) => d.or === i))).join("")}</section>` +
      wl("warn", "Warnings", "With warnings", "Without warnings") + wl("faces", "Faces", "With faces", "Without faces") + wl("dups", "Near-duplicates", "In a duplicate group", "Not in a group");
    let right = "";
    for (const f of FACETS) {
      const items = facetLists[f] || [];
      right += `<section><h4>${FACET_TITLE[f]}</h4>` + (items.length ? items.map((it) => chk(f, it.key, it.label, it.n, it.raw.length > 1 || (it.raw[0] && it.raw[0] !== it.label) ? "As written: " + it.raw.join(", ") : "")).join("")
        : `<p class="note">${facets && facets.pending ? "Reading the camera data…" : "Not recorded in these files."}</p>`) + "</section>";
    }
    pop.className = "fb-pop fb-more-pop";
    pop.innerHTML = `<div class="fb-cols"><div>${left}</div><div>${right}</div></div>`;
    placePop(pop, $("fb-more"));
  }

  pop.addEventListener("click", (e) => {
    e.stopPropagation();
    const t = e.target.closest("button");
    if (!t) return;
    if (t.dataset.list) toggle(t.dataset.list, t.dataset.v);
  });
  document.addEventListener("click", (e) => { if (moreOpen && !pop.contains(e.target) && !e.target.closest("#fb-more")) closePops(); });
  window.addEventListener("resize", () => { if (moreOpen) closePops(); });

  /* ---------- Key faces panel ---------- */

  const kf = document.createElement("section");
  kf.id = "keyfaces"; kf.hidden = true;
  $("detail").appendChild(kf);
  let kfOpen = lsGet("shotsieve.kf.open", "1") === "1";
  let kfScope = lsGet("shotsieve.kf.scope", "shoot");
  const kfSize = 64;   // one fixed face size (the size slider is gone)
  const imgFaces = new Map();         // image id -> [person ids]

  async function currentPeople() {
    const im = S.byId.get(S.sel);
    if (!im || !pmap) return [];
    if (!imgFaces.has(im.id)) {
      const ids = [];
      for (const [pid, set] of pmap) if (set.has(im.id)) ids.push(pid);
      imgFaces.set(im.id, ids);
    }
    return imgFaces.get(im.id);
  }
  async function renderKeyFaces() {
    if (!pmap || !people.length) { kf.hidden = true; kf.dataset.sig = ""; return; }
    const counts = new Map();
    if (kfScope === "image") {
      for (const pid of await currentPeople()) counts.set(pid, 1);
    } else {
      for (const [pid, set] of pmap) {
        let n = 0;
        for (const id of set) if (visibleIds.has(id)) n++;
        if (n) counts.set(pid, n);
      }
    }
    const list = people.filter((p) => counts.has(p.id)).sort((a, b) => counts.get(b.id) - counts.get(a.id) || a.id - b.id);
    const sig = JSON.stringify([kfOpen, kfScope, list.map((p) => [p.id, counts.get(p.id), p.name || ""]), F.people]);
    kf.hidden = false;
    if (sig === kf.dataset.sig) return;
    kf.dataset.sig = sig;
    const MAX = 80;
    kf.style.setProperty("--kf", kfSize + "px");
    kf.innerHTML =
      `<button class="kf-head" id="kf-toggle" aria-expanded="${kfOpen}"><span>Key faces (${list.length})</span><span class="fb-chev">${kfOpen ? "▴" : "▾"}</span></button>` +
      (kfOpen ? `<div class="kf-ctl"><div class="fb-seg" role="group" aria-label="Scope"><button data-kfs="image" class="${kfScope === "image" ? "on" : ""}" aria-pressed="${kfScope === "image"}">This image</button>` +
        `<button data-kfs="shoot" class="${kfScope === "shoot" ? "on" : ""}" aria-pressed="${kfScope === "shoot"}">Whole shoot</button></div>` +
        `</div>` +
        `<div class="kf-grid">${list.slice(0, MAX).map((p) => `<button class="kf-face${F.people.includes(p.id) ? " on" : ""}" data-pid="${p.id}" aria-pressed="${F.people.includes(p.id)}" title="${esc(pName(p))} · ${counts.get(p.id)} photo${counts.get(p.id) === 1 ? "" : "s"}">` +
          `<img alt="" loading="lazy" decoding="async" src="${esc(avatar(p, Math.min(256, kfSize * 2)))}"></button>`).join("")}</div>` +
        (list.length > MAX ? `<p class="note">+ ${list.length - MAX} more (use the People filter)</p>` : "") : "");
  }
  kf.addEventListener("click", (e) => {
    const t = e.target.closest("button");
    if (!t) return;
    if (t.id === "kf-toggle") { kfOpen = !kfOpen; lsSet("shotsieve.kf.open", kfOpen ? "1" : "0"); renderKeyFaces(); return; }
    if (t.dataset.kfs) { kfScope = t.dataset.kfs; lsSet("shotsieve.kf.scope", kfScope); renderKeyFaces(); return; }
    if (t.dataset.pid) togglePerson(Number(t.dataset.pid));
  });

  /* ---------- detail panel label picker + context menu ---------- */

  const pick = document.createElement("div");
  pick.id = "d-labelrow"; pick.className = "labelrow"; pick.hidden = true;
  const verdict = document.querySelector("#info .verdict");
  if (verdict) verdict.insertAdjacentElement("afterend", pick);
  function pickerHtml(cur) {
    return LABELS.map(([k, n, c]) => `<button class="lp-dot${cur === k ? " on" : ""}" data-label="${k}" style="--c:${c}" title="${n} (${{ red: 6, yellow: 7, green: 8, blue: 9 }[k] || "menu"})" aria-pressed="${cur === k}" aria-label="${n}"></button>`).join("") +
      `<button class="lp-none${cur ? "" : " on"}" data-label="none" title="No label" aria-label="No label">none</button>`;
  }
  function renderPicker() {
    const im = S.byId.get(S.sel);
    pick.hidden = !im;
    if (!im) return;
    pick.innerHTML = `<span class="muted">Label</span>${pickerHtml(im.label || "")}`;
  }
  pick.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-label]");
    if (!b || S.sel === null) return;
    const cur = (S.byId.get(S.sel) || {}).label || "";
    setLabel([S.sel], b.dataset.label === cur ? "none" : b.dataset.label);
  });
  $("grid").addEventListener("contextmenu", (e) => {
    const cell = e.target.closest(".cell");
    if (!cell) return;
    e.preventDefault();
    const id = Number(cell.dataset.id), im = S.byId.get(id);
    if (S.sel !== id) select(id, false);
    const m = $("ctx-menu");
    m.querySelectorAll(".ctx-extra").forEach((x) => x.remove());
    const extra = `<div class="ctx-extra"><div class="ctx-sep"></div><div class="ctx-label"><span class="muted">Label</span>${pickerHtml(im.label || "")}</div>` +
      `<div class="ctx-stars">${[1, 2, 3, 4, 5].map((n) => `<button data-stars="${n}" class="${im.rating === n ? "on" : ""}" title="${n} star${n === 1 ? "" : "s"}">${n}★</button>`).join("")}<button data-stars="0" title="Clear the rating">0</button></div></div>`;
    if (m.hidden) m.innerHTML = extra; else m.insertAdjacentHTML("beforeend", extra);
    m.style.left = Math.min(e.clientX, innerWidth - 230) + "px";
    m.style.top = Math.min(e.clientY, innerHeight - 190) + "px";
    m.hidden = false;
    for (const b of m.querySelectorAll("button[data-label], button[data-stars]")) {
      b.onclick = (ev) => {
        ev.stopPropagation();
        m.hidden = true;
        if (b.dataset.label) setLabel([id], b.dataset.label === (im.label || "") ? "none" : b.dataset.label);
        else if (S.sel === id) { const n = Number(b.dataset.stars); if (n || im.rating) decide(null, n); }
      };
    }
  });

  /* ---------- keyboard ---------- */

  document.addEventListener("keydown", (e) => {
    const typing = e.target && e.target.closest && e.target.closest("input, textarea, select");
    if ((e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "f") {   // Cmd+F: search
      e.preventDefault(); e.stopPropagation();
      $("search").focus(); $("search").select();
      return;
    }
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.shiftKey && e.key.toLowerCase() === "f") {   // Shift+F: show / hide the bar
      e.preventDefault(); e.stopPropagation();
      setBar(!barOn);
      return;
    }
    if (e.shiftKey) return;
    if (LABEL_KEYS[e.key] && ["overlay-compare", "overlay-people", "overlay-cameras", "overlay-shotlist", "overlay-edit", "overlay-lr", "overlay-jersey", "overlay-help"].every((id) => $(id).hidden) && $("crop-ui").hidden) {
      e.preventDefault(); e.stopPropagation();
      labelKey(e.key);
      return;
    }
    if (e.key === "0" && $("overlay-loupe").hidden && $("overlay-compare").hidden && $("crop-ui").hidden && $("overlay-people").hidden) {
      e.preventDefault(); e.stopPropagation();   // 0: clear the stars (nothing to do when unrated)
      const im = S.byId.get(S.sel);
      if (im && im.rating) decide(null, 0);
    }
  }, true);

  function setBar(on) {
    barOn = on;
    lsSet("shotsieve.filterbar", on ? "1" : "0");
    if (!on) closePops();
    render();
    if (!on && anyActive()) toast("Filters are still active. Shift+F shows the filter bar again.", 4000);
  }

  /* ---------- loading ---------- */

  let scanKick = null;
  function ensureScan(now) {   // the back-fill reads EXIF of every image: start it a few seconds after load, or at once for the More panel
    if (!facets || !facets.pending || facets.scanning) return;
    clearTimeout(scanKick);
    const folder = S.folder;
    scanKick = setTimeout(() => { if (S.folder === folder) api("/api/shoot/facets/scan", { method: "POST", body: { folder } }).catch(() => {}); }, now ? 0 : 3000);
  }
  async function loadFacets() {
    try {
      facets = await api("/api/shoot/facets" + q({ folder: S.folder }));
    } catch (e) { facets = null; return; }
    if (facets.pending || facets.scanning) {
      ensureScan(moreOpen);
      clearTimeout(scanTimer);
      const folder = S.folder;
      scanTimer = setTimeout(async () => {
        if (S.folder !== folder) return;
        await loadFacets();
        if (facets && !facets.pending && !facets.scanning) finish();   // done: one re-layout, not one per poll
        else if (moreOpen) { derive(); drawMore(); }
      }, 2500);
    }
  }
  let labelsSig = "{}";
  function finish() {   // new data arrived: re-derive, re-layout only when the grid can change (the selection is kept)
    derive();
    const sig = JSON.stringify(facets ? facets.labels : {});
    const grid = anyActive() || sig !== labelsSig;
    labelsSig = sig;
    if (!grid) { render(); renderKeyFaces(); return; }
    relayout();
    if (S.sel !== null && S.rowOf.has(S.sel)) renderPicker();
    else if (!S.flat.length) { S.sel = null; renderDetail(); }
    render();
  }
  async function onShoot() {
    barOn = lsGet("shotsieve.filterbar", "1") === "1";
    if (forFolder !== S.folder) { forFolder = S.folder; restore(); facets = null; pmap = null; people = []; imgFaces.clear(); labelsSig = "{}"; }
    derive();
    render();
    const folder = S.folder;
    await Promise.all([loadFacets(), loadPeople()]);
    if (S.folder !== folder) return;
    imgFaces.clear();
    finish();
  }

  function afterLayout() {
    visibleIds = new Set();
    for (const im of S.images) if (matches(im)) visibleIds.add(im.id);
    updateCount();
    renderKeyFaces();
    renderPicker();
  }
  function onSelect() {
    renderPicker();
    if (kfScope === "image") renderKeyFaces();
  }
  let lastPeopleOn = null, lastPeopleSig = null;
  async function peopleState(on, sig) {   // called by app.js's people poll
    if (on === lastPeopleOn && sig === lastPeopleSig) return;
    const first = lastPeopleOn === null;
    lastPeopleOn = on; lastPeopleSig = sig;
    if (first && (pmap !== null) === on) return;   // onShoot already loaded exactly this
    if (!S.folder || !S.images.length) return;
    await loadPeople();
    imgFaces.clear();
    if (F.people.length) finish(); else { render(); renderKeyFaces(); }   // membership only matters to the filter
  }

  /* people.js "show photos" and the face popover use this: one filter state for everything */
  window.ShotFilters = {
    pass, onShoot, afterLayout, onSelect, peopleState, cellBadges, setLabel, setPeople, togglePerson, setPmode, clearAll, toggle, setBar,
    state: () => F, derived: () => D, isActive: anyActive, renderPicker, counts: () => visibleIds.size,
  };
})();
