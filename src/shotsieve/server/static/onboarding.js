/* ShotSieve first-run wizard (T19) and the "?" menu actions. Loaded after app.js: it shares
   app.js's globals (S, $, api, toast, esc, loadShoot, pollIndex, checkShotlist). */
"use strict";

(() => {
  const STEP_TITLES = { models: "Welcome", folder: "Your shoot", catalog: "Lightroom (optional)",
    privacy: "Privacy and extras", cull: "Cull", lightroom: "Lightroom" };
  const STAGE_NAMES = { scan: "Reading the folder", features: "Measuring sharpness", faces: "Finding faces",
    embed: "Understanding the scenes", group: "Grouping bursts", score: "Choosing the best", write: "Writing ratings",
    video: "Looking at videos", images: "Learning from your picks" };

  const root = $("overlay-onboarding");
  const body = $("wiz-body");
  const stepsEl = $("wiz-steps");
  const W = {
    open: false, steps: [], i: 0, features: {}, models: null,
    folder: "", scan: null, catalog: "", trainJob: null, trainDone: null, trainFromOverrides: false,
    people: false, search: false, ocr: false, crop: false, writeXmp: true, wroteXmp: false,
    cullDone: false, dlSearch: null, lr: null, timer: null,
  };

  const fmtMB = (b) => (b >= 1e8 ? Math.round(b / 1e6) : Math.max(1, Math.round(b / 1e6))) + " MB";
  const fmtEta = (s) => (s === null || s === undefined || !isFinite(s) ? "" : s < 90 ? `about ${Math.max(1, Math.round(s))} s left` : `about ${Math.round(s / 60)} min left`);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const html = (s) => { body.innerHTML = s; const p = body.querySelector("[autofocus]"); if (p) p.focus(); };
  const progressHtml = (id, label) => `<div class="wiz-prog"><div class="bar" role="progressbar" aria-label="${esc(label)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="${id}-bar"><div id="${id}-fill"></div></div><p class="wiz-live" id="${id}-text" aria-live="polite"></p></div>`;
  function setProgress(id, frac, text) {
    const bar = $(id + "-bar"); if (!bar) return;
    const pct = Math.round(Math.max(0, Math.min(1, frac)) * 100);
    bar.setAttribute("aria-valuenow", String(pct));
    $(id + "-fill").style.width = pct + "%";
    $(id + "-text").textContent = text;
  }
  function saveStep() { api("/api/onboarding", { method: "POST", body: { last_step: W.i } }).catch(() => {}); }

  /* ---------- shell ---------- */
  function renderShell() {
    stepsEl.textContent = `Step ${W.i + 1} of ${W.steps.length}: ${STEP_TITLES[W.steps[W.i]]}`;
  }
  async function go(i) {
    W.i = i; renderShell(); saveStep();
    await STEPS[W.steps[i]]();
  }
  const next = () => go(Math.min(W.i + 1, W.steps.length - 1));

  async function show(force = false) {
    let ob;
    try { ob = await api("/api/onboarding"); } catch (e) { return; }
    W.features = ob.features || {};
    W.bundled = !!ob.bundled;  // release app: every model ships inside it, nothing to download
    if (!force && !ob.show) return;
    try { W.models = await api("/api/models"); } catch (e) { W.models = null; }
    W.steps = ob.models_ready ? ["folder", "catalog", "privacy", "cull", "lightroom"]
      : ["models", "folder", "catalog", "privacy", "cull", "lightroom"];
    W.open = true; root.hidden = false;
    W.folder = S.folder || ""; W.scan = null; W.cullDone = false;
    await go(0);
  }
  function close() { W.open = false; root.hidden = true; clearInterval(W.timer); }
  async function skip() {
    await api("/api/onboarding", { method: "POST", body: { skipped: true, last_step: W.i } }).catch(() => {});
    close();
  }
  async function finish() {
    await api("/api/onboarding", { method: "POST", body: { completed: true, last_step: W.i } }).catch(() => {});
    close();
    const g = $("grid"); if (g) g.focus();
  }

  /* ---------- step 1: models ---------- */
  async function stepModels() {
    const core = W.models && W.models.groups.core;
    const size = core ? fmtMB(core.bytes) : "131 MB";
    html(`<h2 id="wiz-title">Welcome to ShotSieve</h2>
      <p>ShotSieve works entirely on this Mac: your photos never leave it. To start it needs to download its models once (${size}).</p>
      <div id="wiz-dl-area"><button id="wiz-download" class="primary" autofocus>Download (${size})</button></div>`);
    $("wiz-download").onclick = startDownload;
    const cur = W.models && W.models.download;
    if (cur && (cur.state === "running" || cur.state === "queued") && (cur.result.groups || []).includes("core")) watchDownload(cur.job_id, "core");
  }
  async function startDownload() {
    $("wiz-dl-area").innerHTML = progressHtml("wiz-dl", "Download progress");
    setProgress("wiz-dl", 0, "Starting…");
    try {
      const r = await api("/api/models/download", { method: "POST", body: { groups: ["core"] } });
      watchDownload(r.job_id, "core");
    } catch (e) { downloadFailed(e.message); }
  }
  async function watchDownload(id, group) {
    if (!$("wiz-dl-area")) return;
    $("wiz-dl-area").innerHTML = progressHtml("wiz-dl", "Download progress");
    for (;;) {
      let j;
      try { j = await api("/api/jobs/" + id); } catch (e) { return downloadFailed(e.message); }
      const frac = j.total ? j.done / j.total : 0;
      const r = j.result || {};
      setProgress("wiz-dl", frac, `${Math.round(frac * 100)}% · ${r.mb_s ? r.mb_s.toFixed(1) + " MB/s" : "…"} ${fmtEta(r.eta_s)}`);
      if (j.state === "done") { setProgress("wiz-dl", 1, "100% · done"); W.models = await api("/api/models").catch(() => W.models); return next(); }
      if (j.state === "error") return downloadFailed(j.error);
      await sleep(250);
    }
  }
  function downloadFailed(msg) {
    const area = $("wiz-dl-area"); if (!area) return;
    area.innerHTML = `<p class="wiz-err" role="alert">${esc(msg)}</p><button id="wiz-retry" class="primary" autofocus>Retry</button>`;
    $("wiz-retry").onclick = startDownload;
    $("wiz-retry").focus();
  }

  /* ---------- file browser (server-side: browsers cannot give absolute paths) ---------- */
  async function browse(box, kind, dir, onPick) {
    let d;
    try { d = await api("/api/fs/list" + q({ dir, kind })); }
    catch (e) { box.innerHTML = `<p class="wiz-err" role="alert">${esc(e.message)}</p>`; return; }
    const rows = d.entries.map((en, n) => en.dir
      ? `<li><button class="link" data-open="${n}">${esc(en.name)}</button> <span class="muted">${en.n_images ? en.n_images + " images" : ""}</span>${kind === "folder" ? ` <button data-pick="${n}" class="small">Use</button>` : ""}</li>`
      : `<li><button class="link" data-pick="${n}">${esc(en.name)}</button> <span class="muted">catalog</span></li>`).join("");
    box.innerHTML = `<div class="wiz-browser"><div class="wiz-crumb"><span class="muted">${esc(d.dir)}</span>
      ${d.parent ? '<button class="small" data-up="1">Up</button>' : ""}
      ${d.roots.map((r, n) => `<button class="small" data-root="${n}">${esc(r === "/Volumes" ? "Disks" : "Home")}</button>`).join("")}
      ${kind === "folder" ? '<button class="small" data-here="1">Use this folder</button>' : ""}</div>
      <ul>${rows || '<li class="muted">Nothing here.</li>'}</ul>${d.truncated ? '<p class="muted">Showing the first 500.</p>' : ""}</div>`;
    box.querySelectorAll("[data-open]").forEach((b) => (b.onclick = () => browse(box, kind, d.entries[+b.dataset.open].path, onPick)));
    box.querySelectorAll("[data-pick]").forEach((b) => (b.onclick = () => onPick(d.entries[+b.dataset.pick].path)));
    box.querySelectorAll("[data-up]").forEach((b) => (b.onclick = () => browse(box, kind, d.parent, onPick)));
    box.querySelectorAll("[data-root]").forEach((b) => (b.onclick = () => browse(box, kind, d.roots[+b.dataset.root], onPick)));
    box.querySelectorAll("[data-here]").forEach((b) => (b.onclick = () => onPick(d.dir)));
  }
  async function nativePick(kind) {
    try {
      if (window.shotsieveDesktop && window.shotsieveDesktop.available()) return await window.shotsieveDesktop.pick(kind);
      const r = await api("/api/fs/pick", { method: "POST", body: { kind } });
      return r.cancelled ? null : r.path;
    } catch (e) { toast(e.message, 6000); return null; }
  }

  /* ---------- step 2: folder ---------- */
  function scanLine(s) {
    if (!s) return "";
    if (!s.n_images) return "No supported photos found in this folder.";
    const f = s.by_format, parts = [];
    if (f.raw) parts.push(`${f.raw.toLocaleString()} RAW`);
    if (f.dng) parts.push(`${f.dng.toLocaleString()} DNG`);
    if (f.jpeg) parts.push(`${f.jpeg.toLocaleString()} JPEG`);
    if (f.heic) parts.push(`${f.heic.toLocaleString()} HEIC`);
    if (f.other) parts.push(`${f.other.toLocaleString()} other`);
    let t = `${s.n_images.toLocaleString()} photos: ${parts.join(", ")}`;
    if (f.video) t += ` · ${f.video} video${f.video > 1 ? "s" : ""}`;
    return t + (s.culled ? " · already culled" : "");
  }
  async function setFolder(path) {
    W.folder = path; W.scan = null;
    const input = $("wiz-path"); if (input) input.value = path;
    const nm = $("wiz-path-name"); if (nm) { nm.textContent = path.replace(/\/+$/, "").split("/").pop(); nm.title = path; }
    const line = $("wiz-scan"), btn = $("wiz-continue");
    if (btn) btn.disabled = true;
    if (line) { line.className = "wiz-live"; line.textContent = "Looking…"; }
    try {
      W.scan = await api("/api/fs/scan" + q({ folder: path }));
      if (line) { line.textContent = scanLine(W.scan); line.className = "wiz-live" + (W.scan.n_images ? " ok" : " wiz-err"); }
      W.writeXmp = (W.scan.by_format.raw + W.scan.by_format.dng) > 0;
      if (btn) btn.disabled = !W.scan.n_images;
      document.querySelectorAll("[data-recent]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.recent === path)));
    } catch (e) {
      W.scan = null;
      if (line) { line.textContent = e.message; line.className = "wiz-live wiz-err"; }
    }
  }
  async function stepFolder() {
    let recent = [];
    try { recent = (await api("/api/recent")).folders; } catch (e) { /* none */ }
    html(`<h2 id="wiz-title">Choose your shoot folder</h2>
      <p>The folder with the photos from one shoot. ShotSieve never changes your originals.</p>
      ${recent.length ? `<div class="wiz-recent" role="group" aria-label="Recent folders">${recent.map((r, n) => `<button id="wiz-recent-${n}" data-recent="${esc(r.path)}" aria-pressed="false" title="${esc(r.path)}">${esc(r.name)}</button>`).join("")}</div>` : ""}
      <div class="wiz-row"><input id="wiz-path" class="browser-only" type="text" spellcheck="false" placeholder="/Users/you/Pictures/Wedding" aria-label="Folder path" value="${esc(W.folder)}">
        <button id="wiz-pick">Choose Folder…</button><span id="wiz-path-name" class="path-name native-only" title="${esc(W.folder)}">${esc(W.folder ? W.folder.replace(/\/+$/, "").split("/").pop() : "")}</span><button id="wiz-browse-toggle" class="browser-only">Browse</button></div>
      <div id="wiz-browser" hidden></div>
      <p id="wiz-scan" class="wiz-live" aria-live="polite">${esc(scanLine(W.scan))}</p>
      <div class="wiz-actions"><button id="wiz-continue" class="primary" ${W.scan && W.scan.n_images ? "" : "disabled"}>Continue</button></div>`);
    if (!recent.length) { if (window.shotsieveDesktop && window.shotsieveDesktop.available()) $("wiz-pick").focus(); else $("wiz-path").focus(); } else $("wiz-recent-0").focus();
    document.querySelectorAll("[data-recent]").forEach((b) => (b.onclick = () => setFolder(b.dataset.recent)));
    $("wiz-path").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); setFolder($("wiz-path").value.trim()); } });
    $("wiz-path").addEventListener("change", () => { const v = $("wiz-path").value.trim(); if (v && v !== W.folder) setFolder(v); });
    $("wiz-pick").onclick = async () => { const p = await nativePick("folder"); if (p) setFolder(p); };
    $("wiz-browse-toggle").onclick = () => {
      const box = $("wiz-browser"); box.hidden = !box.hidden;
      if (!box.hidden) browse(box, "folder", W.folder && W.scan ? W.folder : "", (p) => { box.hidden = true; setFolder(p); });
    };
    $("wiz-continue").onclick = next;
    if (W.folder && !W.scan) setFolder(W.folder);
  }

  /* ---------- step 3: Lightroom catalog + training ---------- */
  async function stepCatalog() {
    let cats = [];
    if (W.features.lr) {
      try { const r = await api("/api/lr/catalogs"); cats = (r.catalogs || r || []).map((c) => (typeof c === "string" ? { path: c } : c)); } catch (e) { /* none */ }
    }
    html(`<h2 id="wiz-title">Learn from your Lightroom picks (optional)</h2>
      <p>If you cull in Lightroom Classic, ShotSieve can learn which photos you tend to keep. It only reads the catalog, never writes to it. You can skip this.</p>
      ${cats.length ? `<div class="wiz-recent" role="group" aria-label="Catalogs found">${cats.map((c, n) => `<button data-cat="${esc(c.path)}" aria-pressed="false" title="${esc(c.path)}">${esc(c.name || c.path.split("/").pop())}</button>`).join("")}</div>` : ""}
      <div class="wiz-row"><input id="wiz-cat" class="browser-only" type="text" spellcheck="false" placeholder="/Users/you/Pictures/Lightroom/Catalog.lrcat" aria-label="Lightroom catalog path" value="${esc(W.catalog)}">
        <button id="wiz-cat-pick">Choose Catalog…</button><span id="wiz-cat-name" class="path-name native-only" title="${esc(W.catalog)}">${esc(W.catalog ? W.catalog.split("/").pop() : "")}</span><button id="wiz-cat-browse" class="browser-only">Browse</button></div>
      <div id="wiz-cat-box" hidden></div>
      ${W.scan && W.scan.culled ? `<label class="wiz-chk"><input type="checkbox" id="wiz-ovr" ${W.trainFromOverrides ? "checked" : ""}> Also learn from my choices in this folder</label>` : ""}
      <div class="wiz-actions"><button id="wiz-train" ${W.catalog ? "" : "disabled"}>Train my personal model</button></div>
      <div id="wiz-train-area" aria-live="polite"></div>
      <div class="wiz-actions"><button id="wiz-continue" class="primary" autofocus>Continue</button></div>`);
    const setCat = (p) => { W.catalog = p; $("wiz-cat").value = p; $("wiz-cat-name").textContent = p.split("/").pop(); $("wiz-cat-name").title = p; $("wiz-train").disabled = !p; document.querySelectorAll("[data-cat]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.cat === p))); };
    document.querySelectorAll("[data-cat]").forEach((b) => (b.onclick = () => setCat(b.dataset.cat)));
    $("wiz-cat").addEventListener("input", () => { W.catalog = $("wiz-cat").value.trim(); $("wiz-train").disabled = !W.catalog; });
    $("wiz-cat-pick").onclick = async () => { const p = await nativePick("lrcat"); if (p) setCat(p); };
    $("wiz-cat-browse").onclick = () => {
      const box = $("wiz-cat-box"); box.hidden = !box.hidden;
      if (!box.hidden) browse(box, "lrcat", "", (p) => { box.hidden = true; setCat(p); });
    };
    const ovr = $("wiz-ovr"); if (ovr) ovr.onchange = () => { W.trainFromOverrides = ovr.checked; };
    $("wiz-train").onclick = startTrain;
    $("wiz-continue").onclick = next;
    if (W.trainJob && !W.trainDone) watchTrain(W.trainJob); else if (W.trainDone) showTrainResult(W.trainDone);
  }
  async function startTrain() {
    $("wiz-train").disabled = true;
    const bodyReq = { catalog: W.catalog || null, xmp_folders: [], from_overrides: W.trainFromOverrides && W.folder ? [W.folder] : [] };
    $("wiz-train-area").innerHTML = progressHtml("wiz-tr", "Training progress");
    setProgress("wiz-tr", 0, "Starting…");
    try {
      const r = await api("/api/train", { method: "POST", body: bodyReq });
      W.trainJob = r.job_id; W.trainDone = null;
      watchTrain(r.job_id);
    } catch (e) {
      $("wiz-train-area").innerHTML = `<p class="wiz-err" role="alert">${esc(e.message)}</p>`;
      $("wiz-train").disabled = false;
    }
  }
  function showTrainResult(r) {
    const a = $("wiz-train-area"); if (!a) return;
    a.innerHTML = `<p class="wiz-live ${r.trained ? "ok" : ""}" id="wiz-train-result">${esc(r.message)}</p>`;
  }
  async function watchTrain(id) {
    for (;;) {
      let j;
      try { j = await api("/api/jobs/" + id); } catch (e) { return; }
      if (j.state === "done") { W.trainDone = j.result; W.trainJob = null; const b = $("wiz-train"); if (b) b.disabled = !W.catalog; return showTrainResult(j.result); }
      if (j.state === "error") { W.trainJob = null; const a = $("wiz-train-area"); if (a) a.innerHTML = `<p class="wiz-err" role="alert">${esc(j.error)}</p>`; const b = $("wiz-train"); if (b) b.disabled = false; return; }
      if ($("wiz-tr-bar")) {
        const frac = j.total ? j.done / j.total : 0, r = j.result || {};
        setProgress("wiz-tr", frac, `${STAGE_NAMES[j.stage] || "Working"}: ${j.done.toLocaleString()} of ${j.total.toLocaleString()} photos${r.img_s ? " · " + r.img_s + " img/s" : ""} ${fmtEta(r.eta_s)}`);
      }
      await sleep(400);
    }
  }

  /* ---------- step 4: privacy and extras ---------- */
  async function stepPrivacy() {
    try { W.models = await api("/api/models"); } catch (e) { /* keep old */ }
    const g = (W.models && W.models.groups) || {};
    const sSize = g.search ? fmtMB(g.search.bytes) : "690 MB", oSize = g.ocr ? fmtMB(g.ocr.bytes) : "10 MB";
    const cSize = g.crop ? fmtMB(g.crop.bytes) : "36 MB";
    const note = (grp) => (g[grp] && g[grp].installed ? " (already downloaded)" : "");
    html(`<h2 id="wiz-title">Privacy and extras</h2>
      <p>These are all off until you turn them on. Nothing is sent anywhere; everything stays on this Mac.</p>
      <label class="wiz-opt"><input type="checkbox" id="wiz-people" ${W.people ? "checked" : ""}><span><b>Recognise people</b><br><span class="muted">Groups the same face across this shoot so you can name people and find them. Face data stays in this shoot's folder and you can delete it any time.</span></span></label>
      <label class="wiz-opt"><input type="checkbox" id="wiz-search" ${W.search ? "checked" : ""}><span><b>Search by description</b><br><span class="muted">Type "bride laughing" to find photos. ${W.bundled ? "Included with the app, works offline" : "Downloads " + sSize + note("search")}; indexing runs in the background after the cull.</span></span></label>
      <label class="wiz-opt"><input type="checkbox" id="wiz-ocr" ${W.ocr ? "checked" : ""}><span><b>Read jersey numbers</b><br><span class="muted">For sports: finds players by number. ${W.bundled ? "Included with the app, works offline." : "Downloads " + oSize + note("ocr") + "."}</span></span></label>
      <label class="wiz-opt"><input type="checkbox" id="wiz-crop" ${W.crop ? "checked" : ""}><span><b>Smart crop</b><br><span class="muted">Press R on a photo for AI crop suggestions. ${W.bundled ? "Includes the object detector that finds people and animals." : "Works without this download; the object detector (" + cSize + note("crop") + ") finds people and animals better."}</span></span></label>
      <div class="wiz-actions"><button id="wiz-continue" class="primary" autofocus>Continue</button></div>
      <p id="wiz-msg" class="wiz-live" aria-live="polite"></p>`);
    $("wiz-continue").onclick = async () => {
      W.people = $("wiz-people").checked; W.search = $("wiz-search").checked; W.ocr = $("wiz-ocr").checked; W.crop = $("wiz-crop").checked;
      $("wiz-continue").disabled = true;
      try {
        if (W.people && W.folder) await api("/api/people/consent", { method: "POST", body: { folder: W.folder, on: true } });
        const need = [];
        if (W.search && !(g.search && g.search.installed)) need.push("search");
        if (W.ocr && !(g.ocr && g.ocr.installed)) need.push("ocr");
        if (W.crop && !(g.crop && g.crop.installed)) need.push("crop");
        if (need.length) {
          const r = await api("/api/models/download", { method: "POST", body: { groups: need } }).catch((e) => { if (e.status === 409) return null; throw e; });
          if (r) watchBackgroundDownload(r.job_id);
        }
      } catch (e) { toast(e.message, 6000); }
      next();
    };
  }
  async function watchBackgroundDownload(id) {
    W.dlSearch = id;
    for (;;) {
      let j;
      try { j = await api("/api/jobs/" + id); } catch (e) { return; }
      if (j.state === "done") { W.dlSearch = null; if (W.cullDone) startIndexing(); else toast("Extras downloaded; they will be indexed after the cull.", 4000); return; }
      if (j.state === "error") { W.dlSearch = null; toast("Download failed: " + j.error, 8000); return; }
      const pill = $("wiz-bg-dl");
      if (pill) pill.textContent = `Downloading extras: ${j.total ? Math.round((100 * j.done) / j.total) : 0}%`;
      await sleep(500);
    }
  }
  async function startIndexing() {
    if (!W.folder || !(W.search || W.ocr)) return;
    try { await api("/api/index/start", { method: "POST", body: { folder: W.folder, people: W.people, ocr: W.ocr } }); pollIndex(); }
    catch (e) { /* models still missing: nothing to do */ }
  }

  /* ---------- step 5: cull ---------- */
  async function stepCull() {
    const s = W.scan;
    html(`<h2 id="wiz-title">Cull your shoot</h2>
      <p>${esc(scanLine(s))}<br><span class="muted">${esc(W.folder)}</span></p>
      <label class="wiz-chk"><input type="checkbox" id="wiz-xmp" ${W.writeXmp ? "checked" : ""}> Write ratings for Lightroom</label>
      <p class="muted">ShotSieve writes a small .xmp file next to each RAW. Your originals are never modified.</p>
      <div id="wiz-cull-area"><div class="wiz-actions"><button id="wiz-cull" class="primary" autofocus>Cull</button></div></div>
      <p id="wiz-bg-dl" class="muted" aria-live="polite"></p>`);
    $("wiz-cull").onclick = runCull;
  }
  async function waitForTraining() {
    while (W.trainJob) {
      setProgress("wiz-cull", 0, "Waiting for training to finish first…");
      await sleep(500);
    }
  }
  async function runCull() {
    W.writeXmp = $("wiz-xmp").checked; W.wroteXmp = W.writeXmp;
    $("wiz-cull-area").innerHTML = progressHtml("wiz-cull", "Cull progress");
    setProgress("wiz-cull", 0, "Starting…");
    await waitForTraining();
    S.folder = W.folder; setFolderField(W.folder);
    try { localStorage.setItem("shotsieve.folder", W.folder); } catch (e) { /* ignore */ }
    let r;
    for (let tries = 0; ; tries++) {
      try { r = await api("/api/cull", { method: "POST", body: { folder: W.folder, write_xmp: W.writeXmp, video: false } }); break; }
      catch (e) {
        if (e.status === 409 && tries < 240) { setProgress("wiz-cull", 0, "Waiting for the current job to finish…"); await sleep(1000); continue; }
        return cullFailed(e.message);
      }
    }
    const t0 = performance.now();
    for (;;) {
      let j;
      try { j = await api("/api/jobs/" + r.job_id); } catch (e) { return cullFailed(e.message); }
      const frac = j.total ? j.done / j.total : 0, el = (performance.now() - t0) / 1000;
      const rate = j.done && el > 1 ? j.done / el : 0, eta = rate ? (j.total - j.done) / rate : null;
      setProgress("wiz-cull", frac, `${STAGE_NAMES[j.stage] || "Working"}: ${j.done.toLocaleString()} of ${j.total.toLocaleString()}${rate ? " · " + rate.toFixed(1) + " img/s" : ""} ${fmtEta(eta)}`);
      if (j.state === "done") break;
      if (j.state === "error") return cullFailed(j.error);
      await sleep(350);
    }
    W.cullDone = true;
    await loadShoot(); pollIndex(); checkShotlist();
    if (W.search || W.ocr) startIndexing();
    next();
  }
  function cullFailed(msg) {
    $("wiz-cull-area").innerHTML = `<p class="wiz-err" role="alert">${esc(msg)}</p><button id="wiz-cull" class="primary" autofocus>Retry</button>`;
    $("wiz-cull").onclick = runCull; $("wiz-cull").focus();
  }

  /* ---------- step 6: Lightroom card ---------- */
  function lrCard() {
    const f = (W.scan && W.scan.by_format) || {};
    const parts = [];
    if (f.raw) {
      parts.push(`<div class="wiz-card"><b>RAW files (${f.raw.toLocaleString()})</b><p>In Lightroom Classic: <b>Add</b> this folder in Import${W.wroteXmp ? " (ratings and colour labels come with it)" : ""}. Already imported? Select all, then Metadata, then <b>Read Metadata from Files</b>.</p>${W.wroteXmp ? "" : '<p class="muted">You did not write ratings; use "Write XMP" in the top bar first.</p>'}</div>`);
    }
    const other = (f.jpeg || 0) + (f.dng || 0) + (f.heic || 0);
    if (other) {
      parts.push(`<div class="wiz-card"><b>JPEG, DNG and HEIC files (${other.toLocaleString()})</b><p>Lightroom does not read ShotSieve's ratings for these files. Review them here (Keep tab); after you finish in Lightroom, use <b>Sync from Lightroom</b> so ShotSieve learns.</p></div>`);
    }
    if (!parts.length) parts.push('<div class="wiz-card"><p>Open your photos in any editor; ShotSieve only writes small sidecar files next to them.</p></div>');
    return parts.join("");
  }
  async function stepLightroom() {
    if (!W.scan && S.folder) { W.folder = S.folder; try { W.scan = await api("/api/fs/scan" + q({ folder: S.folder })); } catch (e) { /* card shows the generic text */ } }
    html(`<h2 id="wiz-title">What to do in Lightroom</h2>${lrCard()}
      ${W.features.lr ? '<div class="wiz-actions"><button id="wiz-sync">Sync from Lightroom…</button></div><div id="wiz-sync-area" aria-live="polite"></div>' : ""}
      <div class="wiz-actions"><button id="wiz-done" class="primary" autofocus>Done</button></div>`);
    $("wiz-done").onclick = finish;
    const sync = $("wiz-sync"); if (sync) sync.onclick = lrSync;
  }
  async function lrSync() {
    const area = $("wiz-sync-area");
    let catalog = W.catalog;
    if (!catalog) { // standalone card: the remembered / first detected catalog, else ask
      try {
        const c = await api("/api/lr/catalogs" + q({ folder: W.folder }));
        catalog = c.remembered || (c.catalogs || []).map((x) => (typeof x === "string" ? x : x.path))[0] || "";
      } catch (e) { /* none */ }
    }
    if (!catalog) { area.innerHTML = '<p class="wiz-err">Choose your catalog in the Lightroom step first (Setup assistant).</p>'; return; }
    area.textContent = "Reading the catalog…";
    try {
      const r = await api("/api/lr/sync", { method: "POST", body: { folder: W.folder, catalog, dry_run: true } });
      const j = await waitJob(r.job_id);
      const res = j.result || {};
      const none = !res.n_writes;
      area.innerHTML = `<p id="wiz-sync-msg">${none ? "Nothing to sync. " : "Preview (nothing changed yet): "}${esc(res.summary || "")}</p>` +
        (none ? "" : '<button id="wiz-sync-apply" class="primary">Apply</button>');
      const ap = $("wiz-sync-apply");
      if (ap) ap.onclick = async () => {
        ap.disabled = true;
        try {
          const r2 = await api("/api/lr/sync", { method: "POST", body: { folder: W.folder, catalog, dry_run: false } });
          const j2 = await waitJob(r2.job_id); const r3 = j2.result || {};
          area.innerHTML = `<p id="wiz-sync-msg">Applied: ${esc(r3.summary || "")}</p>`;
          await loadShoot();
        } catch (e) { area.innerHTML = `<p class="wiz-err" role="alert">${esc(e.message)}</p>`; }
      };
    } catch (e) { area.innerHTML = `<p class="wiz-err" role="alert">${esc(e.message)}</p>`; }
  }
  async function waitJob(id) {
    for (;;) {
      const j = await api("/api/jobs/" + id);
      if (j.state === "done") return j;
      if (j.state === "error") throw new Error(j.error);
      await sleep(400);
    }
  }

  const STEPS = { models: stepModels, folder: stepFolder, catalog: stepCatalog, privacy: stepPrivacy, cull: stepCull, lightroom: stepLightroom };

  /* ---------- wiring ---------- */
  root.addEventListener("keydown", (e) => e.stopPropagation()); // app.js shortcuts stay out of the wizard
  $("wiz-skip").onclick = skip;
  $("btn-setup").onclick = () => { $("overlay-help").hidden = true; show(true); };
  $("btn-lrcard").onclick = async () => {
    $("overlay-help").hidden = true;
    W.steps = ["lightroom"]; W.i = 0; W.open = true; root.hidden = false; W.scan = null;
    try { const ob = await api("/api/onboarding"); W.features = ob.features || {}; } catch (e) { /* ignore */ }
    W.folder = S.folder || ""; renderShell(); await stepLightroom();
  };
  $("btn-quit").onclick = async () => {
    $("btn-quit").disabled = true;
    try { await api("/api/quit", { method: "POST", body: {} }); } catch (e) { /* the server may close first */ }
    document.body.innerHTML = '<div style="padding:60px;text-align:center;font:16px -apple-system,sans-serif;color:#e6e8ea">ShotSieve has quit. You can close this tab.</div>';
  };
  window.ShotSieveSetup = { show, state: W };
  show(false);
})();
