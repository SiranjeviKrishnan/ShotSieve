/* ShotSieve faces (T23d): the "Faces" row of the detail panel, the click popover (name,
   merge, show photos, not this person) and the optional name chips on grid cells. Everything that
   reaches innerHTML goes through esc(). Esc closes the popover first (preventDefault). */
"use strict";

(function () {
  const KEY_CHIPS = "shotsieve.chips";
  let chipsOn = false;
  try { chipsOn = localStorage.getItem(KEY_CHIPS) === "1"; } catch (e) { /* ignore */ }
  const cache = new Map();        // image id -> payload of /api/image/{id}/faces
  let chipMap = new Map();        // image id -> [names]
  const SOFT = 0.25;              // normalised face sharpness below this reads "soft"
  const url = (id, path) => `/api/image/${id}/${path}${q({ folder: S.folder })}`;
  const faceThumb = (id, k, size = 64) => `/api/image/${id}/face/${k}${q({ size, folder: S.folder })}`;

  async function load(id) {
    if (cache.has(id)) return cache.get(id);
    const p = api(url(id, "faces")).catch(() => ({ faces: [], people_consent: false }));
    cache.set(id, p);                          // share the in-flight request
    const d = await p;
    cache.set(id, d);
    return d;
  }
  const hidden = new Set();       // `${image}:${idx}` of faces just marked "Not a person" (hidden from the row this session)
  const have = (id) => { const v = cache.get(id); return v && !(v instanceof Promise) ? v : null; };

  function pct(v) { return v === null || v === undefined ? "n/a" : Math.round(v * 100) + " %"; }
  function scoreText(f) {
    const bits = [];
    bits.push("sharp " + pct(f.sharp));
    bits.push("eyes open " + pct(f.eyes_open));
    if (f.smile !== null && f.smile !== undefined) bits.push("smile " + pct(f.smile));
    return bits.join(" · ");
  }
  const driveText = (f) => !f.drove ? "" : f.drove_reason === "eyes_closed" ? "This face's closed eyes counted against the photo." : "This face being soft counted against the photo.";
  const labelOf = (f) => f.name || (f.person_id ? "Person " + f.person_id : f.important ? "Unknown" : "Small face");

  /* T23h: the loupe face overlay and its F key are gone (S = subject mode replaced them); the faces
     stay in the detail panel row, the popover below, the people strip, People and Key Faces. */
  const pop = $("face-pop");

  /* ---------- detail panel row ---------- */

  async function panel(im) {
    const box = $("d-faces");
    if (!im || !im.n_faces) { box.hidden = true; return; }
    const id = im.id;
    const d = await load(id);
    if (S.sel !== id) return;
    const list = d.faces.filter((f) => f.important && !hidden.has(id + ":" + f.idx));
    box.hidden = !list.length;
    $("d-faces-list").innerHTML = list.map((f) => {
      const eyes = f.eyes_open === null ? "" : f.eyes_open < 0.45 ? '<span class="fi bad" title="Eyes closed">eyes closed</span>' : '<span class="fi ok" title="Eyes open">eyes open</span>';
      const sharp = f.sharp === null ? "" : f.sharp < SOFT ? '<span class="fi bad" title="Soft focus">soft</span>' : '<span class="fi ok" title="Sharp">sharp</span>';
      return `<button class="fchip${f.drove ? " drove" : ""}" data-idx="${f.idx}" title="${esc(scoreText(f))}">` +
        `<img alt="" loading="lazy" src="${esc(faceThumb(id, f.idx))}"><span class="fn">${esc(labelOf(f))}</span>` +
        `<span class="fis">${eyes}${sharp}</span></button>`;
    }).join("");
  }
  $("d-faces-list").addEventListener("click", (e) => {
    const b = e.target.closest(".fchip");
    if (!b || S.sel === null) return;
    const d = have(S.sel);
    const f = d && d.faces.find((x) => String(x.idx) === b.dataset.idx);
    if (f) openPop(S.sel, f, b.getBoundingClientRect());
  });

  $("d-faces-list").addEventListener("contextmenu", (e) => {   // right-click / Ctrl-click: the person menu (Same person as…, Not a person, …)
    const b = e.target.closest(".fchip");
    if (!b || S.sel === null) return;
    const d = have(S.sel);
    const f = d && d.faces.find((x) => String(x.idx) === b.dataset.idx);
    if (!f || f.person_id === null || f.person_id === undefined || !window.ShotPeople) return;
    e.preventDefault(); e.stopPropagation();
    closePop();
    window.ShotPeople.personMenu(e.clientX, e.clientY, f.person_id, { face: { id: S.sel, idx: f.idx } });
  }, true);

  /* ---------- popover ---------- */

  function closePop() { if (pop.hidden) return false; pop.hidden = true; pop.textContent = ""; return true; }
  function place(rect) {
    const pw = pop.offsetWidth, ph = pop.offsetHeight;
    let x = rect.left + rect.width / 2 - pw / 2, y = rect.bottom + 8;
    if (y + ph > innerHeight - 8) y = Math.max(8, rect.top - ph - 8);
    x = Math.min(Math.max(8, x), innerWidth - pw - 8);
    pop.style.left = x + "px"; pop.style.top = y + "px";
  }
  async function peopleNow() {
    try { return (await api("/api/people" + q({ folder: S.folder }))).people; } catch (e) { return []; }
  }
  async function afterChange(msg) {
    cache.clear();
    window.dispatchEvent(new CustomEvent("people-changed"));
    if (msg) toast(msg);
    if (S.sel !== null) { const im = S.byId.get(S.sel); panel(im); }
  }

  async function openPop(id, f, rect) {
    const people = f.person_id !== null || f.face_idx !== null ? await peopleNow() : [];
    pop.hidden = false;
    const named = !!f.name;
    const consent = (have(id) || {}).people_consent !== false;
    pop.innerHTML =
      `<div class="fp-head"><img alt="" src="${esc(faceThumb(id, f.idx, 72))}"><div><b>${esc(labelOf(f))}</b>` +
      `<div class="fp-scores">${esc(scoreText(f))}</div>${f.drove ? `<div class="fp-drove">${esc(driveText(f))}</div>` : ""}</div></div>` +
      (!consent ? `<p class="note">Face grouping is off, so faces cannot be named.</p><div class="fp-row"><button id="fp-consent">Turn on face grouping</button></div>`
        : f.face_idx === null ? `<p class="note">This face is too small to be recognised.</p>`
        : `<label class="fp-lab">${f.person_id ? "Name this person" : "This face is not in a group yet. Name it"}` +
          `<input id="fp-name" type="text" list="fp-names" autocomplete="off" spellcheck="false" value="${esc(f.name || "")}" placeholder="name"></label>` +
          `<datalist id="fp-names">${[...new Set(people.map((p) => p.name).filter(Boolean))].map((n) => `<option value="${esc(n)}">`).join("")}</datalist>` +
          `<div id="fp-merge" class="fp-merge" hidden></div>` +
          `<div class="fp-row"><button id="fp-save" class="primary small">Save</button>` +
          (f.person_id ? `<button id="fp-show" class="small">Show all photos${named ? " of " + esc(f.name) : ""}</button><button id="fp-not" class="small">Not this person</button>` : "") + `</div>`) +
      `<div class="fp-row"><button id="fp-close" class="small">Close <kbd>Esc</kbd></button></div>`;
    place(rect);
    const inp = $("fp-name");
    if (inp) inp.focus();
    $("fp-close").onclick = closePop;
    const cs = $("fp-consent");
    if (cs) cs.onclick = async () => { await api("/api/people/consent", { method: "POST", body: { folder: S.folder, on: true } }); closePop(); afterChange("Face grouping is on. Grouping runs with the search index."); };
    if (!inp) return;
    const other = () => {
      const v = inp.value.trim().toLowerCase();
      return v ? people.find((p) => (p.name || "").toLowerCase() === v && p.id !== f.person_id) : null;
    };
    const doSave = async (mode) => {
      const name = inp.value.trim();
      const match = other();
      try {
        if (f.person_id) {
          if (match && mode === "merge") {
            await api("/api/people/" + f.person_id, { method: "POST", body: { folder: S.folder, merge_into: match.id } });
            closePop(); return afterChange(`Merged into ${match.name}`);
          }
          if (match && mode !== "rename") { showMerge(match); return; }
          await api("/api/people/" + f.person_id, { method: "POST", body: { folder: S.folder, name } });
        } else {
          await api(`/api/image/${id}/faces/${f.idx}/assign`, { method: "POST", body: { folder: S.folder, person_id: match ? match.id : null, name } });
        }
        closePop(); afterChange(name ? `Named ${name}` : "Name cleared");
      } catch (e) { toast(e.message, 5000); }
    };
    const showMerge = (m) => {
      const box = $("fp-merge");
      box.hidden = false;
      box.innerHTML = `<span>There is already a person called ${esc(m.name)}.</span> <button id="fp-do-merge" class="small primary">Merge into ${esc(m.name)}</button> <button id="fp-do-rename" class="small">Keep separate</button>`;
      $("fp-do-merge").onclick = () => doSave("merge");
      $("fp-do-rename").onclick = () => doSave("rename");
    };
    $("fp-save").onclick = () => doSave();
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); doSave(); } });
    inp.addEventListener("input", () => { $("fp-merge").hidden = true; });
    const show = $("fp-show");
    if (show) show.onclick = () => { closePop(); if (window.ShotPeople) window.ShotPeople.filterPerson({ id: f.person_id, name: f.name }); };
    const not = $("fp-not");
    if (not) not.onclick = async () => {
      try {
        await api(`/api/image/${id}/faces/${f.idx}/detach`, { method: "POST", body: { folder: S.folder } });
        closePop(); afterChange("Detached: this face is no longer part of that person.");
      } catch (e) { toast(e.message, 5000); }
    };
  }
  document.addEventListener("keydown", (e) => {     // capture: the popover gets Esc before the loupe / sheets
    if (e.key !== "Escape" || pop.hidden) return;
    e.preventDefault(); e.stopPropagation();
    closePop();
  }, true);
  document.addEventListener("mousedown", (e) => { if (!pop.hidden && !pop.contains(e.target) && !e.target.closest(".face-box, .fchip")) closePop(); }, true);

  /* ---------- grid chips ---------- */
  async function loadChips() {
    if (!chipsOn || !S.peopleOn) { chipMap = new Map(); return; }
    try { chipMap = new Map(Object.entries((await api("/api/people/chips" + q({ folder: S.folder }))).chips).map(([k, v]) => [Number(k), v])); }
    catch (e) { chipMap = new Map(); }
  }
  function chipHtml(id) {
    if (!chipsOn) return "";
    const n = chipMap.get(id);
    if (!n || !n.length) return "";
    const shown = n.slice(0, 2).map((x) => `<span>${esc(x)}</span>`).join("") + (n.length > 2 ? `<span>+${n.length - 2}</span>` : "");
    return `<div class="nchips">${shown}</div>`;
  }
  async function setChips(v) {
    chipsOn = !!v;
    try { localStorage.setItem(KEY_CHIPS, chipsOn ? "1" : "0"); } catch (e) { /* ignore */ }
    $("mi-chips").setAttribute("aria-checked", String(chipsOn));
    await loadChips();
    renderRows(true);
  }
  $("mi-chips").addEventListener("click", () => setChips(!chipsOn));
  $("mi-chips").setAttribute("aria-checked", String(chipsOn));
  window.addEventListener("people-changed", async () => { cache.clear(); if (S.sel !== null && S.byId.get(S.sel)) panel(S.byId.get(S.sel)); await loadChips(); if (chipsOn) renderRows(true); });
  setInterval(() => { if (chipsOn && S.peopleOn && !chipMap.size) loadChips().then(() => renderRows(true)); }, 4000);

  function hidePerson(pid) {
    const d = S.sel !== null ? have(S.sel) : null;
    if (d) d.faces.forEach((f) => { if (f.person_id === pid) hidden.add(S.sel + ":" + f.idx); });
  }
  function unhide() { hidden.clear(); if (S.sel !== null && S.byId.get(S.sel)) panel(S.byId.get(S.sel)); }

  window.ShotFaces = { hidePerson, unhide, panel, chipHtml, load, closePop, invalidate: () => { cache.clear(); } };
})();
