/* ShotSieve People view (T23d): every recognised person in a scrollable sheet (Shift+P). Lazy
   thumbnails (IntersectionObserver), editable names, photo / keep counts, merge by drag or by
   "Same person as...", "Not a person" (ignore), click to filter the grid. Works for 300+ people. */
"use strict";

(function () {
  const sheet = $("people-sheet"), overlay = $("overlay-people");
  let data = { people: [], unsorted: 0, ignored: 0 };
  let needle = "";
  let loaded = false;
  let io = null;
  const openSecs = new Map();      // section id -> user's open/closed choice, kept for the session across re-renders
  let anchor = null;               // {id, top, focus}: card to keep at the same place after the next render
  const secOpen = (id, hasItems) => (openSecs.has(id) ? openSecs.get(id) : hasItems);

  const faceUrl = (p, size = 96) => `/api/people/${p.id}/face${q({ size, folder: S.folder, v: (p.cover_id || 0) + "_" + (p.cover_idx || 0) })}`;
  const label = (p) => p.name || (p.n === 1 ? "Unknown" : "Person " + p.id);

  async function load() {
    try { data = await api("/api/people" + q({ folder: S.folder })); loaded = true; }
    catch (e) { data = { people: [], unsorted: 0, ignored: 0, error: e.status === 403 ? "off" : e.message }; loaded = true; }
  }

  function entryHtml(p) {
    return `<div class="pv-person" draggable="true" data-id="${p.id}">` +
      `<img class="pv-face" alt="" width="56" height="56" data-src="${esc(faceUrl(p))}" title="Show their photos">` +
      `<div class="pv-main"><input class="pv-name" data-id="${p.id}" type="text" value="${esc(p.name || "")}" placeholder="Name (${p.n} face${p.n === 1 ? "" : "s"})" aria-label="Name" spellcheck="false">` +
      `<div class="pv-counts"><span class="pv-photos">${p.photos} photo${p.photos === 1 ? "" : "s"}</span> · <span class="pv-keeps">${p.keeps} keep${p.keeps === 1 ? "" : "s"}</span></div>` +
      `<div class="pv-acts"><button class="small pv-show" data-id="${p.id}">Show photos</button>` +
      `<button class="small pv-same" data-id="${p.id}" aria-haspopup="listbox">Same person as…</button>` +
      `<button class="small pv-ignore" data-id="${p.id}" title="This is not a person (a poster, a statue...): leave it out">Not a person</button></div></div></div>`;
  }

  function render() {
    const oldList = sheet.querySelector(".pv-list");
    const keepTop = oldList ? oldList.scrollTop : 0;
    const ae = document.activeElement;
    const keepFocus = ae && sheet.contains(ae) && ae.dataset && ae.dataset.id && /pv-(name|same|show|ignore)/.test(ae.className) ? { id: ae.dataset.id, cls: ae.className.match(/pv-(name|same|show|ignore)/)[0] } : null;
    const keepAnchor = anchor; anchor = null;
    const n = needle.trim().toLowerCase();
    const all = data.people.filter((p) => !n || (p.name || "").toLowerCase().includes(n) || String(p.id) === n);
    const main = all.filter((p) => p.name || p.photos > 1);
    const others = all.filter((p) => !p.name && p.photos <= 1);
    sheet.innerHTML =
      `<div class="pv-head"><h2>People <span class="muted" id="pv-count">${data.people.length} recognised</span></h2>` +
      `<input id="pv-filter" type="search" placeholder="Find a name" value="${esc(needle)}" aria-label="Find a person" spellcheck="false">` +
      `<span class="spacer"></span><button id="pv-close" class="small">Close <kbd>Esc</kbd></button></div>` +
      (data.error === "off" ? `<p class="note">Face grouping is off. Turn it on from the strip under the toolbar.</p>` : data.error ? `<p class="note">${esc(data.error)}</p>` : "") +
      `<div class="pv-list" id="pv-list">` +
      (main.length ? main.map(entryHtml).join("") : (data.people.length ? "" : `<p class="note">No people yet. Faces are grouped when the search index has run; every face gets an entry you can name.</p>`)) +
      (others.length ? `<details id="pv-others" class="pv-others" data-sec="others"${n || secOpen("others", true) ? " open" : ""}><summary>Others (${others.length}): faces seen once (name them, or merge with "Same person as...")</summary><div class="pv-grid">${others.map(entryHtml).join("")}</div></details>` : "") +
      `</div><div class="pv-foot">` +
      (data.ignored ? ` · <span id="pv-ign">${data.ignored} ignored face(s)</span> <button id="pv-restore" class="small">Restore ignored</button>` : "") + `</div>`;
    const list = $("pv-list");
    list.scrollTop = keepTop;
    if (keepAnchor) {
      const card = list.querySelector(`.pv-person[data-id="${keepAnchor.id}"]`);
      if (card) {
        list.scrollTop += card.getBoundingClientRect().top - list.getBoundingClientRect().top - keepAnchor.top;
        const f = keepAnchor.focus ? card.querySelector(keepAnchor.focus === "same" ? ".pv-same" : ".pv-name") : null;
        if (f) f.focus({ preventScroll: true });
      }
    } else if (keepFocus) {
      const f = list.querySelector(`.${keepFocus.cls}[data-id="${keepFocus.id}"]`);
      if (f && document.activeElement !== f) f.focus({ preventScroll: true });
    }
    list.querySelectorAll("details[data-sec]").forEach((d) => d.addEventListener("toggle", () => { if (!needle.trim()) openSecs.set(d.dataset.sec, d.open); }));
    observe(list);
    $("pv-close").onclick = close;
    $("pv-filter").oninput = (e) => { needle = e.target.value; const pos = e.target.selectionStart; render(); const f = $("pv-filter"); f.focus(); f.setSelectionRange(pos, pos); };
    const rs = $("pv-restore");
    if (rs) rs.onclick = async () => {
      try { const r = await api("/api/people/restore_ignored", { method: "POST", body: { folder: S.folder } }); toast(`${r.restored} face(s) restored; they are grouped again at the next index run.`, 5000); window.dispatchEvent(new CustomEvent("people-changed")); } catch (e) { toast(e.message); }
    };
  }

  function observe(list) {
    if (io) io.disconnect();
    io = new IntersectionObserver((ents) => {
      for (const en of ents) {
        if (!en.isIntersecting) continue;
        const img = en.target;
        io.unobserve(img);
        if (img.dataset.src) { img.src = img.dataset.src; delete img.dataset.src; }
      }
    }, { root: list, rootMargin: "300px" });
    list.querySelectorAll("img.pv-face[data-src]").forEach((i) => io.observe(i));
    list.querySelectorAll("details").forEach((d) => d.addEventListener("toggle", () => list.querySelectorAll("img.pv-face[data-src]").forEach((i) => io.observe(i))));
    list.addEventListener("error", (e) => { if (e.target.tagName === "IMG") e.target.style.visibility = "hidden"; }, true);
  }

  const find = (id) => data.people.find((p) => p.id === Number(id));

  async function post(id, body) {
    return api("/api/people/" + id, { method: "POST", body: { folder: S.folder, ...body } });
  }
  // remember the neighbour of card `id` (next, else previous) and where it sits, so render() can keep it in place
  function setAnchor(id, focus, skip) {
    const list = $("pv-list"), card = list && list.querySelector(`.pv-person[data-id="${id}"]`);
    if (!card) return;
    const cards = [...list.querySelectorAll(".pv-person")], i = cards.indexOf(card);
    const ok = (c) => c && c.dataset.id !== String(skip);   // not the merge target: its count changes and it may move
    const nb = [cards[i + 1], cards[i + 2], cards[i - 1], cards[i - 2]].find(ok);
    if (nb) anchor = { id: nb.dataset.id, top: nb.getBoundingClientRect().top - list.getBoundingClientRect().top, focus };
  }
  async function changed(msg) {
    if (msg) toast(msg);
    await load();
    window.dispatchEvent(new CustomEvent("people-changed"));
    if (!overlay.hidden) render();
  }
  async function mergeInto(a, b, how) {
    try {
      if (a.name && !b.name) await post(b.id, { name: a.name });   // the merged person keeps the name that was given
      await post(a.id, { merge_into: b.id });
      setAnchor(a.id, how, b.id);
      await changed(`Merged ${label(a)} into ${label(b)}`);
    } catch (e) { toast(e.message, 5000); }
  }

  async function notAPerson(p) {
    try {
      if (window.ShotFaces) window.ShotFaces.hidePerson(p.id);
      const r = await post(p.id, { ignore: true });
      setAnchor(p.id, "name");
      await changed();
      toast(`${label(p)} will be left out of People. `, 9000);
      const t = $("toast"), b = document.createElement("button");
      b.className = "small"; b.textContent = "Undo";
      b.onclick = async () => {
        try {
          await api("/api/people/unignore", { method: "POST", body: { folder: S.folder, faces: r.ignored_faces || [] } });
          t.hidden = true;
          if (window.ShotFaces) window.ShotFaces.unhide();
          toast("Restored; the faces are grouped again at the next index run.", 5000);
          window.dispatchEvent(new CustomEvent("people-changed"));
        } catch (err) { toast(err.message, 5000); }
      };
      t.appendChild(b);
    } catch (err) { toast(err.message, 5000); }
  }

  function chooser(anchorBtn, p) {
    const old = sheet.querySelector(".pv-chooser");
    if (old) old.remove();
    const box = document.createElement("div");
    box.className = "pv-chooser";
    box.setAttribute("role", "listbox");
    box.innerHTML = `<input type="search" placeholder="Same person as… (type a name)" aria-label="Choose the person" spellcheck="false"><div class="pv-opts"></div>`;
    const opts = box.querySelector(".pv-opts"), inp = box.querySelector("input");
    const fill = () => {
      const n = inp.value.trim().toLowerCase();
      const list = data.people.filter((x) => x.id !== p.id && (!n || (x.name || "").toLowerCase().includes(n) || ("person " + x.id).includes(n))).slice(0, 60);
      opts.innerHTML = list.map((x) => `<button class="pv-opt" role="option" data-id="${x.id}">${esc(label(x))} <span class="muted">${x.photos} photo${x.photos === 1 ? "" : "s"}</span></button>`).join("") || `<div class="none">No match</div>`;
    };
    inp.addEventListener("input", fill);
    opts.addEventListener("click", (e) => {
      const b = e.target.closest(".pv-opt");
      if (!b) return;
      box.remove();
      mergeInto(p, find(b.dataset.id), "same");
    });
    fill();
    const host = anchorBtn.closest(".pv-person");
    host.appendChild(box);
    inp.focus();
    box.scrollIntoView({ block: "nearest" });
  }

  sheet.addEventListener("click", async (e) => {
    const t = e.target;
    const same = t.closest(".pv-same");
    if (same) { chooser(same, find(same.dataset.id)); return; }
    if (!t.closest(".pv-chooser")) { const c = sheet.querySelector(".pv-chooser"); if (c) c.remove(); }
    const show = t.closest(".pv-show") || (t.matches("img.pv-face") ? t.closest(".pv-person") : null);
    if (show) {
      const id = show.dataset.id;
      return filterPerson(find(id));
    }
    const ign = t.closest(".pv-ignore");
    if (ign) {
      await notAPerson(find(ign.dataset.id));
    }
  });
  sheet.addEventListener("change", async (e) => {
    const inp = e.target.closest(".pv-name");
    if (!inp) return;
    try { await post(inp.dataset.id, { name: inp.value }); setAnchor(inp.dataset.id, null); await changed(inp.value.trim() ? `Named ${inp.value.trim()}` : "Name cleared"); } catch (err) { toast(err.message, 5000); }
  });
  sheet.addEventListener("keydown", (e) => {
    if (e.target.classList.contains("pv-name") && e.key === "Enter") { e.preventDefault(); e.target.blur(); }
  });
  // drag one person onto another = the same person
  let dragId = null;
  sheet.addEventListener("dragstart", (e) => {
    const c = e.target.closest && e.target.closest(".pv-person");
    if (!c) return;
    dragId = c.dataset.id;
    e.dataTransfer.setData("text/plain", dragId);
    e.dataTransfer.effectAllowed = "move";
    c.classList.add("dragging");
  });
  sheet.addEventListener("dragend", () => { dragId = null; sheet.querySelectorAll(".dragging, .drop").forEach((x) => x.classList.remove("dragging", "drop")); });
  sheet.addEventListener("dragover", (e) => {
    const c = e.target.closest && e.target.closest(".pv-person");
    if (!c || dragId === null || c.dataset.id === dragId) return;
    e.preventDefault();
    sheet.querySelectorAll(".drop").forEach((x) => { if (x !== c) x.classList.remove("drop"); });
    c.classList.add("drop");
  });
  sheet.addEventListener("drop", (e) => {
    const c = e.target.closest && e.target.closest(".pv-person");
    if (!c || dragId === null || c.dataset.id === dragId) return;
    e.preventDefault();
    const a = find(dragId), b = find(c.dataset.id);
    dragId = null;
    if (a && b) mergeInto(a, b, "name");
  });

  async function open() {
    if (!S.folder) return;
    overlay.hidden = false;
    needle = "";
    sheet.innerHTML = `<h2>People</h2><p class="note">Loading…</p>`;
    await load();
    render();
  }
  function close() { overlay.hidden = true; if (io) io.disconnect(); }
  function toggle() { if (overlay.hidden) open(); else close(); }

  async function filterPerson(p) {
    if (!p) return;
    close();
    if (window.ShotFilters && p.id !== undefined && p.id !== null) {   // T23e: one filter state (People filter)
      if (S.search) clearSearch();
      window.ShotFilters.setPeople([p.id], "any");
      return;
    }
    const dup = data.people.filter((x) => p.name && (x.name || "").toLowerCase() === p.name.toLowerCase()).length > 1;
    if (p.name && !dup) {                                   // the existing person:"Name" search path
      $("search").value = `person:"${p.name}"`;
      return runSearch();
    }
    try {
      const r = await api(`/api/people/${p.id}/images` + q({ folder: S.folder }));
      S.search = new Map(r.ids.filter((i) => S.byId.has(i)).map((i) => [i, 0]));
      S.searchQ = label(p);
      $("search").value = "";
      relayout(true);
      if (S.flat.length) select(S.flat[0]); else { S.sel = null; renderDetail(); }
      toast(`${S.search.size} photo(s) of ${label(p)} (Esc shows everything)`, 5000);
    } catch (e) { toast(e.message, 5000); }
  }

  window.addEventListener("people-changed", async () => {
    if (overlay.hidden || sheet.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;
    await load(); render();
  });
  document.addEventListener("keydown", (e) => {            // capture: Shift+P opens People; keys do not leak to the grid behind it
    const typing = e.target.closest && e.target.closest("input, textarea, select");
    if (e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey && e.key.toLowerCase() === "p" && !typing) {
      e.preventDefault(); e.stopPropagation(); toggle(); return;
    }
    if (!overlay.hidden && !typing && !["Escape", "Tab", "Enter"].includes(e.key) && !(e.metaKey || e.ctrlKey)) e.stopPropagation();
  }, true);
  $("btn-people").addEventListener("click", toggle);

  /* ---------- right-click / Ctrl-click menu on a person (People sheet, people strip, Key faces) ---------- */
  let cm = null;
  function cmClose() { if (cm) { cm.remove(); cm = null; } }
  function cmPlace(x, y) {
    const r = cm.getBoundingClientRect();
    cm.style.left = Math.max(4, Math.min(x, window.innerWidth - r.width - 4)) + "px";
    cm.style.top = Math.max(4, Math.min(y, window.innerHeight - r.height - 4)) + "px";
  }
  async function personMenu(x, y, id, opts = {}) {
    cmClose();
    await load();
    const p = find(id);
    if (!p) return;
    cm = document.createElement("div");
    cm.className = "pv-cm";
    cm.setAttribute("role", "menu");
    cm.dataset.id = String(p.id);
    document.body.appendChild(cm);
    const view = (html, after) => { cm.innerHTML = html; cmPlace(x, y); if (after) after(); };
    const main = () => view(
      `<div class="pv-cm-title">${esc(label(p))}</div>` +
      `<button role="menuitem" data-act="same">Same person as…</button><button role="menuitem" data-act="ignore">Not a person</button>` +
      (opts.face ? `<button role="menuitem" data-act="notthis">Not this person</button>` : ""),
      () => cm.querySelector("button").focus());
    const same = () => {
      view(`<input type="search" placeholder="Same person as… (type a name)" aria-label="Choose the person" spellcheck="false"><div class="pv-opts"></div>`, () => {
        const inp = cm.querySelector("input"), opts = cm.querySelector(".pv-opts");
        const fill = () => {
          const n = inp.value.trim().toLowerCase();
          const rows = data.people.filter((o) => o.id !== p.id && (!n || (o.name || "").toLowerCase().includes(n) || ("person " + o.id).includes(n)))
            .sort((a, b) => (!!b.name - !!a.name) || (a.name || "").localeCompare(b.name || "") || b.photos - a.photos).slice(0, 80);
          opts.innerHTML = rows.map((o) => `<button class="pv-opt" role="option" data-id="${o.id}"><img alt="" width="28" height="28" src="${esc(faceUrl(o, 56))}"><span>${esc(label(o))}</span> <span class="muted">${o.photos} photo${o.photos === 1 ? "" : "s"}</span></button>`).join("") || `<div class="none">No match</div>`;
        };
        inp.addEventListener("input", fill);
        fill();
        inp.focus();
      });
    };
    main();
    cm.addEventListener("click", async (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      if (b.dataset.act === "same") return same();
      if (b.classList.contains("pv-opt")) { const t = find(b.dataset.id); cmClose(); setAnchor(p.id, "same", b.dataset.id); if (t) mergeInto(p, t, "same"); return; }
      if (b.dataset.act === "ignore") { cmClose(); return notAPerson(p); }
      if (b.dataset.act === "notthis") {
        cmClose();
        try {
          await api(`/api/image/${opts.face.id}/faces/${opts.face.idx}/detach`, { method: "POST", body: { folder: S.folder } });
          if (window.ShotFaces) window.ShotFaces.invalidate();
          await changed("Detached: this face is no longer part of that person.");
        } catch (err) { toast(err.message, 5000); }
      }
    });
    cm.addEventListener("keydown", async (e) => {
      if ((e.key === "ArrowDown" || e.key === "ArrowUp") && !e.target.matches("input")) {
        const bs = [...cm.querySelectorAll("button")], i = bs.indexOf(document.activeElement);
        const nx = bs[(i + (e.key === "ArrowDown" ? 1 : -1) + bs.length) % bs.length];
        if (nx) { e.preventDefault(); nx.focus(); }
      }
    });
  }
  document.addEventListener("keydown", (e) => { if (cm && e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cmClose(); } }, true);
  document.addEventListener("mousedown", (e) => { if (cm && !cm.contains(e.target)) cmClose(); }, true);
  window.addEventListener("resize", cmClose);
  document.addEventListener("contextmenu", (e) => {
    const t = e.target instanceof Element ? e.target : e.target.parentElement;
    if (!t) return;
    let id = null;
    const c = t.closest(".pv-person"), s = t.closest("#people-strip .person"), k = t.closest("#keyfaces .kf-face");
    if (c && sheet.contains(c)) id = c.dataset.id;
    else if (s) id = (s.querySelector("[data-id]") || {}).dataset?.id;
    else if (k) id = k.dataset.pid;
    if (id === null || id === undefined || t.closest("#d-faces-list")) return;
    e.preventDefault(); e.stopPropagation();
    personMenu(e.clientX, e.clientY, id);
  }, true);

  window.ShotPeople = { open, close, filterPerson, faceUrl, personMenu };
})();
