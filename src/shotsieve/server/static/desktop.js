/* Native-window hooks: only active inside the ShotSieve Mac app (window.pywebview exists).
   - links to other sites open in the default browser; the window never leaves the UI
   - window.shotsieveDesktop.pick(kind) -> native folder / .lrcat dialog (null = cancelled) */
(function () {
  "use strict";
  function bridge() { return window.pywebview && window.pywebview.api ? window.pywebview.api : null; }
  window.shotsieveDesktop = {
    available() { return !!(bridge() && bridge().pick); },
    onDrop(plan) { window.dispatchEvent(new CustomEvent("shotsieve-drop", { detail: plan })); }, // called from Python
    async pick(kind) { const b = bridge(); return b && b.pick ? b.pick(kind) : null; },
  };
  function sync() { document.documentElement.classList.toggle("native", window.shotsieveDesktop.available()); }
  sync();
  window.addEventListener("pywebviewready", sync); // the bridge appears a moment after load in the real app
  function external(url) {
    try { const u = new URL(url, location.href); return /^https?:$/.test(u.protocol) && u.origin !== location.origin ? u.href : null; }
    catch (e) { return null; }
  }
  function toBrowser(href) {
    const b = bridge();
    if (b && b.open_external) b.open_external(href);
  }
  document.addEventListener("click", (e) => {
    const a = e.target.closest && e.target.closest("a[href]");
    if (!a) return;
    const href = external(a.getAttribute("href"));
    if (href && (window.pywebview || a.target === "_blank")) { e.preventDefault(); toBrowser(href); }
  }, true);
  const nativeOpen = window.open;
  window.open = function (url) {
    const href = url && external(url);
    if (href && window.pywebview) { toBrowser(href); return null; }
    return nativeOpen.apply(window, arguments);
  };
})();
