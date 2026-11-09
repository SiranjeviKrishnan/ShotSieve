# ruff: noqa: E501
"""T23g: the native macOS menu bar of the ShotSieve window.

Every item calls `window.shotsieveMenu(id)` in the page (static/menu.js) through evaluate_js. The ids
are a fixed allow-list (ACTION_IDS + recent:0..9), checked here before anything reaches the page and
JSON-encoded, never interpolated from user data. Menu items carry NO key equivalents (pywebview has no
API for them); the page keeps handling its own keys (P, X, F, Shift+P, Shift+F, Cmd+O, Cmd+L, ?). The
only key the menu owns is Cmd+W on "Close Window", which the page never handled, so nothing fires twice.
The page reports its toggle states, recent folders and which optional tools apply through
JsApi.menu_state; the menu is rebuilt from that (check marks, optional items) on the main thread.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any

RECENT_RE = re.compile(r"^recent:[0-9]$")
PY_ONLY = frozenset({"close"})  # handled here, never sent to the page
CHECKS = ("subject", "chips", "filterbar", "denoise", "crops", "video", "autoxmp")  # ids shown with a check mark when on
OPTIONAL = ("jersey", "lrsync", "auditdone", "undotrash")  # listed only when the page says they apply
ACTION_IDS = frozenset({
    "choose", "autoxmp", "crops", "video", "lropen", "subject", "chips", "filterbar", "cameras", "people", "shotlist",
    "edit", "matchref", "denoise", "compare", "jersey", "lrsync", "auditdone", "help", "setup", "lrcard", "trash", "undotrash",
})
KEY_EQUIV = {"Close Window": "w"}  # Cmd+<key>, applied natively after the bar is built
DEFAULT_STATE: dict[str, Any] = {
    "subject": False, "chips": False, "filterbar": True, "denoise": False, "crops": True, "video": True, "autoxmp": True, "jersey": False, "lrsync": False,
    "auditdone": False, "undotrash": False, "recent": [],
}

Item = tuple  # ("a", id, title) | ("sep",) | ("sub", title, [Item])


def clean_state(raw: Any) -> dict[str, Any]:
    """Whatever the page sent, reduced to known keys and plain types."""
    st = dict(DEFAULT_STATE)
    if not isinstance(raw, dict):
        return st
    for k in (*CHECKS, *OPTIONAL):
        if k in raw:
            st[k] = bool(raw[k])
    rec = raw.get("recent")
    if isinstance(rec, list):
        st["recent"] = [str(x)[:80] for x in rec[:10]]
    return st


def tree(state: dict[str, Any] | None = None) -> dict[str, list[Item]]:
    """{'File': items, 'View': items, 'Tools': items, 'Help': items}; Edit is native (Cut / Copy ...)."""
    st = clean_state(state)

    def chk(i: str, title: str) -> Item:
        return ("a", i, ("✓ " if st[i] else "") + title)

    recent: list[Item] = [("a", f"recent:{n}", name) for n, name in enumerate(st["recent"])]
    file_items: list[Item] = [
        ("a", "choose", "Choose Folder…"),
        ("sub", "Open Recent", recent or [("off", "No recent folders")]),
        ("sep",),
        ("a", "lropen", "Open in Lightroom"),
        ("sep",),
        ("a", "trash", "Move to Trash"),  # T23i: recoverable (Finder's Put Back); the page asks first
        *([("a", "undotrash", "Undo Move to Trash")] if st["undotrash"] else []),
        ("sep",),
        ("a", "close", "Close Window"),
    ]
    view_items: list[Item] = [
        chk("subject", "Subject mode"),
        chk("crops", "Show crop suggestions"),
        chk("chips", "Name chips on grid cells"),
        chk("filterbar", "Filter bar"),
        chk("denoise", "Denoise"),
        ("a", "compare", "Compare with the group"),
    ]
    tools: list[Item] = [
        chk("video", "Include videos"), chk("autoxmp", "Write XMP sidecars automatically"), ("sep",),
        ("a", "cameras", "Cameras…"), ("a", "people", "People…"),
        ("a", "shotlist", "Shot list…"), ("a", "edit", "Edit profile…"),
        ("a", "matchref", "Colour match: set this photo as reference"),
    ]
    if st["jersey"]:
        tools.append(("a", "jersey", "Jersey numbers…"))
    if st["lrsync"]:
        tools.append(("a", "lrsync", "Sync with Lightroom Classic catalog…"))
    if st["auditdone"]:
        tools += [("sep",), ("a", "auditdone", "Finish audit")]
    help_items: list[Item] = [
        ("a", "help", "Keyboard Shortcuts"), ("a", "setup", "Setup Assistant"),
        ("a", "lrcard", "What to do in Lightroom"),
    ]
    return {"File": file_items, "View": view_items, "Tools": tools, "Help": help_items}


def all_ids(items: list[Item]) -> set[str]:
    out: set[str] = set()
    for it in items:
        if it[0] == "a":
            out.add(it[1])
        elif it[0] == "sub":
            out |= all_ids(it[2])
    return out


class MenuController:
    def __init__(self, wv: Any, get_window: Any) -> None:
        self.wv = wv
        self.get_window = get_window
        self.state = dict(DEFAULT_STATE)
        self.patched = False
        self.rebuilds = 0
        self._lock = threading.Lock()

    # -- actions -----------------------------------------------------------------------------
    def dispatch(self, action: str) -> bool:
        if not (action in ACTION_IDS or action in PY_ONLY or RECENT_RE.match(action)):
            return False
        win = self.get_window()
        if win is None:
            return False
        try:
            if action == "close":
                win.destroy()
            else:
                win.evaluate_js("window.shotsieveMenu(" + json.dumps(action) + ")")
        except Exception:  # noqa: BLE001 - a menu click must never end the app
            return False
        return True

    def _action(self, action: str, title: str) -> Any:
        def run() -> None:
            self.dispatch(action)
        run.__name__ = "menu_" + re.sub(r"\W", "_", action)
        return run

    # -- menu objects ------------------------------------------------------------------------
    def _convert(self, items: list[Item]) -> list[Any]:
        wv = self.wv
        out: list[Any] = []
        for it in items:
            if it[0] == "sep":
                out.append(wv.menu.MenuSeparator())
            elif it[0] == "sub":
                out.append(wv.menu.Menu(it[1], self._convert(it[2])))
            elif it[0] == "off":  # a disabled-looking placeholder: does nothing
                out.append(wv.menu.MenuAction(it[1], lambda: None))
            else:
                out.append(wv.menu.MenuAction(it[2], self._action(it[1], it[2])))
        return out

    def menus(self, with_view: bool | None = None) -> list[Any]:
        """The Menu list for webview.start(menu=...). The native View menu (Enter Full Screen) is
        extended in place when the pywebview patch is active, otherwise View is a normal menu."""
        t = tree(self.state)
        show_view = (not self.patched) if with_view is None else with_view
        names = ["File", *(["View"] if show_view else []), "Tools", "Help"]
        return [self.wv.menu.Menu(n, self._convert(t[n])) for n in names]

    def view_menu_items(self) -> list[Any]:
        return self._convert(tree(self.state)["View"])

    # -- page -> menu ------------------------------------------------------------------------
    def update(self, raw: Any) -> None:
        with self._lock:
            new = clean_state(raw)
            if new == self.state:
                return
            self.state = new
        self.rebuild()

    def rebuild(self) -> None:
        self.rebuilds += 1
        try:
            from PyObjCTools import AppHelper
            from webview.platforms import cocoa
        except Exception:  # noqa: BLE001 - no Cocoa (tests, other platforms)
            return
        if getattr(self.wv, "__name__", "") != "webview":
            return

        def apply() -> None:
            try:
                bv = cocoa.BrowserView
                insts = list(bv.instances.values())
                if not insts:
                    return
                menus = self.menus()
                for i in insts:
                    i.menu = menus
                bv.current_menu = menus
                bv.app.setMainMenu_(insts[0]._recreate_menus(menus))
            except Exception:  # noqa: BLE001
                pass

        AppHelper.callAfter(apply)

    # -- Cocoa patch: extend pywebview's own View / Edit menus, order the bar ------------------
    def patch_cocoa(self) -> bool:
        try:
            import AppKit
            from webview.platforms import cocoa
        except Exception:  # noqa: BLE001
            return False
        bv = cocoa.BrowserView
        if getattr(bv, "_shotsieve_menu", None) is self.__class__:
            return True
        ctl = self
        o_view, o_edit, o_re = bv._add_view_menu, bv._add_edit_menu, bv._recreate_menus

        def add_view(view: Any, main: Any) -> None:
            o_view(view, main)
            try:
                vm = main.itemAtIndex_(1).submenu()
                vm.addItem_(AppKit.NSMenuItem.separatorItem())
                view._process_menu_items(ctl.view_menu_items(), vm)
            except Exception:  # noqa: BLE001
                pass

        def add_edit(view: Any, main: Any) -> None:
            o_edit(view, main)
            try:
                em = main.itemAtIndex_(1).submenu()
                em.insertItemWithTitle_action_keyEquivalent_atIndex_("Undo", "undo:", "z", 0)
                redo = em.insertItemWithTitle_action_keyEquivalent_atIndex_("Redo", "redo:", "z", 1)
                redo.setKeyEquivalentModifierMask_(
                    AppKit.NSEventModifierFlagCommand | AppKit.NSEventModifierFlagShift)
                em.insertItem_atIndex_(AppKit.NSMenuItem.separatorItem(), 2)
            except Exception:  # noqa: BLE001
                pass

        def recreate(view: Any, user_menu: Any) -> Any:
            main = o_re(view, user_menu)
            try:  # App, File, Edit, View, Tools, Help; Cmd+W on Close Window
                for n in range(main.numberOfItems()):
                    it = main.itemAtIndex_(n)
                    if it.title() == "File" and n != 1:
                        main.removeItemAtIndex_(n)
                        main.insertItem_atIndex_(it, 1)
                        break
                for n in range(main.numberOfItems()):
                    sm = main.itemAtIndex_(n).submenu()
                    for m in range(sm.numberOfItems() if sm else 0):
                        mi = sm.itemAtIndex_(m)
                        k = KEY_EQUIV.get(str(mi.title()))
                        if k:
                            mi.setKeyEquivalent_(k)
                            mi.setKeyEquivalentModifierMask_(AppKit.NSEventModifierFlagCommand)
            except Exception:  # noqa: BLE001
                pass
            return main

        bv._add_view_menu, bv._add_edit_menu, bv._recreate_menus = add_view, add_edit, recreate
        bv._shotsieve_menu = self.__class__
        self.patched = True
        return True
