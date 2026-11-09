# ruff: noqa: E501, F401, F811
"""T23g: the native menu tree (fake pywebview, no display): every action id is present, every id is
in the page's allow-list, ids are checked before they reach evaluate_js, and state drives the menu."""

from __future__ import annotations

import json
import re
from pathlib import Path

from shotsieve import desktop
from shotsieve import desktop_menu as dm
from tests import fake_webview as fw
from tests.test_desktop import _run_in_thread, fake  # noqa: F401

STATIC = Path(dm.__file__).parent / "server" / "static"
FULL = {"jersey": True, "lrsync": True, "auditdone": True, "undotrash": True, "recent": ["A", "B"]}


def _titles(items):
    out = []
    for it in items:
        out.append(getattr(it, "title", "-"))
        out += _titles(getattr(it, "items", []))
    return out


def _ids_of(ctl):
    """action id behind every MenuAction, via its function name."""
    seen = set()

    def walk(items):
        for it in items:
            if isinstance(it, fw.menu.MenuAction):
                seen.add(it.function.__name__.removeprefix("menu_").replace("recent_", "recent:"))
            elif isinstance(it, fw.menu.Menu):
                walk(it.items)
    for m in ctl.menus(with_view=True):
        walk(m.items)
    return seen


def test_menu_tree_has_every_action_and_each_is_allow_listed():
    js = (STATIC / "menu.js").read_text()
    body = js[js.index("const HANDLERS = {"):js.index("const RECENT")]
    js_ids = set(re.findall(r"^\s{4}(\w+):", body, re.M))
    assert js_ids == set(dm.ACTION_IDS)
    ctl = dm.MenuController(fw, lambda: None)
    ctl.state = dm.clean_state(FULL)
    ids = _ids_of(ctl)
    assert dm.ACTION_IDS | dm.PY_ONLY <= ids
    assert all(i in js_ids or i in dm.PY_ONLY or dm.RECENT_RE.match(i) for i in ids)
    assert [m.title for m in ctl.menus(with_view=True)] == ["File", "View", "Tools", "Help"]
    assert "Open Recent" in _titles(ctl.menus(with_view=True)[0].items)
    assert "Write XMP" not in _titles([i for m in ctl.menus(with_view=True) for i in m.items])  # the button is gone
    def plain(items):
        return [t.removeprefix("✓ ") for t in _titles(items)]

    for needed in ("Choose Folder…", "Open in Lightroom", "Show crop suggestions", "Include videos", "Compare with the group", "Write XMP sidecars automatically", "Close Window", "Cameras…",
                   "People…", "Shot list…", "Edit profile…", "Jersey numbers…",
                   "Sync with Lightroom Classic catalog…", "Keyboard Shortcuts", "Move to Trash", "Undo Move to Trash"):
        assert needed in plain([i for m in ctl.menus(with_view=True) for i in m.items]), needed


def test_state_drives_checks_and_optional_items():
    ctl = dm.MenuController(fw, lambda: None)
    base = _titles([i for m in ctl.menus(with_view=True) for i in m.items])
    assert "Subject mode" in base and "✓ Subject mode" not in base and "Faces in the loupe" not in base and "Name chips on grid cells" in base
    assert "Jersey numbers…" not in base and "Finish audit" not in base
    ctl.update({"chips": True, "subject": True, "jersey": True, "auditdone": True, "recent": ["Wedding B"], "evil": 1})
    t = _titles([i for m in ctl.menus(with_view=True) for i in m.items])
    assert "✓ Name chips on grid cells" in t and "✓ Subject mode" in t
    assert "Jersey numbers…" in t and "Finish audit" in t and "Wedding B" in t
    assert ctl.rebuilds == 1
    ctl.update({"chips": True, "subject": True, "jersey": True, "auditdone": True, "recent": ["Wedding B"]})
    assert ctl.rebuilds == 1  # unchanged state: no rebuild


def test_dispatch_only_sends_allow_listed_ids_json_encoded():
    win = fw.Window("t", "u")
    ctl = dm.MenuController(fw, lambda: win)
    assert ctl.dispatch("cameras") and ctl.dispatch("recent:3")
    assert win.js_calls == ['window.shotsieveMenu("cameras")', 'window.shotsieveMenu("recent:3")']
    for bad in ("x", "recent:10", "cameras');alert(1);('", "", "close ", "constructor"):
        assert not ctl.dispatch(bad)
    assert len(win.js_calls) == 2
    assert json.dumps("cameras") in win.js_calls[0]
    assert dm.clean_state({"recent": ["a" * 500] * 20})["recent"][0] == "a" * 80
    assert len(dm.clean_state({"recent": ["a"] * 20})["recent"]) == 10
    assert ctl.dispatch("close")  # Close Window destroys the window itself


def test_run_app_passes_the_menu_to_start(fake):
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    assert [m.title for m in fw.start_kwargs["menu"]] == ["File", "View", "Tools", "Help"]
    assert desktop.JsApi(fw).menu_state({"chips": True}) is False  # no controller on a bare api
    fw.windows[-1].simulate_close()
    t.join(15)
    assert not t.is_alive()
