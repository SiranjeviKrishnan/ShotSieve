# ruff: noqa: E501
"""Native (Mac app) UI behaviour with a stubbed `window.pywebview` bridge: Finder pickers replace
the typed path boxes, plus drag and drop. Plain-browser behaviour is covered by test_ui.py."""

from __future__ import annotations

import json
import shutil

import pytest

from tests.test_ui import Server, Ui

pytestmark = pytest.mark.ui

STUB = """window.__picks = %s;
window.pywebview = {api: {pick: async (k) => (window.__picks[k] || null), open_external: () => true}};"""


def _ctx(browser, picks):
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    ctx.add_init_script(STUB % json.dumps(picks))
    return ctx


def _copy(shoot_dir, dest):
    shutil.copytree(shoot_dir, dest, ignore=shutil.ignore_patterns(".shotsieve"))
    return dest


def test_choose_folder_recent_and_hidden_text_boxes(browser, tmp_path, shoot_dir, models_ready):
    a, b = _copy(shoot_dir, tmp_path / "empty_a"), _copy(shoot_dir, tmp_path / "Wedding B")
    srv = Server(a)
    ctx = _ctx(browser, {"folder": str(b)})
    try:
        u = Ui(ctx.new_page(), srv).open(ready=False)
        page = u.page
        page.wait_for_selector("#btn-choose")
        assert page.is_hidden("#folder")
        assert page.is_visible("#btn-choose") and page.is_visible("#btn-recent")
        assert page.locator("#btn-cull").count() == 0  # T25a: the cull starts by itself
        page.click("#btn-choose")  # pick -> chip -> cull
        page.wait_for_selector(".cell", timeout=180_000)
        assert page.inner_text("#folder-chip") == "Wedding B"
        assert page.get_attribute("#folder-chip", "title") == str(b)
        assert page.input_value("#folder") == str(b)  # hidden input still feeds app.js
        # layout fits at 1100x720
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        # recent menu lists it and re-culls
        page.click("#btn-recent")
        page.wait_for_selector("#recent-menu button")
        assert "Wedding B" in page.inner_text("#recent-menu")
        page.click("#recent-menu button >> nth=0")
        page.wait_for_selector("#progress:not([hidden])")
        page.wait_for_function("S.culling === false", timeout=180_000)
        u.expect_uncullled_404()
        u.check()
    finally:
        ctx.close()
        srv.stop()


def test_lr_catalog_and_edit_pickers(browser, tmp_path, shoot_dir, models_ready, make_fixtures):
    shoot = _copy(shoot_dir, tmp_path / "S")
    cat = tmp_path / "Lightroom" / "My Catalog.lrcat"
    cat.parent.mkdir()
    make_fixtures.build_lrcat(
        cat, [dict(root=str(shoot), base=p.stem, ext="jpg", rating=3, pick=0, label="")
              for p in sorted(shoot.glob("*.jpg"))])
    other = tmp_path / "xmp_one"
    other.mkdir()
    srv = Server(shoot, env={"SHOTSIEVE_FAKE_LR_APPS": "classic"})
    ctx = _ctx(browser, {"folder": str(shoot), "lrcat": str(cat)})
    try:
        u = Ui(ctx.new_page(), srv).open(ready=False)
        page = u.page
        page.click("#btn-choose")
        page.wait_for_selector(".cell", timeout=180_000)
        page.evaluate("shotsieveMenu('lrsync')")
        page.wait_for_selector("#lr-pick")
        assert page.is_hidden("#lr-path")
        page.click("#lr-pick")
        assert page.inner_text("#lr-name") == "My Catalog.lrcat"
        page.click("#lr-dry")
        page.wait_for_function("document.getElementById('lr-msg').textContent.startsWith('Preview')", timeout=120_000)
        page.keyboard.press("Escape")
        # edit profile: catalog button + several XMP folders as removable chips
        page.evaluate("shotsieveMenu('edit')")
        page.wait_for_selector("#edit-cat-pick")
        assert page.is_hidden("#edit-cat") and page.is_hidden("#edit-xmp")
        page.evaluate(f"window.__picks.folder = {json.dumps(str(other))}")
        page.click("#edit-xmp-add")
        page.evaluate(f"window.__picks.folder = {json.dumps(str(shoot))}")
        page.click("#edit-xmp-add")
        assert page.locator(".xmp-chip").count() == 2
        page.click(".xmp-chip button >> nth=0")
        assert page.locator(".xmp-chip").count() == 1 and "S" == page.inner_text(".xmp-chip span")
        u.expect_uncullled_404()
        u.check()
    finally:
        ctx.close()
        srv.stop()


def test_drop_overlay_folder_and_file_list(browser, tmp_path, shoot_dir, models_ready):
    shoot = _copy(shoot_dir, tmp_path / "S")
    srv = Server(shoot)
    ctx = _ctx(browser, {})
    drag = """(t) => { const dt = new DataTransfer(); dt.items.add(new File(["x"], "a.jpg", {type: "image/jpeg"}));
        window.dispatchEvent(new DragEvent(t, {dataTransfer: dt, bubbles: true, cancelable: true})); }"""
    try:
        u = Ui(ctx.new_page(), srv).open(ready=False)
        page = u.page
        page.wait_for_selector("#btn-choose")
        page.evaluate(drag, "dragenter")
        assert page.is_visible("#drop-overlay") and "Drop a folder or photos" in page.inner_text("#drop-overlay")
        page.evaluate(drag, "dragleave")
        assert page.is_hidden("#drop-overlay")
        page.evaluate(drag, "dragenter")
        page.evaluate(drag, "drop")  # native: the real paths come from Python, no browser toast
        assert page.is_hidden("#drop-overlay")
        # Python hands the plan over: only 3 files of the folder are culled
        files = [str(shoot / n) for n in ("landscape.jpg", "black.jpg", "burst0_0.jpg")]
        page.evaluate("(p) => window.shotsieveDesktop.onDrop(p)", {"folder": str(shoot), "files": files, "notes": ["2 unsupported file(s) ignored."]})
        page.wait_for_selector("#progress:not([hidden])")
        assert "unsupported" in page.inner_text("#toast")
        page.wait_for_function("document.querySelectorAll('.cell').length === 3", timeout=180_000)
        page.wait_for_function("document.querySelectorAll('.cell.dropped').length === 3")  # highlighted
        # a drop while culling is ignored with a note
        page.wait_for_function("S.culling === false", timeout=60_000)
        page.evaluate("S.culling = true")
        page.evaluate("(p) => window.shotsieveDesktop.onDrop(p)", {"folder": str(shoot), "files": None, "notes": []})
        assert "A cull is running" in page.inner_text("#toast")
        # whole folder drop culls everything
        page.evaluate("S.culling = false")
        page.evaluate("(p) => window.shotsieveDesktop.onDrop(p)", {"folder": str(shoot), "files": None, "notes": []})
        page.wait_for_function("document.querySelectorAll('.cell').length > 3", timeout=180_000)
        u.expect_uncullled_404()
        u.check()
    finally:
        ctx.close()
        srv.stop()


def test_browser_mode_keeps_text_box_and_drop_toast(browser, tmp_path, shoot_dir, models_ready):
    shoot = _copy(shoot_dir, tmp_path / "S")
    srv = Server(shoot)
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    try:
        u = Ui(ctx.new_page(), srv).open(ready=False)
        page = u.page
        page.wait_for_selector("#folder")
        assert page.is_visible("#folder") and page.is_hidden("#btn-choose")
        page.evaluate("""() => { const dt = new DataTransfer(); dt.items.add(new File(["x"], "a.jpg"));
            window.dispatchEvent(new DragEvent("drop", {dataTransfer: dt, bubbles: true, cancelable: true})); }""")
        assert "Drop works in the ShotSieve app" in page.inner_text("#toast")
        u.expect_uncullled_404()
        u.check()
    finally:
        ctx.close()
        srv.stop()
