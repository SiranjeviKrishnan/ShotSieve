# ruff: noqa: E501, F401, F811
"""T23g: the top-right shows only "?" (native) and "?" + the hamburger (browser); every action is
reachable through window.shotsieveMenu (native menu bar) or the hamburger (browser)."""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import expect

from tests.test_ui import Ui, culled, server  # noqa: F401

pytestmark = pytest.mark.ui

STUB = """window.__calls = [];
window.pywebview = {api: {pick: async () => null, open_external: () => true, menu_state: (s) => { window.__menu = s; return true; }}};"""

# id -> (selector that must become visible, key to close) ; toggles are checked separately
SHEETS = {
    "cameras": "#overlay-cameras", "people": "#overlay-people", "shotlist": "#overlay-shotlist",
    "edit": "#overlay-edit", "help": "#overlay-help", "setup": "#wizard",
}


def _top_right_buttons(page):
    return page.evaluate("""() => [...document.querySelectorAll('#topbar button, #topbar a, #topbar [role=button]')]
      .filter(b => b.offsetParent !== null && b.getBoundingClientRect().left > window.innerWidth * 0.6)
      .map(b => b.id)""")


def _top_right_text(page):
    return page.evaluate("""() => { const out = []; const w = document.createTreeWalker(document.getElementById('topbar'), NodeFilter.SHOW_TEXT);
      for (let n; (n = w.nextNode());) { const t = n.textContent.trim(); const e = n.parentElement;
        if (t && e.offsetParent !== null && e.getBoundingClientRect().left > window.innerWidth * 0.6) out.push(t); } return out; }""")


def test_native_top_right_is_only_the_question_mark_and_menu_ids_work(browser, server):
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    ctx.add_init_script(STUB)
    try:
        u = Ui(ctx.new_page(), server).open()
        p = u.page
        assert _top_right_buttons(p) == ["btn-help"], _top_right_buttons(p)
        assert _top_right_text(p) == ["?"]
        assert p.is_hidden("#btn-hamburger")
        for gone in ("#btn-cameras", "#btn-people", "#btn-shotlist", "#btn-edit", "#btn-xmp", "#btn-view", "#btn-more", "#stats", "#btn-compare", "#btn-model", "#chk-xmp"):
            assert p.locator(gone).count() == 0 or p.is_hidden(gone), gone
        # counts live in the tabs, once
        for k in ("keep", "review", "reject"):
            assert int(p.inner_text(f"#c-{k}")) >= 0
        assert p.locator("#n-keep").count() == 0
        assert p.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        # ids open the sheets
        for mid, sel in SHEETS.items():
            if mid == "setup":
                continue
            assert p.evaluate(f"shotsieveMenu('{mid}')") is True
            expect(p.locator(sel)).to_be_visible()
            p.keyboard.press("Escape")
            expect(p.locator(sel)).to_be_hidden()
        # toggles
        assert p.evaluate("window.ShotSubject.isOn()") is False
        assert p.evaluate("shotsieveMenu('faces')") is False  # gone in T23h
        assert p.evaluate("shotsieveMenu('chips')") and p.get_attribute("#mi-chips", "aria-checked") == "true"
        p.evaluate("shotsieveMenu('chips')")
        p.evaluate("shotsieveMenu('filterbar')")
        expect(p.locator("#filterbar")).to_be_hidden()
        p.evaluate("shotsieveMenu('filterbar')")
        expect(p.locator("#filterbar")).to_be_visible()
        # state reported to the native menu
        p.wait_for_function("window.__menu && window.__menu.filterbar === true && window.__menu.subject === false", timeout=6000)
        assert p.evaluate("window.__menu.jersey") in (True, False)
        # xmp / unknown / injection
        assert p.evaluate("shotsieveMenu('xmp')") is False  # Write XMP is gone (automatic)
        assert p.evaluate("shotsieveMenu('autoxmp')") is True and p.evaluate("shotsieveMenu('autoxmp')") is True
        assert p.evaluate("shotsieveMenu('crops')") is True and p.evaluate("shotsieveMenu('crops')") is True
        assert p.evaluate("shotsieveMenu('video')") is True and p.evaluate("shotsieveMenu('video')") is True
        assert p.evaluate("shotsieveMenu('compare')") is True
        p.keyboard.press("Escape")
        for bad in ("nope", "recent:99", "toString", "__proto__", None, 5):
            assert p.evaluate("(b) => shotsieveMenu(b)", bad) is False
        # jersey / lr / audit exist as ids
        for mid in ("jersey", "lrsync", "auditdone", "lropen", "lrcard"):
            assert p.evaluate(f"typeof shotsieveMenu('{mid}')") == "boolean"
            p.keyboard.press("Escape")
        u.errors[:] = [e for e in u.errors if "lightroom" not in e.lower()]
        u.check()
    finally:
        ctx.close()


def test_browser_hamburger_reaches_every_action(browser, server):
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    try:
        u = Ui(ctx.new_page(), server).open()
        p = u.page
        assert _top_right_buttons(p) == ["btn-hamburger", "btn-help"], _top_right_buttons(p)
        assert _top_right_text(p) == ["☰", "?"]
        # optional items listed only when they apply: force them on for the walk
        p.evaluate("document.getElementById('mi-lr').hidden = false; document.getElementById('mi-jersey').hidden = false")
        for mid, sel in SHEETS.items():
            if mid == "help":
                continue  # covered below
            p.click("#btn-hamburger")
            r = p.locator("#hamburger-menu").bounding_box()
            assert r["x"] >= 0 and r["x"] + r["width"] <= 1100
            p.click(f'#hamburger-menu [data-menu="{mid}"]')
            if mid == "setup":
                p.wait_for_selector("#overlay-onboarding:not([hidden])", timeout=5000)
                p.click("#wiz-skip")
                p.wait_for_selector("#overlay-onboarding", state="hidden")
            else:
                expect(p.locator(sel)).to_be_visible()
            p.keyboard.press("Escape")
        p.evaluate("document.getElementById('mi-lr').hidden = false; document.getElementById('mi-jersey').hidden = false")
        p.click("#btn-hamburger")
        have = set(p.eval_on_selector_all("#hamburger-menu [data-menu]", "els => els.map(e => e.dataset.menu)"))
        assert have >= {"autoxmp", "crops", "video", "compare", "lropen", "subject", "chips", "filterbar", "cameras", "people", "shotlist", "edit", "jersey", "lrsync", "help", "setup", "lrcard"}, have
        assert "choose" not in have  # the Finder picker exists only in the app
        # toggle shows its state
        p.click('#hamburger-menu [data-menu="chips"]')
        p.click("#btn-hamburger")
        assert "✓" in p.inner_text('#hamburger-menu [data-menu="chips"]')
        p.click('#hamburger-menu [data-menu="chips"]')
        p.click('#btn-hamburger')
        p.keyboard.press("Escape")
        expect(p.locator("#hamburger-menu")).to_be_hidden()
        p.click("#btn-hamburger")
        p.click('#hamburger-menu [data-menu="help"]')
        expect(p.locator("#overlay-help")).to_be_visible()
        p.keyboard.press("Escape")
        # counts in the tabs only
        assert p.locator("#stats").count() == 0
        assert int(p.inner_text("#c-keep")) + int(p.inner_text("#c-review")) + int(p.inner_text("#c-reject")) == int(p.inner_text("#c-all"))
        assert p.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        u.errors[:] = [e for e in u.errors if not ("/api/people" in e and "403" in e)]  # People sheet without consent
        u.check()
    finally:
        ctx.close()


def test_main_view_denoise_and_use_ai_decision_button(browser, server):
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    try:
        u = Ui(ctx.new_page(), server).open()
        p = u.page
        p.locator(".cell").first.click()
        # D works in the main view: the preview swaps to /api/denoise, badge shows the backend
        expect(p.locator("#btn-dn-main")).to_be_visible()
        assert p.inner_text("#btn-dn-main .dn-badge") == "Off"
        with p.expect_response(lambda r: "/api/denoise/" in r.url and r.status == 200, timeout=120_000):
            u.key("d")
        assert "/api/denoise/" in p.get_attribute("#loupe", "src")
        p.wait_for_function("['AI','Basic'].includes(document.querySelector('#btn-dn-main .dn-badge').textContent)", timeout=120_000)
        assert p.get_attribute("#btn-dn-main", "aria-pressed") == "true"
        # the View menu state follows
        p.click("#btn-hamburger")
        assert "✓" in p.inner_text('#hamburger-menu [data-menu="denoise"]')
        p.keyboard.press("Escape")
        u.key("d")
        p.wait_for_function("document.getElementById('loupe').getAttribute('src').includes('/api/preview/')")
        assert p.inner_text("#btn-dn-main .dn-badge") == "Off"
        # disabled in crop mode
        u.key("r")
        expect(p.locator("#crop-ui")).to_be_visible()
        p.evaluate("shotsieveMenu('denoise')")
        assert p.get_attribute("#btn-dn-main", "aria-pressed") == "false"
        u.key("Escape")
        u.key("Escape")
        # no Model/U button; P again goes back to the AI decision (pressed state + tooltip)
        u.key("ArrowRight")
        assert p.locator('.actions button[data-act="model"], #btn-model').count() == 0
        assert p.locator(".actions button:visible").count() in (4, 5)  # Keep, Reject, Crop, Compare (+ Lightroom when installed)
        assert p.locator("#btn-cmp-detail").is_visible()
        expect(p.locator('.actions button[data-act="keep"]')).to_have_attribute("aria-pressed", "false")
        u.key("p")
        expect(p.locator('.actions button[data-act="keep"]')).to_have_attribute("aria-pressed", "true")
        assert "go back to ShotSieve's decision" in p.get_attribute('.actions button[data-act="keep"]', "title")
        with p.expect_response(lambda r: r.url.endswith("/api/override") and r.request.post_data_json["decision"] == "model"):
            u.key("p")
        expect(p.locator('.actions button[data-act="keep"]')).to_have_attribute("aria-pressed", "false")
        # right-click menu offers "Use AI decision" for an overridden photo
        u.key("x")
        p.locator(".cell.sel").first.click(button="right")
        expect(p.locator('#ctx-menu button[data-ai="1"]')).to_be_visible()
        with p.expect_response(lambda r: r.url.endswith("/api/override") and r.request.post_data_json["decision"] == "model"):
            p.click('#ctx-menu button[data-ai="1"]')
        assert "(yours)" not in p.inner_text("#d-decision")
        # the checkboxes are gone from the window
        assert p.locator("#topbar input[type=checkbox]:visible").count() == 0
        assert p.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        u.check()
    finally:
        ctx.close()
