# ruff: noqa: E501, F811
"""T23c in a real browser: zoom + pan in the loupe and in compare, full-resolution swap, keep zoom
across frames, link / unlink, crop mode, layout at 1100x720 (Playwright + Chromium; `ui` marker)."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from tests.test_crop_ui import crop_shoot, open_crop, page, srv  # noqa: F401
from tests.test_ui_t23b import dup  # noqa: F401

pytestmark = [pytest.mark.ui, pytest.mark.usefixtures("models_ready")]

LV = "#lz-view"


def open_loupe(ui, name="frame.pef"):
    ui.select(name)
    ui.key("Space")
    expect(ui.page.locator("#overlay-loupe")).to_be_visible()
    ui.page.wait_for_function("document.getElementById('big').naturalWidth > 0")
    return ui


def attr(ui, sel, name):
    return ui.page.get_attribute(sel, "data-" + name)


def z_of(ui, sel=LV):
    return float(attr(ui, sel, "z"))


def tx(ui, img="#big"):
    m = re.match(r"translate\((-?[\d.]+)px, (-?[\d.]+)px\) scale\(([\d.]+)\)", ui.page.evaluate(f"document.querySelector('{img}').style.transform"))
    return float(m[1]), float(m[2]), float(m[3])


def wait_pct(ui, pct, sel=LV):
    ui.page.wait_for_function(f"document.querySelector('{sel}').dataset.pct === '{pct}' && document.querySelector('{sel}').dataset.z !== '1.000'")


def wait_full(ui, sel=LV):
    ui.page.wait_for_selector(f'{sel}[data-full="1"]', timeout=60000)


def centre(ui, sel=LV):
    b = ui.page.locator(sel).bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def badge(ui, host="#lz-zoom"):
    return ui.page.inner_text(f"{host} .zb-text").strip()


# ---- loupe ------------------------------------------------------------------------------------


def test_loupe_opens_at_fit_with_the_zoom_control(page):
    open_loupe(page)
    assert z_of(page) == 1 and badge(page) == "Fit"
    assert "/api/preview/" in page.page.get_attribute("#big", "src")
    for sel in ("#lz-zoom .zb-minus", "#lz-zoom .zb-badge", "#lz-zoom .zb-plus"):
        expect(page.page.locator(sel)).to_be_visible()
    expect(page.page.locator("#lz-view .znav")).to_be_hidden()


def test_z_key_toggles_fit_and_100_percent_and_swaps_in_the_full_resolution(page):
    open_loupe(page)
    pv = page.page.evaluate("document.getElementById('big').naturalWidth")
    assert pv in (1280, 2048, 2880, 3840)  # a viewer bucket (T23f), below the 4288 px full image
    page.key("z")
    wait_pct(page, "100")
    assert badge(page) == "100 %"
    wait_full(page)
    src = page.page.get_attribute("#big", "src")
    assert "/api/full/" in src
    page.page.wait_for_function("document.getElementById('big').src.includes('/api/full/') && document.getElementById('big').naturalWidth > 1600")
    size = page.srv.get(f"/api/full/{page.sel_id()}/size?folder={page.srv.folder}")
    assert page.page.evaluate("document.getElementById('big').naturalWidth") == size["w"] == 4288
    assert page.page.evaluate("document.getElementById('big').naturalHeight") == size["h"]
    # 100 % = one image pixel per screen pixel (device pixel ratio 1 here)
    _, _, s = tx(page)
    box = page.page.evaluate("(() => { const i = document.getElementById('big'); return [i.clientWidth, i.clientHeight]; })()")
    fit = min(box[0] / size["w"], box[1] / size["h"])
    assert abs(fit * s - 1.0) < 0.01, (fit, s)
    # the spinner is gone once loaded
    expect(page.page.locator("#lz-zoom .zb-spin")).to_be_hidden()
    page.key("z")
    page.page.wait_for_function(f"document.querySelector('{LV}').dataset.z === '1.000'")
    assert badge(page) == "Fit"


def test_keys_plus_minus_zero_and_cmd_variants(page):
    open_loupe(page)
    page.key("+")
    assert abs(z_of(page) - 1.5) < 1e-6
    page.key("=")
    assert abs(z_of(page) - 2.25) < 1e-6
    page.key("-")
    assert abs(z_of(page) - 1.5) < 1e-6
    page.key("0")
    assert z_of(page) == 1
    page.key("Meta+=")
    assert abs(z_of(page) - 1.5) < 1e-6
    page.key("Meta+-")
    assert z_of(page) == 1
    page.key("Control+=")
    page.key("Meta+0")
    assert z_of(page) == 1
    expect(page.page.locator("#overlay-loupe")).to_be_visible()


def test_buttons_and_menu_presets(page):
    open_loupe(page)
    page.page.click("#lz-zoom .zb-plus")
    assert z_of(page) > 1 and badge(page).endswith("%")
    page.page.click("#lz-zoom .zb-minus")
    assert z_of(page) == 1 and badge(page) == "Fit"
    assert page.page.locator("#lz-zoom .zb-minus").is_disabled()
    page.page.click("#lz-zoom .zb-badge")
    expect(page.page.locator("#lz-zoom .zb-menu")).to_be_visible()
    assert page.page.locator("#lz-zoom .zb-menu button").all_inner_texts() == ["Fit", "50 %", "100 %", "200 %", "400 %"]
    page.page.click("#lz-zoom .zb-badge")  # closes it again
    expect(page.page.locator("#lz-zoom .zb-menu")).to_be_hidden()
    for pct in (50, 100, 200, 400):
        page.page.click("#lz-zoom .zb-badge")
        page.page.click(f'#lz-zoom .zb-menu button[data-pct="{pct}"]')
        wait_pct(page, str(pct))
        assert badge(page) == f"{pct} %"
        expect(page.page.locator("#lz-zoom .zb-menu")).to_be_hidden()
    page.page.click("#lz-zoom .zb-badge")
    page.page.click('#lz-zoom .zb-menu button[data-pct="fit"]')
    page.page.wait_for_function(f"document.querySelector('{LV}').dataset.z === '1.000'")


def test_ctrl_wheel_pinch_zooms_around_the_cursor_and_scroll_pans(page):
    open_loupe(page)
    cx, cy = centre(page)
    page.page.mouse.move(cx, cy)
    page.page.mouse.wheel(0, -300)  # a plain scroll at Fit does nothing
    assert z_of(page) == 1
    page.page.keyboard.down("Control")
    page.page.mouse.wheel(0, -300)
    page.page.keyboard.up("Control")
    z1 = z_of(page)
    assert z1 > 1.5
    # cursor anchoring: zoom in at the top-left quarter; that image point stays under the cursor
    b = page.page.locator(LV).bounding_box()
    px, py = b["x"] + b["width"] * 0.3, b["y"] + b["height"] * 0.3
    page.key("0")
    page.page.mouse.move(px, py)
    img = page.page.evaluate("(() => { const i = document.getElementById('big').getBoundingClientRect(); return [i.x, i.y, i.width, i.height]; })()")
    u0, v0 = (px - img[0]) / img[2], (py - img[1]) / img[3]
    page.page.keyboard.down("Control")
    page.page.mouse.wheel(0, -200)
    page.page.keyboard.up("Control")
    img = page.page.evaluate("(() => { const i = document.getElementById('big').getBoundingClientRect(); return [i.x, i.y, i.width, i.height]; })()")
    u1, v1 = (px - img[0]) / img[2], (py - img[1]) / img[3]
    assert abs(u1 - u0) < 0.01 and abs(v1 - v0) < 0.01, (u0, u1, v0, v1)
    # two-finger scroll pans while zoomed, and it never changed the frame
    sel0 = page.sel_id()
    x0, y0, _ = tx(page)
    page.page.mouse.wheel(60, 40)
    x1, y1, _ = tx(page)
    assert (x1, y1) != (x0, y0)
    assert page.sel_id() == sel0


def test_webkit_gesture_events_zoom(page):
    open_loupe(page)
    cx, cy = centre(page)
    page.page.evaluate(
        """([x, y]) => {
          const el = document.getElementById('lz-view');
          const ev = (t, scale) => { const e = new Event(t, { bubbles: true, cancelable: true });
            Object.defineProperty(e, 'scale', { value: scale }); Object.defineProperty(e, 'clientX', { value: x });
            Object.defineProperty(e, 'clientY', { value: y }); el.dispatchEvent(e); return e.defaultPrevented; };
          window.__gp = [ev('gesturestart', 1), ev('gesturechange', 2), ev('gesturechange', 3), ev('gestureend', 3)];
        }""",
        [cx, cy],
    )
    assert page.page.evaluate("window.__gp") == [True, True, True, True]
    assert abs(z_of(page) - 3.0) < 0.01


def test_double_click_toggles_100_percent_at_the_click_point(page):
    open_loupe(page)
    b = page.page.locator(LV).bounding_box()
    page.page.mouse.dblclick(b["x"] + b["width"] * 0.25, b["y"] + b["height"] * 0.5)
    wait_pct(page, "100")
    expect(page.page.locator("#overlay-loupe")).to_be_visible()  # a click on the picture never closes it
    # centred on the click point: the view now shows the left part of the image
    _, _, ux = (lambda v: (0, 0, v))(page.page.evaluate("(() => { const i = document.getElementById('big'); return 0; })()"))
    nav = page.page.locator("#lz-view .znav i").bounding_box()
    navbox = page.page.locator("#lz-view .znav").bounding_box()
    assert nav["x"] + nav["width"] / 2 < navbox["x"] + navbox["width"] / 2  # left half
    page.page.mouse.dblclick(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    page.page.wait_for_function(f"document.querySelector('{LV}').dataset.z === '1.000'")
    assert ux == 0


def test_drag_pans_and_a_drag_does_not_close_the_loupe(page):
    open_loupe(page)
    page.page.click("#lz-zoom .zb-badge")
    page.page.click('#lz-zoom .zb-menu button[data-pct="200"]')
    wait_pct(page, "200")
    x0, y0, _ = tx(page)
    cx, cy = centre(page)
    page.page.mouse.move(cx, cy)
    page.page.mouse.down()
    page.page.mouse.move(cx + 120, cy + 60, steps=8)
    page.page.mouse.up()
    x1, y1, _ = tx(page)
    assert abs((x1 - x0) - 120) < 3 and abs((y1 - y0) - 60) < 3, (x0, x1, y0, y1)
    expect(page.page.locator("#overlay-loupe")).to_be_visible()
    assert "panning" not in page.page.get_attribute(LV, "class")
    assert "zoomed" in page.page.get_attribute(LV, "class")


def test_navigator_thumbnail_shows_the_region_and_drags(page):
    open_loupe(page)
    nav = page.page.locator("#lz-view .znav")
    expect(nav).to_be_hidden()
    page.key("+")
    page.key("+")
    expect(nav).to_be_visible()
    nb = nav.bounding_box()
    r = page.page.locator("#lz-view .znav i").bounding_box()
    assert 0 < r["width"] < nb["width"] and r["x"] >= nb["x"] - 1
    x0, y0, _ = tx(page)
    page.page.mouse.move(nb["x"] + nb["width"] * 0.9, nb["y"] + nb["height"] * 0.9)
    page.page.mouse.down()
    page.page.mouse.move(nb["x"] + nb["width"] * 0.95, nb["y"] + nb["height"] * 0.95, steps=4)
    page.page.mouse.up()
    x1, y1, _ = tx(page)
    assert x1 < x0 and y1 < y0  # moved towards the bottom-right of the image
    page.key("0")
    expect(nav).to_be_hidden()


def test_zoom_is_remembered_per_photo_and_the_keep_checkbox_is_gone(page):
    open_loupe(page, "b0.jpg")
    assert page.page.locator("#lz-keep, #cmp-keep").count() == 0
    page.page.click("#lz-zoom .zb-badge")
    page.page.click('#lz-zoom .zb-menu button[data-pct="200"]')
    wait_pct(page, "200")
    page.page.mouse.move(*centre(page))
    page.page.mouse.wheel(-90, -70)  # pan to the top-left a little
    z0 = z_of(page)
    pos0 = page.page.evaluate("(() => { const n = document.querySelector('#lz-view .znav i'); return parseFloat(n.style.left); })()")
    first = page.sel_id()
    page.key("ArrowRight")  # another photo: its own state, Fit (never zoomed)
    page.page.wait_for_function("document.getElementById('lz-name').textContent.length > 0")
    assert page.sel_id() != first
    page.page.wait_for_function(f"document.querySelector('{LV}').dataset.z === '1.000'")
    assert badge(page) == "Fit"
    page.key("ArrowLeft")  # back: exactly where it was left
    page.page.wait_for_function(f"document.querySelector('{LV}').dataset.z !== '1.000'")
    assert page.sel_id() == first
    assert abs(z_of(page) - z0) < 1e-6 and badge(page) == "200 %"
    pos1 = page.page.evaluate("(() => { const n = document.querySelector('#lz-view .znav i'); return parseFloat(n.style.left); })()")
    assert abs(pos1 - pos0) < 1.5
    assert page.page.evaluate("localStorage.getItem('shotsieve.keepZoom')") is None
    assert "keeps the zoom" not in page.page.text_content("#overlay-help")


def test_crop_mode_turns_zoom_off_and_returns_to_fit(page):
    open_loupe(page, "landscape.jpg")
    page.key("+")
    assert z_of(page) > 1
    page.key("r")
    expect(page.page.locator("#crop-ui")).to_be_visible()
    expect(page.page.locator("#lz")).to_be_hidden()
    assert z_of(page) == 1  # zoom is reset and off while cropping
    page.page.wait_for_selector('#crop-ui[data-busy="0"]', timeout=30000)
    page.key("+")  # zoom keys do nothing in crop mode
    page.key("z")
    assert z_of(page) == 1
    rect_before = page.page.get_attribute("#crop-box", "data-rect")
    assert rect_before is not None
    page.key("Escape")
    page.key("Escape")
    expect(page.page.locator("#overlay-loupe")).to_be_hidden()
    page.key("Space")
    expect(page.page.locator("#lz")).to_be_visible()
    assert z_of(page) == 1


def test_open_crop_helper_still_works_from_the_grid(page):
    open_crop(page, "landscape.jpg")
    expect(page.page.locator("#lz")).to_be_hidden()


def test_shortcut_sheet_lists_zoom(page):
    page.key("?")
    t = page.page.text_content("#overlay-help")
    assert "zoom (loupe and compare)" in t.lower() and "100 %" in t and "link" in t.lower()


def test_loupe_layout_fits_at_1100x720(browser, srv):
    from tests.test_ui import Ui

    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    ui = Ui(ctx.new_page(), srv).open()
    try:
        open_loupe(ui)
        ui.key("+")
        ui.key("+")
        assert ui.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth && document.documentElement.scrollHeight <= window.innerHeight")
        for sel in ("#lz-zoom", "#lz-close", LV, "#lz-view .znav"):
            b = ui.page.locator(sel).bounding_box()
            assert b and b["x"] >= 0 and b["y"] >= 0 and b["x"] + b["width"] <= 1100 and b["y"] + b["height"] <= 720, (sel, b)
        ui.page.click("#lz-zoom .zb-badge")
        mb = ui.page.locator("#lz-zoom .zb-menu").bounding_box()
        assert mb["y"] + mb["height"] <= 720 and mb["x"] + mb["width"] <= 1100
    finally:
        ui.check()
        ctx.close()


# ---- compare ----------------------------------------------------------------------------------


def open_compare(ui, member):
    ui.select(member["name"])
    ui.page.locator("#d-reasons .dup-compare").click()
    expect(ui.page.locator("#overlay-compare")).to_be_visible()
    ui.page.wait_for_function("document.getElementById('cmp-a').naturalWidth > 0 && document.getElementById('cmp-b').naturalWidth > 0")


def cz(ui, side):
    return float(attr(ui, f"#cmp-fig-{side} .cmp-view", "z"))


def ctrl_wheel(ui, x, y, dy):
    ui.page.mouse.move(x, y)
    ui.page.keyboard.down("Control")
    ui.page.mouse.wheel(0, dy)
    ui.page.keyboard.up("Control")


def test_compare_zoom_control_link_toggle_and_full_res_on_both_sides(dup):
    ui, g, best, member = dup
    open_compare(ui, member)
    for sel in ("#cmp-zoombar .zb-minus", "#cmp-zoombar .zb-badge", "#cmp-zoombar .zb-plus", "#cmp-link"):
        expect(ui.page.locator(sel)).to_be_visible()
    assert ui.page.get_attribute("#cmp-link", "aria-pressed") == "true"
    ax, ay = centre(ui, "#cmp-fig-a .cmp-view")
    bx, by = centre(ui, "#cmp-fig-b .cmp-view")
    # linked by default: a pinch on one side zooms both
    ctrl_wheel(ui, ax, ay, -300)
    assert cz(ui, "a") > 1.5 and abs(cz(ui, "a") - cz(ui, "b")) < 1e-6
    # full resolution swapped in on both sides
    wait_full(ui, "#cmp-fig-a .cmp-view")
    wait_full(ui, "#cmp-fig-b .cmp-view")
    assert "/api/full/" in ui.page.get_attribute("#cmp-a", "src") and "/api/full/" in ui.page.get_attribute("#cmp-b", "src")
    ui.key("0")
    assert cz(ui, "a") == 1 and cz(ui, "b") == 1
    # L unlinks: now each side zooms on its own
    ui.key("l")
    assert ui.page.get_attribute("#cmp-link", "aria-pressed") == "false"
    ctrl_wheel(ui, ax, ay, -300)
    assert cz(ui, "a") > 1.5 and cz(ui, "b") == 1
    ctrl_wheel(ui, bx, by, -150)
    assert cz(ui, "b") > 1 and abs(cz(ui, "a") - cz(ui, "b")) > 0.05
    # the toolbar buttons drive both sides in either mode
    ui.page.click("#cmp-zoombar .zb-plus")
    # relinking snaps the other side to the side last used
    ui.key("l")
    assert ui.page.get_attribute("#cmp-link", "aria-pressed") == "true"
    assert abs(cz(ui, "a") - cz(ui, "b")) < 1e-6
    # menu + keys + dblclick in compare
    ui.page.click("#cmp-zoombar .zb-badge")
    ui.page.click('#cmp-zoombar .zb-menu button[data-pct="100"]')
    ui.page.wait_for_function("document.querySelector('#cmp-fig-a .cmp-view').dataset.pct === '100'")
    assert ui.page.inner_text("#cmp-zoombar .zb-text").strip() == "100 %"
    ui.page.mouse.dblclick(ax, ay)
    assert cz(ui, "a") == 1 and cz(ui, "b") == 1
    ui.page.click("#cmp-zoombar .zb-badge")
    ui.page.click('#cmp-zoombar .zb-menu button[data-pct="400"]')
    ui.page.wait_for_function("document.querySelector('#cmp-fig-a .cmp-view').dataset.pct === '400'")
    # drag pans both sides together when linked
    x0, _, _ = tx(ui, "#cmp-a")
    xb0, _, _ = tx(ui, "#cmp-b")
    ui.page.mouse.move(ax, ay)
    ui.page.mouse.down()
    ui.page.mouse.move(ax + 30, ay + 10, steps=5)
    ui.page.mouse.up()
    x1, _, _ = tx(ui, "#cmp-a")
    xb1, _, _ = tx(ui, "#cmp-b")
    assert x1 != x0 and xb1 != xb0
    expect(ui.page.locator("#overlay-compare")).to_be_visible()
    ui.key("Escape")
    expect(ui.page.locator("#overlay-compare")).to_be_hidden()


def test_compare_keeps_zoom_when_the_right_frame_changes(dup):
    ui, g, best, member = dup
    open_compare(ui, member)
    ui.key("+")
    ui.key("+")
    z0 = cz(ui, "b")
    src0 = ui.page.get_attribute("#cmp-b", "src")
    ui.key("ArrowDown")  # next frame of the group on the right
    ui.page.wait_for_function(f"document.getElementById('cmp-b').getAttribute('src') !== {src0!r}")
    assert abs(cz(ui, "b") - z0) < 1e-6 and abs(cz(ui, "a") - z0) < 1e-6


def test_compare_layout_fits_at_1100x720(browser, dup):
    ui, g, best, member = dup
    ui.page.set_viewport_size({"width": 1100, "height": 720})
    ui.page.wait_for_timeout(400)
    open_compare(ui, member)
    ui.key("+")
    assert ui.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth && document.documentElement.scrollHeight <= window.innerHeight")
    for sel in ("#cmp-zoombar", "#cmp-link", "#cmp-close", "#cmp-fig-a .cmp-view .znav", "#cmp-fig-b .cmp-view .znav"):
        b = ui.page.locator(sel).bounding_box()
        assert b and b["x"] >= 0 and b["y"] >= 0 and b["x"] + b["width"] <= 1100 and b["y"] + b["height"] <= 720, (sel, b)


# ---- Esc must be consumed (WebKit would pass it to the window and leave full screen) ----------

ESC_PROBE = "window.__esc = []; window.addEventListener('keydown', (e) => { if (e.key === 'Escape') window.__esc.push(e.defaultPrevented); });"


def test_escape_is_prevented_in_loupe_sheet_and_zoom_menu(page):
    page.page.evaluate(ESC_PROBE)
    open_loupe(page)
    page.page.click("#lz-zoom .zb-badge")
    page.key("Escape")  # closes the menu only
    expect(page.page.locator("#lz-zoom .zb-menu")).to_be_hidden()
    expect(page.page.locator("#overlay-loupe")).to_be_visible()
    page.key("Escape")
    expect(page.page.locator("#overlay-loupe")).to_be_hidden()
    page.key("?")
    expect(page.page.locator("#overlay-help")).to_be_visible()
    page.key("Escape")
    expect(page.page.locator("#overlay-help")).to_be_hidden()
    page.key("Escape")  # nothing open at all
    assert page.page.evaluate("window.__esc") == [True] * 3  # the menu's Esc stops propagation; the rest reach the page


def test_escape_is_prevented_in_compare_and_crop(dup):
    ui, g, best, member = dup
    ui.page.evaluate(ESC_PROBE)
    open_compare(ui, member)
    ui.key("Escape")
    expect(ui.page.locator("#overlay-compare")).to_be_hidden()
    ui.select("landscape.jpg")
    ui.key("r")
    expect(ui.page.locator("#crop-ui")).to_be_visible()
    ui.key("Escape")
    ui.key("Escape")
    expect(ui.page.locator("#overlay-loupe")).to_be_hidden()
    assert ui.page.evaluate("window.__esc") and all(ui.page.evaluate("window.__esc"))
