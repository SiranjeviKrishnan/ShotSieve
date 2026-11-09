# ruff: noqa: E501, E702, F811
"""T23h S = subject mode in a real browser (Playwright + Chromium; `ui` marker): S toggles, drag draws a
box, move / resize / delete, boxes stay glued to the picture at 100 % / 200 % after panning, reload
keeps them, crop mode uses them, the sharpness score is shown, F does nothing, Esc leaves the mode."""

from __future__ import annotations

import json
import shutil

import pytest
from playwright.sync_api import expect

from tests.test_crop_ui import crop_shoot  # noqa: F401
from tests.test_ui import Server, Ui

pytestmark = [pytest.mark.ui, pytest.mark.usefixtures("models_ready")]


@pytest.fixture()
def srv(crop_shoot, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(crop_shoot, dest)
    s = Server(dest)
    yield s
    s.stop()


@pytest.fixture()
def page(browser, srv):
    ctx = browser.new_context(viewport={"width": 1300, "height": 850})
    ui = Ui(ctx.new_page(), srv).open()
    yield ui
    try:
        ui.check()
    finally:
        ctx.close()


def open_loupe(ui, name="landscape.jpg"):
    iid = ui.select(name)
    ui.key("Space")
    expect(ui.page.locator("#overlay-loupe")).to_be_visible()
    ui.page.wait_for_function("document.getElementById('big').naturalWidth > 0")
    return iid


def view_box(ui):
    return ui.page.locator("#lz-view").bounding_box()


def drag(ui, fx0, fy0, fx1, fy1):
    """Drag between two points given as fractions of the view box."""
    vb = view_box(ui)
    pts = [(vb["x"] + vb["width"] * a, vb["y"] + vb["height"] * b) for a, b in ((fx0, fy0), (fx1, fy1))]
    m = ui.page.mouse
    m.move(*pts[0])
    m.down()
    m.move(pts[0][0] + (pts[1][0] - pts[0][0]) / 2, pts[0][1] + (pts[1][1] - pts[0][1]) / 2, steps=4)
    m.move(*pts[1], steps=4)
    m.up()


def stored(srv, iid, n=None):
    """The saved boxes; with `n`, wait (saves are asynchronous) until there are n of them."""
    import time

    for _ in range(60):
        b = srv.get(f"/api/crop/{iid}/boxes?folder={srv.folder}")["boxes"]
        if n is None or len(b) == n:
            return b
        time.sleep(0.1)
    return b


def expected(ui, nb):
    """Where a normalised box must be on screen, from the real DOM rect of the <img> (not zoom.js's maths)."""
    return ui.page.evaluate(
        """(nb) => { const im = document.getElementById('big'); const r = im.getBoundingClientRect();
        const iw = im.naturalWidth, ih = im.naturalHeight; const f = Math.min(r.width / iw, r.height / ih);
        const cw = iw * f, ch = ih * f, ox = r.left + (r.width - cw) / 2, oy = r.top + (r.height - ch) / 2;
        return [ox + nb[0] * cw, oy + nb[1] * ch, (nb[2] - nb[0]) * cw, (nb[3] - nb[1]) * ch]; }""", nb)


def actual(ui, i=0):
    b = ui.page.locator(".sbox").nth(i).bounding_box()
    return [b["x"], b["y"], b["width"], b["height"]]


def aligned(ui, nb, i=0, tol=3.0):
    e, a = expected(ui, nb), actual(ui, i)
    return all(abs(x - y) <= tol for x, y in zip(e, a, strict=True)), e, a


def test_s_toggles_the_mode_and_f_does_nothing(page):
    open_loupe(page)
    p = page.page
    btn = p.locator("#lz-subject")
    expect(btn).to_have_attribute("aria-pressed", "false")
    page.key("s")
    expect(btn).to_have_attribute("aria-pressed", "true")
    assert p.evaluate("getComputedStyle(document.getElementById('subj-layer')).cursor") == "crosshair"
    assert p.evaluate("window.ShotSubject.isOn()") is True
    page.key("s")
    expect(btn).to_have_attribute("aria-pressed", "false")
    btn.click()
    expect(btn).to_have_attribute("aria-pressed", "true")
    assert p.evaluate("shotsieveMenu('subject')") is True
    expect(btn).to_have_attribute("aria-pressed", "false")
    page.key("f")
    p.wait_for_timeout(150)
    assert p.locator("#face-layer, .face-box, #lz-faces").count() == 0
    expect(p.locator("#overlay-loupe")).to_be_visible()


def test_s_in_the_main_view_opens_the_loupe_in_subject_mode(page):
    page.select("landscape.jpg")
    page.key("s")
    expect(page.page.locator("#overlay-loupe")).to_be_visible()
    expect(page.page.locator("#lz-subject")).to_have_attribute("aria-pressed", "true")


def test_drag_draws_a_box_that_is_saved_as_the_crop_user_box_and_scored(page):
    iid = open_loupe(page)
    page.key("s")
    drag(page, 0.30, 0.30, 0.60, 0.65)
    expect(page.page.locator(".sbox")).to_have_count(1)
    page.page.wait_for_function("document.querySelector('.sbox .sbox-label').textContent.includes('in focus') || document.querySelector('.sbox .sbox-label').textContent.includes('soft')")
    label = page.page.inner_text(".sbox .sbox-label")
    assert label.startswith("Subject 1: ") and any(ch.isdigit() for ch in label)
    b = stored(page.srv, iid, 1)
    assert len(b) == 1 and 0.0 <= b[0][0] < b[0][2] <= 1.0
    ok, e, a = aligned(page, b[0])
    assert ok, (e, a)


def test_move_resize_delete_and_the_limit_of_six(page):
    iid = open_loupe(page)
    page.key("s")
    drag(page, 0.30, 0.30, 0.50, 0.50)
    first = stored(page.srv, iid, 1)[0]
    box = page.page.locator(".sbox").first
    bb = box.bounding_box()
    # move: drag from inside the box
    m = page.page.mouse
    m.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
    m.down(); m.move(bb["x"] + bb["width"] / 2 + 60, bb["y"] + bb["height"] / 2 + 30, steps=5); m.up()
    moved = None
    for _ in range(40):
        cur = stored(page.srv, iid)
        if cur and cur[0] != first:
            moved = cur[0]
            break
        page.page.wait_for_timeout(100)
    assert moved and moved[0] > first[0] and moved[1] > first[1]
    assert abs((moved[2] - moved[0]) - (first[2] - first[0])) < 1e-3  # a move keeps the size
    # resize: drag the SE corner outwards
    bb = box.bounding_box()
    m.move(bb["x"] + bb["width"], bb["y"] + bb["height"])
    m.down(); m.move(bb["x"] + bb["width"] + 50, bb["y"] + bb["height"] + 40, steps=5); m.up()
    for _ in range(40):
        cur = stored(page.srv, iid)[0]
        if cur[2] > moved[2] + 1e-3:
            break
        page.page.wait_for_timeout(100)
    assert cur[2] > moved[2] and cur[3] > moved[3] and cur[0] == pytest.approx(moved[0], abs=1e-3)
    # delete with X (the photo must not be rejected by that key), then with Delete
    before = page.image("landscape.jpg")["decision"]
    page.key("x")
    assert stored(page.srv, iid, 0) == [] and page.page.locator(".sbox").count() == 0
    assert page.image("landscape.jpg")["decision"] == before
    drag(page, 0.2, 0.2, 0.35, 0.4)
    expect(page.page.locator(".sbox")).to_have_count(1)
    page.key("Delete")
    expect(page.page.locator(".sbox")).to_have_count(0)
    # at most 6
    for i in range(6):
        drag(page, 0.05 + 0.1 * i, 0.1, 0.09 + 0.1 * i, 0.3)
    expect(page.page.locator(".sbox")).to_have_count(6)
    drag(page, 0.2, 0.6, 0.4, 0.8)
    page.page.wait_for_timeout(300)
    assert page.page.locator(".sbox").count() == 6 and len(stored(page.srv, iid, 6)) == 6


def test_boxes_stay_glued_to_the_picture_at_100_and_200_percent_after_panning(page):
    iid = open_loupe(page, "two.jpg")
    page.key("s")
    drag(page, 0.40, 0.35, 0.60, 0.60)
    expect(page.page.locator(".sbox")).to_have_count(1)
    nb = stored(page.srv, iid, 1)[0]
    assert aligned(page, nb)[0]  # Fit
    p = page.page
    for pct in ("100", "200"):
        if pct == "100":
            p.click("#lz-zoom .zb-badge")
            p.click('#lz-zoom .zb-menu [data-pct="100"]')
        else:
            p.click("#lz-zoom .zb-badge")
            p.click('#lz-zoom .zb-menu [data-pct="200"]')
        p.wait_for_function(f"document.querySelector('#lz-view').dataset.pct === '{pct}'")
        p.wait_for_timeout(200)
        ok, e, a = aligned(page, nb)
        assert ok or not _on_screen(p), (pct, e, a)
        vb = view_box(page)
        p.mouse.move(vb["x"] + vb["width"] / 2, vb["y"] + vb["height"] / 2)
        for dx, dy in ((120, 60), (-300, 90), (40, -160)):  # two-finger scroll pans while zoomed (also in subject mode)
            p.mouse.wheel(dx, dy)
            p.wait_for_timeout(120)
            ok, e, a = aligned(page, nb)
            assert ok or not _on_screen(p), (pct, dx, dy, e, a)
    # centre the box so it is certainly visible, then check once more
    p.evaluate("(nb) => { const v = loupeView; const f = v.frame(); const cx = f.x + (nb[0]+nb[2])/2*f.w, cy = f.y + (nb[1]+nb[3])/2*f.h; const el = document.getElementById('lz-view'); v.panBy(el.clientWidth/2 - cx, el.clientHeight/2 - cy); }", nb)
    p.wait_for_timeout(250)
    ok, e, a = aligned(page, nb)
    assert ok and p.locator(".sbox").first.is_visible(), (e, a)


def _on_screen(p):
    return p.evaluate("""() => { const b = document.querySelector('.sbox').getBoundingClientRect(); const v = document.getElementById('lz-view').getBoundingClientRect();
        return b.right > v.left && b.left < v.right && b.bottom > v.top && b.top < v.bottom; }""")


def test_boxes_survive_a_reload_and_drive_the_crop_suggestions(page):
    iid = open_loupe(page)
    page.key("s")
    drag(page, 0.55, 0.30, 0.80, 0.70)
    expect(page.page.locator(".sbox")).to_have_count(1)
    nb = stored(page.srv, iid, 1)[0]
    page.page.reload()
    page.page.wait_for_selector(".cell")
    open_loupe(page)
    expect(page.page.locator(".sbox")).to_have_count(1)  # shown (passively) without subject mode
    assert aligned(page, nb)[0]
    # crop mode: the same boxes are the drawn boxes, and the suggest request carries them
    seen = []
    page.page.on("request", lambda r: seen.append(r.post_data) if r.method == "POST" and "/suggest" in r.url else None)
    page.key("r")
    expect(page.page.locator("#crop-ui")).to_be_visible()
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(1)
    for _ in range(60):
        if seen:
            break
        page.page.wait_for_timeout(100)
    body = json.loads(seen[-1])
    assert len(body["boxes"]) == 1 and body["boxes"][0]["x0"] == pytest.approx(nb[0], abs=1e-4)
    # server side: the suggestions are built around the box
    r = page.srv.get(f"/api/crop/{iid}/boxes?folder={page.srv.folder}")
    assert r["boxes"] == [nb]
    page.key("Escape")


def test_esc_leaves_subject_mode_but_keeps_the_loupe_and_the_boxes(page):
    iid = open_loupe(page)
    page.key("s")
    drag(page, 0.3, 0.3, 0.5, 0.5)
    expect(page.page.locator(".sbox")).to_have_count(1)
    page.key("Escape")
    expect(page.page.locator("#lz-subject")).to_have_attribute("aria-pressed", "false")
    expect(page.page.locator("#overlay-loupe")).to_be_visible()
    assert page.page.locator(".sbox").count() == 1 and len(stored(page.srv, iid, 1)) == 1
    assert page.page.evaluate("getComputedStyle(document.getElementById('subj-layer')).cursor") != "crosshair"
    page.key("Escape")
    expect(page.page.locator("#overlay-loupe")).to_be_hidden()
