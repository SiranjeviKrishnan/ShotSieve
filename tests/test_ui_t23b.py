# ruff: noqa: E501, F811
"""T23b in a real browser: manual subject boxes in crop mode, near-identical links + compare view,
shortcut sheet, dark mode and layout at the native-window sizes (Playwright + Chromium; `ui` marker)."""

from __future__ import annotations

import shutil

import pytest
from playwright.sync_api import expect

from shotsieve.pipeline.cache import Cache
from tests.test_crop_ui import crop_shoot, open_crop, page, rect, settle, srv, stage_xy  # noqa: F401
from tests.test_ui import Server, Ui

pytestmark = [pytest.mark.ui, pytest.mark.usefixtures("models_ready")]


def drag(ui, a, b, steps=8):
    x0, y0 = stage_xy(ui, *a)
    x1, y1 = stage_xy(ui, *b)
    ui.page.mouse.move(x0, y0)
    ui.page.mouse.down()
    ui.page.mouse.move(x1, y1, steps=steps)
    ui.page.mouse.up()


def contains(r, b, tol=3e-3):
    return r[0] <= b[0] + tol and r[1] <= b[1] + tol and r[2] >= b[2] - tol and r[3] >= b[3] - tol


def saved_boxes(ui):
    return ui.srv.get(f"/api/crop/{ui.sel_id()}/boxes?folder={ui.srv.folder}")["boxes"]


def ubox_n(ui):
    return ui.page.locator("#crop-uboxes .ubox").count()


# ---- manual subject boxes ---------------------------------------------------------------------


def test_crop_mode_shows_the_hint_crosshair_and_a_distinct_box_style(page):
    open_crop(page, "landscape.jpg")
    hint = page.page.locator("#crop-hint")
    expect(hint).to_contain_text("Click a subject, or drag to draw a box around it")
    assert page.page.evaluate("getComputedStyle(document.getElementById('crop-stage')).cursor") == "crosshair"
    drag(page, (0.15, 0.2), (0.3, 0.5))
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(1)
    style = page.page.evaluate("(() => { const u = getComputedStyle(document.querySelector('#crop-uboxes .ubox')); const s = document.createElement('div'); s.className='subj'; document.body.appendChild(s); const d = getComputedStyle(s); const r = [u.borderTopStyle, d.borderTopStyle, u.borderTopColor, d.borderTopColor]; s.remove(); return r; })()")
    assert style[0] == "solid" and style[1] == "dashed" and style[2] != style[3]


def test_dragging_draws_a_user_box_that_the_top_crop_contains_and_it_is_saved(page):
    open_crop(page, "two.jpg")
    box = (0.80, 0.12, 0.95, 0.42)
    drag(page, box[:2], box[2:])
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(1)
    page.page.wait_for_timeout(200)
    settle(page)
    r = rect(page)
    assert contains(r, box), (r, box)
    assert page.page.locator("#crop-box").get_attribute("data-may-be-cut") in ("0", None)
    sv = saved_boxes(page)
    assert len(sv) == 1 and all(abs(a - b) < 0.02 for a, b in zip(sv[0], box, strict=True))
    # the suggestion request carried the box
    req = page.srv.get(f"/api/crop/{page.sel_id()}?folder={page.srv.folder}")
    assert req["user_boxes"] == sv


def test_a_tiny_drag_is_a_click_not_a_box_and_a_plain_click_still_points(page):
    open_crop(page, "landscape.jpg")
    drag(page, (0.5, 0.5), (0.5008, 0.5008), steps=2)
    assert ubox_n(page) == 0
    expect(page.page.locator("#crop-subjects .subj.k-point")).to_have_count(1, timeout=30000)


def test_clicking_a_detected_box_toggles_its_selection(page):
    open_crop(page, "two.jpg")
    d = page.srv.get(f"/api/crop/{page.sel_id()}/subjects?folder={page.srv.folder}")
    f = sorted((s for s in d["subjects"] if s["kind"] == "face"), key=lambda s: s["box"][0])[0]["box"]
    c = ((f[0] + f[2]) / 2, (f[1] + f[3]) / 2)
    page.page.mouse.click(*stage_xy(page, *c))
    expect(page.page.locator("#crop-subjects .subj.sel")).to_have_count(1)
    page.page.mouse.click(*stage_xy(page, *c))
    expect(page.page.locator("#crop-subjects .subj.sel")).to_have_count(0)


def test_box_move_resize_delete_clear_and_persistence_across_a_reload(page):
    iid = open_crop(page, "landscape.jpg")
    drag(page, (0.2, 0.2), (0.4, 0.5))
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(1)
    b0 = saved_boxes(page)[0]
    # move: drag the body
    bb = page.page.locator("#crop-uboxes .ubox").bounding_box()
    cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
    page.page.mouse.move(cx, cy)
    page.page.mouse.down()
    page.page.mouse.move(cx + 60, cy + 20, steps=6)
    page.page.mouse.up()
    page.page.wait_for_timeout(250)
    b1 = saved_boxes(page)[0]
    assert b1[0] > b0[0] + 0.02 and abs((b1[2] - b1[0]) - (b0[2] - b0[0])) < 1e-3
    # resize by the south-east corner
    h = page.page.locator('#crop-uboxes .ubox .ubh[data-ubh="se"]').bounding_box()
    hx, hy = h["x"] + h["width"] / 2, h["y"] + h["height"] / 2
    page.page.mouse.move(hx, hy)
    page.page.mouse.down()
    page.page.mouse.move(hx + 50, hy + 40, steps=6)
    page.page.mouse.up()
    page.page.wait_for_timeout(250)
    b2 = saved_boxes(page)[0]
    assert b2[2] > b1[2] + 0.02 and b2[3] > b1[3] + 0.02 and abs(b2[0] - b1[0]) < 1e-3
    # a reload keeps it
    page.page.reload()
    page.page.wait_for_selector(".cell")
    page.select("landscape.jpg")
    page.key("r")
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(1, timeout=30000)
    settle(page)
    # a second box, removed with the x button
    drag(page, (0.6, 0.6), (0.8, 0.9))
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(2)
    page.page.locator("#crop-uboxes .ubox").nth(1).locator(".ubox-x").click()
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(1)
    assert len(saved_boxes(page)) == 1
    # the Delete key removes the active box
    page.key("Delete")
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(0)
    assert saved_boxes(page) == []
    # Clear selection removes every box
    drag(page, (0.1, 0.1), (0.2, 0.3))
    drag(page, (0.5, 0.1), (0.7, 0.3))
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(2)
    page.page.click("#crop-clear")
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(0)
    page.page.wait_for_timeout(250)
    assert saved_boxes(page) == [] and iid


def test_more_than_six_boxes_are_refused_with_a_message(page):
    open_crop(page, "landscape.jpg")
    for k in range(6):
        drag(page, (0.05 + 0.15 * k, 0.1), (0.15 + 0.15 * k, 0.25))
    expect(page.page.locator("#crop-uboxes .ubox")).to_have_count(6)
    drag(page, (0.3, 0.5), (0.5, 0.8))
    expect(page.page.locator("#toast")).to_contain_text("Up to 6 boxes")
    assert ubox_n(page) == 6


def test_moving_the_crop_uses_the_grip_not_a_plain_drag(page):
    open_crop(page, "two.jpg")
    before = rect(page)
    g = page.page.locator("#crop-box .grip").bounding_box()
    x, y = g["x"] + g["width"] / 2, g["y"] + g["height"] / 2
    page.page.mouse.move(x, y)
    page.page.mouse.down()
    page.page.mouse.move(x + 30, y, steps=5)
    page.page.mouse.up()
    assert rect(page)[0] != before[0]
    assert ubox_n(page) == 0


# ---- near-identical link and compare ----------------------------------------------------------


@pytest.fixture()
def dup(browser, crop_shoot, tmp_path):
    """A culled shoot whose non-best group frames carry the near-identical reason."""
    dest = tmp_path / "shoot"
    shutil.copytree(crop_shoot, dest)
    with Cache.for_folder(dest) as c:
        rows = c.load_decisions()
        best_of = {r["group_id"]: p for p, r in rows.items() if r["is_best"]}
        sizes: dict[int, int] = {}
        for r in rows.values():
            sizes[r["group_id"]] = sizes.get(r["group_id"], 0) + 1
        with c._lock:
            for p, r in rows.items():
                if sizes[r["group_id"]] > 1 and not r["is_best"] and "#t=" not in p:
                    c._con.execute("UPDATE decisions SET reasons='[\"duplicate_tie\"]' WHERE path=?", (c._rel(p),))
            c._con.commit()
    s = Server(dest)
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    ui = Ui(ctx.new_page(), s).open()
    shoot = s.shoot()
    by = {i["id"]: i for i in shoot["images"]}
    g = next(g for g in shoot["groups"] if len(g["members"]) >= 3 and by[g["best"]]["video_t"] is None)
    member = next(by[m] for m in g["members"] if m != g["best"])
    yield ui, g, by[g["best"]], member
    try:
        ui.check()
    finally:
        ctx.close()
        s.stop()
        assert best_of


def test_api_carries_duplicate_of_next_to_the_unchanged_text(dup):
    ui, g, best, member = dup
    assert member["duplicate_of"] == {"image_id": best["id"], "name": best["name"]}
    assert best["name"] in " ".join(member["reason_texts"])
    assert best["duplicate_of"] is None
    assert all("metrics" in i and set(i["metrics"]) == {"sharp", "eyes_open", "face", "n_faces"} for i in ui.srv.shoot()["images"])


def test_the_better_frames_file_name_is_a_link_that_opens_it_in_the_loupe(dup):
    ui, g, best, member = dup
    ui.select(member["name"])
    link = ui.page.locator("#d-reasons a.dup-link")
    expect(link).to_have_text(best["name"])
    assert link.get_attribute("data-id") == str(best["id"])
    link.click()
    expect(ui.page.locator("#overlay-loupe")).to_be_visible()
    assert ui.sel_id() == best["id"]
    assert f"/api/preview/{best['id']}" in ui.page.get_attribute("#big", "src")
    ui.key("Escape")
    expect(ui.page.locator("#overlay-loupe")).to_be_hidden()


def test_compare_view_metrics_filmstrip_zoom_keys_and_keep_this_one_instead(dup):
    ui, g, best, member = dup
    ui.select(member["name"])
    ui.page.locator("#d-reasons .dup-compare").click()
    expect(ui.page.locator("#overlay-compare")).to_be_visible()
    assert best["name"] in ui.page.inner_text("#cmp-a-cap") and member["name"] in ui.page.inner_text("#cmp-b-cap")
    for side in "ab":
        t = ui.page.inner_text(f"#cmp-{side}-m")
        assert "Sharpness" in t and "Eyes open" in t and "Face sharpness" in t
    assert ui.page.locator("#cmp-strip .cmp-thumb").count() == len(g["members"])
    # synced zoom: a pinch over one pane zooms both
    v = ui.page.locator("#cmp-fig-a .cmp-view").bounding_box()
    assert v["width"] > 100 and v["height"] > 100, v
    ui.page.mouse.move(v["x"] + v["width"] / 2, v["y"] + v["height"] / 2)
    ui.page.keyboard.down("Control")  # a pinch arrives as ctrl+wheel
    ui.page.mouse.wheel(0, -400)
    ui.page.keyboard.up("Control")
    ui.page.wait_for_function("!document.getElementById('cmp-a').style.transform.includes('scale(1)')")
    ta = ui.page.evaluate("document.getElementById('cmp-a').style.transform")
    assert ta == ui.page.evaluate("document.getElementById('cmp-b').style.transform")
    ui.key("0")
    assert "scale(1)" in ui.page.evaluate("document.getElementById('cmp-a').style.transform")
    # keys: 2 / right choose the right frame, 1 / left the left one
    ui.key("2")
    assert "chosen" in ui.page.get_attribute("#cmp-fig-b", "class")
    ui.key("ArrowLeft")
    assert "chosen" in ui.page.get_attribute("#cmp-fig-a", "class")
    ui.key("ArrowRight")
    assert "chosen" in ui.page.get_attribute("#cmp-fig-b", "class")
    expect(ui.page.locator("#cmp-keep-b")).to_contain_text("Keep this one instead")
    # Enter keeps the right frame and rejects the left one, as user overrides
    with ui.page.expect_response(lambda r: r.url.endswith("/api/override") and r.request.post_data and '"reject"' in r.request.post_data):
        ui.key("Enter")
    ui.page.wait_for_function("document.getElementById('toast').textContent.includes('Kept')")
    by = {i["id"]: i for i in ui.srv.shoot()["images"]}
    assert by[member["id"]]["decision"] == "keep" and by[member["id"]]["overridden"]
    assert by[best["id"]]["decision"] == "reject" and by[best["id"]]["overridden"]
    ui.key("Escape")
    expect(ui.page.locator("#overlay-compare")).to_be_hidden()


def test_compare_group_button_c_toggle_and_filmstrip_choice(dup):
    ui, g, best, member = dup
    ui.select(member["name"])
    ui.key("c")
    expect(ui.page.locator("#overlay-compare")).to_be_visible()
    thumbs = ui.page.locator("#cmp-strip .cmp-thumb")
    third = next(m for m in g["members"] if m not in (best["id"], member["id"]))
    thumbs.filter(has=ui.page.locator("img")).nth(g["members"].index(third)).click()
    assert ui.page.locator("#cmp-strip .cmp-thumb.is-b").get_attribute("data-id") == str(third)
    assert f"/api/preview/{third}" in ui.page.get_attribute("#cmp-b", "src")
    ui.key("ArrowDown")  # next frame of the group on the right
    assert ui.page.locator("#cmp-strip .cmp-thumb.is-b").get_attribute("data-id") != str(third)
    ui.key("1")
    with ui.page.expect_response(lambda r: r.url.endswith("/api/override")) as ri:
        ui.page.click("#cmp-keep-a")
    assert ri.value.status == 200, (ri.value.request.post_data, ri.value.text())
    ui.key("c")
    expect(ui.page.locator("#overlay-compare")).to_be_hidden()


def test_reason_text_with_markup_is_not_injected(dup):
    ui, g, best, member = dup
    ui.select(member["name"])
    assert ui.page.locator("#d-reasons script, #d-reasons img").count() == 0


# ---- shortcut sheet, theme and layout ---------------------------------------------------------


def test_shortcut_sheet_lists_the_new_features(page):
    page.key("?")
    expect(page.page.locator("#overlay-help")).to_be_visible()
    t = page.page.inner_text("#overlay-help")
    for frag in ("Draw a box", "In compare", "Crop mode", "Delete"):
        assert frag in t, frag
    page.key("Escape")


@pytest.mark.parametrize("size", [(1100, 720), (1440, 900)])
def test_layout_fits_the_native_window_sizes(browser, srv, size):
    ctx = browser.new_context(viewport={"width": size[0], "height": size[1]})
    try:
        ui = Ui(ctx.new_page(), srv).open()
        pg = ui.page
        over = pg.evaluate("({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, sh: document.documentElement.scrollHeight, ch: document.documentElement.clientHeight, main: document.getElementById('main').getBoundingClientRect().bottom, sb: document.getElementById('statusbar').getBoundingClientRect().bottom, top: document.getElementById('topbar').getBoundingClientRect().height})")
        assert over["sw"] <= over["cw"] and over["sh"] <= over["ch"], over
        assert over["sb"] <= size[1] + 1 and over["top"] < 120, over
        for sel in ("#btn-help", "#btn-hamburger", "#search", "#btn-cull"):  # nothing is clipped off the right edge
            b = pg.locator(sel).bounding_box()
            assert b and b["x"] + b["width"] <= size[0] + 1, (sel, b)
        ui.select("two.jpg")
        ui.key("r")
        expect(pg.locator("#crop-ui")).to_be_visible()
        pg.wait_for_selector('#crop-ui[data-busy="0"]', timeout=30000)
        geo = pg.evaluate("({st: document.getElementById('crop-stage').getBoundingClientRect(), side: document.getElementById('crop-side').getBoundingClientRect(), sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth})")
        assert geo["st"]["bottom"] <= size[1] + 1 and geo["st"]["right"] <= size[0] - 280, geo
        assert geo["side"]["right"] <= size[0] + 1 and geo["sw"] <= geo["cw"], geo
        pg.keyboard.press("Escape")
        pg.keyboard.press("Escape")
        pg.keyboard.press("c")
        pg.keyboard.press("Escape")
        ui.check()
    finally:
        ctx.close()


def test_dark_and_light_follow_the_system(browser, srv):
    colors = {}
    for scheme in ("dark", "light"):
        ctx = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme=scheme)
        try:
            ui = Ui(ctx.new_page(), srv).open()
            colors[scheme] = ui.page.evaluate("getComputedStyle(document.body).backgroundColor")
            ui.check()
        finally:
            ctx.close()
    assert colors["dark"] != colors["light"]
    assert colors["dark"] == "rgb(21, 23, 26)"


def test_detail_compare_button_sits_next_to_crop_and_opens_compare(dup):
    ui, g, best, member = dup
    ui.select(member["name"])
    crop, cmp_ = ui.page.locator("#btn-crop"), ui.page.locator("#btn-cmp-detail")
    expect(cmp_).to_be_visible()
    assert "Compare" in cmp_.inner_text() and cmp_.locator("kbd").inner_text() == "C"
    bc, bm = crop.bounding_box(), cmp_.bounding_box()
    assert abs(bc["y"] - bm["y"]) < 2 and bm["x"] > bc["x"] + bc["width"] - 1  # same row, to the right
    cmp_.click()
    expect(ui.page.locator("#overlay-compare")).to_be_visible()
    assert f"/api/preview/{best['id']}" in ui.page.get_attribute("#cmp-a", "src")
    assert f"/api/preview/{member['id']}" in ui.page.get_attribute("#cmp-b", "src")
    ui.key("Escape")
    expect(ui.page.locator("#overlay-compare")).to_be_hidden()
    ui.key("c")  # the key still works
    expect(ui.page.locator("#overlay-compare")).to_be_visible()


def test_detail_compare_button_without_a_near_identical_frame_gives_the_c_hint(dup):
    ui, g, best, member = dup
    shoot = ui.srv.shoot()
    single = next(i for i in shoot["images"] if i["video_t"] is None and len(next(x for x in shoot["groups"] if x["id"] == i["group_id"])["members"]) < 2)
    ui.select(single["name"])
    ui.page.click("#btn-cmp-detail")
    expect(ui.page.locator("#overlay-compare")).to_be_hidden()
    expect(ui.page.locator("#toast")).to_contain_text("not part of a group", timeout=10000)
