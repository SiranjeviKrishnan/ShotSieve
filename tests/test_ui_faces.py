# ruff: noqa: E501, F811
"""T23d in a real browser: People view, face overlay (zoom-exact), popover, side-panel faces, grid
chips, Open in Lightroom, catalog sync from the menu (Playwright + Chromium; `ui` marker)."""

from __future__ import annotations

import json
import shutil
import time
import urllib.request

import pytest
from playwright.sync_api import expect

from shotsieve.pipeline.cache import Cache
from tests.test_crop_ui import crop_shoot  # noqa: F401
from tests.test_ui import Server, Ui

pytestmark = [pytest.mark.ui, pytest.mark.usefixtures("models_ready")]


def post(srv, path, body):
    req = urllib.request.Request(
        srv.url + path, data=json.dumps({"folder": str(srv.folder), **body}).encode(),
        headers={"Content-Type": "application/json", "Origin": srv.url}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


@pytest.fixture()
def srv(crop_shoot, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(crop_shoot, dest)
    s = Server(dest, env={"SHOTSIEVE_FAKE_LR_APPS": "cloud"})
    post(s, "/api/people/consent", {"on": True})
    for _ in range(600):  # the all-faces recognition scan (T23h) re-clusters when it ends: wait for it
        sc = s.get(f"/api/people/scan?folder={s.folder}")
        if not sc["running"] and not sc["pending"]:
            break
        s.get(f"/api/people?folder={s.folder}")
        time.sleep(0.5)
    idle = 0
    for _ in range(600):  # let the background index / people clustering finish: it would regroup hand-made people
        st = s.get(f"/api/index/status?folder={s.folder}")
        idle = 0 if st.get("running") else idle + 1
        if idle >= 4:
            break
        time.sleep(0.5)
    with Cache.for_folder(s.folder) as c:  # start from no people: the tests make their own
        c.wipe_people(keep_face_idx=True)
    yield s
    s.stop()


@pytest.fixture()
def page(browser, srv):
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    ui = Ui(ctx.new_page(), srv).open()
    yield ui
    try:
        ui.check()
    finally:
        ctx.close()


def fn_is(page, idx, text):
    page.page.wait_for_function(
        "([i, t]) => { const e = document.querySelector(`#d-faces .fchip[data-idx=\"${i}\"] .fn`); return !!e && e.textContent === t; }",
        arg=[idx, text],
    )


def img_id(srv, name):
    return next(i["id"] for i in srv.shoot()["images"] if i["name"] == name)


def two(srv):
    """Two recognised people on two.jpg (face 0: left = Lena, face 1: right = unnamed)."""
    iid = img_id(srv, "two.jpg")
    faces = srv.get(f"/api/image/{iid}/faces?folder={srv.folder}")["faces"]
    imp = sorted((f for f in faces if f["important"]), key=lambda f: f["box"][0])
    assert len(imp) == 2, faces
    a = post(srv, f"/api/image/{iid}/faces/{imp[0]['idx']}/assign", {"name": "Lena"})["person_id"]
    b = post(srv, f"/api/image/{iid}/faces/{imp[1]['idx']}/assign", {"name": ""})["person_id"]
    return iid, a, b, imp


def open_loupe(ui, name="two.jpg"):
    ui.select(name)
    ui.key("Space")
    expect(ui.page.locator("#overlay-loupe")).to_be_visible()
    ui.page.wait_for_function("document.getElementById('big').naturalWidth > 0")
    return ui


# ---- People view ----------------------------------------------------------------------------------


def test_people_view_lists_everyone_renames_merges_ignores_and_filters(page):
    iid, a, b, imp = two(page.srv)
    page.key("Shift+P")
    expect(page.page.locator("#overlay-people")).to_be_visible()
    rows = page.page.locator(".pv-person")
    expect(rows).to_have_count(2)
    assert "Lena" in page.page.input_value(f'.pv-name[data-id="{a}"]')
    assert page.page.inner_text(f'.pv-person[data-id="{a}"] .pv-counts').startswith("1 photo")
    # rename in place: saved at once, the strip / search / overlay data follow
    page.page.fill(f'.pv-name[data-id="{b}"]', "Bruno")
    page.page.press(f'.pv-name[data-id="{b}"]', "Enter")
    # fill() already put "Bruno" in the input, so the input proves nothing: wait for the saved state (the "Named" toast
    # fires after the POST and the reload), then check the server
    page.page.wait_for_function("document.getElementById('toast') && document.getElementById('toast').textContent.includes('Named Bruno')")
    names = {p["id"]: p["name"] for p in page.srv.get(f"/api/people?folder={page.srv.folder}")["people"]}
    assert names == {a: "Lena", b: "Bruno"}
    # click = filter the grid to their photos (T23e: the People filter of the filter bar)
    page.page.click(f'.pv-show[data-id="{b}"]')
    expect(page.page.locator("#overlay-people")).to_be_hidden()
    assert page.page.input_value("#search") == ""
    page.page.wait_for_function("document.querySelectorAll('.cell').length === 1")
    assert "People" in page.page.inner_text(".fb-chip")
    page.page.click("#fb-clear")
    page.page.wait_for_function("document.querySelectorAll('.cell').length > 1")
    # merge by "Same person as..."
    page.key("Shift+P")
    page.page.click(f'.pv-same[data-id="{b}"]')
    page.page.click(f'.pv-opt[data-id="{a}"]')
    page.page.wait_for_function("document.querySelectorAll('.pv-person').length === 1")
    ppl = page.srv.get(f"/api/people?folder={page.srv.folder}")["people"]
    assert len(ppl) == 1 and ppl[0]["id"] == a and ppl[0]["n"] == 2 and ppl[0]["name"] == "Lena"
    # not a person: ignored, restorable
    page.page.click(f'.pv-ignore[data-id="{a}"]')
    page.page.wait_for_function("document.querySelectorAll('.pv-person').length === 0")
    expect(page.page.locator("#pv-restore")).to_be_visible()
    page.page.click("#pv-restore")
    # the click only starts the POST: wait for the sheet to re-render without the button (state, not timing)
    page.page.wait_for_function("!document.getElementById('pv-restore')")
    assert page.srv.get(f"/api/people?folder={page.srv.folder}")["ignored"] == 0
    page.key("Escape")
    expect(page.page.locator("#overlay-people")).to_be_hidden()


def test_people_view_merges_by_drag_and_drop(page):
    iid, a, b, imp = two(page.srv)
    page.key("Shift+P")
    page.page.drag_and_drop(f'.pv-person[data-id="{b}"] .pv-face', f'.pv-person[data-id="{a}"] .pv-main')
    page.page.wait_for_function("document.querySelectorAll('.pv-person').length === 1")
    assert [p["id"] for p in page.srv.get(f"/api/people?folder={page.srv.folder}")["people"]] == [a]


def _many_singletons(page, n=32):
    iid, a, b, imp = two(page.srv)
    with Cache.for_folder(page.srv.folder) as c:
        with c._lock:
            cover = c._con.execute("SELECT cover_face FROM people WHERE id=?", (a,)).fetchone()[0]
            for _ in range(n):
                c._con.execute("INSERT INTO people (name, cover_face, n) VALUES (?,?,?)", (None, cover, 1))
            c._con.commit()
    page.key("Shift+P")
    page.page.wait_for_function(f"document.querySelectorAll('#pv-others .pv-person').length >= {n}", timeout=20000)
    return a


def _state(page):
    return page.page.evaluate(
        "() => ({open: document.getElementById('pv-others').open, top: document.getElementById('pv-list').scrollTop,"
        " sheet: !document.getElementById('overlay-people').hidden, n: document.querySelectorAll('#pv-others .pv-person').length,"
        " ids: [...document.querySelectorAll('#pv-others .pv-person')].map(e => e.dataset.id)})"
    )


def test_people_view_keeps_others_open_scroll_and_focus_after_merge_rename_drag(page):
    a = _many_singletons(page)
    pg = page.page
    pg.evaluate("() => { const l = document.getElementById('pv-list'); l.scrollTop = document.getElementById('pv-others').offsetTop + 400; }")
    pg.wait_for_timeout(200)
    st = _state(page)
    assert st["open"] and st["top"] > 300
    ids = st["ids"]
    card_h = pg.evaluate("document.querySelector('#pv-others .pv-person').getBoundingClientRect().height")
    # visible card in the middle of the viewport
    mid = pg.evaluate("() => { const r = document.getElementById('pv-list').getBoundingClientRect(); return [...document.querySelectorAll('#pv-others .pv-person')].findIndex(e => e.getBoundingClientRect().top > r.top + 150) }")
    x = ids[mid]
    nxt = ids[mid + 1]
    top_before = pg.evaluate(f"document.querySelector('.pv-person[data-id=\"{nxt}\"]').getBoundingClientRect().top")
    ph = lambda: pg.get_attribute(f'.pv-person[data-id="{a}"] .pv-name', "placeholder")  # noqa: E731
    before_ph = ph()
    pg.click(f'.pv-same[data-id="{x}"]')
    pg.click(f'.pv-opt[data-id="{a}"]')
    pg.wait_for_function(f"document.querySelectorAll('#pv-others .pv-person').length === {st['n'] - 1}")
    assert pg.locator(".pv-chooser").count() == 0
    st2 = _state(page)
    assert st2["open"] and st2["sheet"]
    top_after = pg.evaluate(f"document.querySelector('.pv-person[data-id=\"{nxt}\"]').getBoundingClientRect().top")
    assert abs(top_after - top_before) <= card_h
    assert pg.evaluate("document.activeElement.className") == "small pv-same"
    assert pg.evaluate("document.activeElement.dataset.id") == nxt
    # rename one: it leaves Others; the sheet, section and position stay
    y = st2["ids"][mid + 2]
    nxt2 = st2["ids"][mid + 3]
    top_b = pg.evaluate(f"document.querySelector('.pv-person[data-id=\"{nxt2}\"]').getBoundingClientRect().top")
    pg.fill(f'.pv-name[data-id="{y}"]', "Zed")
    pg.press(f'.pv-name[data-id="{y}"]', "Enter")
    pg.wait_for_function(f"document.querySelectorAll('#pv-others .pv-person').length === {st2['n'] - 1}")
    st3 = _state(page)
    assert st3["open"] and st3["sheet"]
    # the new named card joined the main list above, so the list grew: the anchored card stays under the cursor
    top_a = pg.evaluate(f"document.querySelector('.pv-person[data-id=\"{nxt2}\"]').getBoundingClientRect().top")
    assert abs(top_a - top_b) <= card_h
    # drag-merge one Others card onto another
    vis = pg.evaluate("() => { const r = document.getElementById('pv-list').getBoundingClientRect(); return [...document.querySelectorAll('#pv-others .pv-person')].filter(e => e.getBoundingClientRect().top > r.top + 100 && e.getBoundingClientRect().bottom < r.bottom - 20).map(e => e.dataset.id) }")
    p1, p2, p3 = vis[0], vis[1], vis[2]
    top_b = pg.evaluate(f"document.querySelector('.pv-person[data-id=\"{p3}\"]').getBoundingClientRect().top")
    pg.drag_and_drop(f'.pv-person[data-id="{p1}"] .pv-face', f'.pv-person[data-id="{p2}"] .pv-main')
    pg.wait_for_function(f"document.querySelectorAll('#pv-others .pv-person').length === {st3['n'] - 1}")
    st4 = _state(page)
    assert st4["open"] and st4["sheet"] and abs(pg.evaluate(f"document.querySelector('.pv-person[data-id=\"{p3}\"]').getBoundingClientRect().top") - top_b) <= card_h
    # a real face merged into Lena: her count goes up (the singletons above have no faces of their own)
    b = next(p["id"] for p in page.srv.get(f"/api/people?folder={page.srv.folder}")["people"] if not p["name"] and p["n"] == 1 and p["photos"] == 1)
    pg.locator(f'.pv-same[data-id="{b}"]').scroll_into_view_if_needed()
    pg.click(f'.pv-same[data-id="{b}"]')
    pg.click(f'.pv-opt[data-id="{a}"]')
    pg.wait_for_function(f"document.querySelector('.pv-person[data-id=\"{a}\"] .pv-name').placeholder === 'Name (2 faces)'")
    assert before_ph == "Name (1 face)"
    # a closed section stays closed across a refresh (a background re-cluster)
    pg.click("#pv-others summary")
    pg.evaluate("window.dispatchEvent(new CustomEvent('people-changed'))")
    pg.wait_for_timeout(600)
    assert not _state(page)["open"]


def _people_n(page):
    return len(page.srv.get(f"/api/people?folder={page.srv.folder}")["people"])


def _menu_items(pg, face=False):
    """The person menu offers exactly these items (no Rename / Show photos)."""
    want = ["Same person as…", "Not a person"] + (["Not this person"] if face else [])
    expect(pg.locator(".pv-cm button[role=menuitem]")).to_have_text(want)


def _menu_merge(page, target, a, face=False):
    pg = page.page
    menu = pg.locator(".pv-cm")
    expect(menu).to_be_visible()
    _menu_items(pg, face)
    box = menu.bounding_box()
    assert box["x"] >= 0 and box["y"] >= 0 and box["x"] + box["width"] <= 1100 and box["y"] + box["height"] <= 720
    menu.get_by_text("Same person as…").click()
    pg.click(f'.pv-cm .pv-opt[data-id="{a}"]')
    expect(pg.locator(".pv-cm")).to_have_count(0)
    for _ in range(40):
        if _people_n(page) == 1:
            break
        pg.wait_for_timeout(250)
    assert _people_n(page) == 1


def test_right_click_same_person_in_people_sheet_keeps_state_and_esc_closes(page):
    a = _many_singletons(page)
    pg = page.page
    pg.evaluate("() => { document.getElementById('pv-list').scrollTop = document.getElementById('pv-others').offsetTop + 300; }")
    ids = _state(page)["ids"]
    x = pg.evaluate("() => { const r = document.getElementById('pv-list').getBoundingClientRect(); return [...document.querySelectorAll('#pv-others .pv-person')].find(e => e.getBoundingClientRect().top > r.top + 100).dataset.id }")
    pg.click(f'.pv-person[data-id="{x}"] .pv-main', button="right", position={"x": 5, "y": 5})
    expect(pg.locator(".pv-cm")).to_be_visible()
    _menu_items(pg)
    pg.keyboard.press("Escape")
    expect(pg.locator(".pv-cm")).to_have_count(0)
    assert _state(page)["sheet"]  # Esc closed only the menu
    n0 = len(ids)
    pg.click(f'.pv-person[data-id="{x}"] .pv-main', button="right", position={"x": 5, "y": 5})
    pg.locator(".pv-cm").get_by_text("Same person as…").click()
    pg.click(f'.pv-cm .pv-opt[data-id="{a}"]')
    pg.wait_for_function(f"document.querySelectorAll('#pv-others .pv-person').length === {n0 - 1}")
    st = _state(page)
    assert st["open"] and st["sheet"]


def test_right_click_same_person_in_people_strip(page):
    iid, a, b, imp = two(page.srv)
    page.page.evaluate("window.dispatchEvent(new CustomEvent('people-changed'))")
    pg = page.page
    pg.wait_for_selector(f'#people-strip img[data-id="{b}"]')
    pg.click(f'#people-strip img[data-id="{b}"]', button="right")
    _menu_merge(page, b, a)
    pg.wait_for_function(f"!document.querySelector('#people-strip img[data-id=\"{b}\"]')")


def test_right_click_same_person_in_key_faces(page):
    iid, a, b, imp = two(page.srv)
    page.page.evaluate("window.dispatchEvent(new CustomEvent('people-changed'))")
    pg = page.page
    page.select("two.jpg")
    pg.wait_for_selector(f'#keyfaces .kf-face[data-pid="{b}"]')
    pg.click(f'#keyfaces .kf-face[data-pid="{b}"]', button="right")
    _menu_merge(page, b, a)
    pg.wait_for_function(f"!document.querySelector('#keyfaces .kf-face[data-pid=\"{b}\"]')")


def test_right_click_faces_row_same_person_and_not_a_person_with_undo(page):
    iid, a, b, imp = two(page.srv)
    pg = page.page
    pg.evaluate("window.dispatchEvent(new CustomEvent('people-changed'))")
    page.select("two.jpg")
    pg.wait_for_selector("#d-faces-list .fchip")
    chips = pg.locator("#d-faces-list .fchip")
    expect(chips).to_have_count(2)
    # unnamed (right) face: Not a person -> gone everywhere, Undo toast
    chips.filter(has_not=pg.locator(".fn", has_text="Lena")).click(button="right")
    expect(pg.locator(".pv-cm")).to_be_visible()
    _menu_items(pg, face=True)
    pg.locator('.pv-cm button[data-act="ignore"]').click()
    pg.wait_for_function("document.querySelectorAll('#d-faces-list .fchip').length === 1")
    pg.wait_for_function(f"!document.querySelector('#people-strip img[data-id=\"{b}\"]')")
    expect(pg.locator("#toast button")).to_have_text("Undo")
    assert _people_n(page) == 1
    assert page.srv.get(f"/api/people?folder={page.srv.folder}")["ignored"] >= 1
    pg.locator("#toast button").click()
    assert page.srv.get(f"/api/people?folder={page.srv.folder}")["ignored"] == 0


def test_right_click_faces_row_same_person_merges(page):
    iid, a, b, imp = two(page.srv)
    pg = page.page
    pg.evaluate("window.dispatchEvent(new CustomEvent('people-changed'))")
    page.select("two.jpg")
    pg.wait_for_selector("#d-faces-list .fchip")
    pg.locator("#d-faces-list .fchip").filter(has_not=pg.locator(".fn", has_text="Lena")).click(button="right")
    _menu_merge(page, b, a, face=True)
    pg.wait_for_function("[...document.querySelectorAll('#d-faces-list .fn')].every(e => e.textContent === 'Lena')")


def test_right_click_key_faces_not_a_person(page):
    iid, a, b, imp = two(page.srv)
    pg = page.page
    pg.evaluate("window.dispatchEvent(new CustomEvent('people-changed'))")
    page.select("two.jpg")
    pg.wait_for_selector(f'#keyfaces .kf-face[data-pid="{b}"]')
    pg.click(f'#keyfaces .kf-face[data-pid="{b}"]', button="right")
    _menu_items(pg)
    pg.locator('.pv-cm button[data-act="ignore"]').click()
    pg.wait_for_function(f"!document.querySelector('#keyfaces .kf-face[data-pid=\"{b}\"]')")
    pg.wait_for_function(f"!document.querySelector('#people-strip img[data-id=\"{b}\"]')")
    expect(pg.locator("#toast button")).to_have_text("Undo")


def test_people_view_handles_300_people_and_loads_thumbnails_lazily(page):
    iid, a, b, imp = two(page.srv)
    with Cache.for_folder(page.srv.folder) as c:
        with c._lock:
            cover = c._con.execute("SELECT cover_face FROM people WHERE id=?", (a,)).fetchone()[0]
            for i in range(320):
                c._con.execute("INSERT INTO people (name, cover_face, n) VALUES (?,?,?)", (f"Guest {i:03d}", cover, 1))
            c._con.commit()
    page.page.evaluate("shotsieveMenu('people')")
    page.page.wait_for_function("document.querySelectorAll('.pv-person').length >= 322", timeout=20000)
    loaded = lambda: page.page.evaluate("document.querySelectorAll('img.pv-face:not([data-src])').length")  # noqa: E731
    page.page.wait_for_timeout(500)
    first = loaded()
    assert 1 <= first < 90  # only what is near the viewport
    page.page.evaluate("document.getElementById('pv-list').scrollTop = 1e6")
    page.page.wait_for_function(f"document.querySelectorAll('img.pv-face:not([data-src])').length > {first}")
    assert page.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    box = page.page.locator("#people-sheet").bounding_box()
    assert box["height"] <= 720 and box["width"] <= 1100
    page.page.fill("#pv-filter", "Guest 31")
    page.page.wait_for_function("document.querySelectorAll('.pv-person').length === 10")


def test_unnamed_single_photo_groups_sit_under_others_but_stay_reachable(page):
    iid, a, b, imp = two(page.srv)
    page.key("Shift+P")
    # two.jpg is one photo with two faces: the unnamed person has 1 photo = "Others"
    expect(page.page.locator("#pv-others summary")).to_contain_text("Others (1)")
    expect(page.page.locator(f'#pv-others .pv-person[data-id="{b}"]')).to_be_visible()
    page.page.click(f'#pv-others .pv-show[data-id="{b}"]')
    page.page.wait_for_function("document.querySelectorAll('.cell').length === 1")
    page.page.click("#fb-clear")


# ---- face overlay ---------------------------------------------------------------------------------


def test_f_does_nothing_and_the_loupe_draws_no_face_overlay(page):
    """T23h: the loupe's face boxes, the F key, its toolbar button and menu item are gone."""
    iid, a, b, imp = two(page.srv)
    open_loupe(page)
    p = page.page
    assert p.locator("#face-layer, .face-box, #lz-faces, #mi-faces").count() == 0
    page.key("f")
    p.wait_for_timeout(200)
    assert p.locator("#face-layer, .face-box").count() == 0
    expect(p.locator("#overlay-loupe")).to_be_visible()
    assert p.evaluate("shotsieveMenu('faces')") is False
    page.key("Escape")
    expect(p.locator("#overlay-loupe")).to_be_hidden()


def test_chip_click_opens_the_popover_with_the_scores_and_esc_closes_only_it(page):
    iid, a, b, imp = two(page.srv)
    page.select("two.jpg")
    chip = page.page.locator(f'#d-faces .fchip[data-idx="{imp[0]["idx"]}"]')
    assert "sharp" in chip.get_attribute("title") and "eyes open" in chip.get_attribute("title")
    chip.click()
    pop = page.page.locator("#face-pop")
    expect(pop).to_be_visible()
    expect(pop.locator("#fp-name")).to_have_value("Lena")
    page.key("Escape")
    expect(pop).to_be_hidden()


def test_popover_rename_autocomplete_merge_show_photos_and_not_this_person(page):
    iid, a, b, imp = two(page.srv)
    page.select("two.jpg")
    page.page.locator(f'#d-faces .fchip[data-idx="{imp[1]["idx"]}"]').click()
    pop = page.page.locator("#face-pop")
    expect(pop).to_be_visible()
    assert "Lena" in page.page.evaluate("[...document.querySelectorAll('#fp-names option')].map(o => o.value)")  # autocomplete source
    page.page.fill("#fp-name", "Bruno")
    page.page.click("#fp-save")
    expect(pop).to_be_hidden()
    fn_is(page, imp[1]["idx"], "Bruno")
    assert {p["name"] for p in page.srv.get(f"/api/people?folder={page.srv.folder}")["people"]} == {"Lena", "Bruno"}
    # typing an existing name offers a merge
    page.page.locator(f'#d-faces .fchip[data-idx="{imp[1]["idx"]}"]').click()
    page.page.fill("#fp-name", "lena")
    page.page.click("#fp-save")
    expect(page.page.locator("#fp-merge")).to_be_visible()
    page.page.click("#fp-do-merge")
    fn_is(page, imp[1]["idx"], "Lena")
    ppl = page.srv.get(f"/api/people?folder={page.srv.folder}")["people"]
    assert len(ppl) == 1 and ppl[0]["n"] == 2
    # not this person: detach the right-hand face
    page.page.locator(f'#d-faces .fchip[data-idx="{imp[1]["idx"]}"]').click()
    page.page.click("#fp-not")
    fn_is(page, imp[1]["idx"], "Unknown")
    assert page.srv.get(f"/api/people?folder={page.srv.folder}")["people"][0]["n"] == 1
    # show all photos of the left face's person
    page.page.locator(f'#d-faces .fchip[data-idx="{imp[0]["idx"]}"]').click()
    page.page.click("#fp-show")
    assert "People" in page.page.inner_text(".fb-chip")
    page.page.wait_for_function("document.querySelectorAll('.cell').length === 1")


def test_side_panel_faces_row_opens_the_same_popover(page):
    iid, a, b, imp = two(page.srv)
    page.select("two.jpg")
    row = page.page.locator("#d-faces .fchip")
    expect(row).to_have_count(2)
    names = [t for t in page.page.locator("#d-faces .fchip .fn").all_inner_texts()]
    assert "Lena" in names
    assert page.page.locator("#d-faces .fchip .fi").count() >= 2  # eyes / sharp indicators
    row.first.click()
    expect(page.page.locator("#face-pop")).to_be_visible()
    page.key("Escape")
    expect(page.page.locator("#face-pop")).to_be_hidden()
    page.select("landscape.jpg")
    expect(page.page.locator("#d-faces")).to_be_hidden()


def test_name_chips_on_grid_cells_are_off_by_default_and_follow_renames(page):
    iid, a, b, imp = two(page.srv)
    assert page.page.locator(".nchips").count() == 0
    page.page.click("#btn-hamburger")
    page.page.click('#hamburger-menu [data-menu="chips"]')
    page.page.wait_for_function("document.querySelectorAll('.nchips').length === 1")
    assert page.page.inner_text(".nchips") == "Lena"
    post(page.srv, f"/api/people/{b}", {"name": "Bruno"})
    page.page.evaluate("window.dispatchEvent(new CustomEvent('people-changed'))")
    page.page.wait_for_function("document.querySelector('.nchips').textContent.includes('Bruno')")
    page.page.click("#btn-hamburger")
    page.page.click('#hamburger-menu [data-menu="chips"]')
    page.page.wait_for_function("document.querySelectorAll('.nchips').length === 0")


# ---- Open in Lightroom + menus -----------------------------------------------------------------


def test_open_in_lightroom_has_no_consent_sheet_and_calls_the_api(page):
    page.select("landscape.jpg")
    expect(page.page.locator("#btn-lrop")).to_be_visible()
    with page.page.expect_response(lambda r: r.url.endswith("/api/lightroom/open")) as ri:
        page.key("Meta+l")
    out = ri.value.json()
    assert out["ok"] and out["app"] == "cloud" and "needs_consent" not in out
    assert page.page.locator("#overlay-lrask").count() == 0  # the one-time sheet is gone
    assert (page.srv.folder / "landscape.xmp").exists()  # automatic XMP: already current
    expect(page.page.locator("#toast")).to_contain_text("Opened 1 file in Lightroom")
    open_loupe(page, "landscape.jpg")
    with page.page.expect_response(lambda r: r.url.endswith("/api/lightroom/open")) as ri:
        page.page.click("#lz-lr")
    assert ri.value.json()["ok"] is True
    page.key("Escape")


def test_grid_context_menu_opens_in_lightroom_and_hides_when_not_installed(browser, srv, page):
    cell = page.page.locator('.cell[title="landscape.jpg"]').first
    cell.click(button="right")
    expect(page.page.locator("#ctx-menu button[data-app]")).to_have_text("Open in Lightroom")
    with page.page.expect_response(lambda r: r.url.endswith("/api/lightroom/open")) as ri:
        page.page.click("#ctx-menu button[data-app]")
    assert ri.value.json()["ok"] is True
    page.key("Escape")
    # catalog sync: hidden without Lightroom Classic, jersey item hidden without OCR models
    page.page.click("#btn-hamburger")
    assert page.page.locator('#hamburger-menu [data-menu="lrsync"]').count() == 0
    page.key("Escape")
    srv2_env = {"SHOTSIEVE_FAKE_LR_APPS": ""}  # neither installed
    s2 = Server(srv.folder, env=srv2_env)
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    try:
        u = Ui(ctx.new_page(), s2).open()
        u.select("landscape.jpg")
        page_ = u.page
        page_.wait_for_timeout(500)
        assert page_.is_hidden("#btn-lrop")  # no Lightroom at all: the button is hidden
        u.key("Meta+l")
        page_.wait_for_timeout(200)
        assert page_.locator("#overlay-lrask").count() == 0
        u.check()
    finally:
        ctx.close()
        s2.stop()


def test_catalog_sync_is_reachable_from_the_menu_when_classic_is_installed(browser, srv):
    s2 = Server(srv.folder, env={"SHOTSIEVE_FAKE_LR_APPS": "classic"})
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    try:
        u = Ui(ctx.new_page(), s2).open()
        assert u.page.locator("#btn-lr").count() == 0  # not in the toolbar any more
        u.page.wait_for_selector("#mi-lr:not([hidden])", state="attached")
        u.page.click("#btn-hamburger")
        expect(u.page.locator('#hamburger-menu [data-menu="lrsync"]')).to_be_visible()
        u.page.click('#hamburger-menu [data-menu="lrsync"]')
        expect(u.page.locator("#overlay-lr")).to_be_visible()
        u.page.wait_for_function("document.getElementById('lr-sheet').textContent.trim().length > 0")
        u.page.keyboard.press("Escape")
        expect(u.page.locator("#overlay-lr")).to_be_hidden()
        assert u.page.is_hidden("#btn-lrop") is False  # Classic counts as Lightroom for Open
        assert u.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        u.check()
    finally:
        ctx.close()
        s2.stop()


def test_menus_close_with_escape_and_the_toolbar_fits_1100(page):
    for btn, menu in (("#btn-hamburger", "#hamburger-menu"),):
        page.page.click(btn)
        expect(page.page.locator(menu)).to_be_visible()
        r = page.page.locator(menu).bounding_box()
        assert r["x"] >= 0 and r["x"] + r["width"] <= 1100
        page.key("Escape")
        expect(page.page.locator(menu)).to_be_hidden()
    assert page.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    tb = page.page.locator("#topbar").bounding_box()
    assert tb["height"] < 140
    page.page.click("#btn-help")
    assert "Shift" in page.page.inner_text("#overlay-help")
    page.key("Escape")
