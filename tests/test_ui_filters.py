# ruff: noqa: E501, E702, E731
"""T23e in a real browser (Playwright + Chromium; `ui` marker): filter bar (people any / all,
flags, colour labels, stars, More panel), chips, Key faces, cell badges, keys, persistence.
A model-free seeded shoot of 12 photos (see `fshoot`)."""

from __future__ import annotations

import json
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import piexif
import pytest
from playwright.sync_api import expect

from shotsieve.config import Settings
from shotsieve.features.record import RawFeatures
from shotsieve.features.subject import FaceObs
from shotsieve.io.exif import ImageMeta
from shotsieve.pipeline.cache import Cache, CachedImage
from shotsieve.server import filters as filters_server
from tests.test_api import _row
from tests.test_faces_api import make_person, put_emb, put_raw
from tests.test_ui import Server, Ui

pytestmark = pytest.mark.ui
N = 12
SRC = ["landscape", "burst0_0", "burst0_1", "burst0_2", "burst1_0", "burst1_1"]


def post(srv, path, body):
    req = urllib.request.Request(
        srv.url + path, data=json.dumps({"folder": str(srv.folder), **body}).encode(),
        headers={"Content-Type": "application/json", "Origin": srv.url}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def dims(i: int) -> tuple[int, int]:
    return (4000, 6000) if i % 3 == 0 else (1000, 1000) if i == 4 else (6000, 4000)


@pytest.fixture(scope="module")
def fshoot(tmp_path_factory, shoot_dir) -> Path:
    """t00..t11. decisions keep/review/reject = i % 3; blur on 2 and 5; groups (0,1) and (2,3);
    faces on 0-5; Sony (0-5, #1 spelled SONY) / Canon (6-9) / no make (10, 11); portrait when
    i % 3 == 0, square for 4; lens on 0-3; rating 3 on t00, 5 on t01 (kept by you), t07 rejected
    by you; people Ana = t00-t05, Ben = t03-t08, Zed = t11 only."""
    f = tmp_path_factory.mktemp("filt") / "shoot"
    f.mkdir()
    for i in range(N):
        shutil.copy(shoot_dir / f"{SRC[i % len(SRC)]}.jpg", f / f"t{i:02d}.jpg")
    for i in range(4):
        p = f / f"t{i:02d}.jpg"
        ex = piexif.load(str(p))
        ex["Exif"][piexif.ExifIFD.LensModel] = b"FE 24-70mm F2.8 GM"
        ex["Exif"][piexif.ExifIFD.LensSerialNumber] = b"SN-1234"
        piexif.insert(piexif.dump(ex), str(p))
    group = {0: 0, 1: 0, 2: 1, 3: 1}
    rows = []
    for i in range(N):
        r = _row(f / f"t{i:02d}.jpg", ["keep", "review", "reject"][i % 3], group.get(i, 10 + i), best=i not in (1, 3),
                 reasons=["blur"] if i in (2, 5) else [])
        r["features"]["n_faces"] = 1.0 if i < 6 else 0.0
        rows.append(r)
    with Cache.for_folder(f) as c:
        c.save_decisions(rows)
        for i in range(N):
            p = f / f"t{i:02d}.jpg"
            make = "SONY" if i == 1 else "Sony" if i < 6 else "Canon" if i < 10 else ""
            model = "ILCE-7M4" if i < 6 else "EOS R5" if i < 10 else ""
            w, h = dims(i)
            st = p.stat()
            c.put(str(p), st.st_size, st.st_mtime_ns, CachedImage(
                ImageMeta(timestamp=1.0 + i, make=make, model=model), RawFeatures(width=w, height=h),
                np.zeros(384, np.float32), np.zeros((0, 128), np.float32)))
        c.commit()
        c.set_override(str(f / "t00.jpg"), "keep", 1.0, "ui", 3)
        c.set_override(str(f / "t01.jpg"), "keep", 1.0, "ui", 5)
        c.set_override(str(f / "t07.jpg"), "reject", 1.0, "ui", None)
        slots = {i: ["Ana"] for i in range(3)} | {i: ["Ana", "Ben"] for i in range(3, 6)} | {i: ["Ben"] for i in range(6, 9)} | {11: ["Zed"]}
        for i in range(N):
            names = slots.get(i, [])
            box = lambda k: FaceObs((100 + 250 * k, 200, 200, 200), 0.99, 100.0, 0.1, 0.1, 0.3, True, True)  # noqa: E731
            put_raw(c, f / f"t{i:02d}.jpg", *dims(i), [box(k) for k in range(len(names))])
            if names:
                put_emb(c, f / f"t{i:02d}.jpg", len(names))
        c.set_meta("people_consent", True)
        for who in ("Ana", "Ben", "Zed"):
            faces = [(f / f"t{i:02d}.jpg", names.index(who)) for i, names in slots.items() if who in names]
            make_person(c, None if who == "Zed" else who, faces)
    filters_server.scan(f, Settings())
    return f


@pytest.fixture()
def srv(fshoot, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(fshoot, dest)
    s = Server(dest)
    yield s
    s.stop()


@pytest.fixture()
def ui(browser, srv):
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    u = Ui(ctx.new_page(), srv).open()
    u.page.wait_for_selector("#filterbar:not([hidden])")
    yield u
    try:
        u.check()
    finally:
        ctx.close()


def labels_of(srv) -> dict:
    return srv.get(f"/api/shoot/facets?folder={srv.folder}")["labels"]


def wait_for(fn, what="condition", t=6.0):
    end = time.time() + t
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError("timed out: " + what)


def shown(ui) -> str:
    return ui.page.inner_text("#fb-count")


def wait_count(ui, n, total=N):
    expect(ui.page.locator("#fb-count")).to_have_text(f"Showing {n} of {total}")


def wait_people(ui):
    ui.page.wait_for_function("document.querySelector('#people-strip img[data-id]')")


def open_more(ui):
    ui.page.click("#fb-more")
    ui.page.wait_for_selector(".fb-more-pop .fb-opt")


def opt(ui, text):
    return ui.page.locator(".fb-pop .fb-opt", has_text=text).first


# ---- layout ------------------------------------------------------------------------------------


def test_bar_fits_one_row_at_1100x720(browser, srv):
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    ui = Ui(ctx.new_page(), srv).open()
    try:
        ui.page.wait_for_selector("#people-strip img[data-id]")
        r = ui.page.evaluate("""() => { const b = document.querySelector('#filterbar'), row = b.querySelector('.fb-row');
          const kids = [...row.children].filter(k => !k.hidden && k.offsetParent); const rect = row.getBoundingClientRect();
          return { h: b.getBoundingClientRect().height, rowW: row.scrollWidth, rowC: row.clientWidth, docW: document.documentElement.scrollWidth,
                   tops: new Set(kids.map(k => { const r = k.getBoundingClientRect(); return Math.round((r.top + r.bottom) / 2 / 6); })).size, right: Math.max(...kids.map(k => k.getBoundingClientRect().right)) }; }""")
        assert r["h"] < 44 and r["tops"] == 1, r
        assert r["rowW"] <= r["rowC"] + 1 and r["docW"] <= 1100 and r["right"] <= 1100, r
        wait_count(ui, N)
        ui.check()
    finally:
        ctx.close()


# ---- stars, flags, labels ----------------------------------------------------------------------


def test_stars_flags_and_and_or_semantics(ui):
    p = ui.page
    wait_count(ui, 12)
    p.click('.fb-star[data-v="3"]')
    wait_count(ui, 1)
    p.click('.fb-star[data-v="5"]')          # OR inside the group
    wait_count(ui, 2)
    p.click('.fb-star[data-v="0"]')          # 0 = unrated: everything except the two rated
    wait_count(ui, 12)
    p.click('.fb-star[data-v="3"]'); p.click('.fb-star[data-v="5"]')
    wait_count(ui, 10)
    p.click('.fb-tog[data-v="rejected"]')    # AND between groups: unrated AND rejected-by-you (t07)
    wait_count(ui, 1)
    p.click('.fb-tog[data-v="picked"]')      # picked OR rejected inside the flags group, still unrated: t07 only
    wait_count(ui, 1)
    p.click('.fb-tog[data-v="rejected"]')
    wait_count(ui, 0)
    p.click("#fb-clear")
    wait_count(ui, 12)
    assert p.locator(".fb-chip").count() == 0


def test_label_keys_pickers_filter_and_persist(ui, srv):
    p = ui.page
    ui.select("t05.jpg")
    for key, name in (("6", "red"), ("7", "yellow"), ("8", "green"), ("9", "blue")):
        ui.key(key)
        p.wait_for_function(f"S.byId.get(S.sel).label === '{name}'")
        sid = str(ui.sel_id())
        wait_for(lambda sid=sid, name=name: labels_of(srv).get(sid) == name, "label saved")
    expect(p.locator('.cell.sel .cf-dot')).to_have_count(1)
    ui.key("9")                              # the same key again removes it
    p.wait_for_function("S.byId.get(S.sel).label === ''")
    expect(p.locator('.cell.sel .cf-dot')).to_have_count(0)
    # the side-panel picker (purple) and the context menu
    p.click('#d-labelrow [data-label="purple"]')
    p.wait_for_function("S.byId.get(S.sel).label === 'purple'")
    wait_for(lambda: len([1 for v in labels_of(srv).values() if v == "purple"]) == 1, "first purple saved")
    p.wait_for_timeout(300)
    ui.select("t06.jpg")
    p.click(".cell.sel", button="right")
    p.click('#ctx-menu [data-label="purple"]')
    p.wait_for_function("S.byId.get(S.sel).label === 'purple'")
    # filter: purple (2) / none (10) / a dot that nothing has
    p.click('.fb-dot[data-v="purple"]')
    wait_count(ui, 2)
    p.click('.fb-dot[data-v="red"]')
    wait_count(ui, 2)
    p.click('.fb-dot[data-v="purple"]'); p.click('.fb-dot[data-v="red"]')
    p.click('.fb-dot[data-v="none"]')
    wait_count(ui, 10)
    p.click('.fb-dot[data-v="purple"]')
    wait_count(ui, 12)
    # persisted server side and across a reload
    p.reload(); p.wait_for_selector(".cell")
    p.wait_for_function("[...S.byId.values()].filter(i => i.label === 'purple').length === 2")
    assert len([k for k, v in labels_of(srv).items() if v == "purple"]) == 2


def test_zero_key_clears_stars_and_other_keys_still_work(ui):
    ui.select("t00.jpg")
    assert ui.image("t00.jpg")["rating"] == 3
    ui.key("0")
    ui.page.wait_for_function("S.byId.get(S.sel).rating === null")
    wait_for(lambda: ui.image("t00.jpg")["rating"] in (0, None), "rating cleared")
    ui.select("t04.jpg")
    ui.key("0")                              # unrated: nothing happens (no phantom override)
    time.sleep(0.3)
    assert ui.image("t04.jpg")["overridden"] is False
    ui.key("4")
    ui.page.wait_for_function("S.byId.get(S.sel).rating === 4")
    ui.key("x")
    ui.page.wait_for_function("S.byId.get(S.sel).decision === 'reject'")


# ---- people ------------------------------------------------------------------------------------


def test_people_strip_filters_with_any_all_and_no_filter_bar_button(ui):
    p = ui.page
    assert p.locator("#fb-people").count() == 0  # the redundant filter-bar button is gone
    wait_people(ui)
    faces = p.locator("#people-strip img[data-id]")
    ids = p.evaluate("[...document.querySelectorAll('#people-strip img[data-id]')].map(i => i.dataset.id)")
    names = p.evaluate("Object.fromEntries(S.people.people.map(x => [x.id, x.name]))")
    ana = next(i for i in ids if names[i] == "Ana")
    ben = next(i for i in ids if names[i] == "Ben")
    assert p.locator("#people-pm").is_hidden()
    p.click(f'#people-strip img[data-id="{ana}"]')
    wait_count(ui, 6)
    assert p.locator(".fb-chip").first.inner_text().startswith("People (any)")
    assert p.locator(f'#people-strip img[data-id="{ana}"].on').count() == 1 and faces.count() >= 2
    p.click(f'#people-strip img[data-id="{ben}"]')
    wait_count(ui, 9)                        # any of them: t00-t08
    p.click('#people-pm [data-pm="all"]')
    wait_count(ui, 3)                        # all of them: t03-t05
    assert p.locator(".fb-chip").first.inner_text().startswith("People (all)")
    p.click(".fb-chip button")               # the chip removes the filter
    wait_count(ui, 12)
    assert p.locator("#people-strip img.on").count() == 0 and p.locator("#people-pm").is_hidden()


def test_people_strip_scrolls_with_40_people_and_keeps_buttons_pinned(browser, fshoot, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(fshoot, dest)
    with Cache.for_folder(dest) as c:
        k = 0
        for n in range(40):
            make_person(c, f"Extra {n:02d}", [(dest / f"t{k % N:02d}.jpg", 2 + k // N)])
            k += 1
    s = Server(dest)
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    try:
        p = Ui(ctx.new_page(), s).open().page
        p.wait_for_selector("#people-scroll .person >> nth=39")
        r = p.evaluate("""() => { const sc = document.querySelector('#people-scroll'), st = document.querySelector('#people-strip');
            return { over: sc.scrollWidth > sc.clientWidth, docW: document.documentElement.scrollWidth,
                     stripH: st.getBoundingClientRect().height, lazy: sc.querySelectorAll('img[loading=lazy]').length,
                     right: document.querySelector('#btn-people-del').getBoundingClientRect().right }; }""")
        assert r["over"] and r["docW"] <= 1100 and r["stripH"] < 100 and r["lazy"] >= 40 and r["right"] <= 1100, r
        assert p.locator("#ps-right").is_visible() and p.locator("#ps-left").is_hidden()
        p.hover("#people-scroll")
        for _ in range(60):  # a plain vertical wheel scrolls the strip sideways
            p.mouse.wheel(0, 400)
        p.wait_for_function("(() => { const s = document.querySelector('#people-scroll'); return s.scrollLeft + s.clientWidth >= s.scrollWidth - 2; })()")
        last = p.locator("#people-scroll .person").last
        box, sbox = last.bounding_box(), p.locator("#people-scroll").bounding_box()
        assert box["x"] + box["width"] <= sbox["x"] + sbox["width"] + 1
        assert p.locator("#ps-left").is_visible() and p.locator("#ps-right").is_hidden()
        assert p.locator("#btn-people-all").is_visible() and p.locator("#btn-people-del").is_visible()  # pinned, not scrolled away
        p.click("#ps-left")
        p.wait_for_function("document.querySelector('#people-scroll').scrollLeft < document.querySelector('#people-scroll').scrollWidth - document.querySelector('#people-scroll').clientWidth - 10")
    finally:
        ctx.close()
        s.stop()


def test_people_view_show_photos_uses_the_same_filter(ui):
    p = ui.page
    pid = p.evaluate("S.people && S.people.people ? S.people.people[0].id : null") or p.evaluate("fetch('/api/people?folder='+encodeURIComponent(S.folder)).then(r=>r.json()).then(d=>d.people[0].id)")
    p.evaluate(f"ShotPeople.filterPerson({{id: {pid}, name: 'x'}})")
    expect(p.locator("#fb-count")).to_have_text(f"Showing 6 of {N}")
    assert p.locator(".fb-chip").count() == 1 and p.evaluate("S.search") is None
    p.click(".fb-chip button")
    wait_count(ui, 12)


# ---- More panel + chips ------------------------------------------------------------------------


def test_more_panel_lists_and_filters(ui):
    p = ui.page
    open_more(ui)
    box = p.locator(".fb-more-pop").bounding_box()
    assert box["x"] + box["width"] <= 1500 and box["y"] + box["height"] <= 900
    assert p.evaluate("getComputedStyle(document.querySelector('.fb-more-pop')).overflowY") == "auto"
    assert p.locator(".fb-cols").evaluate("e => getComputedStyle(e).gridTemplateColumns.split(' ').length") == 2
    sony = opt(ui, "Sony")
    assert sony.inner_text().split("\n")[-1] == "6"          # "Sony" + "SONY" are one entry
    assert "SONY" in sony.get_attribute("title")
    assert opt(ui, "Canon").inner_text().split("\n")[-1] == "4"
    assert opt(ui, "Others").first.count() == 1
    assert p.locator('.fb-pop [data-list="lens"]').count() == 2      # the lens, and Others (no lens)
    sony.click()
    wait_count(ui, 6)
    opt(ui, "Portrait").click()              # AND with the camera: t00 (portrait, Sony) and t03
    wait_count(ui, 2)
    opt(ui, "Landscape").click()             # OR within orientation
    wait_count(ui, 5)
    opt(ui, "Landscape").click(); opt(ui, "Portrait").click()
    wait_count(ui, 6)
    opt(ui, "With warnings").click()         # t02 and t05
    wait_count(ui, 2)
    opt(ui, "With warnings").click()
    opt(ui, "In a duplicate group").click()  # t00-t03 are grouped
    wait_count(ui, 4)
    opt(ui, "Not in a group").click()        # both = no restriction
    wait_count(ui, 6)
    opt(ui, "Not in a group").click(); opt(ui, "In a duplicate group").click()
    opt(ui, "Without faces").click()
    wait_count(ui, 0)
    opt(ui, "Without faces").click()
    wait_count(ui, 6)
    p.locator('.fb-pop [data-list="lens"]').first.click()    # FE 24-70mm: t00-t03
    wait_count(ui, 4)
    p.click("#fb-more")                      # closes
    assert p.locator(".fb-more-pop").count() == 1 and not p.locator(".fb-more-pop").is_visible()
    chips = p.locator(".fb-chip").all_inner_texts()
    assert any("Camera make: Sony" in c for c in chips) and any("Lens model: FE 24-70mm" in c for c in chips)
    p.locator(".fb-chip", has_text="Lens model").locator("button").click()   # removable chip
    wait_count(ui, 6)
    p.click("#fb-clear")
    wait_count(ui, 12)
    assert p.locator(".fb-chip").count() == 0


def test_file_type_filter(ui):
    open_more(ui)
    opt(ui, "RAW").click()
    wait_count(ui, 0)
    opt(ui, "RAW").click(); opt(ui, "JPEG").click()
    wait_count(ui, 12)


# ---- Key faces ---------------------------------------------------------------------------------


def test_key_faces_panel_slider_scope_and_click(ui, browser):
    p = ui.page
    p.wait_for_selector("#keyfaces .kf-face")
    assert "Key faces (3)" in p.inner_text("#keyfaces .kf-head")
    assert p.locator("#keyfaces .kf-face").count() == 3
    p.locator('#keyfaces .kf-face').first.hover()
    assert "photo" in p.locator('#keyfaces .kf-face').first.get_attribute("title")
    assert p.locator("#kf-size").count() == 0  # the size slider is gone: one fixed face size
    w = p.locator("#keyfaces .kf-face").first.bounding_box()["width"]
    assert 56 <= w <= 80
    # respects the filters: only Ana's photos t00-t05 (Ben appears through t03-t05, Zed disappears)
    p.click('#keyfaces .kf-face[title^="Ana"]')
    wait_count(ui, 6)
    assert p.locator('#keyfaces .kf-face[aria-pressed="true"]').count() == 1
    assert "Key faces (2)" in p.inner_text("#keyfaces .kf-head")
    p.click('#keyfaces .kf-face[title^="Ana"]')
    wait_count(ui, 12)
    # this image only
    ui.select("t07.jpg")
    p.click('[data-kfs="image"]')
    p.wait_for_function("document.querySelectorAll('#keyfaces .kf-face').length === 1")
    assert "Key faces (1)" in p.inner_text("#keyfaces .kf-head")
    assert p.locator("#keyfaces .kf-face").first.get_attribute("title").startswith("Ben")
    # remembered across a reload
    p.reload(); p.wait_for_selector("#keyfaces .kf-face")
    assert p.locator('[data-kfs="image"].on').count() == 1
    p.click("#kf-toggle")                    # collapse
    assert p.locator("#keyfaces .kf-face").count() == 0


# ---- badges ------------------------------------------------------------------------------------


def test_star_pick_label_badges_legible_and_not_covering_decision(ui):
    p = ui.page
    ui.select("t01.jpg")
    ui.key("8")
    p.wait_for_function("S.byId.get(S.sel).label === 'green'")
    cell = p.locator('.cell[title="t01.jpg"]')
    expect(cell.locator(".cf-dot")).to_have_count(1)
    pill = cell.locator(".rate-pill")
    assert pill.inner_text() == "★★★★★"
    assert p.locator('.cell[title="t05.jpg"] .rate-pill').count() == 0   # only when rated
    assert cell.locator(".cf-pick").count() == 1 and cell.locator(".cf-dot").count() == 1
    bg = pill.evaluate("e => getComputedStyle(e).backgroundColor")
    assert bg.startswith("rgba(12, 13, 15")  # a dark scrim: readable on light and dark photos, both themes
    cb = cell.bounding_box(); pb = pill.bounding_box()
    assert cb["x"] <= pb["x"] and pb["x"] + pb["width"] <= cb["x"] + cb["width"] and cb["y"] <= pb["y"] and pb["y"] + pb["height"] <= cb["y"] + cb["height"]
    for sel in (".badge", ".star", ".cell-flags"):
        for other in cell.locator(sel).all():
            ob = other.bounding_box()
            if ob and sel != ".cell-flags":
                assert pb["y"] >= ob["y"] + ob["height"] - 1 or pb["x"] >= ob["x"] + ob["width"] or pb["x"] + pb["width"] <= ob["x"], (sel, ob, pb)
    fb = cell.locator(".cell-flags").bounding_box()
    assert fb["y"] + fb["height"] < pb["y"]  # flags sit at the top, the pill above the caption
    ui.page.emulate_media(color_scheme="dark")
    assert pill.evaluate("e => getComputedStyle(e).backgroundColor").startswith("rgba(12, 13, 15")
    # the decision outline is untouched
    assert "dec-keep" in cell.get_attribute("class")


# ---- keys, persistence, perf -------------------------------------------------------------------


def test_keys_cmd_f_shift_f_and_help(ui):
    p = ui.page
    p.keyboard.press("Control+f")
    assert p.evaluate("document.activeElement.id") == "search"
    p.keyboard.press("Escape")
    p.evaluate("document.activeElement.blur()")
    p.keyboard.press("Shift+F")
    expect(p.locator("#filterbar")).to_be_hidden()
    p.keyboard.press("Shift+F")
    expect(p.locator("#filterbar")).to_be_visible()
    # plain F is gone (T23h: no face overlay); it must not touch the filter bar either
    p.keyboard.press("f")
    expect(p.locator("#filterbar")).to_be_visible()
    p.keyboard.press("?")
    txt = p.inner_text("#overlay-help")
    assert "Colour label" in txt and "Shift" in txt and "filter bar" in txt
    p.keyboard.press("Escape")


def test_filters_remembered_per_shoot_after_reload(ui):
    p = ui.page
    p.click('.fb-star[data-v="3"]'); p.click('.fb-tog[data-v="picked"]')
    wait_count(ui, 1)
    p.reload(); p.wait_for_selector("#filterbar:not([hidden])")
    wait_count(ui, 1)
    assert p.locator(".fb-chip").count() == 2
    assert p.locator('.fb-star[data-v="3"].on').count() == 1
    p.click("#fb-clear"); p.reload(); p.wait_for_selector(".cell")
    wait_count(ui, 12)


def test_filtering_10k_images_is_fast(ui):
    p = ui.page
    ms = p.evaluate("""() => {
      const imgs = [], groups = [];
      for (let i = 0; i < 10000; i++) imgs.push({ id: 100000 + i, name: 'x' + i + '.arw', decision: 'keep', overridden: i % 5 === 0, rating: i % 6,
        label: ['', 'red', 'green'][i % 3], reasons: i % 7 ? [] : ['blur'], reason_texts: [], n_faces: i % 2, video_t: null, group_id: i });
      S.images = imgs; S.byId = new Map(imgs.map(i => [i.id, i])); S.groups = imgs.map(i => ({ id: i.id, members: [i.id], best: i.id }));
      ShotFilters.onShoot();
      imgs.forEach((i) => { i.label = ['', 'red', 'green'][i.id % 3]; });
      const F = ShotFilters.state();
      F.stars = [3, 4, 5]; F.labels = ['red', 'green']; F.flags = ['picked', 'unflagged']; F.ftype = ['raw']; F.warn = ['without']; F.faces = ['with'];
      ShotFilters.toggle('dups', 'without');
      const t = performance.now();
      let n = 0;
      for (let r = 0; r < 5; r++) { n = 0; for (const im of S.images) if (ShotFilters.pass(im)) n++; }
      return [(performance.now() - t) / 5, n];
    }""")
    assert ms[0] < 100, ms
    assert ms[1] > 0
