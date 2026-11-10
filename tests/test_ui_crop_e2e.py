# ruff: noqa: E501
"""T25b: the crop function end to end, as a user drives it (Playwright + Chromium; `ui` marker).

Real CC0 photos (tests/fixtures/crop*) and RAW masters: cull -> R -> suggestions -> pick subjects
(click, Shift-click, drag a box) -> accept -> XMP sidecar keys + the original byte-identical ->
reset; plus EXIF-rotated (portrait) JPEGs and a portrait-orientation RAW, whose `crs:Crop*` values
must be in the stored (sensor) frame."""

from __future__ import annotations

import hashlib
import re
import shutil
import struct
from pathlib import Path

import pytest
from playwright.sync_api import expect

from tests.test_ui import ROOT, Server, Ui

pytestmark = [pytest.mark.ui, pytest.mark.usefixtures("models_ready")]

CROP = ROOT / "tests" / "fixtures" / "crop"
USER_XMP = """<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" xmp:Rating="4" crs:Version="15.0" crs:ProcessVersion="11.0" crs:HasSettings="True" crs:Exposure2012="+0.65" crs:Contrast2012="+12" crs:WhiteBalance="As Shot"/></rdf:RDF></x:xmpmeta>
<?xpacket end="w"?>
"""
VIEWPORTS = [(1100, 720), (1440, 900)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def set_dng_orientation(src: Path, dst: Path, orientation: int) -> None:
    """Copy a DNG (a TIFF) with its IFD0 Orientation tag patched: a RAW whose sensor frame is
    landscape but which is displayed rotated (portrait), like a camera held on its side."""
    d = bytearray(src.read_bytes())
    bo = "<" if d[:2] == b"II" else ">"
    off = struct.unpack(bo + "I", d[4:8])[0]
    n = struct.unpack(bo + "H", d[off : off + 2])[0]
    for i in range(n):
        e = off + 2 + 12 * i
        if struct.unpack(bo + "H", d[e : e + 2])[0] == 274:
            d[e + 8 : e + 10] = struct.pack(bo + "H", orientation)
            dst.write_bytes(d)
            return
    raise AssertionError("no Orientation tag in the DNG")


@pytest.fixture(scope="session")
def e2e_shoot(tmp_path_factory, gen_dir) -> Path:
    from shotsieve.cli import main

    folder = tmp_path_factory.mktemp("cropE2E") / "shoot"
    folder.mkdir()
    shutil.copy(CROP / "real" / "group_03.jpg", folder / "group_03.jpg")
    for name in ("orient6_portrait_01", "orient8_portrait_05"):
        shutil.copy(CROP / "real" / f"{name}.jpg", folder / f"{name}.jpg")
    dng = gen_dir / "raw" / "pentax_kx.dng"
    shutil.copy(dng, folder / "raw_user.pef")  # landscape RAW that already has the photographer's XMP
    (folder / "raw_user.xmp").write_text(USER_XMP)
    set_dng_orientation(dng, folder / "raw_rot6.pef", 6)  # portrait-orientation RAW
    assert main(["cull", str(folder)]) == 0
    return folder


@pytest.fixture()
def srv(e2e_shoot, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(e2e_shoot, dest)
    s = Server(dest)
    yield s
    s.stop()


def make_page(browser, srv, size):
    ctx = browser.new_context(viewport={"width": size[0], "height": size[1]})
    return ctx, Ui(ctx.new_page(), srv).open()


def settle(ui, timeout=30000):
    ui.page.wait_for_selector('#crop-ui[data-busy="0"]', timeout=timeout)


def open_crop(ui, name):
    cell = ui.page.locator(f'.cell[title="{name}"]').first
    cell.scroll_into_view_if_needed()
    cell.click()
    iid = int(cell.get_attribute("data-id"))
    ui.key("r")
    expect(ui.page.locator("#crop-ui")).to_be_visible()
    ui.page.wait_for_function("n => document.getElementById('crop-name').textContent.trim() === n", arg=name, timeout=30000)
    ui.page.wait_for_selector("#crop-box:not([hidden])", timeout=30000)
    settle(ui)
    ui.page.wait_for_timeout(500)
    settle(ui)
    return iid


def rect(ui):
    return [float(v) for v in ui.page.locator("#crop-box").get_attribute("data-rect").split(",")]


def stage_xy(ui, nx, ny):
    b = ui.page.locator("#crop-stage").bounding_box()
    return b["x"] + nx * b["width"], b["y"] + ny * b["height"]


def api_crop(ui, iid):
    return ui.srv.get(f"/api/crop/{iid}?folder={ui.srv.folder}")


def inside(r, b, tol=2e-3):
    return r[0] <= b[0] + tol and r[1] <= b[1] + tol and r[2] >= b[2] - tol and r[3] >= b[3] - tol


def crs_attrs(xmp: Path) -> dict[str, str]:
    return dict(re.findall(r'(crs:\w+)="([^"]*)"', xmp.read_text()))


@pytest.mark.parametrize("size", VIEWPORTS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_pick_subjects_accept_reset_on_a_multi_person_photo(browser, srv, size):
    ctx, ui = make_page(browser, srv, size)
    try:
        iid = open_crop(ui, "group_03.jpg")
        pg = ui.page
        # every ratio chip gives a suggestion, and the controls stay inside the window
        for ratio in ("1:1", "4:5", "5:4", "2:3", "16:9", "free", "original"):
            pg.click(f'#crop-ratios button[data-ratio="{ratio}"]')
            settle(ui)
            assert pg.locator("#crop-cands .cand").count() >= 1, ratio
            assert pg.locator("#crop-box").get_attribute("data-rect"), ratio
        for sel in ("#crop-accept", "#crop-reject", "#crop-reset"):
            b = pg.locator(sel).bounding_box()
            assert b["x"] >= 0 and b["y"] >= 0 and b["x"] + b["width"] <= size[0] and b["y"] + b["height"] <= size[1], (sel, b)
        pg.click('#crop-ratios button[data-ratio="4:5"]')
        settle(ui)
        faces = sorted((s for s in ui.srv.get(f"/api/crop/{iid}/subjects?folder={ui.srv.folder}")["subjects"] if s["kind"] == "face"), key=lambda s: s["box"][0])
        assert len(faces) >= 2
        f1, f2 = faces[0]["box"], faces[-1]["box"]
        pg.mouse.click(*stage_xy(ui, (f1[0] + f1[2]) / 2, (f1[1] + f1[3]) / 2))
        expect(pg.locator("#crop-subjects .subj.sel").first).to_be_visible()
        settle(ui)
        pg.wait_for_timeout(400)
        assert inside(rect(ui), f1)
        pg.keyboard.down("Shift")  # Shift-click adds a second subject
        pg.mouse.click(*stage_xy(ui, (f2[0] + f2[2]) / 2, (f2[1] + f2[3]) / 2))
        pg.keyboard.up("Shift")
        pg.wait_for_timeout(400)
        settle(ui)
        r = rect(ui)
        warned = pg.locator("#crop-warn").is_visible() and "fit" in pg.inner_text("#crop-warn")
        assert (inside(r, f1) and inside(r, f2)) or warned, (r, f1, f2)  # contains both, or the amber warning
        # draw a subject box on a spot with no detected face
        x0, y0 = stage_xy(ui, 0.58, 0.55)
        x1, y1 = stage_xy(ui, 0.68, 0.72)
        pg.mouse.move(x0, y0)
        pg.mouse.down()
        pg.mouse.move(x1, y1, steps=8)
        pg.mouse.up()
        expect(pg.locator("#crop-uboxes > *")).to_have_count(1)
        settle(ui)
        pg.wait_for_timeout(400)
        settle(ui)
        r = rect(ui)
        warned = pg.locator("#crop-warn").is_visible() and "fit" in pg.inner_text("#crop-warn")
        assert inside(r, [0.58, 0.55, 0.68, 0.72]) or warned, r
        # accept -> a version, JPEG: ShotSieve only; reset -> full frame; history kept
        orig = sha(ui.srv.folder / "group_03.jpg")
        ui.key("Enter")
        expect(pg.locator("#toast")).to_contain_text("Crop v1 saved", timeout=15000)
        st = api_crop(ui, iid)
        assert st["current"]["version"] == 1 and st["current"]["rect"]
        assert ui.image("group_03.jpg")["crop"] == pytest.approx(st["current"]["rect"])
        pg.click("#crop-reset")
        expect(pg.locator("#toast")).to_contain_text("Crop reset (v2)", timeout=15000)
        expect(pg.locator("#crop-box")).to_be_hidden()
        assert [h["version"] for h in api_crop(ui, iid)["history"]] == [1, 2]
        assert sha(ui.srv.folder / "group_03.jpg") == orig
        ui.check()
    finally:
        ctx.close()


def test_raw_accept_writes_crs_keys_keeps_the_photographers_xmp_and_the_original(browser, srv):
    folder = srv.folder
    raw, xmp = folder / "raw_user.pef", folder / "raw_user.xmp"
    raw_sha = sha(raw)
    ctx, ui = make_page(browser, srv, (1440, 900))
    try:
        iid = open_crop(ui, "raw_user.pef")
        pg = ui.page
        pg.click('#crop-ratios button[data-ratio="4:5"]')
        settle(ui)
        pg.mouse.click(*stage_xy(ui, 0.5, 0.5))  # point at something: a point subject
        expect(pg.locator("#crop-subjects .subj.k-point")).to_have_count(1, timeout=30000)
        settle(ui)
        pg.wait_for_timeout(400)
        settle(ui)
        shown = rect(ui)
        ui.key("Enter")
        expect(pg.locator("#toast")).to_contain_text("Crop v1 saved", timeout=15000)
        a = crs_attrs(xmp)
        assert a["crs:HasCrop"] == "True" and a["crs:CropAngle"] == "0"
        # unrotated RAW: crs values are the displayed rect
        assert (float(a["crs:CropLeft"]), float(a["crs:CropTop"]), float(a["crs:CropRight"]), float(a["crs:CropBottom"])) == pytest.approx(shown, abs=2e-6)
        # the photographer's own settings and rating are untouched
        assert a["crs:Exposure2012"] == "+0.65" and a["crs:Contrast2012"] == "+12" and a["crs:WhiteBalance"] == "As Shot"
        assert 'xmp:Rating="4"' in xmp.read_text()
        assert sha(raw) == raw_sha
        assert api_crop(ui, iid)["xmp"]["has_crop"] is True
        # grid: the cropped badge + wrapper; the toggle shows full frames
        ui.key("Escape")
        ui.key("Escape")
        cell = pg.locator('.cell[title="raw_user.pef"]')
        expect(cell.locator(".badge.crop")).to_have_count(1)
        expect(cell.locator(".cropwrap")).to_have_count(1)
        pg.evaluate("shotsieveMenu('crops')")
        expect(pg.locator('.cell[title="raw_user.pef"] .cropwrap')).to_have_count(0)
        pg.evaluate("shotsieveMenu('crops')")
        # reset: HasCrop False + LR's no-crop defaults, other keys still intact
        ui.select("raw_user.pef")
        ui.key("r")
        settle(ui)
        pg.click("#crop-reset")
        expect(pg.locator("#toast")).to_contain_text("Crop reset (v2)", timeout=15000)
        a = crs_attrs(xmp)
        assert a["crs:HasCrop"] == "False" and (a["crs:CropLeft"], a["crs:CropTop"], a["crs:CropRight"], a["crs:CropBottom"]) == ("0", "0", "1", "1")
        assert a["crs:Exposure2012"] == "+0.65"
        assert sha(raw) == raw_sha
        ui.check()
    finally:
        ctx.close()


def test_portrait_orientation_raw_writes_the_crop_in_the_stored_sensor_frame(browser, srv):
    """EXIF orientation 6 (rotate 90 CW to display): display (X, Y) = (1 - y, x) of the sensor.
    So crs Left/Right = display y0/y1 and crs Top/Bottom = 1 - display x1 / 1 - display x0."""
    folder = srv.folder
    raw, xmp = folder / "raw_rot6.pef", folder / "raw_rot6.xmp"
    raw_sha = sha(raw)
    ctx, ui = make_page(browser, srv, (1440, 900))
    try:
        iid = open_crop(ui, "raw_rot6.pef")
        st = api_crop(ui, iid)
        assert st["orientation"] == 6 and st["full"]["h"] > st["full"]["w"]  # displayed as a portrait
        ui.page.click('#crop-ratios button[data-ratio="4:5"]')
        settle(ui)
        x0, y0, x1, y1 = rect(ui)
        b = ui.page.locator("#crop-stage").bounding_box()
        assert b["height"] > b["width"]  # the stage shows the rotated (portrait) frame
        ui.key("Enter")
        expect(ui.page.locator("#toast")).to_contain_text("Crop v1 saved", timeout=15000)
        a = crs_attrs(xmp)
        assert a["crs:HasCrop"] == "True"
        got = (float(a["crs:CropLeft"]), float(a["crs:CropTop"]), float(a["crs:CropRight"]), float(a["crs:CropBottom"]))
        assert got == pytest.approx((y0, 1 - x1, y1, 1 - x0), abs=2e-6), (got, (x0, y0, x1, y1))
        assert sha(raw) == raw_sha
        # a later decision write (rating) keeps the crop keys
        ui.key("4")
        ui.page.wait_for_timeout(2500)
        assert crs_attrs(xmp)["crs:HasCrop"] == "True"
        ui.check()
    finally:
        ctx.close()


@pytest.mark.parametrize("name", ["orient6_portrait_01.jpg", "orient8_portrait_05.jpg"])
def test_exif_rotated_jpegs_are_framed_in_the_displayed_orientation(browser, srv, name):
    ctx, ui = make_page(browser, srv, (1100, 720))
    try:
        iid = open_crop(ui, name)
        pg = ui.page
        st = api_crop(ui, iid)
        assert st["orientation"] in (6, 8) and st["full"]["h"] > st["full"]["w"]
        sb = pg.locator("#crop-stage").bounding_box()
        assert sb["height"] > sb["width"]  # portrait on screen
        face = next(s for s in ui.srv.get(f"/api/crop/{iid}/subjects?folder={ui.srv.folder}")["subjects"] if s["kind"] == "face")
        assert face["box"][3] - face["box"][1] > 0.1
        for ratio in ("1:1", "4:5", "16:9"):
            pg.click(f'#crop-ratios button[data-ratio="{ratio}"]')
            settle(ui)
            r = rect(ui)
            warned = pg.locator("#crop-warn").is_visible() and "fit" in pg.inner_text("#crop-warn")
            assert inside(r, face["box"]) or warned, (ratio, r, face["box"])
        ui.key("Enter")
        expect(pg.locator("#toast")).to_contain_text("Crop v1 saved", timeout=15000)
        ui.key("Escape")
        ui.key("Escape")
        cell = pg.locator(f'.cell[title="{name}"]')
        expect(cell.locator(".cropwrap")).to_have_count(1)
        wb = cell.locator(".cropwrap").bounding_box()
        assert wb["height"] > wb["width"] * 0.9  # the cropped grid thumb is still upright (portrait-ish)
        ui.check()
    finally:
        ctx.close()
