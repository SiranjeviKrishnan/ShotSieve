# ruff: noqa: E501
"""T22 crop mode in a real browser (Playwright + Chromium; `ui` marker): R, subject picking, ratio
chips, alternatives, adjusting, accept / reject / reset / restore, show crops, burst, no JS errors."""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest
from playwright.sync_api import expect

from tests.test_ui import Server, Ui

pytestmark = [pytest.mark.ui, pytest.mark.usefixtures("models_ready")]


@pytest.fixture(scope="session")
def crop_shoot(tmp_path_factory, gen_dir, shoot_dir) -> Path:
    """A culled shoot: a two-face image, a 3-frame burst, a landscape, a RAW master and a video."""
    from shotsieve.cli import main
    from tests.video_synth import make_three_scene_video

    folder = tmp_path_factory.mktemp("cropui") / "shoot"
    folder.mkdir()
    a = cv2.resize(cv2.imread(str(gen_dir / "src" / "lena.jpg")), (640, 640))
    b = cv2.resize(cv2.imread(str(shoot_dir / "burst0_0.jpg")), (640, 640))
    cv2.imwrite(str(folder / "two.jpg"), np.hstack([a, b]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    for i in range(3):
        shutil.copy(shoot_dir / f"burst0_{i}.jpg", folder / f"b{i}.jpg")
    shutil.copy(shoot_dir / "landscape.jpg", folder / "landscape.jpg")
    shutil.copy(gen_dir / "raw" / "pentax_kx.dng", folder / "frame.pef")
    make_three_scene_video(folder / "clip.mp4", gen_dir / "src")
    assert main(["cull", str(folder), "--video"]) == 0
    return folder


@pytest.fixture()
def srv(crop_shoot, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(crop_shoot, dest)
    s = Server(dest)
    yield s
    s.stop()


@pytest.fixture()
def page(browser, srv):
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    ui = Ui(ctx.new_page(), srv).open()
    yield ui
    try:
        ui.check()
    finally:
        ctx.close()


def settle(ui, timeout=30000):
    ui.page.wait_for_selector('#crop-ui[data-busy="0"]', timeout=timeout)


def open_crop(ui, name):
    iid = ui.select(name)
    ui.key("r")
    expect(ui.page.locator("#crop-ui")).to_be_visible()
    ui.page.wait_for_selector("#crop-subjects .subj, #crop-box:not([hidden])", timeout=30000)
    settle(ui)
    return iid


def rect(ui):
    return [float(v) for v in ui.page.locator("#crop-box").get_attribute("data-rect").split(",")]


def stage_xy(ui, nx, ny):
    b = ui.page.locator("#crop-stage").bounding_box()
    return b["x"] + nx * b["width"], b["y"] + ny * b["height"]


def faces(ui):
    d = ui.srv.get(f"/api/crop/{ui.sel_id()}/subjects?folder={ui.srv.folder}")
    return sorted((s for s in d["subjects"] if s["kind"] == "face"), key=lambda s: s["box"][0])


def click_face(ui, face, shift=False, alt=False):
    b = face["box"]
    for k, on in (("Shift", shift), ("Alt", alt)):
        if on:
            ui.page.keyboard.down(k)
    ui.page.mouse.click(*stage_xy(ui, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    for k, on in (("Shift", shift), ("Alt", alt)):
        if on:
            ui.page.keyboard.up(k)


def test_r_opens_crop_mode_with_subject_boxes_and_a_suggestion(page):
    open_crop(page, "two.jpg")
    expect(page.page.locator("#crop-subjects .subj.k-face")).to_have_count(2)
    expect(page.page.locator("#crop-subjects .subj.sel")).to_have_count(0)
    assert page.page.locator("#crop-box").get_attribute("data-rect")  # the auto suggestion
    assert page.page.locator("#crop-cands .cand").count() >= 1
    page.key("Escape")
    expect(page.page.locator("#crop-ui")).to_be_hidden()
    page.key("Escape")  # nothing selected (crop mode closed): leaves the loupe too
    expect(page.page.locator("#overlay-loupe")).to_be_hidden()


def test_click_selects_shift_click_adds_and_the_suggestion_contains_both_faces(page):
    open_crop(page, "two.jpg")
    f1, f2 = faces(page)
    click_face(page, f1)
    expect(page.page.locator("#crop-subjects .subj-badge")).to_have_count(1)
    settle(page)
    r1 = rect(page)
    assert r1[0] <= f1["box"][0] and r1[2] >= f1["box"][2]
    click_face(page, f2, shift=True)
    expect(page.page.locator("#crop-subjects .subj-badge")).to_have_count(2)
    page.page.wait_for_timeout(300)
    settle(page)
    r2 = rect(page)
    for f in (f1, f2):
        b = f["box"]
        assert r2[0] <= b[0] and r2[2] >= b[2] and r2[1] <= b[1] and r2[3] >= b[3], (r2, b)
    click_face(page, f2, shift=True)  # shift-click again toggles it off
    expect(page.page.locator("#crop-subjects .subj-badge")).to_have_count(1)


def test_option_click_on_a_face_selects_the_face_only(page):
    open_crop(page, "two.jpg")
    f1, _ = faces(page)
    click_face(page, f1, alt=True)
    expect(page.page.locator("#crop-subjects .subj.sel")).to_have_count(1)
    settle(page)
    r = rect(page)
    assert r[0] <= f1["box"][0] and r[2] >= f1["box"][2] and r[1] <= f1["box"][1] and r[3] >= f1["box"][3]
    click_face(page, f1, alt=True)  # clicking the only selected subject again deselects it
    expect(page.page.locator("#crop-subjects .subj.sel")).to_have_count(0)


def test_click_on_empty_background_adds_a_point_subject(page):
    open_crop(page, "landscape.jpg")
    page.page.mouse.click(*stage_xy(page, 0.5, 0.5))
    expect(page.page.locator("#crop-subjects .subj.k-point")).to_have_count(1, timeout=30000)
    expect(page.page.locator("#crop-subjects .subj.k-point.sel")).to_have_count(1)
    settle(page)
    assert page.page.locator("#crop-box").get_attribute("data-rect")


def test_ratio_chips_custom_ratio_flip_and_exact_overlay_aspect(page):
    open_crop(page, "two.jpg")
    page.page.click('#crop-ratios button[data-ratio="1:1"]')
    settle(page)
    box = page.page.locator("#crop-box")
    expect(box).to_have_attribute("data-ratio", "1:1")
    bb = box.bounding_box()
    assert abs(bb["width"] - bb["height"]) <= 1.0, bb
    page.page.fill("#crop-w", "7")
    page.page.fill("#crop-h", "5")
    page.page.click("#crop-custom-apply")
    settle(page)
    expect(box).to_have_attribute("data-ratio", "7:5")
    page.page.click("#crop-flip")
    settle(page)
    expect(box).to_have_attribute("data-ratio", "5:7")
    bb = box.bounding_box()
    assert abs(bb["width"] / bb["height"] - 5 / 7) < 0.02, bb
    page.page.fill("#crop-w", "9")
    page.page.fill("#crop-h", "1")
    page.page.click("#crop-custom-apply")
    expect(page.page.locator("#crop-ratio-err")).not_to_be_empty()  # inline error: outside 1:4 .. 4:1
    page.page.fill("#crop-w", "1.91")
    page.page.fill("#crop-h", "1")
    page.page.click("#crop-custom-apply")
    settle(page)
    expect(box).to_have_attribute("data-ratio", "1.91:1")
    page.page.click('#crop-ratios button[data-ratio="4:5"]')
    settle(page)
    page.key("x")  # X flips, as in Lightroom (the global X = reject is suspended)
    settle(page)
    expect(box).to_have_attribute("data-ratio", "4:5")
    assert page.image("two.jpg")["decision"] == page.image("two.jpg")["model_decision"]  # nothing rejected


def test_bracket_keys_cycle_the_alternatives(page):
    open_crop(page, "two.jpg")
    f1, _ = faces(page)
    click_face(page, f1)
    settle(page)
    n = page.page.locator("#crop-cands .cand").count()
    assert n >= 2
    r0 = rect(page)
    page.key("]")
    r1 = rect(page)
    assert r1 != r0
    expect(page.page.locator('#crop-cands .cand[aria-pressed="true"]')).to_have_attribute("data-alt", "1")
    page.key("[")
    assert rect(page) == r0


def test_dragging_a_corner_keeps_the_ratio_and_marks_the_crop_adjusted(page):
    open_crop(page, "two.jpg")
    page.page.click('#crop-ratios button[data-ratio="4:5"]')
    settle(page)
    box = page.page.locator("#crop-box")
    before = box.bounding_box()
    h = page.page.locator('#crop-box .handle[data-h="se"]').bounding_box()
    x, y = h["x"] + h["width"] / 2, h["y"] + h["height"] / 2
    page.page.mouse.move(x, y)
    page.page.mouse.down()
    page.page.mouse.move(x - 40, y - 30, steps=6)
    page.page.mouse.up()
    expect(box).to_have_attribute("data-source", "ai_adjusted")
    after = box.bounding_box()
    assert after["width"] < before["width"] - 5
    ratio = box.get_attribute("data-ratio")
    a, b = (float(v) for v in ratio.split(":"))
    assert abs(after["width"] / after["height"] - a / b) < 0.02, (after, ratio)
    # drag the "move crop" bar = move (a plain drag inside the crop now draws a subject box)
    g = page.page.locator("#crop-box .grip").bounding_box()
    cx, cy = g["x"] + g["width"] / 2, g["y"] + g["height"] / 2
    r_before = rect(page)
    page.page.mouse.move(cx, cy)
    page.page.mouse.down()
    page.page.mouse.move(cx + 25, cy, steps=5)
    page.page.mouse.up()
    r_after = rect(page)
    assert r_after[0] > r_before[0] and abs((r_after[2] - r_after[0]) - (r_before[2] - r_before[0])) < 1e-5
    assert page.srv.get(f"/api/crop/{page.sel_id()}?folder={page.srv.folder}")["history"] == []  # nothing saved yet


def test_accept_reject_reset_restore_history_badges_and_show_crops_toggle(page):
    iid = open_crop(page, "frame.pef")
    page.page.click('#crop-ratios button[data-ratio="4:5"]')
    settle(page)
    page.page.click("#crop-flip")  # a portrait crop of a landscape RAW
    settle(page)
    r1 = rect(page)
    page.key("Enter")
    expect(page.page.locator("#toast")).to_contain_text("Crop v1 saved", timeout=15000)
    st = page.srv.get(f"/api/crop/{iid}?folder={page.srv.folder}")
    assert st["current"]["version"] == 1 and st["current"]["source"] == "ai"
    assert 'crs:HasCrop="True"' in (page.srv.folder / "frame.xmp").read_text()
    assert page.image("frame.pef")["crop"] == pytest.approx(st["current"]["rect"])
    # reject keeps the saved crop
    page.page.click('#crop-ratios button[data-ratio="1:1"]')
    settle(page)
    page.key("Backspace")
    expect(page.page.locator("#crop-msg")).to_contain_text("dismissed", timeout=10000)
    assert page.srv.get(f"/api/crop/{iid}?folder={page.srv.folder}")["current"]["version"] == 1
    assert rect(page) == pytest.approx(st["current"]["rect"], abs=1e-6)
    page.key("Escape")  # nothing selected: leaves crop mode
    expect(page.page.locator("#crop-ui")).to_be_hidden()
    page.key("Escape")
    cell = page.page.locator('.cell[title="frame.pef"]')
    expect(cell.locator(".badge.crop")).to_have_count(1)
    expect(cell.locator(".cropwrap")).to_have_count(1)
    wrap_on = cell.locator(".cropwrap").bounding_box()
    page.page.evaluate("shotsieveMenu('crops')")  # View > Show crop suggestions off
    expect(page.page.locator('.cell[title="frame.pef"] .cropwrap')).to_have_count(0)
    expect(cell.locator(".badge.crop")).to_have_count(1)  # the badge says "cropped" either way
    img_off = cell.locator("img").first.bounding_box()
    assert abs(img_off["width"] - wrap_on["width"]) > 2 or abs(img_off["height"] - wrap_on["height"]) > 2
    page.page.evaluate("shotsieveMenu('crops')")
    expect(page.page.locator('.cell[title="frame.pef"] .cropwrap')).to_have_count(1)
    # second accept (adjusted source), then reset, then restore v1
    page.select("frame.pef")
    page.key("r")
    settle(page)
    page.page.click('#crop-ratios button[data-ratio="1:1"]')
    settle(page)
    page.page.click("#crop-accept")
    expect(page.page.locator("#toast")).to_contain_text("Crop v2 saved", timeout=15000)
    page.page.click("#crop-reset")
    expect(page.page.locator("#toast")).to_contain_text("Crop reset (v3)", timeout=15000)
    expect(page.page.locator("#crop-box")).to_be_hidden()  # the full frame is shown
    assert 'crs:HasCrop="False"' in (page.srv.folder / "frame.xmp").read_text()
    expect(page.page.locator('.cell[title="frame.pef"] .badge.crop')).to_have_count(0)
    hist = page.srv.get(f"/api/crop/{iid}?folder={page.srv.folder}")["history"]
    assert [h["version"] for h in hist] == [1, 2, 3] and hist[2]["rect"] is None
    page.page.select_option("#crop-history", "1")
    page.page.click("#crop-restore")
    expect(page.page.locator("#toast")).to_contain_text("Restored v1 as v4", timeout=15000)
    hist = page.srv.get(f"/api/crop/{iid}?folder={page.srv.folder}")["history"]
    assert len(hist) == 4 and hist[3]["rect"] == hist[0]["rect"] and r1 is not None


def test_esc_clears_the_selection_first_then_leaves_and_backspace_changes_nothing(page):
    iid = open_crop(page, "two.jpg")
    f1, _ = faces(page)
    click_face(page, f1)
    expect(page.page.locator("#crop-subjects .subj-badge")).to_have_count(1)
    page.key("Escape")
    expect(page.page.locator("#crop-subjects .subj-badge")).to_have_count(0)
    expect(page.page.locator("#crop-ui")).to_be_visible()
    page.key("Escape")
    expect(page.page.locator("#crop-ui")).to_be_hidden()
    page.key("Escape")
    open_crop(page, "two.jpg")
    page.key("Backspace")
    assert page.srv.get(f"/api/crop/{iid}?folder={page.srv.folder}")["history"] == []
    page.key("p")  # the global keep shortcut is suspended while cropping
    assert page.image("two.jpg")["overridden"] is False


def test_video_items_have_the_crop_button_disabled(page):
    vid = next(i for i in page.srv.shoot()["images"] if i["video_t"] is not None)
    page.select(vid["name"])
    expect(page.page.locator("#btn-crop")).to_be_disabled()
    page.key("r")
    page.page.wait_for_timeout(200)
    expect(page.page.locator("#crop-ui")).to_be_hidden()
    assert page.srv.get(f"/api/shoot?folder={page.srv.folder}")["images"]


def test_crop_mode_fits_1280_by_800(browser, srv):
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    ui = Ui(ctx.new_page(), srv).open()
    try:
        open_crop(ui, "two.jpg")
        for sel in ("#crop-stage", "#crop-side", "#crop-top"):
            b = ui.page.locator(sel).bounding_box()
            assert b["x"] >= 0 and b["y"] >= 0 and b["x"] + b["width"] <= 1280 + 1 and b["y"] + b["height"] <= 800 + 1, (sel, b)
        for sel in ("#crop-accept", "#crop-reject", "#crop-reset"):
            b = ui.page.locator(sel).bounding_box()
            assert b["y"] + b["height"] <= 800, (sel, b)
        ui.check()
    finally:
        ctx.close()


def test_apply_to_burst_gives_proposals_that_are_accepted_per_frame_or_all(page):
    iid = open_crop(page, "b0.jpg")
    expect(page.page.locator("#crop-burst-btn")).to_be_visible()
    assert "Apply to burst (2)" in page.page.inner_text("#crop-burst-btn")
    page.page.click('#crop-ratios button[data-ratio="1:1"]')
    settle(page)
    page.key("Enter")
    expect(page.page.locator("#toast")).to_contain_text("Crop v1 saved", timeout=15000)
    page.page.click("#crop-burst-btn")
    page.page.wait_for_selector("#crop-burst .burst-item", timeout=60000)
    items = page.page.locator("#crop-burst .burst-item")
    assert items.count() == 2
    others = [i for i in page.srv.shoot()["images"] if i["name"] in ("b1.jpg", "b2.jpg")]
    for o in others:  # proposals only: nothing saved yet
        assert page.srv.get(f"/api/crop/{o['id']}?folder={page.srv.folder}")["history"] == []
    pend = page.page.locator("#crop-burst .burst-item.pending")
    if pend.count():
        pend.first.locator('button[data-do="acc"]').click()
        expect(page.page.locator("#toast")).to_contain_text("burst crop", timeout=15000)
    accepted = [o for o in others if page.srv.get(f"/api/crop/{o['id']}?folder={page.srv.folder}")["history"]]
    assert accepted and accepted[0]["id"] != iid
    h = page.srv.get(f"/api/crop/{accepted[0]['id']}?folder={page.srv.folder}")["history"]
    assert h[0]["source"] == "burst" and h[0]["ratio"] == "1:1"


def test_detector_notice_is_hidden_when_the_model_is_installed(page):
    from shotsieve.models import object_detect

    open_crop(page, "two.jpg")
    if object_detect.available():
        expect(page.page.locator("#crop-detector")).to_be_hidden()
    else:
        expect(page.page.locator("#crop-detector")).to_be_visible()
    json.loads(urllib.request.urlopen(page.srv.url + "/api/health").read())
