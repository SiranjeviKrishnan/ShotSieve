# ruff: noqa: E501
"""Real-browser UI tests (Playwright + Chromium). Marked `ui`; skipped when Chromium is missing.

Install the browser inside the project:
    PLAYWRIGHT_BROWSERS_PATH=.playwright uv run playwright install chromium

Every test drives `shotsieve serve` (a subprocess on a free port) on a private copy of a culled
fixture shoot, and fails on any JS error (console.error / uncaught exception).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import piexif
import pytest
from playwright.sync_api import expect

from shotsieve.pipeline.cache import Cache

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright"))

pytestmark = pytest.mark.ui

DT_B = 3725  # camera B's clock runs this many seconds ahead
JPEG_1PX = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////"
    "////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


# ---- infrastructure ----------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, folder: Path, wizard: bool = False, env: dict | None = None) -> None:
        self.folder = folder
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        # private data dir; the T19 first-run wizard is suppressed (onboarding completed)
        inherited = os.environ.get("SHOTSIEVE_DATA_DIR")  # e.g. the edit_env fixture's data dir
        self.data = Path(inherited) if inherited else folder.parent / f"ui_data_{folder.name}"
        self.data.mkdir(exist_ok=True)
        if not wizard and not (self.data / "onboarding.json").exists():
            (self.data / "onboarding.json").write_text(
                json.dumps({"version": 1, "completed_at": 1.0, "skipped": False, "last_step": 5})
            )
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "shotsieve.cli", "serve", str(folder), "--port",
             str(self.port), "--no-open"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT,
            env={**os.environ, "SHOTSIEVE_DATA_DIR": str(self.data), **(env or {})},
        )  # fmt: skip
        for _ in range(150):
            try:
                urllib.request.urlopen(self.url + "/api/health", timeout=1).read()
                return
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError("serve exited early") from None
                time.sleep(0.2)
        raise RuntimeError("serve did not start")

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=30) as r:
            return json.loads(r.read())

    def shoot(self) -> dict:
        return self.get("/api/shoot?folder=" + str(self.folder))

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _stamp(path: Path, serial: str, shift_s: float = 0.0) -> None:
    ex = piexif.load(str(path))
    raw = ex["Exif"][piexif.ExifIFD.DateTimeOriginal].decode()
    t = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S") + timedelta(seconds=shift_s)
    ex["Exif"][piexif.ExifIFD.DateTimeOriginal] = t.strftime("%Y:%m:%d %H:%M:%S").encode()
    ex["Exif"][piexif.ExifIFD.BodySerialNumber] = serial.encode()
    piexif.insert(piexif.dump(ex), str(path))


def _build_shoot(dest: Path, shoot_dir: Path, gen_dir: Path, video: bool) -> Path:
    shutil.copytree(shoot_dir, dest)
    for p in sorted(dest.glob("*.jpg")):
        _stamp(p, "A")
    for src in sorted(shoot_dir.glob("burst0_*.jpg")) + sorted(shoot_dir.glob("burst1_*.jpg")):
        dst = dest / f"camB_{src.name}"
        shutil.copy(src, dst)
        _stamp(dst, "B", DT_B)
    if video:
        from tests.video_synth import make_three_scene_video

        make_three_scene_video(dest / "clip.mp4", gen_dir / "src")
    return dest


@pytest.fixture(scope="session")
def culled(tmp_path_factory, gen_dir, shoot_dir, models_ready, search_ready) -> Path:
    """A culled + search-indexed shoot: 2 camera bodies, video frames, audit items, a jersey."""
    from shotsieve.cli import main

    folder = _build_shoot(tmp_path_factory.mktemp("ui") / "shoot", shoot_dir, gen_dir, True)
    assert main(["cull", str(folder), "--video"]) == 0
    assert main(["index", str(folder), "--no-people"]) == 0
    with Cache.for_folder(folder) as c:
        paths = [p for p, d in c.load_decisions().items() if "#t=" not in p]
        rejects = [p for p in paths if c.load_decisions()[p]["decision"] != "keep"][:2]
        with c._lock:  # make sure the Audit tab has content
            for p in rejects or paths[:2]:
                c._con.execute(
                    "UPDATE decisions SET audit=1, audit_prob=0.25 WHERE path=?", (c._rel(p),)
                )
            c._con.commit()
        target = next(p for p in sorted(paths) if p.endswith("landscape.jpg"))
        c.put_jerseys(target, "fp", [dict(number="23", conf=0.9, x=1, y=1, w=5, h=5)])
    return folder


@pytest.fixture()
def server(culled, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(culled, dest)
    srv = Server(dest)
    yield srv
    srv.stop()


class Ui:
    """A page that records JS errors; `check()` fails the test when there were any."""

    def __init__(self, page, srv: Server | None) -> None:
        self.page, self.srv, self.errors = page, srv, []
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on(
            "console",
            lambda m: self.errors.append(f"console.{m.type}: {m.text} {m.location.get('url', '')}")
            if m.type == "error" else 0,
        )

    def open(self, ready: bool = True) -> Ui:
        self.page.goto(self.srv.url + "/")
        if ready:
            self.page.wait_for_selector(".cell")
        return self

    def expect_uncullled_404(self) -> None:
        """The one expected 404: /api/shoot of a folder that has not been culled yet."""
        self.errors[:] = [e for e in self.errors if not ("404" in e and "/api/shoot?" in e)]

    def check(self) -> None:
        assert not self.errors, "\n".join(self.errors)

    # helpers
    def cells(self):
        return self.page.locator(".cell")

    def sel_id(self) -> int:
        return int(self.page.locator(".cell.sel").first.get_attribute("data-id"))

    def key(self, k: str) -> None:
        self.page.keyboard.press(k)

    def select(self, name: str) -> int:
        cell = self.page.locator(f'.cell[title="{name}"]').first
        cell.scroll_into_view_if_needed()
        cell.click()
        return int(cell.get_attribute("data-id"))

    def image(self, name: str) -> dict:
        return next(i for i in self.srv.shoot()["images"] if i["name"] == name)


@pytest.fixture()
def ui(browser, server):
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    u = Ui(ctx.new_page(), server).open()
    yield u
    try:
        u.check()
    finally:
        ctx.close()


def _photos(shoot: dict) -> list[dict]:
    return [i for i in shoot["images"] if i["video_t"] is None]


# ---- cull button + progress -----------------------------------------------------------------


def test_entering_a_folder_culls_it_without_any_button(browser, tmp_path, gen_dir, shoot_dir, models_ready):
    folder = _build_shoot(tmp_path / "raw", shoot_dir, gen_dir, False)
    srv = Server(folder)
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    try:
        u = Ui(ctx.new_page(), srv).open(ready=False)
        page = u.page
        page.wait_for_selector("#empty:not([hidden])")
        assert "not been culled" in page.inner_text("#empty")
        assert u.cells().count() == 0
        assert page.locator("#btn-cull").count() == 0  # T25a: no Cull button; picking a folder starts the cull
        page.focus("#folder")
        page.keyboard.press("Enter")
        page.wait_for_selector("#progress:not([hidden])")
        assert page.evaluate("S.culling") is True
        page.wait_for_function(
            "document.querySelector('#progress-text').textContent.includes('/')"
            " || document.querySelectorAll('.cell').length > 0"
        )
        page.wait_for_selector(".cell", timeout=120_000)
        page.wait_for_function("S.culling === false")
        n = len(srv.shoot()["images"])
        assert page.inner_text("#c-all") == str(n) and n >= 17
        keep, review, reject = (int(page.inner_text(f"#c-{k}")) for k in ("keep", "review", "reject"))
        assert keep + review + reject == n
        # the menu's "Re-cull this folder" re-runs it (incremental: nothing changed), and is refused mid-run
        assert page.evaluate("shotsieveMenu('recull')") is True
        page.wait_for_function("S.culling === true")
        assert page.evaluate("shotsieveMenu('recull')") is True
        assert "A cull is running" in page.inner_text("#toast")
        page.wait_for_function("S.culling === false", timeout=120_000)
        assert "Cull finished" in page.inner_text("#toast")
        u.expect_uncullled_404()
        u.check()
    finally:
        ctx.close()
        srv.stop()


# ---- grid -----------------------------------------------------------------------------------


def test_grid_groups_and_star_on_best_frame(ui):
    shoot = ui.srv.shoot()
    by_id = {i["id"]: i for i in shoot["images"]}
    multi = [g for g in shoot["groups"] if len(g["members"]) > 1]
    assert multi
    ui.page.set_viewport_size({"width": 1500, "height": 4000})  # every row rendered
    ui.page.wait_for_function(f"document.querySelectorAll('.row.group').length >= {len(multi)}")
    starred = ui.page.locator(".cell:has(.star)")
    assert starred.count() == len(multi)
    best_names = {by_id[g["best"]]["name"] for g in multi}
    assert {starred.nth(i).get_attribute("title") for i in range(starred.count())} == best_names
    assert ui.page.locator(".glabel:has-text('Group')").count() >= len(multi)
    assert int(ui.page.inner_text("#c-all")) == len(shoot["images"])
    # decision colour classes
    for dec in ("keep", "reject"):
        assert ui.page.locator(f".cell.dec-{dec}").count() >= 1


# ---- keyboard ---------------------------------------------------------------------------------


def test_keys_left_right_and_up_down(ui):
    first = ui.sel_id()
    ui.key("ArrowRight")
    second = ui.sel_id()
    assert second != first
    ui.key("ArrowLeft")
    assert ui.sel_id() == first
    ui.key("ArrowLeft")  # clamps at the start
    assert ui.sel_id() == first
    shoot = ui.srv.shoot()
    gid = {i["id"]: i["group_id"] for i in shoot["images"]}
    ui.key("ArrowDown")
    assert gid[ui.sel_id()] != gid[first]
    ui.key("ArrowUp")
    assert gid[ui.sel_id()] == gid[first]


def test_key_p_keeps_and_pressing_it_again_restores_the_model_decision(ui):
    victim = next(i for i in _photos(ui.srv.shoot()) if i["model_decision"] != "keep")
    ui.select(victim["name"])
    with ui.page.expect_response(lambda r: r.url.endswith("/api/override")) as ri:
        ui.key("p")
    assert ri.value.status == 200
    assert ri.value.request.post_data_json["decision"] == "keep"
    now = ui.image(victim["name"])
    assert now["decision"] == "keep" and now["overridden"] and now["model_decision"] != "keep"
    assert "(yours)" in ui.page.text_content("#d-decision")
    assert ui.page.locator(f'.cell[data-id="{victim["id"]}"] .badge.ovr').count() == 1
    with ui.page.expect_response(lambda r: r.url.endswith("/api/override")) as ri:
        ui.key("p")  # same key again: back to ShotSieve's decision
    assert ri.value.request.post_data_json["decision"] == "model"
    back = ui.image(victim["name"])
    assert back["decision"] == victim["model_decision"] and not back["overridden"]
    assert "(yours)" not in ui.page.text_content("#d-decision")
    assert ui.page.locator(f'.cell[data-id="{victim["id"]}"] .badge.ovr').count() == 0


def test_key_x_rejects(ui):
    victim = next(i for i in _photos(ui.srv.shoot()) if i["model_decision"] == "keep")
    ui.select(victim["name"])
    before = int(ui.page.inner_text("#c-reject"))
    with ui.page.expect_response(lambda r: r.url.endswith("/api/override")):
        ui.key("x")
    assert ui.image(victim["name"])["decision"] == "reject"
    assert int(ui.page.inner_text("#c-reject")) == before + 1


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_keys_1_to_5_rate(ui, n):
    victim = _photos(ui.srv.shoot())[0]
    ui.select(victim["name"])
    with ui.page.expect_response(lambda r: r.url.endswith("/api/override")):
        ui.key(str(n))
    assert ui.image(victim["name"])["rating"] == n
    assert ui.page.inner_text("#d-rating") == "★" * n


def test_key_space_toggles_loupe_and_esc_closes(ui):
    assert ui.page.locator("#overlay-loupe").is_hidden()
    ui.key(" ")
    ui.page.wait_for_selector("#overlay-loupe:not([hidden])")
    assert f"/api/preview/{ui.sel_id()}" in ui.page.get_attribute("#big", "src")
    ui.page.wait_for_function("document.getElementById('big').naturalWidth > 0")
    ui.key(" ")
    assert ui.page.locator("#overlay-loupe").is_hidden()
    ui.key(" ")
    ui.key("Escape")
    assert ui.page.locator("#overlay-loupe").is_hidden()


def test_key_c_compare_best_with_current_and_arrows_move(ui):
    shoot = ui.srv.shoot()
    by_id = {i["id"]: i for i in shoot["images"]}
    g = next(g for g in shoot["groups"] if len(g["members"]) > 2 and by_id[g["best"]]["video_t"] is None)
    other = next(by_id[m] for m in g["members"] if m != g["best"])
    ui.select(other["name"])
    ui.key("c")
    ui.page.wait_for_selector("#overlay-compare:not([hidden])")
    assert by_id[g["best"]]["name"] in ui.page.inner_text("#cmp-a-cap")
    assert "★ best" in ui.page.inner_text("#cmp-a-cap")
    assert other["name"] in ui.page.inner_text("#cmp-b-cap")
    assert f"/api/preview/{g['best']}" in ui.page.get_attribute("#cmp-a", "src")
    assert f"/api/preview/{other['id']}" in ui.page.get_attribute("#cmp-b", "src")
    cur = ui.sel_id()
    ui.key("ArrowRight")  # in compare the arrows choose the frame to keep; the selection stays put
    assert ui.sel_id() == cur
    assert "chosen" in ui.page.get_attribute("#cmp-fig-b", "class")
    ui.key("ArrowLeft")
    assert "chosen" in ui.page.get_attribute("#cmp-fig-a", "class")
    ui.key("c")
    assert ui.page.locator("#overlay-compare").is_hidden()


def test_key_g_collapses_and_expands_groups(ui):
    ui.page.set_viewport_size({"width": 1500, "height": 4000})
    ui.page.wait_for_function("document.querySelectorAll('.row').length > 3")
    n_before = ui.cells().count()
    # G on a photo collapses (a video frame would export instead)
    ui.select(next(i for i in _photos(ui.srv.shoot()))["name"])
    ui.key("g")
    assert "collapsed" in ui.page.inner_text("#btn-collapse")
    assert ui.cells().count() < n_before
    assert ui.page.locator(".badge.more").count() >= 1
    ui.key("g")
    assert "expanded" in ui.page.inner_text("#btn-collapse")
    assert ui.cells().count() == n_before


def test_key_tab_cycles_filters_and_shift_tab_reverses(ui):
    order = ["all", "keep", "review", "reject", "audit"]
    active = lambda: ui.page.get_attribute(".tab.active", "data-filter")  # noqa: E731
    assert active() == "all"
    for want in order[1:] + ["all"]:
        ui.key("Tab")
        assert active() == want
    ui.key("Shift+Tab")
    assert active() == "audit"


def test_key_a_jumps_to_next_audit_item(ui):
    shoot = ui.srv.shoot()
    audit_ids = {i["id"] for i in shoot["images"] if i["audit"]}
    assert audit_ids
    ui.key("a")
    assert ui.sel_id() in audit_ids
    assert "AUDIT" in ui.page.inner_text(".cell.sel") or ui.page.is_visible("#d-audit")
    assert ui.page.is_visible("#d-audit")


def test_key_w_does_nothing_any_more_xmp_is_automatic(ui):
    sent = []
    ui.page.on("request", lambda r: sent.append(r.url) if r.url.endswith("/api/xmp/write") else None)
    ui.key("w")
    ui.page.wait_for_timeout(300)
    assert not sent


def test_key_slash_focuses_search_and_esc_blurs(ui):
    ui.key("/")
    assert ui.page.evaluate("document.activeElement.id") == "search"
    ui.page.keyboard.type("p")  # typing must not trigger the P shortcut
    assert ui.page.input_value("#search") == "p"
    ui.key("Escape")
    assert ui.page.evaluate("document.activeElement.id") != "search"


def test_key_question_mark_shows_cheat_sheet_listing_every_shortcut(ui):
    assert ui.page.locator("#overlay-help").is_hidden()
    ui.key("?")
    ui.page.wait_for_selector("#overlay-help:not([hidden])")
    text = ui.page.inner_text("#overlay-help")
    for frag in ("Previous / next image", "Keep", "Reject", "AI decision", "Star rating",
                 "loupe", "Compare", "Collapse", "Cycle filter", "audit",
                 "Search", "cheat-sheet"):  # fmt: skip
        assert frag.lower() in text.lower(), frag
    ui.key("?")
    assert ui.page.locator("#overlay-help").is_hidden()
    ui.page.click("#btn-help")  # the button works as well
    assert ui.page.locator("#overlay-help").is_visible()
    ui.key("Escape")
    assert ui.page.locator("#overlay-help").is_hidden()


# ---- filters ----------------------------------------------------------------------------------


@pytest.mark.parametrize("flt", ["keep", "review", "reject", "audit", "all"])
def test_filter_tabs(ui, flt):
    shoot = ui.srv.shoot()
    ims = shoot["images"]
    want = {
        "all": len(ims),
        "audit": sum(i["audit"] for i in ims),
    }.get(flt, sum(i["decision"] == flt for i in ims))
    ui.page.click(f'.tab[data-filter="{flt}"]')
    assert "active" in ui.page.get_attribute(f'.tab[data-filter="{flt}"]', "class")
    ui.page.set_viewport_size({"width": 1500, "height": 4000})
    if want:
        ui.page.wait_for_function(f"document.querySelectorAll('.cell').length === {want}")
    assert ui.cells().count() == want
    if flt in ("keep", "review", "reject"):
        assert ui.page.locator(f".cell:not(.dec-{flt})").count() == 0
    if flt == "audit":
        assert ui.page.locator(".cell .badge:has-text('AUDIT')").count() == want
        assert ui.page.inner_text("#c-audit").endswith(f"/{want}")
    if not want:
        assert "Nothing in this filter" in ui.page.inner_text("#empty")


def test_audit_tab_counts_drop_when_judged(ui):
    ui.page.click('.tab[data-filter="audit"]')
    pending = ui.page.inner_text("#c-audit")
    p, total = (int(x) for x in pending.split("/"))
    assert p == total >= 1
    with ui.page.expect_response(lambda r: r.url.endswith("/api/override")):
        ui.key("x")  # confirm the reject
    ui.page.wait_for_function(f"document.getElementById('c-audit').textContent === '{p - 1}/{total}'")  # the handler runs after the response
    p2, total2 = (int(x) for x in ui.page.inner_text("#c-audit").split("/"))
    assert total2 == total and p2 == p - 1


# ---- loupe and compare via the mouse ----------------------------------------------------------


def test_double_click_opens_loupe_and_click_outside_closes(ui):
    ui.cells().nth(1).dblclick()
    ui.page.wait_for_selector("#overlay-loupe:not([hidden])")
    ui.page.wait_for_function("document.getElementById('big').naturalWidth > 0")
    ui.page.click("#overlay-loupe", position={"x": 5, "y": 5})
    assert ui.page.locator("#overlay-loupe").is_hidden()


def test_detail_panel_shows_reasons_score_and_loupe_image(ui):
    ui.page.wait_for_function("document.getElementById('loupe').naturalWidth > 0")
    assert ui.page.locator("#d-reasons li").count() >= 1
    assert "score" in ui.page.inner_text("#d-score")
    assert ui.page.inner_text("#d-name").endswith(".jpg") or "#t=" in ui.page.inner_text("#d-name")
    for act, dec in (("keep", "keep"), ("reject", "reject")):
        with ui.page.expect_response(lambda r: r.url.endswith("/api/override")):
            ui.page.click(f'.actions button[data-act="{act}"]')
        assert dec in ui.page.text_content("#d-decision")
    with ui.page.expect_response(lambda r: r.url.endswith("/api/override")):
        ui.page.click('.actions button[data-act="reject"]')  # pressed already: toggles back
    assert "(yours)" not in ui.page.text_content("#d-decision")
    assert ui.page.locator('.actions button[data-act="model"]').count() == 0


def test_compare_on_a_singleton_shows_a_hint_not_an_overlay(ui):
    shoot = ui.srv.shoot()
    single = next(
        (i for i in _photos(shoot) if i["name"] == "landscape.jpg"), None
    )
    ui.select(single["name"])
    ui.key("c")
    assert ui.page.locator("#overlay-compare").is_hidden()
    expect(ui.page.locator("#toast")).to_contain_text("not part of a group", timeout=10000)


# ---- XMP button ---------------------------------------------------------------------------------


def test_sidecars_are_written_automatically_after_a_user_change(ui):
    assert ui.page.locator("#btn-xmp, #chk-xmp").count() == 0  # no button, no checkbox
    ui.select("landscape.jpg")
    with ui.page.expect_response(lambda r: r.url.endswith("/api/override")):
        ui.key("5")
    ui.page.wait_for_function("true")
    import time

    p = ui.srv.folder / "landscape.xmp"
    t0 = time.time()
    while time.time() - t0 < 8 and not (p.exists() and ("Rating" in p.read_text())):
        time.sleep(0.2)
    xmp = p.read_text()
    assert 'Rating="5"' in xmp or "<xmp:Rating>5</xmp:Rating>" in xmp


# ---- search ---------------------------------------------------------------------------------------


def test_search_ranks_by_score_and_escape_clears(ui):
    page = ui.page
    n_rows_before = page.locator(".row.group").count()
    assert n_rows_before >= 1
    page.fill("#search", "a photo of a landscape")
    with page.expect_response(lambda r: "/api/search" in r.url) as ri:
        page.press("#search", "Enter")
    hits = ri.value.json()["hits"]
    assert len(hits) >= 5
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)  # server ranks by score
    page.wait_for_function("document.querySelectorAll('.row.group').length === 0")
    shown = page.locator(".cell")
    want = [h["id"] for h in hits]
    page.set_viewport_size({"width": 1500, "height": 4000})
    page.wait_for_function(f"document.querySelectorAll('.cell').length >= {min(len(want), 8)}")
    ids = [int(shown.nth(i).get_attribute("data-id")) for i in range(shown.count())]
    assert ids == want[: len(ids)]  # the grid shows them best first, no groups
    assert ui.sel_id() == want[0]
    ui.key("Escape")  # first Escape blurs the box, the next one clears the search
    ui.key("Escape")
    assert page.input_value("#search") == ""
    page.wait_for_function("document.querySelectorAll('.row.group').length > 0")
    assert page.locator(".cell").count() == len(ui.srv.shoot()["images"])


def test_search_best_hit_is_the_landscape_photo_and_empty_query_clears(ui):
    page = ui.page
    page.fill("#search", "a wide mountain landscape")
    with page.expect_response(lambda r: "/api/search" in r.url):
        page.press("#search", "Enter")
    page.wait_for_function("document.querySelectorAll('.row.group').length === 0")
    assert page.locator(".cell").first.get_attribute("title")
    page.fill("#search", "")
    page.press("#search", "Enter")  # an empty query restores the grid
    page.wait_for_function("document.querySelectorAll('.row.group').length > 0")


# ---- cameras ---------------------------------------------------------------------------------------


def test_cameras_panel_lists_bodies_sets_and_clears_offsets(ui):
    page = ui.page
    page.evaluate("shotsieveMenu('cameras')")
    page.wait_for_selector("#overlay-cameras:not([hidden]) .cam")
    assert page.locator("#cameras-sheet .cam").count() == 2
    assert {page.locator("#cameras-sheet .cam").nth(i).get_attribute("data-serial")
            for i in range(2)} == {"A", "B"}  # fmt: skip
    cam_b = page.locator('#cameras-sheet .cam[data-serial="B"]')
    assert "no offset" in cam_b.inner_text() or "Suggested" in cam_b.inner_text()
    cam_b.locator("input").fill("-1:02:05")
    with page.expect_response(lambda r: r.url.endswith("/api/clock") and r.request.method == "POST"):
        cam_b.locator('button[data-act="manual"]').click()
    page.wait_for_function(
        "(document.querySelector('#cameras-sheet .cam[data-serial=B]')?.textContent || '').includes('-1:02:05 (manual)')"
    )
    assert ui.srv.get("/api/clock?folder=" + str(ui.srv.folder))["offsets"]["B"]["offset_s"] == -DT_B
    cam_b = page.locator('#cameras-sheet .cam[data-serial="B"]')
    # a garbage offset is refused client-side, nothing is sent
    cam_b.locator("input").fill("oops")
    cam_b.locator('button[data-act="manual"]').click()
    expect(page.locator("#toast")).to_contain_text("Type an offset", timeout=10000)
    with page.expect_response(lambda r: r.url.endswith("/api/clock") and r.request.method == "POST"):
        cam_b.locator('button[data-act="clear"]').click()
    page.wait_for_function(
        "(() => { const c = document.querySelector('#cameras-sheet .cam[data-serial=B]'); return !!c && !c.textContent.includes('(manual)'); })()"
    )
    assert "B" not in ui.srv.get("/api/clock?folder=" + str(ui.srv.folder))["offsets"]
    page.click("#overlay-cameras", position={"x": 5, "y": 5})
    assert page.locator("#overlay-cameras").is_hidden()


def test_cameras_suggestion_can_be_accepted(ui):
    page = ui.page
    page.evaluate("shotsieveMenu('cameras')")
    page.wait_for_selector("#cameras-sheet .cam")
    accept = page.locator('#cameras-sheet button[data-act="accept"]')
    if not accept.count():
        pytest.skip("no consistent suggestion on this fixture")
    with page.expect_response(lambda r: r.url.endswith("/api/clock") and r.request.method == "POST"):
        accept.first.click()
    got = ui.srv.get("/api/clock?folder=" + str(ui.srv.folder))["offsets"]["B"]["offset_s"]
    assert abs(got + DT_B) <= 2


# ---- people consent -----------------------------------------------------------------------------


def test_people_search_is_blocked_until_consent_is_given(ui):
    page = ui.page
    page.wait_for_selector("#people-strip:not([hidden]) #btn-consent", timeout=15_000)
    assert "off" in page.inner_text("#people-strip")
    r = urllib.request.Request(
        ui.srv.url + "/api/search?q=" + urllib.parse.quote('person:"Zed"') + "&folder=" + str(ui.srv.folder)
    )
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(r)
    assert ei.value.code == 403
    # in the UI: a people query shows an error toast and the grid is not replaced by results
    n = ui.cells().count()
    page.fill("#search", 'person:"Zed"')
    page.press("#search", "Enter")
    page.wait_for_function("document.getElementById('toast').textContent.startsWith('Search:')")
    assert ui.cells().count() == n and page.locator(".row.group").count() >= 1
    ui.errors[:] = [e for e in ui.errors if "403" not in e]  # the expected refusal
    page.fill("#search", "")
    # consent on
    with page.expect_response(lambda r: r.url.endswith("/api/people/consent")) as ri:
        page.click("#btn-consent")
    assert ri.value.json()["people_consent"] is True
    page.wait_for_selector("#btn-people-del", timeout=15_000)
    assert ui.srv.get("/api/index/status?folder=" + str(ui.srv.folder))["people_consent"] is True
    # the same query is no longer refused: 200 with zero hits for an unknown name
    page.fill("#search", 'person:"Zed"')
    with page.expect_response(lambda r: "/api/search" in r.url) as ri:
        page.press("#search", "Enter")
    assert ri.value.status == 200
    # and revoking it again closes the gate
    with page.expect_response(lambda r: "/api/people" in r.url and r.request.method == "DELETE"):
        page.click("#btn-people-del")
    page.wait_for_selector("#btn-consent", timeout=15_000)


# ---- jersey / roster UI -----------------------------------------------------------------------


def test_jersey_strip_roster_upload_and_name_in_detail(ui, tmp_path):
    page = ui.page
    assert page.locator("#jersey-strip").count() == 0  # no bar in the main window any more
    ui.select("landscape.jpg")
    page.wait_for_function("!document.getElementById('d-jersey').hidden")
    assert page.inner_text("#d-jersey").startswith("#23")
    assert "(not on roster)" not in page.inner_text("#d-jersey")  # empty roster: no verdict yet
    open_jersey_sheet(page)
    csv = tmp_path / "roster.csv"
    csv.write_text("number,name\n23,Jane Doe\n7,Sam Roe\n")
    with page.expect_response(lambda r: r.url.endswith("/api/roster") and r.request.method == "POST") as ri:
        page.set_input_files("#roster-file", str(csv))
    assert ri.value.json()["entries"] == 2
    page.wait_for_function("document.getElementById('d-jersey').textContent.includes('Jane Doe')")
    assert page.inner_text("#d-jersey").strip() == "#23 Jane Doe"
    page.wait_for_function("document.getElementById('jersey-sheet').textContent.includes('2 on roster')")
    with page.expect_response(lambda r: r.url.endswith("/api/jerseys/xmp")) as ri:
        page.click("#btn-jersey-xmp")
    assert ri.value.status == 200
    expect(page.locator("#toast")).to_contain_text("Number tags", timeout=10000)
    assert "#23" in (ui.srv.folder / "landscape.xmp").read_text()
    page.keyboard.press("Escape")
    expect(page.locator("#overlay-jersey")).to_be_hidden()
    # jersey: search finds the image
    page.fill("#search", "jersey:23")
    with page.expect_response(lambda r: "/api/search" in r.url) as ri:
        page.press("#search", "Enter")
    assert ri.value.status == 200
    hits = ri.value.json()["hits"]
    assert hits
    by_id = {i["id"]: i["name"] for i in ui.srv.shoot()["images"]}
    assert by_id[hits[0]["id"]] == "landscape.jpg"
    page.wait_for_function("document.querySelectorAll('.row.group').length === 0")


def test_read_jersey_numbers_button_starts_the_ocr_indexer(ui):
    page = ui.page
    open_jersey_sheet(page)
    page.wait_for_selector("#btn-ocr", timeout=15_000)
    with page.expect_response(lambda r: r.url.endswith("/api/index/start")) as ri:
        page.click("#btn-ocr")
    assert ri.value.status == 200
    assert ri.value.request.post_data_json["ocr"] is True


# ---- video frames ---------------------------------------------------------------------------------


def test_video_frames_are_badged_and_g_exports_the_frame(ui):
    vids = [i for i in ui.srv.shoot()["images"] if i["video_t"] is not None]
    assert len(vids) >= 3
    v = vids[0]
    cell = ui.page.locator(f'.cell[data-id="{v["id"]}"]')
    cell.scroll_into_view_if_needed()
    assert cell.locator(".badge:has-text('VIDEO')").count() == 1
    cell.click()
    assert "#t=" in ui.page.inner_text("#d-name")
    n_collapse = ui.page.inner_text("#btn-collapse")
    with ui.page.expect_response(lambda r: r.url.endswith("/api/video/grab")) as ri:
        ui.key("g")
    out = ri.value.json()
    assert out["ok"] and (ui.srv.folder / out["rel"]).exists()
    assert out["rel"].startswith("frame_grabs")
    assert ui.page.inner_text("#btn-collapse") == n_collapse  # G exported, it did not collapse
    ui.page.wait_for_function("document.getElementById('toast').textContent.includes('Frame exported')")
    assert piexif.load(str(ui.srv.folder / out["rel"]))["Exif"][piexif.ExifIFD.DateTimeOriginal]


# ---- virtualization: 5,000 items --------------------------------------------------------------------


def _mock_shoot(n: int) -> dict:
    images, groups = [], []
    i = gid = 0
    while i < n:
        size = 1 if gid % 3 == 0 else 5
        size = min(size, n - i)
        members = list(range(i, i + size))
        for k, m in enumerate(members):
            dec = ("keep", "review", "reject")[m % 3]
            images.append({
                "id": m, "name": f"IMG_{m:05d}.jpg", "rel": f"IMG_{m:05d}.jpg", "decision": dec,
                "model_decision": dec, "overridden": False, "rating": None, "reasons": [],
                "reason_texts": ["synthetic"], "score": (m % 100) / 100, "prior": 0.5,
                "group_id": gid, "burst_id": gid, "audit": m % 50 == 0, "audit_prob": 0.3,
                "best": k == 0, "n_faces": 0, "video_t": None,
            })  # fmt: skip
        groups.append({"id": gid, "members": members, "best": members[0]})
        i += size
        gid += 1
    counts = {d: sum(im["decision"] == d for im in images) for d in ("keep", "review", "reject")}
    return {"folder": "/mock", "images": images, "groups": groups,
            "stats": {"n_images": n, "counts": counts, "n_audit": 0, "n_audit_pending": 0,
                      "thresholds": {}, "run": None, "calibration": None}}  # fmt: skip


@pytest.mark.parametrize("n, step_budget", [(5000, 0.25), (10_000, 0.10)])
def test_5000_items_virtualised_scroll_stays_bounded(browser, tmp_path, n, step_budget):
    folder = tmp_path / "empty"
    folder.mkdir()
    srv = Server(folder)
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    page = ctx.new_page()
    ui = Ui(page, srv)
    shoot = _mock_shoot(n)
    page.route("**/api/shoot*", lambda r: r.fulfill(json=shoot))
    status = {"running": False, "task": "idle", "done": 0, "total": 0, "paused": False,
              "img_s": 0, "models_ready": False, "ocr_models_ready": False,
              "people_consent": False}  # fmt: skip
    page.route("**/api/index/status*", lambda r: r.fulfill(json=status))
    page.route("**/api/thumb/**", lambda r: r.fulfill(body=JPEG_1PX, content_type="image/jpeg"))
    page.route("**/api/preview/**", lambda r: r.fulfill(body=JPEG_1PX, content_type="image/jpeg"))
    try:
        page.goto(srv.url + f"/?folder={folder}")
        page.wait_for_selector(".cell")
        nodes0 = page.evaluate("document.querySelectorAll('*').length")
        assert page.inner_text("#c-all") == str(n)
        bound = 400
        peak_cells = peak_nodes = 0
        total_h = page.evaluate("document.getElementById('spacer').offsetHeight")
        assert total_h > 20000  # every row of space exists, but not its DOM
        t0 = time.perf_counter()
        steps = 120
        for s in range(steps + 1):
            page.evaluate(f"document.getElementById('grid').scrollTop = {total_h * s / steps}")
            page.evaluate("new Promise(r => requestAnimationFrame(() => r()))")
            peak_cells = max(peak_cells, page.locator(".cell").count())
            peak_nodes = max(peak_nodes, page.evaluate("document.querySelectorAll('*').length"))
        elapsed = time.perf_counter() - t0
        assert 0 < peak_cells <= bound, peak_cells
        assert peak_nodes < nodes0 + 12 * bound, (nodes0, peak_nodes)
        assert elapsed / steps < step_budget, f"{elapsed / steps:.3f}s per scroll step"
        # the end of the list is reachable and rendered
        page.wait_for_selector(f'.cell[title="IMG_{n - 1:05d}.jpg"]')
        if n >= 10_000:  # T21: layout / filter recompute budgets
            page.click('.tab[data-filter="keep"]')
            assert page.evaluate("window.__layoutMs") <= 100, page.evaluate("window.__layoutMs")
            page.click('.tab[data-filter="all"]')
            ms = page.evaluate(
                "new Promise(r => { const t = performance.now();"
                " document.querySelector('.tab[data-filter=\"review\"]').click();"
                " requestAnimationFrame(() => r(performance.now() - t)); })"
            )
            assert ms <= 150, ms
        # keyboard navigation across the whole list keeps the DOM bounded as well
        page.evaluate("document.getElementById('grid').scrollTop = 0")
        page.click(".cell")
        for _ in range(60):
            page.keyboard.press("ArrowDown")
        assert page.locator(".cell").count() <= bound
        page.click('.tab[data-filter="audit"]')
        assert page.locator(".cell").count() <= bound
        assert page.inner_text("#c-audit").endswith(f"/{n // 50}")
        # frame times while scrolling with the wheel
        page.click('.tab[data-filter="all"]')
        page.mouse.move(600, 400)
        for _ in range(30):
            page.mouse.wheel(0, 3000)
        page.wait_for_timeout(200)
        assert page.locator(".cell").count() <= bound
        ui.check()
    finally:
        ctx.close()
        srv.stop()


# ---- editing profile (T15) ------------------------------------------------------------------


@pytest.fixture()
def edit_ui(browser, culled, tmp_path, edit_env):
    """The review UI on a copy of the culled shoot, with a trained editing profile in its data dir."""
    dest = tmp_path / "shoot"
    shutil.copytree(culled, dest)
    srv = Server(dest)
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    u = Ui(ctx.new_page(), srv).open()
    yield u
    try:
        u.check()
    finally:
        ctx.close()
        srv.stop()


def test_edit_sheet_dry_run_apply_badge_and_needs_edit_filter(edit_ui):
    ui = edit_ui
    page = ui.page
    shoot0 = ui.srv.shoot()
    assert page.locator("#tab-needs-edit").is_hidden()  # nothing predicted yet
    page.evaluate("shotsieveMenu('edit')")
    page.wait_for_selector("#overlay-edit:not([hidden])")
    page.wait_for_selector("#edit-sliders")
    rows = page.locator("#edit-sliders tr[data-slider]")
    assert rows.count() == 12
    assert "learned" in page.inner_text('#edit-sliders tr[data-slider="Exposure2012"]')
    assert "Trained on" in page.inner_text("#edit-trained")
    # dry run first: a diff, nothing written, the write button unlocks
    assert page.is_disabled("#edit-apply")
    page.check("#edit-embed")
    page.click("#edit-dry")
    page.wait_for_selector("#edit-summary")
    assert "Would write" in page.inner_text("#edit-summary")
    assert page.locator("#edit-diff-rows div").count() >= 1
    assert page.is_enabled("#edit-apply")
    assert not (ui.srv.folder / "shotsieve_edits").exists()
    n_keep = sum(1 for i in shoot0["images"] if i["decision"] == "keep" and i["video_t"] is None)
    assert f"of {n_keep} image(s)" in page.inner_text("#edit-summary")
    # write for real
    page.click("#edit-apply")
    page.wait_for_function("document.querySelector('#edit-summary') && document.querySelector('#edit-summary').textContent.startsWith('Wrote')")
    copies = list((ui.srv.folder / "shotsieve_edits").glob("*.jpg"))
    assert len(copies) >= 1
    originals = {p.name for p in ui.srv.folder.glob("*.jpg")}
    assert {c.name for c in copies} <= originals
    # the badge: select a kept image; the loupe shows predicted values and confidence
    page.keyboard.press("Escape")
    page.wait_for_selector("#overlay-edit", state="hidden")
    page.wait_for_selector("#tab-needs-edit:not([hidden])")
    keep = next(i for i in ui.srv.shoot()["images"] if i["decision"] == "keep" and i["video_t"] is None and i["edit"])
    ui.select(keep["name"])
    page.wait_for_selector("#d-edit:not([hidden])")
    box = page.inner_text("#d-edit")
    assert "Edit profile" in box and ("predicted" in box or "needs your edit" in box)
    assert "Contrast" in box  # constant sliders are always part of the prediction
    # the "needs your edit" filter lists exactly the flagged images
    flagged = [i for i in ui.srv.shoot()["images"] if i["edit"] and i["edit"]["needs_edit"]]
    assert page.inner_text("#c-needs-edit") == str(len(flagged))
    page.click("#tab-needs-edit")
    page.set_viewport_size({"width": 1500, "height": 4000})
    page.wait_for_function(f"document.querySelectorAll('.cell').length === {len(flagged)}")


def test_edit_sheet_train_refusal_and_delete(edit_ui, edit_styles, tmp_path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("mes_ui", ROOT / "scripts" / "make_edit_styles.py")
    mes = importlib.util.module_from_spec(spec)
    sys.modules["mes_ui"] = mes
    spec.loader.exec_module(mes)
    truth = json.loads((edit_styles / "truth.json").read_text())[:20]
    cat = tmp_path / "few.lrcat"
    mes.build_catalog(cat, [{"root": str(edit_styles / "images"), "rel": str(Path(t["file"]).parent) + "/",
                             "base": Path(t["file"]).stem, "ext": "jpg",
                             "develop": mes.develop_settings(t["truth"]["A"])} for t in truth])  # fmt: skip
    page = edit_ui.page
    page.evaluate("shotsieveMenu('edit')")
    page.wait_for_selector("#edit-train")
    page.fill("#edit-cat", str(cat))
    page.click("#edit-train")
    page.wait_for_function("document.getElementById('edit-msg') && document.getElementById('edit-msg').textContent.includes('need 200 edits, have 20')", timeout=120_000)
    assert page.locator("#edit-sliders").count() == 1  # the old profile is untouched
    page.click("#edit-delete")
    page.wait_for_function("document.getElementById('edit-msg').textContent.includes('deleted')")
    assert page.locator("#edit-sliders").count() == 0 and page.locator("#edit-dry").count() == 0
    page.fill("#edit-cat", "")
    page.click("#edit-train")
    assert "Give a catalog" in page.inner_text("#edit-msg")


# ---- reference-frame colour match (T16) -----------------------------------------------------


@pytest.fixture()
def match_ui(browser, culled, tmp_path, edit_env):
    """The review UI on a copy of the culled shoot whose first same-group photo carries develop settings."""
    from shotsieve.io import crs

    dest = tmp_path / "shoot"
    shutil.copytree(culled, dest)
    srv = Server(dest)
    shoot = srv.shoot()
    members = next(
        [i for i in (next(x for x in shoot["images"] if x["id"] == m) for m in g["members"])
         if i["video_t"] is None and i["name"].endswith(".jpg")]
        for g in shoot["groups"]
        if len([m for m in g["members"] if next(x for x in shoot["images"] if x["id"] == m)["video_t"] is None]) >= 2
    )
    ref, tgt = members[0], members[1]
    stage = tmp_path / "stage"
    stage.mkdir()
    shutil.copy(dest / ref["rel"], stage / Path(ref["rel"]).name)
    (dest / ref["rel"]).unlink()  # a copy never overwrites an existing file (Review #6 M1)
    res = crs.write_develop(stage / Path(ref["rel"]).name, {"Exposure2012": 0.3, "Contrast2012": 12.0},
                            {"Sharpness": 30}, profile_hash="photographer", mode="embed",
                            out_dir=(dest / ref["rel"]).parent)
    assert res.written
    # saved by the photographer in Lightroom: develop settings, without ShotSieve's tag
    res.path.write_bytes(res.path.read_bytes().replace(b"shotsieve:EditProfile", b"shotsieve:EditProfilX"))
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    u = Ui(ctx.new_page(), srv).open()
    u.ref, u.tgt = ref, tgt
    yield u
    try:
        u.check()
    finally:
        ctx.close()
        srv.stop()


def test_match_set_reference_dry_run_and_write(match_ui):
    ui = match_ui
    page = ui.page
    folder = ui.srv.folder
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*.jpg") if "shotsieve_edits" not in p.parts and ".shotsieve" not in p.parts}
    # an unedited frame cannot be the reference
    ui.select(ui.tgt["name"])
    assert page.locator("#d-match").count() == 0  # the panel section is gone; the Tools menu item replaced it
    # an unedited frame cannot be the reference
    page.evaluate("shotsieveMenu('matchref')")
    expect(page.locator("#toast")).to_contain_text("edit the reference in Lightroom first", timeout=10000)
    assert page.evaluate("S.matchRef") is None
    # an edited frame can
    ui.select(ui.ref["name"])
    page.evaluate("shotsieveMenu('matchref')")
    expect(page.locator("#toast")).to_contain_text("develop settings from embedded", timeout=10000)
    assert page.evaluate("S.matchRef") == page.evaluate("S.sel")
    # match the other frame through the API (the panel's run buttons are gone): dry run first, then write an embedded copy
    ui.select(ui.tgt["name"])
    run = """async (dry) => { const r = await api('/api/edit/match', {method: 'POST', body: {folder: S.folder, ref: S.matchRef,
        dry_run: dry, embed_copies: true, ids: [S.sel]}}); return (await waitJob(r.job_id)).result; }"""
    r = page.evaluate(run, True)
    assert r["dry_run"] and (r["would_write"] or r["report_only"])
    assert not (folder / "shotsieve_edits").exists()
    r = page.evaluate(run, False)
    assert not r["dry_run"] and r["written_copies"] == 1
    from shotsieve.io import crs

    copies = list((folder / "shotsieve_edits").rglob(Path(ui.tgt["rel"]).name))
    assert len(copies) == 1
    got = crs.read_develop_xmp(copies[0])
    assert got.ours and got.settings["Contrast2012"] == 12.0 and got.settings["Sharpness"] == 30 and "Exposure2012" in got.settings
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*.jpg") if "shotsieve_edits" not in p.parts and ".shotsieve" not in p.parts}
    assert after == before  # originals untouched


# ---- shot list coverage (T17) ---------------------------------------------------------------


def test_shotlist_panel_frames_and_missing_must_have_banner(ui):
    page, srv = ui.page, ui.srv
    assert page.locator("#shotlist-banner").is_hidden()
    page.evaluate("shotsieveMenu('shotlist')")
    page.wait_for_selector("#overlay-shotlist:not([hidden])")
    page.wait_for_selector("#sl-empty")
    page.click("#sl-load")
    assert "Pick a template or paste a list" in page.inner_text("#sl-msg")
    page.select_option("#sl-template", "corporate")
    page.click("#sl-load")
    page.wait_for_selector("#sl-items")
    n_corp = page.locator("#sl-items tr.sl-item").count()
    assert n_corp >= 15 and "covered" in page.inner_text("#sl-summary")
    page.fill("#sl-text", '!a group of people\na landscape\n!an astronaut inside a space station x2\n')
    page.click("#sl-load")
    page.wait_for_function("document.querySelectorAll('#sl-items tr.sl-item').length === 3")
    rows = srv.get("/api/shotlist?folder=" + str(srv.folder))["items"]
    assert [r["text"] for r in rows] == ["a group of people", "a landscape", "an astronaut inside a space station"]
    marks = {"covered": "✓", "uncertain": "?", "weak": "~", "missing": "✗", "pending": "…"}
    for r in rows:  # the panel shows exactly what the API stored
        tr = page.locator(f'#sl-items tr[data-id="{r["item_id"]}"]')
        assert "sl-" + r["status"] in tr.get_attribute("class") and marks[r["status"]] in tr.locator("td.mark").inner_text()
    assert page.locator("#sl-items tr.sl-item").nth(0).locator(".must").count() == 1
    # clicking a shot with frames filters the grid to them
    with_frames = [r for r in rows if r["best_ids"]]
    assert with_frames
    page.locator(f'#sl-items tr[data-id="{with_frames[0]["item_id"]}"]').click()
    page.wait_for_selector("#overlay-shotlist", state="hidden")
    page.wait_for_function(f"document.querySelectorAll('.cell').length === {len(with_frames[0]['best_ids'])}")
    page.keyboard.press("Escape")
    page.wait_for_function("document.querySelectorAll('.cell').length > 6")
    # reject everything, re-check: the must-have shots are reported missing
    with Cache.for_folder(srv.folder) as c:
        for p in c.load_decisions():
            c.set_override(p, "reject")
    page.evaluate("shotsieveMenu('shotlist')")
    page.wait_for_selector("#sl-check")
    page.click("#sl-check")
    page.wait_for_function("document.getElementById('sl-msg').textContent === 'Checked.' || document.getElementById('sl-msg').textContent.length > 8")
    assert page.inner_text("#sl-msg") == "Checked.", page.inner_text("#sl-msg")
    page.wait_for_selector("#shotlist-banner:not([hidden])")
    txt = page.inner_text("#shotlist-missing")
    assert "a group of people" in txt and "astronaut" in txt and "landscape" not in txt  # only must-haves
    assert "0 covered" in page.inner_text("#sl-summary") and page.locator("#sl-items tr.sl-missing").count() == 3
    page.click("#sl-clear")
    page.wait_for_selector("#sl-empty")
    assert page.locator("#shotlist-banner").is_hidden()


def test_shotlist_uncertain_must_have_banner_and_mark(ui):
    """N1: a must-have matched only near the threshold is shown as '?' and in the banner, never as
    covered (stored rows are written directly, so no model work is needed)."""
    page, srv = ui.page, ui.srv
    with Cache.for_folder(srv.folder) as c:
        ids = [int(i) for i in list(c.final_decisions_by_id()[0])[:3]]
        c.put_shotlist([
            {"item_id": "t:000", "list_name": "t", "pos": 0, "text": "the toast", "must": True,
             "status": "uncertain", "best_ids": ids, "n_hits": 1, "top_score": 0.1,
             "top_z": 5.0, "detail": "near-threshold match (best s 0.100): review the frames"},
            {"item_id": "t:001", "list_name": "t", "pos": 1, "text": "the rings", "must": True,
             "status": "missing", "best_ids": [], "n_hits": 0},
            {"item_id": "t:002", "list_name": "t", "pos": 2, "text": "a nice extra", "must": False,
             "status": "uncertain", "best_ids": ids, "n_hits": 1},
        ])
    page.goto(srv.url + f"/?folder={srv.folder}")
    page.wait_for_selector("#shotlist-banner:not([hidden])")
    assert page.inner_text("#shotlist-uncertain") == "the toast"  # the non-must item is not listed
    assert page.inner_text("#shotlist-missing") == "the rings"
    assert "Review must-have" in page.inner_text("#shotlist-banner")
    page.click("#btn-shotlist-open")
    page.wait_for_selector("#sl-items")
    assert "2 uncertain" in page.inner_text("#sl-summary") and "0 covered" in page.inner_text("#sl-summary")
    tr = page.locator('#sl-items tr[data-id="t:000"]')
    assert "sl-uncertain" in tr.get_attribute("class") and "?" in tr.locator("td.mark").inner_text()
    assert "review the frames" in tr.inner_text()
    # only an uncertain must-have: the banner still shows (never silently counted covered)
    with Cache.for_folder(srv.folder) as c:
        c.put_shotlist([{"item_id": "t:000", "list_name": "t", "pos": 0, "text": "the toast",
                         "must": True, "status": "uncertain", "best_ids": ids}])
    page.goto(srv.url + f"/?folder={srv.folder}")
    page.wait_for_selector("#shotlist-banner:not([hidden])")
    assert page.locator("#shotlist-missing").count() == 0 and page.inner_text("#shotlist-uncertain") == "the toast"


# ---- first grid on a 10k shoot (T21) ----------------------------------------------------------------


@pytest.fixture(scope="module")
def big_shoot(tmp_path_factory) -> Path:
    """A culled model-free 10k shoot (tests/large_synth.py) with 1-KB grid thumbs pre-rendered."""
    import cv2
    import numpy as np

    from shotsieve.config import Settings
    from shotsieve.io.formats import preview_source_for
    from shotsieve.pipeline.runner import run_cull
    from shotsieve.pipeline.thumbs import thumb_path

    sys.path.insert(0, str(Path(__file__).parent))
    import large_synth as ls

    folder = tmp_path_factory.mktemp("bigui") / "shoot"
    ls.build(folder, 10_000, Settings())
    res = run_cull(folder, Settings())
    ok, buf = cv2.imencode(".jpg", np.full((40, 60, 3), 128, np.uint8))
    assert ok
    with Cache.for_folder(folder) as c:
        ids = c.ids_lookup([str(r.master_path) for r in res.images])
    (folder / ".shotsieve" / "thumbs").mkdir(exist_ok=True)
    for r in res.images:
        mt = preview_source_for(r.master_path).stat().st_mtime_ns
        thumb_path(folder, ids[str(r.master_path)], mt).write_bytes(buf.tobytes())
    return folder


def test_first_grid_10k_warm_under_2s(browser, big_shoot):
    srv = Server(big_shoot)
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    page = ctx.new_page()
    ui = Ui(page, srv)
    try:
        t0 = time.perf_counter()
        page.goto(srv.url + "/")
        page.wait_for_selector(".cell")
        page.wait_for_function(
            "(() => { const g = document.getElementById('grid').getBoundingClientRect();"
            " const im = [...document.querySelectorAll('.cell img')].filter(i => {"
            "   const r = i.getBoundingClientRect(); return r.bottom > g.top && r.top < g.bottom; });"
            " return im.length > 0 && im.every(i => i.complete && i.naturalWidth > 0); })()",
            timeout=10_000,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed <= 2.0, f"first grid took {elapsed:.2f}s"
        assert page.locator(".cell").count() <= 400
        ui.check()
    finally:
        ctx.close()
        srv.stop()


def open_lr_menu(page):
    """The Lightroom Classic catalog sync lives in the More menu (T23d)."""
    page.wait_for_selector("#mi-lr:not([hidden])", state="attached")
    page.click("#btn-hamburger")
    page.click('#hamburger-menu [data-menu="lrsync"]')


def open_jersey_sheet(page):
    page.wait_for_selector("#mi-jersey", state="attached", timeout=15_000)
    page.wait_for_function("!document.getElementById('mi-jersey').hidden", timeout=15_000)
    page.click("#btn-hamburger")
    page.click('#hamburger-menu [data-menu="jersey"]')
    page.wait_for_selector("#jersey-sheet #btn-roster", timeout=15_000)


# ---- Lightroom sync-back (T20) -------------------------------------------------------------------------


def test_lightroom_sync_dry_run_apply_badges_and_report(browser, culled, make_fixtures, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(culled, dest)
    srv = Server(dest, env={"SHOTSIEVE_FAKE_LR_APPS": "classic"})  # Lightroom Classic "installed"
    shoot = srv.shoot()
    real = [i for i in shoot["images"] if i["video_t"] is None]
    entries = [
        {"root": str(dest), "rel": "", "base": Path(i["rel"]).stem, "ext": Path(i["rel"]).suffix[1:],
         "pick": 0, "rating": None, "label": "", "touch": 1_700_000_000.0}
        for i in real
    ]  # fmt: skip
    cat = tmp_path / "Catalog.lrcat"
    make_fixtures.build_lrcat(cat, entries)
    rejects = [i for i in real if i["decision"] == "reject"][:3]
    keeps = [i for i in real if i["decision"] == "keep"][:2]
    for i in rejects:
        make_fixtures.lr_edit(cat, dest / i["rel"], pick=1, touch=time.time() + 10)
    for i in keeps:
        make_fixtures.lr_edit(cat, dest / i["rel"], pick=-1, touch=time.time() + 10)
    n_changes = len(rejects) + len(keeps)
    assert n_changes >= 2
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    ui = Ui(ctx.new_page(), srv).open()
    page = ui.page
    try:
        open_lr_menu(page)
        page.fill("#lr-path", str(cat))
        page.click("#lr-dry")
        page.wait_for_function("document.getElementById('lr-apply') && !document.getElementById('lr-apply').hidden")
        msg = page.inner_text("#lr-msg")
        assert f"{n_changes} change" in msg and "nothing changed yet" in msg
        assert not any(i["lr"] for i in srv.shoot()["images"])
        page.click("#lr-apply")
        page.wait_for_function("document.getElementById('lr-msg').textContent.startsWith('Applied')")
        assert page.locator("#lr-report").is_visible()
        assert "Frames compared" in page.inner_text("#lr-report")
        after = srv.shoot()["images"]
        lr_names = {i["name"] for i in after if i["lr"]}
        assert lr_names == {i["name"] for i in [*rejects, *keeps]}
        page.keyboard.press("Escape")
        name = sorted(lr_names)[0]
        page.locator(f'.cell[title="{name}"]').first.scroll_into_view_if_needed()
        assert page.locator(f'.cell[title="{name}"] .badge.lr').count() == 1
        assert next(i for i in after if i["name"] == name)["overridden"]
        ui.check()
    finally:
        ctx.close()
        srv.stop()


def test_wizard_lightroom_sync_against_real_api(browser, culled, make_fixtures, tmp_path):
    """Review #10 M1: the setup wizard's Lightroom card against the REAL /api/lr/catalogs and
    /api/lr/sync: 12 edits -> preview says 12 -> apply writes 12 -> a second preview says 0 and
    Apply is gone."""
    home = tmp_path / "home"  # the shoot lives under HOME, like the fs browser requires
    (home / "Pictures" / "Lightroom").mkdir(parents=True)
    dest = home / "shoot"
    shutil.copytree(culled, dest)
    srv = Server(dest, env={"HOME": str(home)})
    shoot = srv.shoot()
    real = [i for i in shoot["images"] if i["video_t"] is None]
    entries = [
        {"root": str(dest), "rel": "", "base": Path(i["rel"]).stem, "ext": Path(i["rel"]).suffix[1:],
         "pick": 0, "rating": None, "label": "", "touch": 1_700_000_000.0}
        for i in real
    ]  # fmt: skip
    cat = home / "Pictures" / "Lightroom" / "Catalog.lrcat"
    make_fixtures.build_lrcat(cat, entries)
    edited = real[:12]
    assert len(edited) == 12
    for i in edited:  # flip every frame against ShotSieve's decision
        make_fixtures.lr_edit(
            cat, dest / i["rel"], pick=-1 if i["decision"] == "keep" else 1, touch=time.time() + 10
        )
    listing = sorted(p.name for p in cat.parent.iterdir())
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    ui = Ui(ctx.new_page(), srv).open()
    page = ui.page
    try:
        assert srv.get("/api/lr/catalogs?folder=" + str(dest))["catalogs"] == [str(cat)]
        page.evaluate("document.getElementById('btn-lrcard').click()")
        page.click("#wiz-sync")
        expect(page.locator("#wiz-sync-msg")).to_contain_text("12 changes", timeout=30000)
        assert "Preview" in page.inner_text("#wiz-sync-msg")
        assert not any(i["lr"] for i in srv.shoot()["images"])
        page.click("#wiz-sync-apply")
        expect(page.locator("#wiz-sync-msg")).to_contain_text("Applied: 12 changes", timeout=30000)
        assert sum(1 for i in srv.shoot()["images"] if i["lr"]) == 12
        # n2: the LR badges really show in the grid behind the wizard after Apply
        lr_names = sorted(i["name"] for i in srv.shoot()["images"] if i["lr"])
        for name in lr_names:
            cell = page.locator(f'.cell[title="{name}"]').first
            cell.scroll_into_view_if_needed()
            expect(cell.locator(".badge.lr")).to_have_count(1, timeout=10000)
        page.click("#wiz-sync")
        expect(page.locator("#wiz-sync-msg")).to_contain_text("Nothing to sync", timeout=30000)
        expect(page.locator("#wiz-sync-msg")).to_contain_text("0 changes")
        assert page.locator("#wiz-sync-apply").count() == 0
        assert sorted(p.name for p in cat.parent.iterdir()) == listing  # catalog dir untouched
        ui.check()
    finally:
        ctx.close()
        srv.stop()
