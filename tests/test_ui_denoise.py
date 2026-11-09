# ruff: noqa: E501
"""T23f denoise toggle in a real browser (Playwright + Chromium; `ui` marker): D in the loupe and in
compare, the badge states, Shift+D for one side, zoom (viewport region first, then the whole
full-resolution image), the remembered strength and persistence across frames with "keep"."""

from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

from shotsieve.pipeline.cache import Cache
from tests.test_api import _row
from tests.test_ui import Server, Ui

pytestmark = pytest.mark.ui


def _noisy_photo(gen_dir, seed: int, w=1800, h=1200) -> np.ndarray:
    src = cv2.cvtColor(cv2.imread(str(gen_dir / "shoot" / "landscape.jpg")), cv2.COLOR_BGR2RGB)
    img = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def dshoot(tmp_path_factory, gen_dir) -> Server | None:
    folder = tmp_path_factory.mktemp("dn") / "shoot"
    folder.mkdir()
    names = ["a.jpg", "b.jpg", "c.jpg"]
    for i, n in enumerate(names):
        Image.fromarray(_noisy_photo(gen_dir, i)).save(folder / n, quality=95)
    rows = [
        _row(folder / "a.jpg", "keep", 0, True),
        _row(folder / "b.jpg", "reject", 0, False),
        _row(folder / "c.jpg", "keep", 1, True),
    ]
    with Cache.for_folder(folder) as c:
        c.save_decisions(rows)
    return folder


@pytest.fixture()
def srv(dshoot, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(dshoot, dest)
    s = Server(dest)
    yield s
    s.stop()


@pytest.fixture()
def ui(browser, srv):
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    u = Ui(ctx.new_page(), srv).open()
    yield u
    try:
        u.check()
    finally:
        ctx.close()


def open_loupe(ui, name="a.jpg"):
    for attempt in range(3):  # the grid may re-render once right after load
        try:
            ui.select(name)
            break
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            ui.page.wait_for_timeout(300)
    ui.key("Space")
    ui.page.wait_for_function("(() => { const i = document.getElementById('big'); return i.complete && i.naturalWidth > 0; })()")
    return ui


def state(ui, host="#lz-denoise"):
    return ui.page.get_attribute(host, "data-state")


def wait_state(ui, want, host="#lz-denoise", timeout=90000):
    ui.page.wait_for_function(
        f"document.querySelector('{host}').dataset.state && {want!r}.split('|').includes(document.querySelector('{host}').dataset.state)",
        timeout=timeout,
    )


def src(ui, sel="#big"):
    return ui.page.get_attribute(sel, "src")


def wait_src(ui, part, sel="#big", timeout=90000):
    ui.page.wait_for_function(f"document.querySelector('{sel}').src.includes({part!r})", timeout=timeout)


def test_d_toggles_the_badge_and_swaps_the_picture(ui):
    open_loupe(ui)
    assert state(ui) == "off" and ui.page.inner_text("#lz-denoise .dn-badge") == "Off"
    assert "/api/preview/" in src(ui)
    ui.key("d")
    wait_state(ui, "loading|ai|basic")
    wait_state(ui, "ai|basic")
    wait_src(ui, "/api/denoise/")
    assert ui.page.inner_text("#lz-denoise .dn-badge") in ("AI", "Basic")
    assert ui.page.get_attribute("#lz-denoise .dn-btn", "aria-pressed") == "true"
    assert "level=medium" in src(ui)
    ui.key("d")
    wait_state(ui, "off")
    wait_src(ui, "/api/preview/")
    assert ui.page.inner_text("#lz-denoise .dn-badge") == "Off"


def test_the_spinner_shows_while_working(ui):
    open_loupe(ui, "c.jpg")
    ui.page.click("#lz-denoise .dn-btn")
    ui.page.wait_for_selector("#lz-denoise .dn-spin:not([hidden])", timeout=5000)
    wait_state(ui, "ai|basic")
    assert ui.page.is_hidden("#lz-denoise .dn-spin")


def test_zoomed_in_the_visible_region_comes_first_then_the_whole_full_resolution_image(ui):
    open_loupe(ui)
    ui.key("d")
    wait_state(ui, "ai|basic")
    ui.key("z")  # 100 %
    ui.page.wait_for_function("document.getElementById('lz-view').dataset.pct === '100'")
    ui.page.wait_for_selector("#lz-view img.dn-region:not([hidden]), #lz-view[data-full='1']", timeout=90000)
    ui.page.wait_for_selector("#lz-view[data-full='1']", timeout=120000)
    assert "/api/denoise/" in src(ui) and "/full" in src(ui)
    assert ui.page.evaluate("document.getElementById('big').naturalWidth") == 1800  # native pixels, not the preview
    assert ui.page.is_hidden("#lz-view img.dn-region")  # the full image replaced the region overlay
    ui.key("d")  # back to the plain full-resolution image, still zoomed
    wait_state(ui, "off")
    wait_src(ui, "/api/full/")


def test_region_overlay_is_placed_over_the_picture(ui):
    open_loupe(ui)
    ui.key("d")
    wait_state(ui, "ai|basic")
    ui.key("z")
    ui.page.wait_for_selector("#lz-view img.dn-region:not([hidden])", timeout=90000)
    r = ui.page.evaluate("""() => { const o = document.querySelector('#lz-view img.dn-region').getBoundingClientRect();
        const v = document.getElementById('lz-view').getBoundingClientRect(); return [o.width, o.height, v.width, v.height]; }""")
    assert r[0] > 100 and r[1] > 100  # a real region of the zoomed picture, laid over it


def test_compare_denoises_both_sides_and_shift_d_one_side(ui):
    ui.select("b.jpg")
    ui.key("c")
    ui.page.wait_for_selector("#overlay-compare:not([hidden])")
    ui.page.wait_for_function("document.getElementById('cmp-a').naturalWidth > 0 && document.getElementById('cmp-b').naturalWidth > 0")
    assert state(ui, "#cmp-denoise") == "off"
    ui.key("d")
    wait_state(ui, "ai|basic", "#cmp-denoise")
    wait_src(ui, "/api/denoise/", "#cmp-a")
    wait_src(ui, "/api/denoise/", "#cmp-b")
    assert ui.page.get_attribute("#cmp-denoise", "data-sides") == "11"
    # Shift+D: only the side under the pointer
    box = ui.page.locator("#cmp-fig-b .cmp-view").bounding_box()
    ui.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    ui.page.keyboard.press("Shift+D")
    wait_src(ui, "/api/preview/", "#cmp-b")
    assert "/api/denoise/" in src(ui, "#cmp-a")
    assert ui.page.get_attribute("#cmp-denoise", "data-sides") == "10"
    ui.page.keyboard.press("Shift+D")
    wait_src(ui, "/api/denoise/", "#cmp-b")
    assert ui.page.get_attribute("#cmp-denoise", "data-sides") == "11"
    ui.key("d")  # both off
    wait_state(ui, "off", "#cmp-denoise")
    wait_src(ui, "/api/preview/", "#cmp-a")
    wait_src(ui, "/api/preview/", "#cmp-b")


def test_denoise_follows_frames(ui):
    open_loupe(ui, "a.jpg")
    ui.key("d")
    wait_state(ui, "ai|basic")
    first = src(ui)
    ui.key("ArrowRight")  # the next frame in the loupe
    ui.page.wait_for_function("document.getElementById('big').src !== " + repr(first))
    wait_src(ui, "/api/denoise/")
    wait_state(ui, "ai|basic")
    assert ui.page.get_attribute("#lz-denoise", "data-on") == "1"


def test_strength_menu_is_remembered(ui):
    open_loupe(ui)
    ui.page.click("#lz-denoise .dn-more")
    ui.page.wait_for_selector("#lz-denoise .dn-menu:not([hidden])")
    assert ui.page.get_attribute("#lz-denoise .dn-menu [data-level=medium]", "aria-checked") == "true"
    ui.page.click("#lz-denoise .dn-menu [data-level=high]")
    assert ui.page.evaluate("localStorage.getItem('shotsieve.denoiseLevel')") == "high"
    ui.key("d")
    wait_src(ui, "/api/denoise/")
    assert "level=high" in src(ui)
    ui.page.reload()
    ui.page.wait_for_selector(".cell")
    open_loupe(ui)
    ui.page.click("#lz-denoise .dn-more")
    assert ui.page.get_attribute("#lz-denoise .dn-menu [data-level=high]", "aria-checked") == "true"
    ui.page.evaluate("localStorage.removeItem('shotsieve.denoiseLevel')")


def test_badge_says_basic_without_the_model(browser, dshoot, tmp_path):
    dest = tmp_path / "shoot"
    shutil.copytree(dshoot, dest)
    s = Server(dest, env={"SHOTSIEVE_DENOISE_BACKEND": "basic"})
    ctx = browser.new_context(viewport={"width": 1500, "height": 900})
    try:
        u = Ui(ctx.new_page(), s).open()
        open_loupe(u)
        u.key("d")
        wait_state(u, "basic")
        assert u.page.inner_text("#lz-denoise .dn-badge") == "Basic"
        u.page.click("#lz-denoise .dn-more")
        u.page.wait_for_selector("#lz-denoise .dn-get")  # offers the AI model
        u.check()
    finally:
        ctx.close()
        s.stop()


def test_d_is_not_a_global_key_outside_the_loupe(ui):
    ui.select("a.jpg")
    ui.key("d")  # grid: nothing happens, no error
    assert ui.page.is_hidden("#overlay-loupe")
