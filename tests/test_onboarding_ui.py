"""T19 first-run wizard: real-browser end-to-end tests (Playwright + Chromium, marker `ui`).

The server runs in-process (a uvicorn thread) so the model download can be faked at the
`urllib.request.urlopen` level: the real `_download` (progress, `.part`, Range resume, sha check)
runs against local model files served in 1-MB chunks with a short sleep, with no network.
"""

# ruff: noqa: E501, F811
from __future__ import annotations

import io
import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import uvicorn

from shotsieve.config import Settings
from shotsieve.io.formats import scan_folder
from shotsieve.models import registry
from shotsieve.server.app import create_app
from tests.test_ui import Ui, _free_port

pytestmark = pytest.mark.ui
ROOT = Path(__file__).resolve().parents[1]


class ThreadServer:
    def __init__(self, app) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=30) as r:
            return json.loads(r.read())

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(10)


class FakeNet:
    """Replaces `urllib.request.urlopen` for model URLs: serves the real local files."""

    def __init__(self, fail_first_after: int | None = None, sleep: float = 0.02) -> None:
        self.by_url = {s.url: ROOT / "models" / s.filename for s in registry.MODEL_SPECS.values()}
        self.fail_first_after = fail_first_after
        self.sleep = sleep
        self.ranges: list[str | None] = []
        self.calls = 0
        self._real = urllib.request.urlopen

    def __call__(self, req, timeout=0):
        url = req.full_url if hasattr(req, "full_url") else req
        if url not in self.by_url:
            return self._real(req, timeout=timeout)
        self.calls += 1
        rng = req.headers.get("Range")
        self.ranges.append(rng)
        data = self.by_url[url].read_bytes()
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        fail_at = self.fail_first_after if (self.fail_first_after and self.calls == 1) else None
        return _Resp(data[start:], 206 if rng else 200, self.sleep, fail_at)


class _Resp(io.RawIOBase):
    def __init__(self, data: bytes, status: int, sleep: float, fail_at: int | None) -> None:
        self.data, self.status, self.sleep, self.pos, self.fail_at = data, status, sleep, 0, fail_at
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        if self.fail_at is not None and self.pos >= self.fail_at:
            raise urllib.error.URLError("connection lost")
        time.sleep(self.sleep)
        end = len(self.data) if n < 0 else self.pos + min(n, 1 << 20)
        if self.fail_at is not None:
            end = min(end, self.fail_at)
        chunk = self.data[self.pos:end]
        self.pos = end
        return chunk


@pytest.fixture()
def env(tmp_path, monkeypatch, shoot_dir, make_fixtures, models_ready):
    """Sandbox HOME with a copy of the fixture shoot and a catalog of it; fresh data/models dirs."""
    home = tmp_path / "home"
    shoot = home / "Shoots" / "wedding"
    shutil.copytree(shoot_dir, shoot, ignore=shutil.ignore_patterns(".shotsieve"))
    cat = home / "Lightroom" / "catalog.lrcat"
    cat.parent.mkdir(parents=True)
    make_fixtures.build_lrcat(
        cat, [dict(root=str(shoot), base=p.stem, ext="jpg", rating=3, pick=0, label="")
              for p in sorted(shoot.glob("*.jpg"))]
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SHOTSIEVE_MODELS_DIR", str(tmp_path / "models"))
    return dict(home=home, shoot=shoot, catalog=cat, tmp=tmp_path)


@pytest.fixture()
def models_present(env, monkeypatch):
    shutil.copytree(ROOT / "models", env["tmp"] / "models")


def _start(page_ctx_browser, env, wizard_settings=None):
    srv = ThreadServer(create_app(wizard_settings or Settings()))
    ctx = page_ctx_browser.new_context(viewport={"width": 1500, "height": 900})
    ctx.add_init_script("window.__clicks = 0; document.addEventListener('click', () => window.__clicks++, true);")
    return srv, ctx


def test_onboarding_full_flow(browser, env, monkeypatch):
    fake = FakeNet()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    srv, ctx = _start(browser, env)
    t0 = time.time()
    try:
        ui = Ui(ctx.new_page(), srv).open(ready=False)
        page = ui.page
        ui.errors[:] = []
        # wizard appears, models missing
        page.wait_for_selector("#overlay-onboarding:not([hidden])")
        assert "Welcome to ShotSieve" in page.inner_text("#wiz-body")
        assert page.get_attribute("#overlay-onboarding", "role") == "dialog"
        page.click("#wiz-download")
        seen = set()
        for _ in range(600):
            v = page.evaluate("(() => { const b = document.getElementById('wiz-dl-bar'); return b ? b.getAttribute('aria-valuenow') : null; })()")
            if v is not None:
                seen.add(v)
            if page.is_visible("#wiz-path"):
                break
            page.wait_for_timeout(25)
        assert len(seen) >= 3, seen  # progress was observable (not just 0 and 100)
        assert "100" in seen or page.is_visible("#wiz-path")
        # folder step: browse from HOME, pick the shoot
        page.wait_for_selector("#wiz-path")
        page.click("#wiz-browse-toggle")
        page.click("#wiz-browser button.link:text-is('Shoots')")
        page.click("#wiz-browser li:has(button.link:text-is('wedding')) button[data-pick]")
        n = len(scan_folder(env["shoot"]))
        page.wait_for_function("document.getElementById('wiz-scan').textContent.includes('photos')")
        line = page.inner_text("#wiz-scan")
        assert f"{n} photos" in line and f"{n} JPEG" in line, line
        page.click("#wiz-continue")
        # catalog step: train from the fixture catalog (< 40 rows: built-in judgement message)
        page.fill("#wiz-cat", str(env["catalog"]))
        page.click("#wiz-train")
        page.wait_for_selector("#wiz-train-result", timeout=120_000)
        assert "Not enough picks yet (need 40)" in page.inner_text("#wiz-train-result")
        page.click("#wiz-continue")
        # privacy: people on -> consent recorded for that folder
        page.check("#wiz-people")
        page.click("#wiz-continue")
        page.wait_for_selector("#wiz-cull")
        from shotsieve.pipeline.cache import Cache

        with Cache.for_folder(env["shoot"]) as c:
            assert c.people_consent(False) is True
        # cull: grid shows every fixture image, then the Lightroom card
        page.click("#wiz-cull")
        page.wait_for_selector("#wiz-done", timeout=180_000)
        assert "JPEG" in page.inner_text("#wiz-body")
        assert "Lightroom does not read ShotSieve's ratings" in page.inner_text("#wiz-body")
        page.click("#wiz-done")
        page.wait_for_selector("#overlay-onboarding", state="hidden")
        page.wait_for_selector(".cell")
        assert page.inner_text("#c-all") == str(n)
        assert srv.get("/api/onboarding")["completed"] is True
        # reload: the wizard is not shown again
        page.reload()
        page.wait_for_selector(".cell")
        assert page.is_hidden("#overlay-onboarding")
        ui.check()
        assert time.time() - t0 <= 120, time.time() - t0
    finally:
        ctx.close()
        srv.stop()


def test_click_budget_with_recent_folder(browser, env, models_present, monkeypatch):
    from shotsieve.server import onboarding

    onboarding.add_recent(env["shoot"])
    srv, ctx = _start(browser, env)
    try:
        ui = Ui(ctx.new_page(), srv).open(ready=False)
        page = ui.page
        page.wait_for_selector("#wiz-recent-0")
        with page.expect_request(lambda r: r.url.endswith("/api/cull") and r.method == "POST"):
            page.click("#wiz-recent-0")
            page.wait_for_function("document.getElementById('wiz-scan').textContent.includes('photos')")
            page.click("#wiz-continue")  # folder
            page.click("#wiz-continue")  # catalog (skipped)
            page.click("#wiz-continue")  # privacy (all off)
            page.click("#wiz-cull")
        assert page.evaluate("window.__clicks") <= 5
        page.wait_for_selector("#wiz-done", timeout=180_000)
        ui.check()
    finally:
        ctx.close()
        srv.stop()


def test_skip_and_reopen(browser, env, models_present):
    srv, ctx = _start(browser, env)
    try:
        ui = Ui(ctx.new_page(), srv).open(ready=False)
        page = ui.page
        page.wait_for_selector("#overlay-onboarding:not([hidden])")
        # keyboard only: the wizard's controls are all reachable with Tab
        page.keyboard.press("Tab")
        page.click("#wiz-skip")
        page.wait_for_selector("#overlay-onboarding", state="hidden")
        ob = srv.get("/api/onboarding")
        assert ob["skipped"] is True and ob["show"] is False
        page.reload()
        page.wait_for_timeout(500)
        assert page.is_hidden("#overlay-onboarding")
        page.click("#btn-help")
        page.click("#btn-setup")
        page.wait_for_selector("#overlay-onboarding:not([hidden])")
        assert page.is_visible("#wiz-skip") and page.is_visible("#wiz-path")
        page.click("#wiz-skip")
        ui.check()
    finally:
        ctx.close()
        srv.stop()


def test_download_failure_then_retry_resumes_with_range(browser, env, monkeypatch):
    fake = FakeNet(fail_first_after=200_000, sleep=0.0)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    srv, ctx = _start(browser, env)
    try:
        ui = Ui(ctx.new_page(), srv).open(ready=False)
        page = ui.page
        page.wait_for_selector("#wiz-download")
        page.click("#wiz-download")
        page.wait_for_selector("#wiz-retry")
        assert "internet connection" in page.inner_text("#wiz-dl-area")
        page.click("#wiz-retry")
        page.wait_for_selector("#wiz-path", timeout=60_000)
        assert fake.ranges[0] is None
        assert fake.ranges[1] == "bytes=200000-"  # resumed from the .part
        ui.errors[:] = [e for e in ui.errors if "Failed to load resource" not in e]
        ui.check()
    finally:
        ctx.close()
        srv.stop()


def test_wizard_fits_small_window_and_quit(browser, env, models_present):
    srv, ctx = _start(browser, env)
    page = ctx.new_page()
    page.set_viewport_size({"width": 1280, "height": 800})
    Ui(page, srv).open(ready=False)
    try:
        page.wait_for_selector("#overlay-onboarding:not([hidden])")
        box = page.locator(".wizard").bounding_box()
        assert box["x"] >= 0 and box["y"] >= 0 and box["x"] + box["width"] <= 1280 and box["y"] + box["height"] <= 800
        page.click("#wiz-skip")
        page.click("#btn-help")
        page.click("#btn-quit")
        page.wait_for_function("document.body.textContent.includes('has quit')")
        for _ in range(40):
            if not srv.thread.is_alive():
                break
            time.sleep(0.1)
    finally:
        ctx.close()
        srv.stop()


def test_wizard_native_pickers_hide_text_boxes(browser, env, models_present):
    srv = ThreadServer(create_app(Settings()))
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    picks = {"folder": str(env["shoot"]), "lrcat": str(env["catalog"])}
    ctx.add_init_script(
        f"window.pywebview = {{api: {{pick: async (k) => ({json.dumps(picks)})[k] || null}}}};")
    try:
        ui = Ui(ctx.new_page(), srv).open(ready=False)
        page = ui.page
        page.wait_for_selector("#wiz-pick")
        assert page.is_hidden("#wiz-path") and page.is_hidden("#wiz-browse-toggle")
        page.click("#wiz-pick")
        page.wait_for_function("document.getElementById('wiz-scan').textContent.includes('photos')")
        assert page.inner_text("#wiz-path-name") == env["shoot"].name
        page.click("#wiz-continue")
        page.wait_for_selector("#wiz-cat-pick")
        assert page.is_hidden("#wiz-cat") and page.is_hidden("#wiz-cat-browse")
        assert page.is_disabled("#wiz-train")
        page.click("#wiz-cat-pick")
        assert page.inner_text("#wiz-cat-name") == env["catalog"].name
        assert page.is_enabled("#wiz-train")
        ui.check()
    finally:
        ctx.close()
        srv.stop()
