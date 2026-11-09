"""T23a: the native window module, without a display (fake pywebview)."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request

import pytest

from shotsieve import desktop
from tests import fake_webview as fw


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    fw.windows.clear()
    fw.created.clear()
    fw.closed_any.clear()
    monkeypatch.setattr(desktop, "get_screens", lambda: [])  # tests set their own layout
    fw.started.clear()
    fw.dialog_calls.clear()
    fw.dialog_result = None
    fw.dialog_raises = None
    fw.on_started = None
    with socket.socket() as s:  # a free ephemeral port instead of 8765-8775
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(desktop, "pick_port", lambda *a: port)
    return port


def _post(port, path):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST")
    return json.load(urllib.request.urlopen(req, timeout=3))


def test_pick_port_skips_busy_and_reports_none():
    with socket.socket() as a, socket.socket() as b:
        for s in (a, b):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        a.bind(("127.0.0.1", 0))
        pa = a.getsockname()[1]
        b.bind(("127.0.0.1", 0))
        pb = b.getsockname()[1]
        a.listen()
        b.listen()
        assert desktop.pick_port([pa, pb]) is None
    assert desktop.pick_port(range(8765, 8776)) in (None, *range(8765, 8776))


def test_is_external():
    assert desktop.is_external("https://example.com/x")
    assert desktop.is_external("http://127.0.0.1:9/x", port=8765)
    assert not desktop.is_external("http://127.0.0.1:8765/x", port=8765)
    assert not desktop.is_external("/static/app.js")
    assert not desktop.is_external("mailto:a@b.c")
    assert not desktop.open_external("file:///etc/passwd")
    assert not desktop.open_external("javascript:alert(1)")


def test_window_state_roundtrip_and_clamp(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path))
    assert desktop.load_window_state() == {"width": 1360, "height": 880}
    w = fw.Window("t", "u", width=1500, height=900, x=40, y=60)
    desktop.save_window_state(w)
    assert desktop.load_window_state() == {"width": 1500, "height": 900, "x": 40, "y": 60}
    (tmp_path / "window.json").write_text(json.dumps({"width": 300, "height": 200,
                                                      "x": 99999, "y": 5}))
    assert desktop.load_window_state() == {"width": 1100, "height": 720}  # min, off-screen x
    (tmp_path / "window.json").write_text("garbage")
    assert desktop.load_window_state()["width"] == 1360


def test_js_api_pick_uses_native_dialogs(fake):
    api = desktop.JsApi(fw)
    assert api.pick("folder") is None  # no window yet
    api.window = fw.Window("t", "u")
    fw.dialog_result = ["/Users/x/Wedding/"]
    assert api.pick("folder") == "/Users/x/Wedding"
    assert fw.dialog_calls[-1][0] == fw.FOLDER_DIALOG
    fw.dialog_result = ("/Users/x/Cat.lrcat",)
    assert api.pick("lrcat") == "/Users/x/Cat.lrcat"
    kind, kw = fw.dialog_calls[-1]
    assert kind == fw.OPEN_DIALOG and "*.lrcat" in kw["file_types"][0]
    fw.dialog_result = None
    assert api.pick("lrcat") is None  # cancelled
    assert api.pick("bogus") is None


def _run_in_thread(box, **kw):
    t = threading.Thread(target=lambda: box.append(desktop.run_app(webview_mod=fw, **kw)))
    t.start()
    return t


def test_window_close_shuts_server_down(fake, tmp_path):
    port = fake
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    win = fw.windows[-1]
    assert win.kw["min_size"] == (1100, 720) and win.title == "ShotSieve"
    assert win.url == f"http://127.0.0.1:{port}/"
    sj = json.loads((tmp_path / "data" / "server.json").read_text())
    assert sj["port"] == port
    assert urllib.request.urlopen(win.url, timeout=3).status == 200  # Host/Origin policy passes
    win.simulate_close()
    t.join(15)
    assert not t.is_alive() and box == [0]
    assert not (tmp_path / "data" / "server.json").exists()
    assert (tmp_path / "data" / "window.json").exists()
    with pytest.raises(OSError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2)


def test_api_quit_closes_window_and_second_launch_focuses(fake, tmp_path):
    port = fake
    box: list[int] = []
    fw.started.clear()
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    win = fw.windows[-1]
    # second launch: no second server, the existing window is brought forward
    assert desktop.run_app(webview_mod=fw) == 0
    t0 = time.time()
    while not win.shown and time.time() - t0 < 5:
        time.sleep(0.05)
    assert win.shown >= 1 and win.restored >= 1
    assert len(fw.windows) == 1
    assert _post(port, "/api/focus")["focused"] is True
    # /api/quit: same path as closing the window
    assert _post(port, "/api/quit")["quitting"]
    t.join(15)
    assert not t.is_alive() and box == [0]
    assert not (tmp_path / "data" / "server.json").exists()


def test_autoclose_knob(fake, monkeypatch):
    monkeypatch.setenv("SHOTSIEVE_APP_AUTOCLOSE_S", "1")
    box: list[int] = []
    t = _run_in_thread(box)
    t.join(30)
    assert not t.is_alive() and box == [0]


def test_stale_server_json_does_not_block(fake, tmp_path):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "server.json").write_text(
        json.dumps({"pid": 999999, "port": fake, "version": "0", "started": 0}))
    assert desktop.running_instance() is None


def test_static_hooks_present():
    from pathlib import Path

    static = Path(desktop.__file__).parent / "server" / "static"
    html = (static / "index.html").read_text()
    assert html.index("desktop.js") < html.index("app.js")
    js = (static / "desktop.js").read_text()
    assert "open_external" in js and "pywebview" in js
    assert "shotsieveDesktop" in (static / "onboarding.js").read_text()


def test_plan_drop_folders_files_and_notes(tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    for n in ("1.jpg", "2.CR3", "3.mp4", "x.txt"):
        (a / n).write_text("x")
    (b / "9.jpg").write_text("x")
    # a folder wins; a second folder is mentioned
    p = desktop.plan_drop([str(a), str(b)])
    assert p["folder"] == str(a) and p["files"] is None and "2 folders" in p["notes"][0]
    # mixed folder + file: folder wins, note
    p = desktop.plan_drop([str(b), str(a / "1.jpg")])
    assert p["folder"] == str(b) and p["files"] is None and p["notes"]
    # files: supported only, unsupported counted
    p = desktop.plan_drop([str(a / "1.jpg"), str(a / "2.CR3"), str(a / "3.mp4"), str(a / "x.txt")])
    assert p["folder"] == str(a) and len(p["files"]) == 3
    assert p["notes"] == ["1 unsupported file(s) ignored."]
    # several parents: the one with most files
    p = desktop.plan_drop([str(b / "9.jpg"), str(a / "1.jpg"), str(a / "2.CR3")])
    assert p["folder"] == str(a) and len(p["files"]) == 2 and "2 folders" in p["notes"][0]
    # nothing usable
    assert "error" in desktop.plan_drop([str(a / "x.txt")])
    assert "error" in desktop.plan_drop([])
    assert desktop.drop_paths({}) == []


def test_drop_handler_reaches_the_page(fake, tmp_path):
    (tmp_path / "S").mkdir()
    (tmp_path / "S" / "a.jpg").write_text("x")
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    win = fw.windows[-1]
    win.events.loaded.fire()  # page loaded: the DOM drop handler is attached
    assert len(win.dom.document.events.drop.handlers) == 1
    h = win.dom.document.events.drop.handlers[0]
    assert h.prevent_default and h.stop_propagation
    win.simulate_drop([str(tmp_path / "S" / "a.jpg")])
    assert len(win.js_calls) == 1 and "shotsieveDesktop.onDrop" in win.js_calls[0]
    assert json.dumps(str(tmp_path / "S")) in win.js_calls[0]
    win.simulate_close()
    t.join(15)
    assert box == [0]


# ---- the "ShotSieve is not launching" bugs: ghost process after close / quit --------------------

# the user's layout: portrait main 1080x1920 at (0,0) + ultrawide to its left. NSScreen frames are
# bottom-left based: ultrawide origin (-3440, 131); pywebview wants top-left of the main screen.
USER_SCREENS = [
    {"x": 0, "y": 0, "w": 1080, "h": 1920, "vx": 0, "vy": 0, "vw": 1080, "vh": 1920},
    {"x": -3440, "y": 349, "w": 3440, "h": 1440, "vx": -3440, "vy": 349, "vw": 3440, "vh": 1440},
]


@pytest.fixture()
def hard_exit(monkeypatch):
    """Short watchdog; the 'process exit' is recorded and releases the fake GUI loop."""
    calls: list[int] = []

    def fake_exit(code=0):
        calls.append(code)
        fw.closed_any.set()
        fw.windows.clear()

    monkeypatch.setattr(desktop, "_hard_exit", fake_exit)
    monkeypatch.setattr(desktop, "WATCHDOG_S", 1.0)
    return calls


def test_read_geometry_never_raises():
    w = fw.Window("t", "u", width=1200, height=800, x=-2270, y=500)
    assert desktop.read_geometry(w) == {"width": 1200, "height": 800, "x": -2270, "y": 500}
    w.geometry_raises = True
    assert desktop.read_geometry(w) is None
    desktop.save_window_state(w)  # must not raise


def test_close_with_raising_geometry_still_shuts_down(fake, tmp_path):
    """Old code: closing handler read window.x after the native window died -> AttributeError,
    the closed event never fired and the process lived on."""
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    win = fw.windows[-1]
    win.geometry_raises = True
    win.simulate_close()
    t.join(15)
    assert not t.is_alive() and box == [0]
    assert not (tmp_path / "data" / "server.json").exists()


def test_geometry_saved_on_move_while_alive_not_on_close(fake, tmp_path):
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    win = fw.windows[-1]
    win.move_to(-2270, 500, 1300, 850)
    time.sleep(0.8)  # debounce
    saved = json.loads((tmp_path / "data" / "window.json").read_text())
    assert saved == {"width": 1300, "height": 850, "x": -2270, "y": 500}
    win.geometry_raises = True  # close cannot read anything now
    win.simulate_close()
    t.join(15)
    assert box == [0]
    assert json.loads((tmp_path / "data" / "window.json").read_text())["x"] == -2270


def test_api_quit_with_destroyed_window_still_exits(fake, tmp_path, hard_exit):
    """Old code: save_window_state raised before window.destroy(); with the window vanished the
    GUI loop never ended. Now the server stops and the watchdog hard-exits."""
    port = fake
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    fw.windows[-1].vanish()
    assert _post(port, "/api/quit")["quitting"]
    t.join(15)
    assert not t.is_alive() and box == [0]
    assert hard_exit == [0], "watchdog did not fire"
    assert not (tmp_path / "data" / "server.json").exists()


def test_watchdog_fires_when_gui_loop_never_ends(fake, hard_exit):
    """Window close handled, but the GUI loop hangs: os._exit(0) after the watchdog delay."""
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    win = fw.windows[-1]
    win.vanish()
    win.events.closing.fire()
    win.events.closed.fire()  # what pywebview delivers; the loop itself keeps running
    t0 = time.time()
    while not hard_exit and time.time() - t0 < 6:
        time.sleep(0.05)
    assert hard_exit == [0]
    t.join(10)
    assert not t.is_alive()


def test_normal_close_cancels_watchdog(fake, hard_exit, monkeypatch):
    monkeypatch.setattr(desktop, "WATCHDOG_S", 0.5)
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    fw.windows[-1].simulate_close()
    t.join(15)
    time.sleep(1.2)
    assert box == [0] and hard_exit == []  # clean exit: no hard exit (cmd_app has its own)


def test_reopen_with_no_window_creates_a_new_one(fake, hard_exit):
    port = fake
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    first = fw.windows[-1]
    first.vanish()  # Dock click arrives: LaunchServices -> /api/focus equivalent (focus callback)
    assert _post(port, "/api/focus")["focused"] is True
    t0 = time.time()
    while len(fw.created) < 2 and time.time() - t0 < 5:
        time.sleep(0.05)
    assert len(fw.created) == 2 and fw.windows == [fw.created[1]]
    assert fw.created[1].url == first.url
    # a visible window just comes forward, no third window
    _post(port, "/api/focus")
    time.sleep(0.5)
    assert len(fw.created) == 2 and fw.created[1].shown >= 1
    # closing the new window ends the app
    fw.created[1].simulate_close()
    t.join(15)
    assert not t.is_alive() and box == [0] and hard_exit == []


def test_second_launch_with_vanished_window_shows_one_again(fake, hard_exit):
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    fw.windows[-1].vanish()
    assert desktop.run_app(webview_mod=fw) == 0  # second launch -> /api/focus
    t0 = time.time()
    while len(fw.created) < 2 and time.time() - t0 < 5:
        time.sleep(0.05)
    assert len(fw.created) == 2
    fw.created[1].simulate_close()
    t.join(15)
    assert box == [0]


def test_place_window_user_layout_keeps_ultrawide_position():
    st = {"width": 1360, "height": 880, "x": -2270, "y": 500}
    pl = desktop.place_window(st, USER_SCREENS)
    assert (pl["x"], pl["y"]) == (-2270, 500)
    assert (pl["width"], pl["height"]) == (1360, 880)
    assert pl["min_size"] == (1100, 720)  # ultrawide fits the minimum


def test_place_window_offscreen_centres_on_main():
    # display unplugged / saved when a 3rd screen existed: nothing connected there
    st = {"width": 1360, "height": 880, "x": 5000, "y": 300}
    pl = desktop.place_window(st, USER_SCREENS)
    assert pl["width"] == 1080 and pl["min_size"][0] == 1080  # portrait main: min clamped
    assert (pl["x"], pl["y"]) == (0, int((1920 - 880) / 2))
    # above the ultrawide (y way negative) is offscreen too
    pl = desktop.place_window({"width": 1360, "height": 880, "x": -2270, "y": -3000},
                              USER_SCREENS)
    assert pl["x"] == 0 and pl["y"] == 520


def test_place_window_single_screen_after_ultrawide_unplugged():
    main_only = USER_SCREENS[:1]
    pl = desktop.place_window({"width": 1360, "height": 880, "x": -2270, "y": 500}, main_only)
    assert 0 <= pl["x"] and pl["x"] + pl["width"] <= 1080  # back on the only screen
    assert pl["y"] == 520


def test_place_window_half_rule_and_between_screens():
    # window straddling the seam at x=-1000..360 (width 1360): 1000 px on ultrawide = 73 % -> keep
    pl = desktop.place_window({"width": 1360, "height": 880, "x": -1000, "y": 500}, USER_SCREENS)
    assert pl["x"] == -1360  # mostly on the ultrawide: nudged fully onto it
    # only 20 % on a screen: not "mostly" -> centred on main
    pl = desktop.place_window({"width": 1360, "height": 880, "x": -4400, "y": 500}, USER_SCREENS)
    assert pl["x"] == 0  # 400 px of 1360 (29 %) on the ultrawide
    # menu bar: visible frame is smaller than the frame
    mac = [{"x": 0, "y": 0, "w": 1440, "h": 900, "vx": 0, "vy": 25, "vw": 1440, "vh": 875}]
    pl = desktop.place_window({"width": 1360, "height": 880, "x": 40, "y": 0}, mac)
    assert pl["height"] == 875 and pl["y"] >= 25


def test_place_window_portrait_main_clamps_min_width_and_size():
    portrait = USER_SCREENS[:1]
    pl = desktop.place_window({"width": 1360, "height": 880}, portrait)
    assert pl["width"] == 1080 and pl["min_size"] == (1080, 720)
    assert (pl["x"], pl["y"]) == (0, 520)
    assert desktop.place_window({"width": 1360, "height": 880}, [])["x"] is None


def test_run_app_uses_screens_for_the_window(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "get_screens", lambda: USER_SCREENS[:1])
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "window.json").write_text(json.dumps(
        {"width": 1360, "height": 880, "x": -2270, "y": 500}))  # ultrawide gone
    box: list[int] = []
    t = _run_in_thread(box)
    assert fw.started.wait(60)
    win = fw.windows[-1]
    assert win.kw["width"] == 1080 and win.kw["min_size"] == (1080, 720)
    assert win.kw["x"] == 0 and win.kw["y"] == 520
    win.simulate_close()
    t.join(15)
    assert box == [0]


def test_picker_uses_filedialog_enum_and_survives_errors(fake, tmp_path):
    api = desktop.JsApi(fw)
    api.window = fw.Window("t", "u")
    fw.dialog_result = ["/Users/x/Wedding/"]
    assert api.pick("folder") == "/Users/x/Wedding"
    assert isinstance(fw.dialog_calls[-1][0], fw.FileDialog)  # not the deprecated constants
    assert fw.dialog_calls[-1][0] is fw.FileDialog.FOLDER
    api.pick("lrcat")
    assert fw.dialog_calls[-1][0] is fw.FileDialog.OPEN
    # a native error (AttributeError like the field crash) -> None + toast, logged, no raise
    fw.dialog_raises = AttributeError("'NoneType' object has no attribute 'frame'")
    desktop.setup_app_log()
    assert api.pick("folder") is None
    assert any("toast(" in c for c in api.window.js_calls)
    log = (tmp_path / "data" / "logs" / "app.log").read_text()
    assert "file dialog (folder) failed" in log and "Traceback" in log and "'frame'" in log


def test_gui_loop_crash_is_logged_with_traceback_and_ends(fake, tmp_path, hard_exit):
    class Boom:
        def __getattr__(self, name):
            return getattr(fw, name)

        @staticmethod
        def start(*a, **k):
            raise AttributeError("'NoneType' object has no attribute 'frame'")

    with pytest.raises(AttributeError):
        desktop.run_app(webview_mod=Boom())
    log = (tmp_path / "data" / "logs" / "app.log").read_text()
    assert "GUI loop crashed" in log and "Traceback" in log
    assert not (tmp_path / "data" / "server.json").exists()


def test_harden_pywebview_wraps_delegate_callbacks():
    pytest.importorskip("webview.platforms.cocoa")
    import webview.platforms.cocoa as cocoa
    assert desktop.harden_pywebview()
    d = cocoa.BrowserView.WindowDelegate.alloc().init()

    class NoScreenWindow:  # windowDidMove_: i.window.screen() is None
        def frame(self):
            return type("F", (), {"size": type("S", (), {"height": 1, "width": 1})(),
                                  "origin": type("O", (), {"x": 0, "y": 0})()})()

        def screen(self):
            return None

    inst = type("I", (), {"window": NoScreenWindow(), "pywebview_window": None})()
    note = type("N", (), {"object": lambda self: "w"})()
    orig_get = cocoa.BrowserView.get_instance
    cocoa.BrowserView.get_instance = staticmethod(lambda *a, **k: inst)
    try:
        d.windowDidMove_(note)  # would raise AttributeError unhardened
    finally:
        cocoa.BrowserView.get_instance = orig_get


def test_cancel_operation_guard_swallows_esc_in_full_screen():
    pytest.importorskip("webview.platforms.cocoa")
    import webview.platforms.cocoa as cocoa

    assert desktop.guard_cancel_operation()
    assert cocoa.BrowserView.WindowHost.instancesRespondToSelector_(b"cancelOperation:")
    full = type("W", (), {"styleMask": lambda self: desktop.FULLSCREEN_MASK | 15})()
    plain = type("W", (), {"styleMask": lambda self: 15})()
    assert desktop.esc_guard_ignores(full) is True
    assert desktop.esc_guard_ignores(plain) is False
    # the installed method does nothing and never raises (no toggleFullScreen_ call)
    calls = []
    full.toggleFullScreen_ = lambda s: calls.append(s)
    cocoa.BrowserView.WindowHost.__dict__["cancelOperation_"].callable(full, None)
    assert calls == []
