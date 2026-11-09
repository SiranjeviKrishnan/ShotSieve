"""`shotsieve app`: ShotSieve as a native Mac window (pywebview/WKWebView) over the local server.

The process is the app: uvicorn runs in a thread of this process, the window owns the main
thread. Closing the last window (or Cmd+Q, or POST /api/quit) stops the server and exits.
Test knobs: SHOTSIEVE_APP_AUTOCLOSE_S=N closes the window after N s; SHOTSIEVE_WEBVIEW_MODULE
names an importable module standing in for `webview` (tests, no display).
"""

from __future__ import annotations

import atexit
import faulthandler
import fcntl
import importlib
import json
import logging
import os
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

APP_NAME = "ShotSieve"
PORT_RANGE = range(8765, 8776)
MIN_SIZE = (1100, 720)
DEFAULT_SIZE = (1360, 880)
LR_TYPES = ("Lightroom catalog (*.lrcat)",)


# ---- pure helpers (unit-tested without a display) ----------------------------------------------


def pick_port(candidates: Any = PORT_RANGE) -> int | None:
    """First free 127.0.0.1 port in the range (same rule as `shotsieve serve`)."""
    from shotsieve.cli import _port_free

    for p in candidates:
        if _port_free(p):
            return p
    return None


def is_external(url: str, port: int | None = None) -> bool:
    """True for an http(s) URL that does not point at the local UI."""
    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        return False
    return not (u.hostname in ("127.0.0.1", "localhost") and (port is None or u.port == port))


def open_external(url: str) -> bool:
    """Open an http(s) link in the default browser; anything else is refused."""
    if urlparse(url).scheme not in ("http", "https"):
        return False
    try:
        return bool(webbrowser.open(url))
    except Exception:  # noqa: BLE001
        return False


def window_state_path() -> Path:
    from shotsieve.config import data_dir

    return data_dir() / "window.json"


def load_window_state() -> dict[str, int]:
    """Saved {width,height[,x,y]}; sizes clamped to the minimum, junk ignored."""
    try:
        raw = json.loads(window_state_path().read_text())
        w, h = int(raw["width"]), int(raw["height"])
    except (OSError, ValueError, KeyError, TypeError):
        return {"width": DEFAULT_SIZE[0], "height": DEFAULT_SIZE[1]}
    state = {"width": max(w, MIN_SIZE[0]), "height": max(h, MIN_SIZE[1])}
    try:
        x, y = int(raw["x"]), int(raw["y"])
        if abs(x) <= 50000 and abs(y) <= 50000:  # junk guard; place_window checks real screens
            state.update(x=x, y=y)
    except (KeyError, ValueError, TypeError):
        pass
    return state


def read_geometry(window: Any) -> dict[str, int] | None:
    """Window {width,height,x,y} or None. Never raises: after the native window is gone pywebview
    raises AttributeError ('NoneType' object has no attribute 'frame')."""
    try:
        return {"width": int(window.width), "height": int(window.height),
                "x": int(window.x), "y": int(window.y)}
    except Exception:  # noqa: BLE001
        return None


def native_geometry() -> dict[str, int] | None:
    """The ShotSieve NSWindow's frame in top-left-of-main-screen coordinates, straight from AppKit.
    (pywebview's window.y is relative to the height of the screen the window was created on, so on
    the second display it is off by the difference of the screen heights.) None if unavailable."""
    try:
        from AppKit import NSApplication, NSScreen

        main_h = NSScreen.screens()[0].frame().size.height
        for w in NSApplication.sharedApplication().windows():
            if w.isVisible() and str(w.title()) == APP_NAME:
                f = w.frame()
                return {"width": int(f.size.width), "height": int(f.size.height),
                        "x": int(f.origin.x), "y": int(main_h - (f.origin.y + f.size.height))}
    except Exception:  # noqa: BLE001
        pass
    return None


def write_window_state(st: dict[str, int]) -> None:
    try:
        window_state_path().write_text(json.dumps(st))
    except Exception:  # noqa: BLE001 - cosmetic
        pass


def save_window_state(window: Any) -> None:
    """Best-effort save of the live window's geometry; any failure just skips the save."""
    st = read_geometry(window)
    if st is not None:
        write_window_state(st)


def get_screens() -> list[dict[str, float]]:
    """Connected screens in pywebview coordinates (origin top-left of the MAIN screen, y down):
    [{x, y, w, h, vx, vy, vw, vh}] (frame, then visible frame without menu bar / Dock). The main
    screen is first. [] when AppKit is unavailable."""
    try:
        from AppKit import NSScreen

        screens = list(NSScreen.screens())
        main_h = screens[0].frame().size.height
        out = []
        for sc in screens:
            f, v = sc.frame(), sc.visibleFrame()
            out.append({
                "x": f.origin.x, "y": main_h - (f.origin.y + f.size.height),
                "w": f.size.width, "h": f.size.height,
                "vx": v.origin.x, "vy": main_h - (v.origin.y + v.size.height),
                "vw": v.size.width, "vh": v.size.height})
        return out
    except Exception:  # noqa: BLE001
        return []


def _overlap(ax: float, ay: float, aw: float, ah: float, s: dict[str, float]) -> float:
    w = min(ax + aw, s["vx"] + s["vw"]) - max(ax, s["vx"])
    h = min(ay + ah, s["vy"] + s["vh"]) - max(ay, s["vy"])
    return max(w, 0.0) * max(h, 0.0)


def place_window(state: dict[str, int], screens: list[dict[str, float]]) -> dict[str, Any]:
    """Size, position and min size for a new window given the CURRENT screens.

    The saved position is kept only when at least half of the window rectangle lies on a
    connected screen's visible frame (negative x is fine: a display left of the main one).
    Otherwise the window is centred on the main screen. Size is clamped to the visible frame of
    the screen it lands on, and the minimum size is clamped to that screen too (a 1080-wide
    portrait main display cannot hold the 1100 px minimum; we shrink the minimum, we do not hop
    to another display). Without screen info the OS places it and sizes are untouched.
    """
    w, h = state["width"], state["height"]
    if not screens:
        return {"width": w, "height": h, "x": None, "y": None, "min_size": MIN_SIZE}
    target = None
    if "x" in state and "y" in state:
        x, y = state["x"], state["y"]
        best = max(screens, key=lambda s: _overlap(x, y, w, h, s))
        # the window is clamped to the screen it mostly sits on before the test
        cw, ch = min(w, best["vw"]), min(h, best["vh"])
        if _overlap(x, y, cw, ch, best) >= 0.5 * cw * ch:
            target = best
    if target is None:
        target = screens[0]
        x = y = None
    w, h = min(w, int(target["vw"])), min(h, int(target["vh"]))
    min_size = (min(MIN_SIZE[0], int(target["vw"])), min(MIN_SIZE[1], int(target["vh"])))
    w, h = max(w, min_size[0]), max(h, min_size[1])
    if x is None or y is None:
        x = int(target["vx"] + (target["vw"] - w) / 2)
        y = int(target["vy"] + (target["vh"] - h) / 2)
    else:  # keep it fully reachable on its screen
        x = int(min(max(x, target["vx"]), target["vx"] + target["vw"] - w))
        y = int(min(max(y, target["vy"]), target["vy"] + target["vh"] - h))
    return {"width": w, "height": h, "x": x, "y": y, "min_size": min_size}


# a stuck GUI loop must never leave a ghost process: this long after shutdown starts we _exit
WATCHDOG_S = 8.0


def _hard_exit(code: int = 0) -> None:
    os._exit(code)


def arm_watchdog(delay: float | None = None) -> threading.Timer:
    t = threading.Timer(WATCHDOG_S if delay is None else delay, lambda: _hard_exit(0))
    t.daemon = True
    t.start()
    return t


def _http(port: int, path: str, method: str = "GET", timeout: float = 2.0) -> Any:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def is_ours(port: int) -> bool:
    try:
        return _http(port, "/api/health").get("app") == "shotsieve"
    except Exception:  # noqa: BLE001
        return False


def running_instance() -> int | None:
    """Port of a live ShotSieve server recorded in data/server.json, else None (stale)."""
    from shotsieve.server.lifecycle import server_json_path

    try:
        info = json.loads(server_json_path().read_text())
        pid, port = int(info["pid"]), int(info["port"])
        os.kill(pid, 0)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return port if is_ours(port) else None


def focus_running(port: int) -> bool:
    try:
        return bool(_http(port, "/api/focus", "POST").get("ok"))
    except Exception:  # noqa: BLE001
        return False


def _lock_instance() -> Any:
    """Exclusive lock for the process lifetime (two simultaneous launches: one wins)."""
    from shotsieve.config import data_dir

    fh = open(data_dir() / "app.lock", "w")  # noqa: SIM115 - kept open on purpose
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def plan_drop(paths: list[str]) -> dict[str, Any]:
    """What to do with paths dropped on the window: {"folder", "files"|None, "notes": [...]}.

    A dropped folder wins (the first one; others are mentioned). Otherwise the supported files
    are grouped by parent folder and the parent holding the most files is culled. `error` is set
    when nothing usable was dropped.
    """
    from shotsieve.io.formats import is_supported, is_video

    notes: list[str] = []
    dirs = [p for p in paths if os.path.isdir(p)]
    files = [p for p in paths if os.path.isfile(p)]
    if dirs:
        folder = os.path.abspath(dirs[0])
        if len(dirs) > 1:
            notes.append(f"{len(dirs)} folders dropped: culling {os.path.basename(folder)}.")
        if files:
            notes.append("Loose files ignored because a folder was dropped.")
        return {"folder": folder, "files": None, "notes": notes}
    good = [os.path.abspath(f) for f in files if is_supported(f) or is_video(f)]
    bad = len(paths) - len(good)
    if not good:
        return {"error": "No supported photos in that drop.", "notes": notes}
    by_parent: dict[str, list[str]] = {}
    for f in good:
        by_parent.setdefault(os.path.dirname(f), []).append(f)
    parent = max(by_parent, key=lambda k: len(by_parent[k]))  # first wins a tie
    if len(by_parent) > 1:
        n = len(by_parent[parent])
        notes.append(f"Photos came from {len(by_parent)} folders: culling the {n} "
                     f"in {os.path.basename(parent)}.")
    if bad:
        notes.append(f"{bad} unsupported file(s) ignored.")
    return {"folder": parent, "files": sorted(by_parent[parent]), "notes": notes}


def drop_paths(event: dict[str, Any]) -> list[str]:
    """Full paths of the files in a pywebview DOM drop event."""
    try:
        files = event["dataTransfer"]["files"]
    except (KeyError, TypeError):
        return []
    return [str(f["pywebviewFullPath"]) for f in files if isinstance(f, dict)
            and f.get("pywebviewFullPath")]


def hook_drop(wv: Any, window: Any) -> None:
    """Send dropped paths (real ones, from pywebview's DOM drop handler) to the page."""
    def on_drop(event: dict[str, Any]) -> None:
        plan = plan_drop(drop_paths(event))
        try:
            window.evaluate_js(
                f"window.shotsieveDesktop && window.shotsieveDesktop.onDrop({json.dumps(plan)})")
        except Exception:  # noqa: BLE001
            pass

    def attach() -> None:
        try:
            window.dom.document.events.drop += wv.dom.DOMEventHandler(on_drop, True, True)
        except Exception:  # noqa: BLE001 - old pywebview: drop just does nothing
            pass

    window.events.loaded += attach


# ---- native bits (pyobjc; all best-effort) -----------------------------------------------------


def _set_app_identity() -> None:
    """Menu bar / Dock say ShotSieve, not Python."""
    try:
        from Foundation import NSBundle, NSProcessInfo

        NSProcessInfo.processInfo().setProcessName_(APP_NAME)
        info = NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = APP_NAME
        info["CFBundleDisplayName"] = APP_NAME
    except Exception:  # noqa: BLE001
        pass


def _set_dock_icon() -> None:
    icon = Path(os.environ.get("SHOTSIEVE_APP_ICON", "")) if os.environ.get(
        "SHOTSIEVE_APP_ICON") else None
    if icon is None or not icon.is_file():
        return
    try:
        from AppKit import NSApplication, NSImage

        img = NSImage.alloc().initWithContentsOfFile_(str(icon))
        if img is not None:
            NSApplication.sharedApplication().setApplicationIconImage_(img)
    except Exception:  # noqa: BLE001
        pass


def activate_app() -> None:
    try:
        from AppKit import NSApplication

        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    except Exception:  # noqa: BLE001
        pass


_log = logging.getLogger("shotsieve.app")


def setup_app_log() -> Path | None:
    """<logs dir>/app.log (data/logs in a checkout, ~/Library/Logs/ShotSieve in the app):
    uncaught exceptions (main + threads) with tracebacks, and native crashes (faulthandler).
    The `error: ...` lines in serve.log carry no traceback."""
    try:
        from shotsieve.config import logs_dir

        path = logs_dir() / "app.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 2_000_000:
            path.replace(path.with_suffix(".log.1"))
        for old in [h for h in _log.handlers if getattr(h, "_shotsieve_app", False)]:
            _log.removeHandler(old)
            old.close()
        h = logging.FileHandler(path)
        h._shotsieve_app = True  # type: ignore[attr-defined]
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s pid=%(process)d %(message)s"))
        _log.addHandler(h)
        _log.setLevel(logging.INFO)
        if not faulthandler.is_enabled():
            faulthandler.enable(open(path, "a"))  # noqa: SIM115 - lives as long as the process
        if getattr(sys.excepthook, "_shotsieve", False):
            return path
        prev_hook, prev_thread_hook = sys.excepthook, threading.excepthook

        def hook(tp: Any, val: Any, tb: Any) -> None:
            _log.error("uncaught exception\n%s", "".join(traceback.format_exception(tp, val, tb)))
            prev_hook(tp, val, tb)

        def thook(args: Any) -> None:
            _log.error("uncaught exception in thread %s\n%s",
                       getattr(args.thread, "name", "?"),
                       "".join(traceback.format_exception(
                           args.exc_type, args.exc_value, args.exc_traceback)))
            prev_thread_hook(args)

        hook._shotsieve = True  # type: ignore[attr-defined]
        sys.excepthook, threading.excepthook = hook, thook
        _log.info("app log started")
        return path
    except Exception:  # noqa: BLE001
        return None


def log_exception(msg: str) -> None:
    try:
        _log.error("%s\n%s", msg, traceback.format_exc())
    except Exception:  # noqa: BLE001
        pass


_WINDOW_DELEGATE_CALLBACKS = (
    "windowDidMove_", "windowDidResize_", "windowDidMiniaturize_", "windowDidDeminiaturize_",
    "windowDidBecomeKey_", "windowDidEnterFullScreen_", "windowDidExitFullScreen_")


FULLSCREEN_MASK = 1 << 14  # NSWindowStyleMaskFullScreen


def esc_guard_ignores(window: Any) -> bool:
    """True when an unhandled Esc (`cancelOperation:`) must be swallowed: always in full screen,
    where AppKit would otherwise leave it. Ctrl+Cmd+F, the green button and View menu still work."""
    try:
        return bool(int(window.styleMask()) & FULLSCREEN_MASK)
    except Exception:  # noqa: BLE001
        return True


def guard_cancel_operation() -> bool:
    """pywebview's WKWebView hands an unhandled Esc up the responder chain to NSWindow
    `cancelOperation:`, which exits macOS full screen. Add a no-op `cancelOperation:` to the window
    class (the UI handles Esc itself)."""
    try:
        import objc
        from webview.platforms import cocoa

        def cancelOperation_(self: Any, sender: Any) -> None:
            if esc_guard_ignores(self):
                return
            # windowed: Esc has no native meaning either, ignore it too

        sel = objc.selector(cancelOperation_, selector=b"cancelOperation:", signature=b"v@:@")
        objc.classAddMethods(cocoa.BrowserView.WindowHost, [sel])
        return True
    except Exception:  # noqa: BLE001
        return False


def harden_pywebview() -> bool:
    """pywebview 6.2.1 callbacks such as windowDidMove_ do `i.window.screen().frame()`; NSWindow
    .screen() is None when the window sits on no connected screen (display unplugged or asleep) and
    the AttributeError ('NoneType' object has no attribute 'frame') escaped the event loop and
    ended the app. Wrap them so a failing callback is logged and ignored."""
    try:
        import objc
        from webview.platforms import cocoa

        cls = cocoa.BrowserView.WindowDelegate
        namespace: dict[str, Any] = {}
        for name in _WINDOW_DELEGATE_CALLBACKS:
            sel = cls.__dict__.get(name)
            orig = getattr(sel, "callable", None)
            if orig is None:
                continue

            def make(orig: Any, name: str) -> Any:
                def safe(self: Any, note: Any) -> None:
                    try:
                        orig(self, note)
                    except Exception:  # noqa: BLE001
                        log_exception(f"pywebview {name} failed (ignored)")
                return objc.selector(safe, selector=name.replace("_", ":").encode(),
                                     signature=b"v@:@")

            namespace[name] = make(orig, name)
        guard_cancel_operation()
        if not namespace:
            return False
        objc.classAddMethods(cls, list(namespace.values()))
        return True
    except Exception:  # noqa: BLE001
        return False


_app_hooks: dict[str, Any] = {}


def install_app_hooks(on_reopen: Any, on_terminate: Any) -> bool:
    """Dock click on a running app (applicationShouldHandleReopen) and Cmd+Q / menu Quit
    (applicationWillTerminate) reach us. Added to pywebview's NSApplication delegate class. Must run
    on the main thread, after the GUI loop started. False when there is no delegate yet."""
    try:
        import objc
        from AppKit import NSApplication

        delegate = NSApplication.sharedApplication().delegate()
        if delegate is None:
            return False
        _app_hooks.update(reopen=on_reopen, terminate=on_terminate)

        class _ShotSieveDelegate(objc.Category(type(delegate))):
            def applicationShouldHandleReopen_hasVisibleWindows_(self, app, has_visible):
                try:
                    _app_hooks["reopen"]()
                except Exception:  # noqa: BLE001
                    pass
                return False

            def applicationWillTerminate_(self, note):
                try:
                    _app_hooks["terminate"]()
                except Exception:  # noqa: BLE001
                    pass

        return True
    except Exception:  # noqa: BLE001
        return False


def terminate_app() -> None:
    """Last resort to end a GUI loop whose window is already gone (real pywebview only)."""
    try:
        from AppKit import NSApplication
        from PyObjCTools import AppHelper

        AppHelper.callAfter(NSApplication.sharedApplication().terminate_, None)
    except Exception:  # noqa: BLE001
        pass


class JsApi:
    """Exposed to the page as window.pywebview.api (methods run on pywebview's JS thread)."""

    def __init__(self, webview_mod: Any) -> None:
        self._wv = webview_mod
        self.window: Any = None
        self._menu: Any = None

    def pick(self, kind: str) -> str | None:
        """Real macOS folder / file dialog. None = cancelled or failed (a failure is logged and
        shown as a toast, never raised into the GUI loop). The server re-validates the path."""
        if self.window is None or kind not in ("folder", "lrcat"):
            return None
        wv = self._wv
        fd = getattr(wv, "FileDialog", None)  # pywebview 6: FileDialog.FOLDER / OPEN
        try:
            # pywebview runs the panel on the main thread itself (callAfter + semaphore) when
            # called from this JS-bridge thread
            if kind == "folder":
                res = self.window.create_file_dialog(
                    fd.FOLDER if fd is not None else wv.FOLDER_DIALOG)
            else:
                res = self.window.create_file_dialog(
                    fd.OPEN if fd is not None else wv.OPEN_DIALOG, allow_multiple=False,
                    file_types=LR_TYPES)
            if not res:
                return None
            p = res[0] if isinstance(res, (list, tuple)) else res
            return str(p).rstrip("/") or "/"
        except BaseException:  # noqa: BLE001 - a picker problem must never end the app
            log_exception(f"file dialog ({kind}) failed")
            try:
                self.window.evaluate_js(
                    "typeof toast==='function' && toast('The file dialog failed. "
                    "Type the path instead.', 6000)")
            except Exception:  # noqa: BLE001
                pass
            return None

    def open_external(self, url: str) -> bool:
        return open_external(str(url))

    def menu_state(self, state: Any) -> bool:
        """The page reports its toggles / recent folders; the native menu follows (T23g)."""
        ctl = self._menu
        if ctl is None:
            return False
        try:
            ctl.update(state)
        except Exception:  # noqa: BLE001
            log_exception("menu update failed")
            return False
        return True


# ---- server in a thread ------------------------------------------------------------------------


class ServerThread:
    def __init__(self, port: int, folder: str | None = None) -> None:
        import uvicorn

        from shotsieve.config import load_settings
        from shotsieve.server.app import create_app
        from shotsieve.server.lifecycle import write_server_json

        self.port = port
        self.app = create_app(settings=load_settings(None), default_folder=folder)
        self.server = uvicorn.Server(
            uvicorn.Config(self.app, host="127.0.0.1", port=port, log_level="warning"))
        self.app.state.uvicorn_server = self.server  # POST /api/quit flips should_exit
        original = self.server.startup

        async def startup(sockets: Any = None) -> None:
            await original(sockets)
            if not self.server.should_exit:
                write_server_json(port)

        self.server.startup = startup  # type: ignore[method-assign]
        self.thread = threading.Thread(target=self._run, name="shotsieve-server", daemon=True)
        self.error: BaseException | None = None

    def _run(self) -> None:
        try:
            self.server.run()
        except BaseException as exc:  # noqa: BLE001
            self.error = exc

    def start(self) -> None:
        self.thread.start()

    def wait_ready(self, timeout: float = 60.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if not self.thread.is_alive():
                return False
            if getattr(self.server, "started", False) and is_ours(self.port):
                return True
            time.sleep(0.1)
        return False

    def stop(self, timeout: float = 5.0) -> None:
        from shotsieve.server.lifecycle import remove_server_json_if_ours

        self.server.should_exit = True
        self.thread.join(timeout)
        remove_server_json_if_ours()


# ---- the app -----------------------------------------------------------------------------------


def _load_webview() -> Any:
    return importlib.import_module(os.environ.get("SHOTSIEVE_WEBVIEW_MODULE") or "webview")


def real_gui_name(wv: Any) -> bool:
    return getattr(wv, "__name__", "") == "webview"


def run_app(folder: str | None = None, webview_mod: Any = None) -> int:
    from shotsieve.config import data_dir

    wv = webview_mod or _load_webview()
    setup_app_log()
    if getattr(wv, "__name__", "") == "webview":
        harden_pywebview()
    lock = _lock_instance()
    if lock is None:  # another launch is starting or running: wait for it, then focus
        for _ in range(100):
            port = running_instance()
            if port is not None:
                focus_running(port)
                return 0
            time.sleep(0.2)
        print("ShotSieve is already starting.", file=sys.stderr)
        return 1
    port = running_instance()
    if port is not None:  # e.g. a headless `shotsieve serve` is up: show it, do not double up
        focus_running(port)
        print(f"ShotSieve is already running on port {port}.", file=sys.stderr)
        return 0
    port = pick_port()
    if port is None:
        print("No free port between 8765 and 8775: quit other programs using them.",
              file=sys.stderr)
        return 1
    try:  # once, before any request thread exists (HOME redirect is then never raced)
        from shotsieve._ort import import_onnxruntime

        import_onnxruntime()
    except ImportError:
        pass

    srv = ServerThread(port, folder)
    api = JsApi(wv)
    from shotsieve.desktop_menu import MenuController

    menu_ctl = MenuController(wv, lambda: holder.get("window"))
    api._menu = menu_ctl
    if real_gui_name(wv):
        menu_ctl.patch_cocoa()
    state = {"closing": False}
    holder: dict[str, Any] = {}
    geo: dict[str, int] = {}  # last known geometry; close never needs to read the native window
    timers: dict[str, Any] = {}

    def ensure_watchdog() -> None:
        if "watchdog" not in timers:
            timers["watchdog"] = arm_watchdog()

    def shutdown() -> None:
        if not state["closing"]:
            state["closing"] = True
            ensure_watchdog()  # whatever hangs below, the process ends
            srv.stop()

    atexit.register(shutdown)  # path that skips window events
    srv.start()
    if not srv.wait_ready():
        shutdown()
        t = timers.pop("watchdog", None)
        if t is not None:
            t.cancel()
        print(f"ShotSieve server did not start: {srv.error!r}", file=sys.stderr)
        return 1

    real_gui = getattr(wv, "__name__", "") == "webview"

    def alive(win: Any) -> bool:
        try:
            return win is not None and win in list(getattr(wv, "windows", [win]))
        except Exception:  # noqa: BLE001
            return False

    def make_window() -> Any:
        screens = get_screens()
        pl = place_window(load_window_state(), screens)
        kw: dict[str, Any] = {}
        if pl["x"] is not None:
            kw.update(x=pl["x"], y=pl["y"])
        geo.update(width=pl["width"], height=pl["height"])
        if pl["x"] is not None:
            geo.update(x=pl["x"], y=pl["y"])
        win = wv.create_window(
            APP_NAME, f"http://127.0.0.1:{port}/", js_api=api, width=pl["width"],
            height=pl["height"], min_size=pl["min_size"], text_select=True, **kw)
        api.window = holder["window"] = win

        def refresh() -> None:  # timer thread: a failed read just skips
            g = (native_geometry() if real_gui else None) or read_geometry(win)
            if g is not None and alive(win) and not state["closing"]:
                geo.update(g)
                write_window_state(dict(geo))

        def on_geometry(*_: Any) -> None:  # main thread: never touch the native window here
            try:
                old = timers.get("geo")
                if old is not None:
                    old.cancel()
                t = threading.Timer(0.3, refresh)
                t.daemon = True
                timers["geo"] = t
                t.start()
            except Exception:  # noqa: BLE001
                pass

        def on_resized(width: Any = None, height: Any = None, *_: Any) -> None:
            try:
                geo.update(width=int(width), height=int(height))
            except Exception:  # noqa: BLE001
                pass
            on_geometry()

        def on_closing() -> None:
            if geo:  # last known geometry (cache), no native read
                write_window_state(dict(geo))

        def on_closed() -> None:  # last window gone: the app is done
            if not any(alive(w) and w is not win for w in list(getattr(wv, "windows", []))):
                shutdown()

        hook_drop(wv, win)
        for name, fn in (("closing", on_closing), ("closed", on_closed),
                         ("moved", on_geometry), ("resized", on_resized),
                         ("shown", on_geometry)):
            ev = getattr(win.events, name, None)
            if ev is not None:
                ev += fn
        return win

    def focus() -> None:
        """Bring the window forward; create a new one when it is gone. Never leaves the app
        windowless and invisible."""
        if state["closing"]:
            return
        activate_app()
        win = holder.get("window")
        if not alive(win):
            try:
                make_window()
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            win.restore()
            win.show()
        except Exception:  # noqa: BLE001
            pass

    srv.app.state.focus = focus
    _set_app_identity()
    make_window()

    def watch(*_: Any) -> None:
        """Window follows the server (/api/quit) and the optional self-close test knob; also
        installs the Dock reopen / Cmd+Q hooks once the GUI loop runs."""
        auto = float(os.environ.get("SHOTSIEVE_APP_AUTOCLOSE_S") or 0)
        t0 = time.monotonic()
        real = getattr(wv, "__name__", "") == "webview"
        if real:
            try:
                from PyObjCTools import AppHelper

                def hooks() -> None:
                    install_app_hooks(focus, shutdown)

                AppHelper.callAfter(hooks)
            except Exception:  # noqa: BLE001
                pass
        while not state["closing"]:
            if not srv.thread.is_alive() or (auto and time.monotonic() - t0 >= auto):
                ensure_watchdog()
                win = holder.get("window")
                try:
                    if not alive(win):
                        raise RuntimeError("window already gone")
                    win.destroy()
                except Exception:  # noqa: BLE001
                    if real:
                        terminate_app()
                return
            time.sleep(0.25)

    _set_dock_icon()
    try:
        wv.start(watch, private_mode=False, storage_path=str(data_dir() / "webview"),
                 menu=menu_ctl.menus())
        shutdown()
    except BaseException:  # noqa: BLE001 - leave a traceback, then end cleanly (never a ghost)
        log_exception("GUI loop crashed")
        shutdown()
        raise
    finally:
        for key in ("geo", "watchdog"):  # cmd_app arms its own final exit timer
            t = timers.pop(key, None)
            if t is not None:
                t.cancel()
    return 0
