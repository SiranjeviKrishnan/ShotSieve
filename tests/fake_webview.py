"""A display-less stand-in for pywebview (`import webview`): enough API for shotsieve.desktop.

Tests use it in-process (pass as `webview_mod`) or in a subprocess via
SHOTSIEVE_WEBVIEW_MODULE=fake_webview with tests/ on PYTHONPATH.
"""

from __future__ import annotations

import enum
import threading
import time


class FileDialog(enum.IntEnum):
    OPEN = 10
    FOLDER = 20
    SAVE = 30


FOLDER_DIALOG = 20
OPEN_DIALOG = 10

started = threading.Event()
windows: list[Window] = []  # live windows, like webview.windows
created: list[Window] = []  # every window ever created
closed_any = threading.Event()
GONE_MSG = "'NoneType' object has no attribute 'frame'"
dialog_result: list[str] | None = None
dialog_calls: list[tuple] = []
dialog_raises: BaseException | None = None
on_started = None  # optional callable(window) run on the start() worker thread


class Event:
    def __init__(self) -> None:
        self.handlers: list = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self

    def fire(self, *args) -> None:
        for h in list(self.handlers):
            h(*args)


class Events:
    def __init__(self) -> None:
        self.closing = Event()
        self.closed = Event()
        self.loaded = Event()
        self.shown = Event()
        self.moved = Event()
        self.resized = Event()


class DOMEventHandler:
    def __init__(self, callback, prevent_default=False, stop_propagation=False, **kw) -> None:
        self.callback = callback
        self.prevent_default, self.stop_propagation = prevent_default, stop_propagation


class _DomEvents:
    def __init__(self) -> None:
        self.drop = Event()  # holds DOMEventHandler objects


class _Dom:
    def __init__(self) -> None:
        self.document = type("Doc", (), {"events": _DomEvents()})()


class dom:  # noqa: N801 - mirrors `webview.dom`
    DOMEventHandler = DOMEventHandler


class menu:  # noqa: N801 - mirrors `webview.menu`
    class Menu:
        def __init__(self, title, items=()) -> None:
            self.title, self.items = title, list(items)

    class MenuAction:
        def __init__(self, title, function) -> None:
            self.title, self.function = title, function

    class MenuSeparator:
        pass


start_kwargs: dict = {}


class Window:
    def __init__(self, title, url, **kw) -> None:
        self.title, self.url, self.kw = title, url, kw
        self._geo = {"width": kw.get("width", 0), "height": kw.get("height", 0),
                     "x": kw.get("x") or 0, "y": kw.get("y") or 0}
        self.geometry_raises = False  # pywebview after the native window is gone
        self.vanished = False
        self.events = Events()
        self.dom = _Dom()
        self.js_calls: list[str] = []
        self.shown = 0
        self.restored = 0
        self._gone = threading.Event()

    def _read(self, key):
        if self.geometry_raises or self.vanished or self._gone.is_set():
            raise AttributeError(GONE_MSG)
        return self._geo[key]

    width = property(lambda self: self._read("width"))
    height = property(lambda self: self._read("height"))
    x = property(lambda self: self._read("x"))
    y = property(lambda self: self._read("y"))

    def move_to(self, x, y, width=None, height=None) -> None:
        """The user drags / resizes the window: native state changes, events fire."""
        self._geo.update(x=x, y=y)
        if width:
            self._geo.update(width=width, height=height)
            self.events.resized.fire(width, height)
        self.events.moved.fire(x, y)

    def vanish(self) -> None:
        """Native window gone without any event, GUI loop still running (the field bug)."""
        self.vanished = True
        if self in windows:
            windows.remove(self)

    def show(self) -> None:
        self.shown += 1

    def restore(self) -> None:
        self.restored += 1

    def destroy(self) -> None:
        if self.vanished:
            return  # nothing native left to close
        self.simulate_close()

    def simulate_close(self) -> None:
        """What a click on the red button does: closing, then closed."""
        if self._gone.is_set():
            return
        self.events.closing.fire()
        self._gone.set()
        if self in windows:
            windows.remove(self)
        closed_any.set()
        self.events.closed.fire()

    def evaluate_js(self, code):
        self.js_calls.append(code)

    def simulate_drop(self, paths):
        """What a Finder drop does once the page is loaded: handlers get the DOM event dict."""
        ev = {"dataTransfer": {"files": [{"name": p, "pywebviewFullPath": p} for p in paths]}}
        for h in list(self.dom.document.events.drop.handlers):
            h.callback(ev)

    def create_file_dialog(self, kind, **kw):
        dialog_calls.append((kind, kw))
        if dialog_raises is not None:
            raise dialog_raises
        return dialog_result


def create_window(title, url, **kw) -> Window:
    w = Window(title, url, **kw)
    windows.append(w)
    created.append(w)
    return w


def start(func=None, *args, **kw) -> None:
    """Block like the real GUI loop until every window is gone."""
    start_kwargs.clear()
    start_kwargs.update(kw)
    started.set()
    if func is not None:
        threading.Thread(target=func, args=args, daemon=True).start()
    if on_started is not None:
        threading.Thread(target=on_started, args=(windows[-1],), daemon=True).start()
    # like the real loop: ends once a window was closed and none is left (a window that merely
    # vanished keeps the loop alive: the stuck-process case)
    while not (closed_any.is_set() and not windows):
        time.sleep(0.05)
