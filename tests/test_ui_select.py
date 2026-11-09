# ruff: noqa: E501, E702
"""T23i in a real browser (Playwright + Chromium; `ui` marker): multi-select (Cmd-click, Shift-click,
Cmd+A, Esc, Shift+arrows), batch Keep / Reject / stars / labels, the context menu, Move to Trash with the
confirm sheet and Undo. The server's trash backend is a directory (SHOTSIEVE_FAKE_TRASH), never the real Trash."""

from __future__ import annotations

import json
import shutil
import time
import urllib.request

import pytest
from playwright.sync_api import expect

from tests.test_ui import Server, Ui

pytestmark = [pytest.mark.ui, pytest.mark.models, pytest.mark.usefixtures("models_ready")]


def post(srv, path, body):
    req = urllib.request.Request(
        srv.url + path, data=json.dumps({"folder": str(srv.folder), **body}).encode(),
        headers={"Content-Type": "application/json", "Origin": srv.url}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def wait_for(fn, what="condition", t=30.0):
    end = time.time() + t
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError("timed out: " + what)


@pytest.fixture()
def srv(tmp_path, shoot_dir):
    dest = tmp_path / "shoot"
    shutil.copytree(shoot_dir, dest)
    s = Server(dest, env={"SHOTSIEVE_FAKE_TRASH": str(tmp_path / "ftrash")})
    jid = post(s, "/api/cull", {})["job_id"]
    wait_for(lambda: s.get(f"/api/jobs/{jid}")["state"] in ("done", "error"), "cull")
    assert s.get(f"/api/jobs/{jid}")["state"] == "done"
    s.trash = tmp_path / "ftrash"
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


def cell(ui, i):
    return ui.page.locator(".cell").nth(i)


def msel(ui):
    return ui.page.locator(".cell.msel")


def files_in_trash(srv):
    return sorted(p.name for p in srv.trash.rglob("*") if p.is_file()) if srv.trash.exists() else []


def by_name(srv):
    return {im["name"]: im for im in srv.shoot()["images"]}


def test_cmd_click_shift_click_cmd_a_esc(ui):
    n = ui.cells().count()
    assert n >= 10
    cell(ui, 0).click()
    expect(msel(ui)).to_have_count(0)  # a single photo is the plain selection
    cell(ui, 2).click(modifiers=["Meta"])
    expect(msel(ui)).to_have_count(2)
    expect(ui.page.locator("#sb-info")).to_contain_text("2 selected")
    expect(ui.page.locator("#multi-info")).to_contain_text("2 photos selected")
    expect(ui.page.locator(".cell.msel .chk")).to_have_count(2)  # the check badge
    cell(ui, 2).click(modifiers=["Meta"])  # toggles it out again
    expect(msel(ui)).to_have_count(0)  # back to one: no multi-selection left
    cell(ui, 3).click(modifiers=["Meta"])
    cell(ui, 6).click(modifiers=["Shift"])  # the range from the anchor (the cmd-clicked 3) in grid order
    expect(msel(ui)).to_have_count(4)
    assert [int(c.get_attribute("data-id")) for c in msel(ui).all()] == [int(cell(ui, i).get_attribute("data-id")) for i in (3, 4, 5, 6)]
    ui.page.keyboard.press("Escape")  # clears it, does not leave anything else
    expect(msel(ui)).to_have_count(0)
    expect(ui.page.locator("#multi-info")).to_be_hidden()
    ui.page.keyboard.press("Meta+a")
    expect(msel(ui)).to_have_count(n)
    expect(ui.page.locator("#sb-info")).to_contain_text(f"{n} selected")
    ui.page.keyboard.press("Escape")
    expect(msel(ui)).to_have_count(0)
    # Shift+arrows extend
    cell(ui, 1).click()
    ui.page.keyboard.press("Shift+ArrowRight"); ui.page.keyboard.press("Shift+ArrowRight")
    expect(msel(ui)).to_have_count(3)
    ui.page.keyboard.press("Shift+ArrowLeft")
    expect(msel(ui)).to_have_count(2)


def test_batch_keys_apply_to_the_whole_selection(ui, srv):
    ids = [int(cell(ui, i).get_attribute("data-id")) for i in (1, 2, 3)]
    cell(ui, 1).click(); cell(ui, 3).click(modifiers=["Shift"])
    expect(msel(ui)).to_have_count(3)
    ui.page.keyboard.press("p")
    wait_for(lambda: all(im["decision"] == "keep" and im["overridden"] for im in srv.shoot()["images"] if im["id"] in ids), "keep")
    ui.page.keyboard.press("x")
    wait_for(lambda: all(im["decision"] == "reject" and im["overridden"] for im in srv.shoot()["images"] if im["id"] in ids), "reject")
    ui.page.keyboard.press("x")  # again: back to the AI decision
    wait_for(lambda: not any(im["overridden"] for im in srv.shoot()["images"] if im["id"] in ids), "model")
    ui.page.keyboard.press("4")
    wait_for(lambda: all(im["rating"] == 4 for im in srv.shoot()["images"] if im["id"] in ids), "stars")
    expect(msel(ui)).to_have_count(3)  # the selection survives the change
    ui.page.keyboard.press("0")
    wait_for(lambda: all(not im["rating"] for im in srv.shoot()["images"] if im["id"] in ids), "no stars")
    ui.page.keyboard.press("7")
    labels = lambda: srv.get(f"/api/shoot/facets?folder={srv.folder}")["labels"]  # noqa: E731
    wait_for(lambda: all(labels().get(str(i)) == "yellow" for i in ids), "yellow on all three")
    expect(msel(ui)).to_have_count(3)
    ui.page.keyboard.press("7")  # again removes it from all
    wait_for(lambda: not any(labels().get(str(i)) for i in ids), "label removed")
    # the panel buttons do the same
    ui.page.click("#multi-info button[data-m=keep]")
    wait_for(lambda: all(im["decision"] == "keep" for im in srv.shoot()["images"] if im["id"] in ids), "panel keep")
    ui.page.click("#multi-info button[data-stars='2']")
    wait_for(lambda: all(im["rating"] == 2 for im in srv.shoot()["images"] if im["id"] in ids), "panel stars")
    ui.page.click("#multi-info button[data-label=green]")
    wait_for(lambda: all(labels().get(str(i)) == "green" for i in ids), "panel label")
    ui.page.click("#multi-info button[data-m=model]")
    wait_for(lambda: not any(im["overridden"] for im in srv.shoot()["images"] if im["id"] in ids), "panel model")
    others = [im for im in srv.shoot()["images"] if im["id"] not in ids]
    assert not any(im["overridden"] for im in others)


def test_context_menu_outside_inside_and_ctrl_click(ui):
    cell(ui, 1).click(); cell(ui, 3).click(modifiers=["Shift"])
    expect(msel(ui)).to_have_count(3)
    cell(ui, 2).click(button="right")  # inside the selection: kept, the menu is for all three
    expect(ui.page.locator("#ctx-menu")).to_be_visible()
    expect(ui.page.locator("#ctx-menu [data-trash]")).to_contain_text("Move 3 Photos to Trash")
    expect(msel(ui)).to_have_count(3)
    ui.page.keyboard.press("Escape")
    expect(ui.page.locator("#ctx-menu")).to_be_hidden()
    expect(msel(ui)).to_have_count(3)  # Esc closed only the menu
    cell(ui, 8).click(button="right")  # outside: selects that photo alone first
    expect(ui.page.locator("#ctx-menu")).to_be_visible()
    expect(msel(ui)).to_have_count(0)
    assert int(ui.page.locator(".cell.sel").first.get_attribute("data-id")) == int(cell(ui, 8).get_attribute("data-id"))
    expect(ui.page.locator("#ctx-menu [data-trash]")).to_have_text("Move to Trash ⌘⌫")
    ui.page.keyboard.press("Escape")
    # Ctrl-click is a right-click on a Mac: the menu opens, nothing is toggled
    cell(ui, 5).click(modifiers=["Control"])
    expect(ui.page.locator("#ctx-menu")).to_be_visible()
    expect(msel(ui)).to_have_count(0)
    assert int(ui.page.locator(".cell.sel").first.get_attribute("data-id")) == int(cell(ui, 5).get_attribute("data-id"))


def test_move_to_trash_confirm_cancel_move_and_undo(ui, srv):
    before = ui.cells().count()
    names = [cell(ui, i).get_attribute("title") for i in (2, 3)]
    ids = [int(cell(ui, i).get_attribute("data-id")) for i in (2, 3)]
    cell(ui, 2).click(); cell(ui, 3).click(modifiers=["Shift"])
    ui.page.keyboard.press("Meta+Backspace")
    sheet = ui.page.locator("#trash-sheet")
    expect(sheet).to_be_visible()
    expect(sheet).to_contain_text("Move 2 photos to the Trash?")
    expect(sheet).to_contain_text("put them back from the Trash in Finder")
    expect(sheet).to_contain_text(names[0])
    expect(ui.page.locator("#tr-pairs")).to_be_checked()
    assert ui.page.evaluate("document.activeElement.id") == "tr-cancel"  # Cancel is the default
    ui.page.keyboard.press("Escape")  # Esc cancels
    expect(sheet).to_be_hidden()
    assert (srv.folder / names[0]).exists() and ui.cells().count() == before and not files_in_trash(srv)
    # confirm
    ui.page.keyboard.press("Meta+Backspace")
    ui.page.click("#tr-go")
    expect(ui.cells()).to_have_count(before - 2)
    wait_for(lambda: sorted(names) == files_in_trash(srv), "files in the (fake) trash")
    assert not (srv.folder / names[0]).exists() and not (srv.folder / names[1]).exists()
    expect(ui.page.locator("#toast")).to_contain_text("Moved 2 photos to Trash")
    expect(ui.page.locator("#toast button")).to_have_text("Undo")
    assert ui.page.locator(".cell.sel").count() == 1  # the next photo is selected
    wait_for(lambda: all(i not in [im["id"] for im in srv.shoot()["images"]] for i in ids), "rows gone")
    ui.page.click("#toast button")
    expect(ui.cells()).to_have_count(before, timeout=30000)
    assert (srv.folder / names[0]).exists() and (srv.folder / names[1]).exists() and not files_in_trash(srv)
    wait_for(lambda: set(ids) <= {im["id"] for im in srv.shoot()["images"]}, "same ids back")
    expect(msel(ui)).to_have_count(2)  # the restored photos are selected


def test_trash_from_context_menu_and_dont_ask_again(ui, srv):
    before = ui.cells().count()
    name = cell(ui, 4).get_attribute("title")
    cell(ui, 4).click(button="right")
    ui.page.click("#ctx-menu [data-trash]")
    expect(ui.page.locator("#trash-sheet")).to_contain_text("Move 1 photo to the Trash?")
    ui.page.check("#tr-noask")
    ui.page.click("#tr-go")
    expect(ui.cells()).to_have_count(before - 1)
    wait_for(lambda: files_in_trash(srv) == [name], "trashed")
    time.sleep(1.5)  # the incremental cull behind it
    cell(ui, 4).click()
    ui.page.keyboard.press("Meta+Backspace")  # no sheet this session
    expect(ui.cells()).to_have_count(before - 2)
    expect(ui.page.locator("#trash-sheet")).to_be_hidden()
    ui.page.keyboard.press("Meta+z")  # Cmd+Z undoes the last trash
    expect(ui.cells()).to_have_count(before - 1, timeout=30000)


def test_layout_1100x720_sheet_panel_and_no_overflow(browser, srv):
    ctx = browser.new_context(viewport={"width": 1100, "height": 720})
    u = Ui(ctx.new_page(), srv).open()
    try:
        p = u.page
        p.locator(".cell").nth(0).click(); p.locator(".cell").nth(5).click(modifiers=["Shift"])
        box = p.locator("#multi-info").bounding_box()
        assert box and box["x"] + box["width"] <= 1100 + 1
        assert p.evaluate("document.documentElement.scrollWidth") <= 1100
        # every panel button is inside the window
        for b in p.locator("#multi-info button").all():
            bb = b.bounding_box()
            assert bb and bb["x"] >= 0 and bb["x"] + bb["width"] <= 1101, b.inner_text()
        p.keyboard.press("Meta+Backspace")
        expect(p.locator("#trash-sheet")).to_be_visible()
        sb = p.locator("#trash-sheet").bounding_box()
        assert sb and sb["x"] >= 0 and sb["y"] >= 0 and sb["x"] + sb["width"] <= 1100 and sb["y"] + sb["height"] <= 720
        p.keyboard.press("Escape")
        p.locator(".cell").nth(2).click(button="right")
        mb = p.locator("#ctx-menu").bounding_box()
        assert mb and mb["x"] + mb["width"] <= 1100 and mb["y"] + mb["height"] <= 720
        u.check()
    finally:
        ctx.close()


def test_compare_button_in_the_multi_panel_compares_the_selected_photos(ui):
    cell(ui, 0).click()
    cell(ui, 2).click(modifiers=["Meta"])
    ids = [int(c.get_attribute("data-id")) for c in msel(ui).all()]
    expect(ui.page.locator('#multi-info button[data-m="compare"]')).to_be_visible()
    ui.page.click('#multi-info button[data-m="compare"]')
    expect(ui.page.locator("#overlay-compare")).to_be_visible()
    assert f"/api/preview/{ids[0]}" in ui.page.get_attribute("#cmp-a", "src")
    assert f"/api/preview/{ids[1]}" in ui.page.get_attribute("#cmp-b", "src")
    expect(ui.page.locator("#cmp-sub")).to_contain_text("2 selected")
    ui.key("c")  # the C key closes it again
    expect(ui.page.locator("#overlay-compare")).to_be_hidden()
    ui.key("c")  # and with the multi-selection still on, opens the selected pair
    expect(ui.page.locator("#overlay-compare")).to_be_visible()
