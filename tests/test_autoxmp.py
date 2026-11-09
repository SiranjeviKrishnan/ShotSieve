# ruff: noqa: E501, F401, F811
"""T23g: automatic XMP sidecars (default on, background, debounced, user-wins, per-shoot switch)."""

from __future__ import annotations

import time

from tests.test_faces_api import client, ids, shoot, spy  # noqa: F401

FOREIGN = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="5"/>'
    "</rdf:RDF></x:xmpmeta>"
)


def wait(cond, t=8.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_default_is_on_and_a_user_change_updates_the_sidecar_in_the_background(client, shoot, ids):
    assert client.get("/api/xmp/status").json()["auto"] is True
    assert not (shoot / "d.xmp").exists()
    t0 = time.time()
    r = client.post("/api/override", json={"image_id": ids["d.jpg"], "decision": "reject"})
    assert r.status_code == 200 and time.time() - t0 < 1.0  # the request never waits for the write
    assert wait(lambda: (shoot / "d.xmp").exists()), "sidecar not written within a few seconds"
    first = (shoot / "d.xmp").read_text()
    assert "shotsieve" in first.lower()
    client.post("/api/override", json={"image_id": ids["d.jpg"], "decision": "keep"})  # toggle: updated
    assert wait(lambda: (shoot / "d.xmp").read_text() != first)
    client.post("/api/label", json={"image_ids": [ids["a.jpg"]], "label": "red"})
    assert wait(lambda: (shoot / "a.xmp").exists())
    assert not any(client.get("/api/xmp/status").json()["pending"] for _ in [0]) or wait(lambda: not client.get("/api/xmp/status").json()["pending"])


def test_switching_off_stops_all_writes_and_the_state_is_per_shoot(client, shoot, ids):
    assert client.post("/api/xmp/auto", json={"on": False}).json()["auto"] is False
    client.post("/api/override", json={"image_id": ids["c.jpg"], "decision": "keep"})
    client.post("/api/label", json={"image_ids": [ids["b.jpg"]], "label": "green"})
    time.sleep(2.0)
    assert not (shoot / "c.xmp").exists() and not (shoot / "b.xmp").exists()
    assert client.get("/api/xmp/status").json()["auto"] is False
    assert client.post("/api/xmp/auto", json={"on": True}).json()["auto"] is True  # back on: writes everything
    assert wait(lambda: (shoot / "c.xmp").exists())


def test_user_wins_a_foreign_rating_is_kept(client, shoot, ids):
    (shoot / "b.xmp").write_text(FOREIGN)
    client.post("/api/xmp/auto", json={"on": False})
    client.post("/api/xmp/auto", json={"on": True})  # queue every image
    assert wait(lambda: (shoot / "a.xmp").exists())
    assert wait(lambda: not client.get("/api/xmp/status").json()["pending"])
    time.sleep(0.5)
    assert 'Rating="5"' in (shoot / "b.xmp").read_text()  # the photographer's rating survives
    st = client.get("/api/xmp/status").json()
    assert st["written"] >= 1 and st["failed"] == 0


def test_open_in_lightroom_flushes_and_asks_nothing(client, shoot, ids, spy):
    r = client.post("/api/lightroom/open", json={"ids": [ids["a.jpg"]]}).json()
    assert r["ok"] and "needs_consent" not in r and (shoot / "a.xmp").exists()


def test_the_write_xmp_button_and_menu_item_are_gone():
    from pathlib import Path

    import shotsieve.desktop_menu as dm

    static = Path(dm.__file__).parent / "server" / "static"
    html = (static / "index.html").read_text()
    assert 'id="btn-xmp"' not in html and 'id="chk-xmp"' not in html and "overlay-lrask" not in html
    assert "xmp" not in dm.ACTION_IDS and "autoxmp" in dm.ACTION_IDS
