# ruff: noqa: E501
"""T23e API: colour labels (set / read back / XMP user-wins), camera and lens facets (EXIF read
by a back-fill that never touches decisions), the person map, filter-bar data at 10k images."""

from __future__ import annotations

import shutil
import time

import numpy as np
import piexif
import pytest
from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.features.record import RawFeatures
from shotsieve.io.exif import ImageMeta, read_lens
from shotsieve.io.xmp import label_key, read_sidecar, user_values, write_sidecar
from shotsieve.pipeline.cache import Cache, CachedImage
from shotsieve.server import filters as filters_server
from shotsieve.server.app import create_app
from tests.test_api import _row, shoot_of

NAMES = ["burst0_0", "burst0_1", "burst0_2", "landscape", "black", "burst1_3"]
# (make, model, w, h) per file: two spellings of one make, a portrait, a square
CAMS = {
    "burst0_0": ("Sony", "ILCE-7M4", 6000, 4000), "burst0_1": ("SONY", "ILCE-7M4", 6000, 4000),
    "burst0_2": ("Sony", "ILCE-7M4", 4000, 6000), "landscape": ("Canon", "EOS R5", 1000, 995),
    "black": ("Canon", "EOS R5", 1000, 1000), "burst1_3": ("", "", 0, 0),
}  # fmt: skip


def _lens(path, model, serial):
    ex = piexif.load(str(path))
    ex["Exif"][piexif.ExifIFD.LensModel] = model.encode()
    ex["Exif"][piexif.ExifIFD.LensSerialNumber] = serial.encode()
    piexif.insert(piexif.dump(ex), str(path))


@pytest.fixture()
def shoot(tmp_path, shoot_dir):
    f = tmp_path / "shoot"
    f.mkdir()
    for n in NAMES:
        shutil.copy(shoot_dir / f"{n}.jpg", f / f"{n}.jpg")
    _lens(f / "burst0_0.jpg", "FE 24-70mm F2.8 GM", "123456")
    _lens(f / "burst0_1.jpg", "FE 24-70mm F2.8 GM", "123456")
    _lens(f / "landscape.jpg", "RF50mm F1.2 L", "00000")
    rows = [
        _row(f / "burst0_0.jpg", "keep", 0, True), _row(f / "burst0_1.jpg", "reject", 0, reasons=["duplicate"]),
        _row(f / "burst0_2.jpg", "reject", 0, reasons=["duplicate_tie"]), _row(f / "landscape.jpg", "keep", 1, True),
        _row(f / "black.jpg", "reject", 2, True, reasons=["accidental"]), _row(f / "burst1_3.jpg", "review", 3, True),
    ]  # fmt: skip
    with Cache.for_folder(f) as c:
        c.save_decisions(rows)
        for n in NAMES:
            make, model, w, h = CAMS[n]
            st = (f / f"{n}.jpg").stat()
            rec = CachedImage(ImageMeta(timestamp=1.0, make=make, model=model),
                              RawFeatures(width=w, height=h), np.zeros(384, np.float32),
                              np.zeros((0, 128), np.float32))  # fmt: skip
            c.put(str(f / f"{n}.jpg"), st.st_size, st.st_mtime_ns, rec)
        c.commit()
    return f


@pytest.fixture()
def client(shoot):
    with TestClient(create_app(Settings(), default_folder=shoot)) as c:
        yield c


def ids_of(client, f):
    return {im["name"]: im["id"] for im in shoot_of(client, f)[0]["images"]}


def scan_now(f):
    filters_server.scan(f, Settings())


# ---- labels ------------------------------------------------------------------------------


def test_label_set_read_back_and_clear(client, shoot):
    ids = ids_of(client, shoot)
    r = client.post("/api/label", json={"folder": str(shoot), "image_ids": [ids["landscape.jpg"], ids["black.jpg"]], "label": "purple"})
    assert r.status_code == 200 and r.json() == {"ok": True, "n": 2, "label": "purple"}
    labels = client.get("/api/shoot/facets", params={"folder": str(shoot)}).json()["labels"]
    assert labels == {str(ids["landscape.jpg"]): "purple", str(ids["black.jpg"]): "purple"}
    client.post("/api/label", json={"folder": str(shoot), "image_ids": [ids["black.jpg"]], "label": "none"})
    labels = client.get("/api/shoot/facets", params={"folder": str(shoot)}).json()["labels"]
    assert list(labels) == [str(ids["landscape.jpg"])]  # newest wins; "none" clears it
    # persisted: a new app instance reads the same
    with TestClient(create_app(Settings(), default_folder=shoot)) as c2:
        assert c2.get("/api/shoot/facets", params={"folder": str(shoot)}).json()["labels"] == labels


def test_label_validation(client, shoot):
    ids = ids_of(client, shoot)
    f = str(shoot)
    assert client.post("/api/label", json={"folder": f, "image_ids": [ids["black.jpg"]], "label": "pink"}).status_code == 422
    assert client.post("/api/label", json={"folder": f, "image_ids": [], "label": "red"}).status_code == 422
    assert client.post("/api/label", json={"folder": f, "image_ids": [99999], "label": "red"}).status_code == 404
    assert client.post("/api/label", json={"folder": "../x", "image_ids": [1], "label": "red"}).status_code == 400


def test_label_written_to_xmp_user_wins_and_reads_back(client, shoot):
    ids = ids_of(client, shoot)
    f = str(shoot)
    # a foreign tool had labelled landscape Blue; black has our own decision label only
    write_sidecar(shoot / "landscape.jpg", label="Blue", record=False)
    client.post("/api/label", json={"folder": f, "image_ids": [ids["landscape.jpg"]], "label": "red"})
    client.post("/api/label", json={"folder": f, "image_ids": [ids["black.jpg"]], "label": "purple"})
    # burst0_0 keeps a foreign label because the user set none in ShotSieve
    write_sidecar(shoot / "burst0_0.jpg", label="Green", record=False)
    out = client.post("/api/xmp/write", json={"folder": f}).json()
    assert out["failed"] == 0
    land = read_sidecar(shoot / "landscape.jpg")
    assert land.label == "Red"  # the user's label replaced the foreign one
    assert read_sidecar(shoot / "black.jpg").label == "Purple"
    # not recorded as ShotSieve's own label: reads back as the photographer's
    assert label_key(user_values(land, Settings())[1]) == "red"
    # foreign label with no user choice is left alone (existing rule)
    assert read_sidecar(shoot / "burst0_0.jpg").label == "Green"
    # clearing writes an empty label
    client.post("/api/label", json={"folder": f, "image_ids": [ids["black.jpg"]], "label": "none"})
    client.post("/api/xmp/write", json={"folder": f})
    assert read_sidecar(shoot / "black.jpg").label in ("", None)


def test_sidecar_labels_read_back_by_scan(client, shoot):
    ids = ids_of(client, shoot)
    write_sidecar(shoot / "landscape.jpg", label="Blue", record=False)
    write_sidecar(shoot / "black.jpg", label="Review", record=False)  # not a colour: ignored
    scan_now(shoot)
    d = client.get("/api/shoot/facets", params={"folder": str(shoot)}).json()
    assert d["labels"] == {str(ids["landscape.jpg"]): "blue"} and d["pending"] == 0
    # ShotSieve's own decision labels (Green for keep) are not the photographer's
    client.post("/api/xmp/write", json={"folder": str(shoot)})
    scan_now(shoot)
    labels = client.get("/api/shoot/facets", params={"folder": str(shoot)}).json()["labels"]
    assert labels == {str(ids["landscape.jpg"]): "blue"}


# ---- camera / lens facets ------------------------------------------------------------------


def test_read_lens_from_exif(shoot):
    assert read_lens(shoot / "burst0_0.jpg") == ("FE 24-70mm F2.8 GM", "123456")
    assert read_lens(shoot / "landscape.jpg") == ("RF50mm F1.2 L", "")  # an all-zero serial is none
    assert read_lens(shoot / "black.jpg") == ("", "")
    assert read_lens(shoot / "missing.jpg") == ("", "")


def test_facets_make_case_normalised_lens_orientation(client, shoot):
    d0 = client.get("/api/shoot/facets", params={"folder": str(shoot)}).json()
    assert d0["pending"] == 6  # lens not read yet
    assert client.post("/api/shoot/facets/scan", json={"folder": str(shoot)}).json() == {"ok": True}
    for _ in range(100):
        d = client.get("/api/shoot/facets", params={"folder": str(shoot)}).json()
        if not d["pending"] and not d["scanning"]:
            break
        time.sleep(0.1)
    assert d["pending"] == 0
    makes = {v["label"]: v for v in d["vals"]["make"]}
    assert set(makes) == {"Sony", "Canon"} and makes["Sony"]["n"] == 3 and makes["Sony"]["raw"] == ["SONY", "Sony"]
    assert {v["label"] for v in d["vals"]["model"]} == {"ILCE-7M4", "EOS R5"}
    assert [v["label"] for v in d["vals"]["lens"]] == ["FE 24-70mm F2.8 GM", "RF50mm F1.2 L"]
    assert [v["label"] for v in d["vals"]["lens_serial"]] == ["123456"]
    ids = ids_of(client, shoot)
    orient = {n: d["rows"][str(i)][4] for n, i in ids.items()}
    assert orient == {"burst0_0.jpg": 0, "burst0_1.jpg": 0, "burst0_2.jpg": 1, "landscape.jpg": 2,
                      "black.jpg": 2, "burst1_3.jpg": -1}  # near-square within 2 %, unknown w/h
    row = d["rows"][str(ids["burst0_2.jpg"])]  # make / model index; lens missing
    assert row[0] == 0 and row[2] == -1


def test_scan_never_touches_decisions_or_features(client, shoot):
    with Cache.for_folder(shoot) as c:
        before = (c.load_decisions(), c.overrides(), c._con.execute("SELECT path, json FROM features ORDER BY path").fetchall(),
                  c._con.execute("SELECT path, meta_json, extractor FROM images ORDER BY path").fetchall())
    scan_now(shoot)
    scan_now(shoot)  # idempotent
    with Cache.for_folder(shoot) as c:
        after = (c.load_decisions(), c.overrides(), c._con.execute("SELECT path, json FROM features ORDER BY path").fetchall(),
                 c._con.execute("SELECT path, meta_json, extractor FROM images ORDER BY path").fetchall())
        assert len(c.exif_extra_rows()) == 6
    assert before == after


# ---- people map ----------------------------------------------------------------------------


def test_people_map_needs_consent(client, shoot):
    ids = ids_of(client, shoot)
    f = str(shoot)
    with Cache.for_folder(shoot) as c:
        c.set_meta("people_consent", False)
    assert client.get("/api/people/map", params={"folder": f}).status_code == 403
    with Cache.for_folder(shoot) as c:
        c.set_meta("people_consent", True)
        pid = c._con.execute("INSERT INTO people (name, cover_face, n) VALUES ('Ana', 'landscape.jpg#0', 2)").lastrowid
        c._con.executemany("INSERT INTO person_faces VALUES (?,?,?,?)",
                           [(pid, "landscape.jpg", 0, .9), (pid, "landscape.jpg", 1, .8), (pid, "black.jpg", 0, .7)])
        c._con.commit()
    m = client.get("/api/people/map", params={"folder": f}).json()["people"]
    assert m == {str(pid): sorted([ids["landscape.jpg"], ids["black.jpg"]])} or {k: sorted(v) for k, v in m.items()} == {str(pid): sorted([ids["landscape.jpg"], ids["black.jpg"]])}


def test_xmp_write_without_labels_unchanged(client, shoot):
    """No user label: the decision label scheme is exactly as before (Green for keep...)."""
    client.post("/api/xmp/write", json={"folder": str(shoot)})
    assert read_sidecar(shoot / "landscape.jpg").label == "Green"
    assert read_sidecar(shoot / "burst0_1.jpg").label == "Red"
