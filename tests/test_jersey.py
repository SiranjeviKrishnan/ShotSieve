"""Jersey roster, burst vote, face association, XMP merge, table + search, API, CLI, indexer."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from shotsieve import cli
from shotsieve.config import Settings
from shotsieve.features import jersey as jf
from shotsieve.io import xmp
from shotsieve.io.roster import RosterError, normalize_number, parse_roster
from shotsieve.models.ocr import ocr_models_ready
from shotsieve.pipeline import jerseys
from shotsieve.pipeline import search as search_mod
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.indexer import Indexer
from shotsieve.server.app import create_app
from tests.jersey_synth import sports_frame
from tests.test_api import seeded  # noqa: F401  (fixture)

needs_models = pytest.mark.skipif(not ocr_models_ready(), reason="needs `--extra ocr` models")


# ---- roster ---------------------------------------------------------------------------------
def test_roster_bom_whitespace_header_and_delims():
    raw = "﻿number, name , team\r\n 23 ,  Jane Doe , Reds\r\n#7;Bob Ray\r\n\r\n07,Dup Seven\r\n"
    got = parse_roster(raw.encode("utf-8"))
    assert [(e.number, e.name, e.team) for e in got] == [
        ("23", "Jane Doe", "Reds"), ("7", "Dup Seven", ""),  # later row wins; header skipped
    ]  # fmt: skip
    semi = parse_roster("10;Ann Lee;Blues\n00;Zero Guy")
    assert [e.number for e in semi] == ["10", "00"]
    tab = parse_roster("5\tTab Man")
    assert tab[0].name == "Tab Man"
    assert normalize_number(" #007 ") == "7" and normalize_number("x") == ""
    for bad in ("", "\n\n", "number,name\n"):
        with pytest.raises(RosterError):
            parse_roster(bad)


def test_roster_from_path(tmp_path):
    p = tmp_path / "r.csv"
    p.write_bytes(b"\xef\xbb\xbf9,Nine Nine\n")
    assert parse_roster(p)[0].name == "Nine Nine"


# ---- burst vote -------------------------------------------------------------------------------
def test_burst_propagation():
    reads = {  # a burst of 4 frames: 23 seen twice (50 %), 9 once (25 %, < 2 frames); 2 singles
        "a": {"23": 0.99}, "b": {"23": 0.97, "9": 0.9}, "c": {}, "d": {},
        "s1": {"5": 0.99}, "s2": {},
    }  # fmt: skip
    bursts = {"a": 0, "b": 0, "c": 0, "d": 0, "s1": 1, "s2": 2}
    out = jf.burst_vote(reads, bursts)
    assert {(r["path"], r["number"]) for r in out} == {("c", "23"), ("d", "23")}
    assert out[0]["conf"] == pytest.approx(0.98)
    # 1 in 4 frames below 30 % of a 10-frame burst: not propagated
    big = {f"f{i}": ({"8": 0.99} if i < 2 else {}) for i in range(10)}
    assert jf.burst_vote(big, dict.fromkeys(big, 0)) == []
    two = {f"f{i}": ({"8": 0.99} if i < 3 else {}) for i in range(10)}
    assert len(jf.burst_vote(two, dict.fromkeys(two, 0))) == 7  # 30 % of 10 -> propagate


def test_associate_face_picks_the_face_above():
    box = (500, 400, 100, 100)
    faces = [(100, 100, 60, 70), (520, 300, 60, 70), (900, 300, 60, 70)]
    assert jf.associate_face(box, faces) == 1
    assert jf.associate_face((500, 900, 80, 80), [(100, 100, 60, 70)]) is None


def test_dominant_number_per_person():
    rows = [{"person_id": 1, "number": "9", "source": "ocr"}] * 3 + [
        {"person_id": 1, "number": "8", "source": "ocr"},
        {"person_id": 2, "number": "5", "source": "ocr"},  # a single read is not enough
    ]
    assert jf.dominant_numbers(rows) == {1: "9"}


def test_roster_annotation_and_tags():
    rows = [{"path": "/p1", "number": "23", "source": "ocr"},
            {"path": "/p1", "number": "99", "source": "ocr"},
            {"path": "/p2", "number": "7", "source": "burst"}]  # fmt: skip
    roster = {"23": ("Jane Doe", "Reds"), "7": ("Bob Ray", "")}
    ann = {r["number"]: r for r in jf.annotate(rows, roster)}
    assert ann["23"]["status"] == "matched" and ann["99"]["status"] == "unmatched"
    tags = jf.xmp_tags(rows, roster)
    assert tags["/p1"] == {"keywords": ["#23", "Jane Doe"], "persons": ["Jane Doe"]}
    assert "99" not in str(tags)  # unmatched numbers are never written when a roster is loaded
    no_roster = jf.xmp_tags(rows, {})
    assert no_roster["/p1"]["keywords"] == ["#23", "#99"] and no_roster["/p1"]["persons"] == []


# ---- XMP keyword merge ------------------------------------------------------------------------
FOREIGN = """<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
 xmp:Rating="4" crs:Exposure2012="+0.50">
<dc:subject><rdf:Bag><rdf:li>wedding</rdf:li><rdf:li>#23</rdf:li></rdf:Bag></dc:subject>
<dc:creator><rdf:Seq><rdf:li>Me</rdf:li></rdf:Seq></dc:creator>
</rdf:Description></rdf:RDF></x:xmpmeta>
<?xpacket end="w"?>
"""


def test_xmp_keyword_merge_is_foreign_safe_and_idempotent(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    sp = xmp.sidecar_path(img)
    sp.write_text(FOREIGN)
    assert xmp.merge_keywords(img, ["#23", "Jane Doe"], ["Jane Doe"]) == 2  # '#23' already there
    text = sp.read_text()
    kws, people = xmp.read_keywords(img)
    assert kws == ["wedding", "#23", "Jane Doe"] and people == ["Jane Doe"]
    assert 'xmp:Rating="4"' in text and "Exposure2012" in text and "<rdf:li>Me</rdf:li>" in text
    before = sp.read_bytes()
    assert xmp.merge_keywords(img, ["#23", "jane doe"], ["Jane Doe"]) == 0
    assert sp.read_bytes() == before  # idempotent: not even rewritten
    # the decision writer keeps the keywords too
    xmp.write_sidecar(img, rating=3, decision="keep")
    assert xmp.read_keywords(img)[0] == ["wedding", "#23", "Jane Doe"]
    # a fresh file
    other = tmp_path / "b.jpg"
    assert xmp.merge_keywords(other, ["#5"], []) == 1
    assert xmp.read_keywords(other) == (["#5"], [])


# ---- table + search ---------------------------------------------------------------------------
@pytest.fixture()
def tagged(seeded):  # noqa: F811
    folder = seeded
    with Cache.for_folder(folder) as c:
        c.put_jerseys(str(folder / "burst0_0.jpg"), "fp", [
            {"number": "23", "conf": 0.99, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1}])  # fmt: skip
        c.put_jerseys(str(folder / "burst0_1.jpg"), "fp", [])  # OCR'd, nothing found
        c.put_jerseys(str(folder / "landscape.jpg"), "fp", [
            {"number": "7", "conf": 0.95, "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}])  # fmt: skip
        stats = jerseys.finalize(c)
    assert stats["direct"] == 2
    return folder


def test_jersey_filter_search_and_burst_rows(tagged):
    def names():
        return sorted(Path(h.path).name for h in search_mod.search(tagged, "jersey:23"))

    # burst0_0 + burst0_1 are 'OCR'd', burst0_2 is not; 1 of 2 frames (50 %) but < 2 frames
    assert names() == ["burst0_0.jpg"]
    assert [Path(h.path).name for h in search_mod.search(tagged, "jersey:#7")] == ["landscape.jpg"]
    assert search_mod.search(tagged, "jersey:99") == []
    with Cache.for_folder(tagged) as c:
        c.put_jerseys(str(tagged / "burst0_1.jpg"), "fp2", [
            {"number": "23", "conf": 0.9, "x": 0, "y": 0, "w": 0.1, "h": 0.1}])  # fmt: skip
        c.put_jerseys(str(tagged / "burst0_2.jpg"), "fp", [])
        jerseys.finalize(c)
        rows = c.jerseys()
    burst = {Path(r["path"]).name for r in rows if r["number"] == "23" and r["source"] == "burst"}
    assert burst == {"burst0_2.jpg"}  # propagated into the third frame of the burst
    assert names() == ["burst0_0.jpg", "burst0_1.jpg", "burst0_2.jpg"]


def test_person_search_resolves_roster_names_without_consent(tagged):
    with Cache.for_folder(tagged) as c:
        c.set_roster([("23", "Jane Doe", "Reds")])
    hits = search_mod.search(tagged, 'person:"jane doe"', settings=Settings())
    assert [Path(h.path).name for h in hits] == ["burst0_0.jpg"]
    with pytest.raises(search_mod.ConsentError):
        search_mod.search(tagged, 'person:"Nobody"', settings=Settings())


def _names(hits):
    return sorted(Path(h.path).name for h in hits)


def _two_teams(folder, team=""):
    with Cache.for_folder(folder) as c:
        c.set_roster([("23", "Home Jane", "Home"), ("23", "Away Bob", "Away")])
        c.set_meta("roster_team", team)


def test_person_search_honours_folder_team(tagged):
    _two_teams(tagged, "Home")
    w: list[str] = []
    jane = search_mod.search(tagged, 'person:"Home Jane"', settings=Settings(), warnings=w)
    assert _names(jane) == ["burst0_0.jpg"]
    assert search_mod.search(tagged, 'person:"Away Bob"', settings=Settings(), warnings=w) == []
    assert w == []
    _two_teams(tagged, "home")  # case-insensitive team match
    assert _names(search_mod.search(tagged, 'person:"home jane"', settings=Settings())) == [
        "burst0_0.jpg"
    ]
    assert search_mod.search(tagged, 'person:"Away Bob"', settings=Settings()) == []


def test_person_search_ambiguous_number_without_team_warns(tagged):
    _two_teams(tagged, "")
    for who in ("Home Jane", "Away Bob"):
        w: list[str] = []
        assert search_mod.search(tagged, f'person:"{who}"', settings=Settings(), warnings=w) == []
        assert w == ["#23 is on 2 teams; set the folder team"]


def test_person_search_ambiguous_teamless_entries_warn_about_names(tagged):
    with Cache.for_folder(tagged) as c:
        # the table is keyed (team, number): the reachable case is one teamless entry + one team
        c.set_roster([("23", "Ann", ""), ("23", "Bob", "Away")])
        c.set_meta("roster_team", "")
    w: list[str] = []
    assert search_mod.search(tagged, 'person:"Ann"', settings=Settings(), warnings=w) == []
    assert w == ["#23 has 2 names on the roster; give them teams"]


def test_person_search_single_team_without_folder_team_still_matches(tagged):
    with Cache.for_folder(tagged) as c:
        c.set_roster([("23", "Jane Doe", "Reds")])
    w: list[str] = []
    hits = search_mod.search(tagged, 'person:"Jane Doe"', settings=Settings(), warnings=w)
    assert _names(hits) == ["burst0_0.jpg"] and w == []


def test_person_search_roster_name_never_touches_faces_without_consent(tagged, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("face data touched without consent")

    monkeypatch.setattr(search_mod, "image_ids_for_person", boom)
    _two_teams(tagged, "Home")
    assert _names(search_mod.search(tagged, 'person:"Home Jane"', settings=Settings())) == [
        "burst0_0.jpg"
    ]


def test_person_search_ambiguity_warning_in_api_and_cli(tagged, capsys):
    _two_teams(tagged, "")
    with TestClient(create_app(Settings(), default_folder=tagged)) as c:
        r = c.get("/api/search", params={"folder": str(tagged), "q": 'person:"Home Jane"'}).json()
    assert r["hits"] == [] and r["warnings"] == ["#23 is on 2 teams; set the folder team"]
    assert cli.main(["search", str(tagged), 'person:"Home Jane"']) == 0
    assert "#23 is on 2 teams; set the folder team" in capsys.readouterr().err


# ---- API / CLI --------------------------------------------------------------------------------
def test_roster_api_and_xmp(tagged):
    with TestClient(create_app(Settings(), default_folder=tagged)) as c:
        f = str(tagged)
        r = c.post("/api/roster", json={"folder": f, "csv": "﻿23, Jane Doe ,Reds\n41,Gone Guy"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["entries"] == 2 and d["images_with_numbers"] == 2 and d["images_matched"] == 1
        assert d["unmatched_numbers"] == ["7"]
        assert c.post("/api/roster", json={"folder": f, "csv": "nothing"}).status_code == 400
        j = c.get("/api/jerseys", params={"folder": f}).json()["jerseys"]
        shoot = c.get("/api/shoot", params={"folder": f}).json()
        by = {im["name"]: im["id"] for im in shoot["images"]}
        assert j[str(by["burst0_0.jpg"])][0]["name"] == "Jane Doe"
        assert j[str(by["landscape.jpg"])][0]["status"] == "unmatched"
        w = c.post("/api/jerseys/xmp", json={"folder": f}).json()
        assert w["images"] == 1 and w["written"] == 1 and w["failed"] == 0
        assert xmp.read_keywords(tagged / "burst0_0.jpg") == (["#23", "Jane Doe"], ["Jane Doe"])
        assert not xmp.sidecar_path(tagged / "landscape.jpg").exists()  # unmatched: not written
        hits = c.get("/api/search", params={"folder": f, "q": 'person:"Jane Doe"'}).json()["hits"]
        assert [h["id"] for h in hits] == [by["burst0_0.jpg"]]
        assert c.get("/api/roster", params={"folder": f}).json()["entries"] == 2
        assert c.delete("/api/roster", params={"folder": f}).json()["entries"] == 0
        h = c.get("/api/index/status", params={"folder": f}).json()
        assert "ocr_models_ready" in h


def test_roster_cli(tagged, tmp_path, capsys):
    csv = tmp_path / "roster.csv"
    csv.write_text("number,name\n7,Bob Ray\n")
    assert cli.main(["roster", str(tagged), str(csv), "--write-xmp", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["entries"] == 1 and out["images_matched"] == 1 and out["xmp"]["written"] == 1
    assert xmp.read_keywords(tagged / "landscape.jpg") == (["#7", "Bob Ray"], ["Bob Ray"])
    assert cli.main(["roster", str(tagged), str(tmp_path / "missing.csv")]) == 1


# ---- indexer end to end (real OCR models, synthetic frames) -----------------------------------
@needs_models
def test_indexer_ocr_task_end_to_end(tmp_path, shoot_dir):
    folder = tmp_path / "game"
    folder.mkdir()
    for i, (num, ang) in enumerate([("23", 0), ("23", 10), ("23", -8)]):
        img, _ = sports_frame([(num, (500, 380))], (720, 1080), 0.08, seed=i, angle=ang)
        cv2.imwrite(str(folder / f"g{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    shutil.copy(shoot_dir / "landscape.jpg", folder / "scenery.jpg")
    # the OCR task is opt-in: nothing happens when it is off
    Indexer(folder, Settings(), images=False, ocr=False).run_blocking()
    with Cache.for_folder(folder) as c:
        assert c.jerseys() == []
    ix = Indexer(folder, Settings(), images=False, ocr=True)
    ix.run_blocking()
    assert ix.error is None and (ix.done, ix.total) == (4, 4)
    with Cache.for_folder(folder) as c:
        rows = c.jerseys()
        assert sorted(Path(r["path"]).name for r in rows if r["number"] == "23") == [
            "g0.jpg", "g1.jpg", "g2.jpg",
        ]  # fmt: skip
        assert all(0 <= r["x"] <= 1 and r["h"] > 0.03 for r in rows)
    again = Indexer(folder, Settings(), images=False, ocr=True)
    again.run_blocking()
    assert again.ocr_n == 0  # resumable: nothing left to read
    assert ix.status()["ocr"] is True


# ---- UI ---------------------------------------------------------------------------------------
def test_ui_javascript_parses_and_ranks_search():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = Path(__file__).parents[1] / "src/shotsieve/server/static/app.js"
    assert subprocess.run([node, "--check", str(js)], capture_output=True).returncode == 0
    src = js.read_text()
    assert "S.search.get(b) - S.search.get(a)" in src  # best score first while a search is active
    assert "new Map(r.hits.map" in src


# ---- Review #4 M6: two-team rosters are keyed by (team, number) ---------------------------------
TWO_TEAMS = "number,name,team\n23,Ann Home,Home\n23,Bob Away,Away\n7,Cy Home,Home\n"


def test_parse_roster_keeps_same_number_on_two_teams():
    got = parse_roster(TWO_TEAMS)
    assert [(e.number, e.name, e.team) for e in got] == [
        ("23", "Ann Home", "Home"), ("23", "Bob Away", "Away"), ("7", "Cy Home", "Home")]


def test_cache_roster_is_keyed_by_team_and_number(tmp_path):
    with Cache.for_folder(tmp_path / "s") as c:
        c.set_roster([("23", "Ann Home", "Home"), ("23", "Bob Away", "Away")])
        assert sorted(c.roster()["23"]) == [("Ann Home", "Home"), ("Bob Away", "Away")]


def test_old_number_primary_key_roster_is_migrated(tmp_path):
    import sqlite3

    f = tmp_path / "s"
    (f / ".shotsieve").mkdir(parents=True)
    con = sqlite3.connect(f / ".shotsieve" / "cull.db")
    con.execute("CREATE TABLE roster (number TEXT PRIMARY KEY, name TEXT NOT NULL, team TEXT)")
    con.execute("INSERT INTO roster VALUES ('9','Old Timer','Reds')")
    con.commit()
    con.close()
    with Cache.for_folder(f) as c:
        assert c.roster() == {"9": [("Old Timer", "Reds")]}
        c.set_roster([("9", "A", "Reds"), ("9", "B", "Blues")])
        assert len(c.roster()["9"]) == 2


def test_ambiguous_number_writes_no_name_and_is_flagged():
    rows = [{"path": "/p1", "number": "23", "source": "ocr"},
            {"path": "/p2", "number": "7", "source": "ocr"}]  # fmt: skip
    roster = {"23": [("Ann Home", "Home"), ("Bob Away", "Away")], "7": [("Cy Home", "Home")]}
    ann = {r["number"]: r for r in jf.annotate(rows, roster)}
    assert ann["23"]["status"] == "ambiguous" and ann["23"]["name"] == ""
    assert ann["7"]["status"] == "matched" and ann["7"]["name"] == "Cy Home"
    tags = jf.xmp_tags(rows, roster)
    assert tags["/p1"] == {"keywords": ["#23"], "persons": []}  # number only, no guessed name
    assert tags["/p2"]["persons"] == ["Cy Home"]


def test_folder_team_resolves_a_shared_number():
    rows = [{"path": "/p1", "number": "23", "source": "ocr"}]
    roster = {"23": [("Ann Home", "Home"), ("Bob Away", "Away")]}
    away = jf.xmp_tags(rows, roster, team="away")  # case-insensitive
    assert away["/p1"] == {"keywords": ["#23", "Bob Away"], "persons": ["Bob Away"]}
    other = jf.annotate(rows, {"5": [("X", "Home")]}, team="Away")
    assert other[0]["status"] == "unmatched"
    # an entry without a team applies to every folder team
    shared = jf.annotate(rows, {"23": [("Solo", "")]}, team="Away")
    assert shared[0]["status"] == "matched" and shared[0]["name"] == "Solo"


def test_same_name_twice_is_not_ambiguous():
    rows = [{"path": "/p1", "number": "23", "source": "ocr"}]
    got = jf.annotate(rows, {"23": [("Ann", "Home"), ("Ann", "Away")]})
    assert got[0]["status"] == "matched"


def test_roster_summary_flags_review_and_team_setting_clears_it(tagged):
    with Cache.for_folder(tagged) as c:
        jerseys.set_roster(c, parse_roster(TWO_TEAMS))
        s = jerseys.roster_summary(c)
        assert s["ambiguous_numbers"] == ["23"] and s["images_ambiguous"] == 1
        assert s["review_images"] == ["burst0_0.jpg"] and s["images_matched"] == 1
        assert jerseys.set_team(c, "Away")["images_ambiguous"] == 0
        s = jerseys.roster_summary(c)
        assert s["team"] == "Away" and s["review_images"] == []


def test_api_two_teams_never_writes_a_guessed_name(tagged):
    with TestClient(create_app(Settings(), default_folder=tagged)) as c:
        f = str(tagged)
        d = c.post("/api/roster", json={"folder": f, "csv": TWO_TEAMS}).json()
        assert d["ambiguous_numbers"] == ["23"] and d["review_images"] == ["burst0_0.jpg"]
        c.post("/api/jerseys/xmp", json={"folder": f})
        assert xmp.read_keywords(tagged / "burst0_0.jpg") == (["#23"], [])
        d = c.post("/api/roster", json={"folder": f, "csv": TWO_TEAMS, "team": "Away"}).json()
        assert d["images_ambiguous"] == 0 and d["team"] == "Away"
        c.post("/api/jerseys/xmp", json={"folder": f})
        assert xmp.read_keywords(tagged / "burst0_0.jpg") == (["#23", "Bob Away"], ["Bob Away"])


def test_cli_team_option(tagged, tmp_path, capsys):
    csv = tmp_path / "r.csv"
    csv.write_text(TWO_TEAMS)
    assert cli.main(["roster", str(tagged), str(csv), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["images_ambiguous"] == 1
    assert cli.main(["roster", str(tagged), "--team", "Home", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["images_ambiguous"] == 0 and out["team"] == "Home"


def test_finalize_does_not_name_a_person_from_an_ambiguous_number(tagged):
    with Cache.for_folder(tagged) as c:
        c.set_roster([("23", "Ann Home", "Home"), ("23", "Bob Away", "Away")])
        out = jerseys.finalize(c, True)
        assert out["people_named"] == 0
