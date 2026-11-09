import xml.etree.ElementTree as ET

from shotsieve.io.xmp import NS, read_sidecar, sidecar_path, write_sidecar


def test_roundtrip(tmp_path):
    master = tmp_path / "IMG_1.CR2"
    master.write_bytes(b"raw")
    sp = write_sidecar(
        master,
        rating=3,
        label="Green",
        decision="keep",
        reasons=["sharp", "a & b <c>"],
        score=0.8123,
    )
    assert sp == tmp_path / "IMG_1.xmp" and sp == sidecar_path(master)
    ET.parse(sp)  # valid XML
    d = read_sidecar(master)
    assert (d.rating, d.label, d.decision) == (3, "Green", "keep")
    assert d.reasons == ["sharp", "a & b <c>"] and abs(d.score - 0.8123) < 1e-6
    assert master.read_bytes() == b"raw"


def test_update_only_given_fields(tmp_path):
    m = tmp_path / "a.jpg"
    write_sidecar(m, rating=1, label="Red", decision="reject", reasons=["blur"])
    write_sidecar(m, rating=-1)
    d = read_sidecar(m)
    assert d.rating == -1 and d.label == "Red" and d.reasons == ["blur"]
    write_sidecar(m, reasons=["x", "y"])
    assert read_sidecar(m).reasons == ["x", "y"]


FOREIGN = """<?xml version="1.0"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"
    crs:Exposure2012="+0.50" xmp:Rating="5">
   <xmp:Label>Blue</xmp:Label>
   <dc:subject><rdf:Bag><rdf:li>wedding</rdf:li></rdf:Bag></dc:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def test_merge_preserves_foreign(tmp_path):
    m = tmp_path / "b.nef"
    sp = tmp_path / "b.xmp"
    sp.write_text(FOREIGN)
    # the user overrode this image in ShotSieve, so the existing rating / label may be replaced
    write_sidecar(
        m, rating=2, label="Yellow", decision="review", reasons=["close call"], override=True
    )
    text = sp.read_text()
    root = ET.fromstring(text[text.index("<x:xmpmeta") : text.rindex("</x:xmpmeta>") + 12])
    assert "crs:Exposure2012" in text and "+0.50" in text and "wedding" in text
    assert "xmlns:crs" in text
    d = read_sidecar(m)
    assert (d.rating, d.label, d.decision) == (2, "Yellow", "review")
    assert d.reasons == ["close call"]
    # rating attribute stays a single value (no duplicates)
    assert text.count("xmp:Rating=") == 1 and text.count("<xmp:Label") == 1
    assert root.tag == "{" + NS["x"] + "}xmpmeta"


def test_read_missing_or_broken(tmp_path):
    assert read_sidecar(tmp_path / "none.jpg") is None
    (tmp_path / "bad.xmp").write_text("<not xml")
    assert read_sidecar(tmp_path / "bad.jpg") is None
    write_sidecar(tmp_path / "bad.jpg", rating=1)  # M4: skipped, never rewritten
    assert (tmp_path / "bad.xmp").read_text() == "<not xml"
    assert len(list(tmp_path.glob("bad.xmp.bak-*"))) == 1


# ---- Review #4 M4: never overwrite an unreadable sidecar ----------------------------------
import re as _re  # noqa: E402


def _bad_sidecar(tmp_path, body=b"<x:xmpmeta><crs:Exposure2012>+0.5</crs:Exposure2012"):
    from shotsieve.io.xmp import sidecar_path

    master = tmp_path / "IMG_1.CR2"
    master.write_bytes(b"raw")
    sp = sidecar_path(master)
    sp.write_bytes(body)  # truncated / foreign: not well-formed XML
    return master, sp


def _baks(sp):
    return sorted(sp.parent.glob(sp.name + ".bak-*"))


def test_unreadable_sidecar_is_skipped_one_backup_per_content_hash(tmp_path):
    from shotsieve.config import Settings
    from shotsieve.io import xmp

    master, sp = _bad_sidecar(tmp_path)
    old = sp.with_name(sp.name + ".bak-20200101T000000000000")  # legacy timestamp backup
    old.write_bytes(b"legacy")
    original = sp.read_bytes()
    out = xmp.WriteOutcome()

    def run():
        xmp.write_sidecar(master, rating=1)
        xmp.write_sidecar(master, rating=2)
        assert xmp.merge_keywords(master, ["#23"], ["Ana"]) == 0
        xmp.write_decision(master, Settings(), "keep", [], 0.9, outcome=out)

    def hashed():
        return [b for b in _baks(sp) if _re.fullmatch(r".*\.xmp\.bak-[0-9a-f]{12}", b.name)]

    run()
    assert sp.read_bytes() == original and out.skipped_unreadable
    assert len(hashed()) == 1 and hashed()[0].read_bytes() == original
    run()
    assert len(hashed()) == 1  # same bytes: no new backup
    sp.write_bytes(original + b"<more")  # different broken bytes
    run()
    assert len(hashed()) == 2
    assert old.read_bytes() == b"legacy" and len(_baks(sp)) == 3  # legacy one untouched


def test_unreadable_sidecar_warns(tmp_path, caplog):
    import logging

    from shotsieve.io import xmp

    master, _ = _bad_sidecar(tmp_path)
    with caplog.at_level(logging.WARNING):
        xmp.merge_keywords(master, ["#1"])
    assert any("unreadable" in r.getMessage().lower() for r in caplog.records)


def test_tmp_names_are_unique_per_writer(tmp_path, monkeypatch):
    from shotsieve.config import Settings
    from shotsieve.io import xmp

    seen = []
    real = xmp.os.replace
    monkeypatch.setattr(xmp.os, "replace", lambda a, b: (seen.append(str(a)), real(a, b))[1])
    master = tmp_path / "IMG_2.CR2"
    master.write_bytes(b"raw")
    xmp.write_decision(master, Settings(), "keep", [], 0.9)
    xmp.merge_keywords(master, ["#7"])
    xmp.merge_keywords(master, ["#8"])
    assert len(seen) == 3 and len(set(seen)) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_backup_by_hash_is_atomic_and_a_truncated_temp_is_never_the_backup(tmp_path, monkeypatch):
    import hashlib
    import os

    from shotsieve.io import xmp

    sp = tmp_path / "a.xmp"
    data = b"<broken" * 100
    sp.write_bytes(data)
    bak = sp.with_name(f"a.xmp.bak-{hashlib.sha256(data).hexdigest()[:12]}")

    def crash(src, dst):  # the process dies after a partial temp write, before the rename
        os.truncate(src, 5)
        raise OSError("crash")

    monkeypatch.setattr(xmp.os, "replace", crash)
    assert xmp.backup_by_hash(sp) is None
    assert not bak.exists()  # no truncated file under the hash name
    assert list(tmp_path.glob("*.tmp")) == []
    monkeypatch.undo()
    assert xmp.backup_by_hash(sp) == bak and bak.read_bytes() == data

    # a crash that leaves a stale temp behind does not block the next backup
    bak.unlink()
    sp.with_name(bak.name + ".tmp").write_bytes(b"x")
    assert xmp.backup_by_hash(sp) == bak and bak.read_bytes() == data
