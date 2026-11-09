import os

import pytest

from shotsieve.ingest import copier
from shotsieve.io.lrcat import read_catalog


def test_lrcat_synthetic(make_fixtures, tmp_path):
    cat = tmp_path / "c.lrcat"
    make_fixtures.build_lrcat(
        cat,
        [
            {
                "root": "/Volumes/Photos",
                "rel": "2024/wedding/",
                "base": "IMG_0001",
                "ext": "CR2",
                "pick": 1,
                "rating": 4,
                "label": "Green",
            },
            {
                "root": "/Volumes/Photos",
                "rel": "2024/wedding/",
                "base": "IMG_0002",
                "ext": "CR2",
                "pick": -1,
                "rating": None,
                "label": "",
            },
            {
                "root": "/Users/me/Pics",
                "rel": "",
                "base": "x",
                "ext": "jpg",
                "pick": 0,
                "rating": 0,
            },
        ],
    )
    entries = read_catalog(cat)
    assert [e.path for e in entries] == [
        "/Volumes/Photos/2024/wedding/IMG_0001.CR2",
        "/Volumes/Photos/2024/wedding/IMG_0002.CR2",
        "/Users/me/Pics/x.jpg",
    ]
    assert (entries[0].rating, entries[0].pick, entries[0].label) == (4, 1, "Green")
    assert (entries[1].rating, entries[1].pick, entries[1].label) == (None, -1, None)
    assert (entries[2].rating, entries[2].pick) == (0, 0)


def test_lrcat_skips_virtual_copies(make_fixtures, tmp_path):
    import sqlite3

    cat = tmp_path / "c.lrcat"
    make_fixtures.build_lrcat(cat, [{"root": "/r", "base": "a", "ext": "jpg", "pick": 0}])
    con = sqlite3.connect(cat)
    con.execute("INSERT INTO Adobe_images VALUES (99, 1, 1, 5, 'Red', 1)")
    con.commit()
    con.close()
    assert len(read_catalog(cat)) == 1


def test_lrcat_fixture_catalog(gen_dir, manifest):
    entries = read_catalog(gen_dir / "catalog.lrcat")
    assert len(entries) == len(manifest)
    assert all(os.path.exists(e.path) for e in entries)
    assert sum(e.pick == 1 for e in entries) == 3 and any(e.pick == -1 for e in entries)


def _card(tmp_path):
    src = tmp_path / "card"
    (src / "DCIM" / "100").mkdir(parents=True)
    (src / "DCIM" / "100" / "a.jpg").write_bytes(os.urandom(5000))
    (src / "DCIM" / "100" / "b.cr2").write_bytes(os.urandom(9000))
    (src / "DCIM" / "notes.txt").write_text("ignore")
    return src


def test_copy_backup_verify_and_skip(tmp_path):
    src = _card(tmp_path)
    dest, bk = tmp_path / "primary", tmp_path / "backup"
    r = copier.ingest(src, dest, backup=bk)
    assert len(r.copied) == 2 and not r.failed and not r.skipped and r.bytes_copied == 2 * 14000
    for root in (dest, bk):
        assert (root / "DCIM" / "100" / "a.jpg").read_bytes() == (
            src / "DCIM/100/a.jpg"
        ).read_bytes()
        assert not (root / "DCIM" / "notes.txt").exists()
    assert not list(dest.rglob("*.part"))
    r2 = copier.ingest(src, dest, backup=bk)
    assert len(r2.skipped) == 2 and not r2.copied and r2.bytes_copied == 0
    (bk / "DCIM/100/a.jpg").unlink()
    r3 = copier.ingest(src, dest, backup=bk)  # restores the missing backup copy
    assert len(r3.copied) == 1 and (bk / "DCIM/100/a.jpg").exists()


def test_corrupted_existing_dest_detected(tmp_path):
    src = _card(tmp_path)
    dest = tmp_path / "primary"
    copier.ingest(src, dest)
    f = dest / "DCIM/100/a.jpg"
    data = bytearray(f.read_bytes())
    data[100] ^= 0xFF
    f.write_bytes(bytes(data))
    r = copier.ingest(src, dest)
    assert len(r.failed) == 1 and "different content" in r.failed[0][1]
    assert f.read_bytes() == bytes(data)  # never overwritten


def test_verify_failure_detected_and_cleaned(tmp_path, monkeypatch):
    src = _card(tmp_path)
    real = copier._hash_file

    def bad(path, nocache=False):
        return "bad" if path.suffix == ".part" else real(path)

    monkeypatch.setattr(copier, "_hash_file", bad)
    r = copier.ingest(src, tmp_path / "primary")
    assert len(r.failed) == 2 and not r.copied
    assert not list((tmp_path / "primary").rglob("*")) or not any(
        p.is_file() for p in (tmp_path / "primary").rglob("*")
    )


def test_by_date_and_template(tmp_path, shoot_dir):
    import shutil

    src = tmp_path / "card"
    src.mkdir()
    shutil.copy(shoot_dir / "burst0_0.jpg", src / "IMG_1.jpg")
    r = copier.ingest(src, tmp_path / "d", by_date=True, rename_template="{date}_{index}_{stem}")
    assert len(r.copied) == 1
    out = list((tmp_path / "d").rglob("*.jpg"))
    assert out[0].parent.name == "2024-06-01" and out[0].name == "2024-06-01_1_IMG_1.jpg"
    assert (src / "IMG_1.jpg").exists()


def test_source_untouched(tmp_path):
    src = _card(tmp_path)
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    copier.ingest(src, tmp_path / "x", backup=tmp_path / "y")
    assert before == {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}


@pytest.mark.parametrize("n", [0])
def test_empty_source(tmp_path, n):
    (tmp_path / "e").mkdir()
    r = copier.ingest(tmp_path / "e", tmp_path / "o")
    assert not r.copied and not r.failed
