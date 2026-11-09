from shotsieve.config import Settings, load_settings, save_settings
from shotsieve.io.formats import RAW_EXTS, is_raw, is_supported, scan_folder


def test_settings_roundtrip(tmp_path):
    s = Settings(burst_gap_s=3.5, label_keep="Blue", keep_per_group=2)
    p = tmp_path / "cfg" / "s.toml"
    save_settings(s, p)
    assert load_settings(p) == s


def test_settings_defaults_and_unknown_keys(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('dup_cos_dist = 0.2\nbogus = 1\nembed_batch = "8"\n')
    s = load_settings(p)
    assert s.dup_cos_dist == 0.2 and s.embed_batch == 8
    assert load_settings(tmp_path / "missing.toml") == Settings()
    assert Settings().t_keep == 0.60 and Settings().burst_gap_s == 2.0


def test_format_sets():
    assert ".cr3" in RAW_EXTS and is_raw("x/IMG.NEF") and is_supported("a.HEIC")
    assert not is_supported("notes.txt")


def test_scan_pairs_and_hidden(tmp_path):
    for n in ["a.CR2", "a.jpg", "b.jpg", "c.dng", "notes.txt", ".hidden.jpg"]:
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / ".shotsieve").mkdir()
    (tmp_path / ".shotsieve" / "t.jpg").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.png").write_bytes(b"x")
    entries = scan_folder(tmp_path)
    by_master = {e.master.name: e for e in entries}
    assert set(by_master) == {"a.CR2", "b.jpg", "c.dng", "d.png"}
    assert by_master["a.CR2"].preview_source.name == "a.jpg" and by_master["a.CR2"].raw_jpeg_pair
    assert not by_master["b.jpg"].raw_jpeg_pair
