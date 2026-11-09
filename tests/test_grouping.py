import numpy as np
import pytest

from shotsieve.grouping.bursts import TimeItem, assign_bursts
from shotsieve.grouping.duplicates import group_duplicates


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_bursts_by_time_gap_and_camera():
    items = [TimeItem("a", t) for t in (0, 0.3, 0.6, 5.0, 5.2, 70.0)] + [TimeItem("b", 0.1)]
    ids = assign_bursts(items, 2.0)
    assert ids[:3] == [0, 0, 0] and ids[3] == ids[4] == 1 and ids[5] == 2
    assert ids[6] not in ids[:6]  # other camera never joins
    assert len(set(ids)) == 4


def test_bursts_unsorted_input_and_mtime_singletons():
    items = [TimeItem("a", 1.0), TimeItem("a", 0.0), TimeItem("a", 0.5)]
    assert len(set(assign_bursts(items))) == 1
    mt = [TimeItem("a", t, "mtime") for t in (0, 0.1, 0.2)]
    assert len(set(assign_bursts(mt))) == 3
    assert assign_bursts([]) == []


def test_bursts_on_fixture_shoot(shoot_dir, manifest):
    from shotsieve.io.exif import read_meta

    items = []
    for m in manifest:
        meta = read_meta(shoot_dir / m["file"])
        items.append(TimeItem(meta.camera_serial, meta.timestamp, meta.ts_source))
    ids = assign_bursts(items, 2.0)
    for b in range(3):
        members = {ids[i] for i, m in enumerate(manifest) if m["burst"] == b}
        assert len(members) == 1
    assert len(set(ids)) == 5  # 3 bursts + landscape + black


def _times(n, step=0.3):
    return [TimeItem("a", i * step) for i in range(n)]


def test_duplicates_groups_similar_frames():
    rng = np.random.default_rng(0)
    base1, base2 = rng.normal(size=384), rng.normal(size=384)
    emb = _unit([base1 + 0.02 * rng.normal(size=384) for _ in range(3)]
                + [base2 + 0.02 * rng.normal(size=384) for _ in range(3)])  # fmt: skip
    res = group_duplicates(emb, _times(6))
    assert res.group_ids[:3] == [0, 0, 0] and res.group_ids[3:] == [1, 1, 1]


def test_duplicates_singletons_get_own_ids_and_empty():
    rng = np.random.default_rng(1)
    emb = _unit(rng.normal(size=(5, 384)))
    assert group_duplicates(emb, _times(5)).group_ids == [0, 1, 2, 3, 4]
    assert group_duplicates(np.zeros((0, 384), np.float32), []).group_ids == []


def test_duplicates_average_linkage_does_not_chain():
    # a - b - c: neighbours close, ends far apart -> average linkage must not merge all three
    a = np.zeros(8)
    a[0] = 1
    c = np.zeros(8)
    c[0] = np.cos(0.8)
    c[1] = np.sin(0.8)
    b = (a + c) / 2  # noqa: E702
    emb = _unit([a, b, c])
    res = group_duplicates(emb, _times(3), cos_dist=0.05)
    assert len(set(res.group_ids)) >= 2


def test_duplicates_same_person_check_blocks_merge():
    emb = _unit(np.tile(np.ones(16), (2, 1)))  # identical scenes
    f1, f2 = _unit(np.eye(128)[0]), _unit(np.eye(128)[1])  # orthogonal faces: different people
    assert group_duplicates(emb, _times(2), [f1, f2]).group_ids == [0, 1]
    assert group_duplicates(emb, _times(2), [f1, f1]).group_ids == [0, 0]
    assert group_duplicates(emb, _times(2), [f1, None]).group_ids == [0, 0]  # no face -> no veto


def test_duplicates_respect_time_window_and_camera():
    emb = _unit(np.ones((3, 16)))
    items = [TimeItem("a", 0), TimeItem("a", 100), TimeItem("b", 1)]
    assert group_duplicates(emb, items, window_s=30).group_ids == [0, 1, 2]
    items = [TimeItem("a", 0), TimeItem("a", 20), TimeItem("a", 40)]
    # 0-40 s is outside the window, so average linkage cannot chain all three into one group
    ids = group_duplicates(emb, items, window_s=30).group_ids
    assert len(set(ids)) == 2 and ids[1] in (ids[0], ids[2])


def test_duplicates_work_is_windowed_not_quadratic():
    n = 3000
    rng = np.random.default_rng(2)
    emb = _unit(rng.normal(size=(n, 32)))
    items = [TimeItem("a", i * 1.0) for i in range(n)]  # 30 s window -> ~30 neighbours each
    res = group_duplicates(emb, items, window_s=30, max_neighbors=40)
    assert res.pairs_compared <= n * 40 and res.pairs_compared < n * n / 20
    assert len(res.group_ids) == n


def test_duplicates_mtime_uses_index_neighbours():
    emb = _unit(np.ones((12, 16)))
    items = [TimeItem("a", 1000.0 + i * 3600, "mtime") for i in range(12)]
    res = group_duplicates(emb, items, mtime_neighbors=3)
    assert 1 <= len(set(res.group_ids)) <= 12
    assert res.pairs_compared <= 12 * 3


@pytest.mark.parametrize("dist", [0.05, 0.3])
def test_duplicates_threshold(dist):
    a = np.zeros(8)
    a[0] = 1
    b = np.zeros(8)
    b[0] = np.cos(0.3)
    b[1] = np.sin(0.3)  # cos distance ~0.044
    res = group_duplicates(_unit([a, b]), _times(2), cos_dist=dist)
    assert (res.group_ids == [0, 0]) == (dist == 0.3 or 0.044 < dist)
