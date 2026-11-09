"""Near-duplicate grouping: windowed cosine comparison + average-linkage inside components.

Only pairs of consecutive frames (per camera) within `window_s` seconds - and at most
`max_neighbors` positions apart - are ever compared, so the work is O(n * w), never O(n^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from shotsieve.grouping.bursts import TimeItem, time_order

FORBIDDEN = 2.0  # distance assigned to pairs that must never merge (different people)
MAX_COMPONENT = 1000  # larger components use leader clustering (guards O(m^2) memory)
MAX_GROUP = 400  # hard cap on the size of a leader-clustered group


@dataclass
class DupResult:
    group_ids: list[int]  # aligned with the input; 0..k-1, singletons included
    pairs_compared: int


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _same_person(a: np.ndarray | None, b: np.ndarray | None, min_cos: float) -> bool:
    if a is None or b is None:
        return True
    return float(a @ b) >= min_cos


def group_duplicates(
    embeddings: np.ndarray,
    items: list[TimeItem],
    face_embs: list[np.ndarray | None] | None = None,
    *,
    window_s: float = 30.0,
    cos_dist: float = 0.12,
    same_person_cos: float = 0.36,
    max_neighbors: int = 40,
    mtime_neighbors: int = 6,
) -> DupResult:
    """Group near-duplicate images.

    embeddings: (N, D) L2-normalised scene embeddings. face_embs: per image, the L2-normalised
    SFace embedding of the main face or None. Two images with faces merge only if their main
    faces are the same person (cosine >= same_person_cos).
    """
    n = len(items)
    face_embs = face_embs or [None] * n
    order = time_order(items)
    dist: dict[tuple[int, int], float] = {}
    uf = _UnionFind(n)
    compared = 0
    for a in range(n):
        i = order[a]
        ia = items[i]
        for b in range(a + 1, min(n, a + 1 + max_neighbors)):
            j = order[b]
            jb = items[j]
            if jb.camera != ia.camera:
                break
            if ia.ts_source == "mtime" or jb.ts_source == "mtime":
                if ia.ts_source != jb.ts_source or b - a > mtime_neighbors:
                    continue
            elif jb.timestamp - ia.timestamp > window_s:
                break
            compared += 1
            d = 1.0 - float(embeddings[i] @ embeddings[j])
            if not _same_person(face_embs[i], face_embs[j], same_person_cos):
                d = FORBIDDEN
            key = (i, j) if i < j else (j, i)
            dist[key] = d
            if d <= cos_dist:
                uf.union(i, j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(uf.find(i), []).append(i)

    comp_pairs: dict[int, list[tuple[int, int, float]]] = {}
    for (i, j), d in dist.items():
        root = uf.find(i)
        if root == uf.find(j):
            comp_pairs.setdefault(root, []).append((i, j, d))

    labels = [-1] * n
    next_label = 0
    for root in sorted(comps):
        members = comps[root]
        if len(members) == 1:
            labels[members[0]] = next_label
            next_label += 1
            continue
        if len(members) > MAX_COMPONENT:
            # A very long continuous sequence chains into one component. Never emit it as one
            # group: walk it in time order and start a new group whenever a frame is no longer
            # close to the group's first frame (so first and last can never be far apart) or
            # the group is full.
            rank_of = {idx: k for k, idx in enumerate(order)}
            anchor, size = -1, 0
            for idx in sorted(members, key=rank_of.__getitem__):
                if (
                    anchor < 0
                    or size >= MAX_GROUP
                    or 1.0 - float(embeddings[anchor] @ embeddings[idx]) > cos_dist
                    or not _same_person(face_embs[anchor], face_embs[idx], same_person_cos)
                ):
                    anchor, size = idx, 0
                    next_label += 1
                labels[idx] = next_label - 1
                size += 1
            continue
        m = len(members)
        pos = {idx: k for k, idx in enumerate(members)}
        mat = np.ones((m, m), dtype=np.float64)
        np.fill_diagonal(mat, 0.0)
        for i, j, d in comp_pairs.get(root, []):
            # identical embeddings give a cosine distance of about -1e-7; linkage needs >= 0
            mat[pos[i], pos[j]] = mat[pos[j], pos[i]] = max(d, 0.0)
        cl = fcluster(linkage(squareform(mat, checks=False), "average"), cos_dist, "distance")
        remap: dict[int, int] = {}
        for k, idx in enumerate(members):
            if cl[k] not in remap:
                remap[cl[k]] = next_label
                next_label += 1
            labels[idx] = remap[cl[k]]

    # renumber by first appearance (original order) for stable, readable ids
    seen: dict[int, int] = {}
    out = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = len(seen)
        out.append(seen[lab])
    return DupResult(out, compared)
