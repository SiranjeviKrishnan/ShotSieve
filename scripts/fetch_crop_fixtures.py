# ruff: noqa: E501
"""Fetch CC0 / public-domain photos for the T22 crop fixtures from Wikimedia Commons.

  uv run python scripts/fetch_crop_fixtures.py candidates   # search + download to a scratch dir
  uv run python scripts/fetch_crop_fixtures.py build        # picks.json -> tests/fixtures/crop/real

Downloads are untrusted data: each candidate lands in its own scratch directory, is only decoded as
an image (never executed), is re-encoded to a JPEG (long edge <= 1280) and verified against the
Commons licence metadata (CC0 or public domain). `build` writes tests/fixtures/crop/LICENSES.md
with source URL, licence, author and the sha256 of the saved file."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
UA = "shotsieve-fixtures/0.1 (local test fixtures; CC0 only)"
OUT = ROOT / "tests" / "fixtures" / "crop" / "real"
PICKS = ROOT / "scripts" / "crop_fixture_picks.json"
EXCLUDE = re.compile(
    r"\b(met|rijksmuseum|painting|oil|drawing|engraving|print|dp\d+|ep\d+|slp\d+|"
    r"miniature|self-portrait|portrait of|cdv|tintype|daguerreotype)\b",
    re.I,
)
QUERIES = {
    "portrait": ["portrait photo man", "woman portrait photograph", "headshot", "smiling girl", "boy portrait", "portrait photo outdoors"],
    "group": ["group photo people", "friends together", "family photo", "team photo", "students group", "wedding guests"],
    "animal": ["dog photo", "cat photo", "horse photo", "bird photo", "cow photo", "elephant photo"],
    "object": ["statue", "lighthouse", "church tower", "fountain", "windmill", "monument"],
    "horizon": ["sunset sea horizon", "beach landscape", "lake landscape", "ocean waves horizon", "prairie landscape", "coast"],
    "action": ["football player", "soccer match", "basketball game", "runner race", "cyclist", "skier"],
}  # fmt: skip


def get(url: str, params: dict | None = None) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def search(term: str, limit: int = 14) -> list[dict]:
    d = json.loads(get(API, {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{term} incategory:CC-Zero filetype:bitmap", "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1280,
    }))  # fmt: skip
    out = []
    for p in (d.get("query", {}).get("pages") or {}).values():
        ii = p["imageinfo"][0]
        em = ii.get("extmetadata", {})
        lic = em.get("LicenseShortName", {}).get("value", "")
        if ii.get("mime") != "image/jpeg" or EXCLUDE.search(p["title"]):
            continue
        if lic not in ("CC0", "Public domain") or min(ii["width"], ii["height"]) < 600:
            continue
        artist = re.sub(r"<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip()
        out.append({
            "title": p["title"], "license": lic, "artist": artist[:80],
            "page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(p["title"].replace(" ", "_")),
            "url": ii.get("thumburl") or ii["url"], "w": ii["width"], "h": ii["height"],
        })  # fmt: skip
    return out


def fetch_jpeg(url: str) -> bytes:
    """Download and re-encode (an untrusted file is only ever decoded as an image)."""
    import cv2
    import numpy as np

    raw = get(url)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image")
    h, w = img.shape[:2]
    s = min(1.0, 1280 / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else b""


def cmd_candidates() -> int:
    import cv2
    import numpy as np

    scratch = ROOT / "data" / "tmp" / "crop_candidates"
    scratch.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    for cat, terms in QUERIES.items():
        seen: set[str] = set()
        n = 0
        for t in terms:
            for c in search(t):
                if c["title"] in seen:
                    continue
                seen.add(c["title"])
                key = f"{cat}_{n:02d}"
                d = scratch / key
                d.mkdir(exist_ok=True)
                try:
                    (d / "img.jpg").write_bytes(fetch_jpeg(c["url"]))
                except Exception as exc:  # noqa: BLE001
                    print("skip", c["title"], exc)
                    continue
                meta[key] = c
                n += 1
    (scratch / "meta.json").write_text(json.dumps(meta, indent=1))
    keys = sorted(meta)
    for i in range(0, len(keys), 20):
        tiles = []
        for k in keys[i : i + 20]:
            im = cv2.imread(str(scratch / k / "img.jpg"))
            s = 240 / max(im.shape[:2])
            im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)))
            tile = np.zeros((260, 240, 3), np.uint8)
            tile[: im.shape[0], : im.shape[1]] = im
            cv2.putText(tile, k, (2, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % 5:
            tiles.append(np.zeros((260, 240, 3), np.uint8))
        rows = [np.hstack(tiles[j : j + 5]) for j in range(0, len(tiles), 5)]
        cv2.imwrite(str(scratch / f"sheet_{i // 20:02d}.jpg"), np.vstack(rows))
    print(f"{len(meta)} candidates in {scratch}")
    return 0


def cmd_build() -> int:
    picks = json.loads(PICKS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "data" / "tmp" / "crop_candidates"
    meta = json.loads((scratch / "meta.json").read_text())
    lines = [
        "# Crop fixtures: sources and licences (T22)",
        "",
        "Photos from Wikimedia Commons, licence verified through the Commons API (`extmetadata`,",
        "CC0 or public domain). Re-encoded to JPEG q62, long edge <= 1280 px. `sha256` is of the file",
        "in this folder. Downloads are untrusted data (decoded as images only, never executed).",
        "",
        "| file | category | source | licence | author | sha256 |",
        "|---|---|---|---|---|---|",
    ]
    manifest = {}
    for name, key in sorted(picks.items()):
        c = meta[key]
        import cv2
        import numpy as np

        im = cv2.imdecode(np.frombuffer((scratch / key / "img.jpg").read_bytes(), np.uint8), 1)
        data = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 62])[
            1
        ].tobytes()  # <= 8 MB total
        (OUT / name).write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        cat = key.split("_")[0]
        manifest[name] = {"category": cat, "sha256": sha}
        lines.append(
            f"| {name} | {cat} | {c['page']} | {c['license']} | {c['artist'] or '-'} | `{sha}` |"
        )
    (ROOT / "tests" / "fixtures" / "crop" / "LICENSES.md").write_text("\n".join(lines) + "\n")
    (ROOT / "tests" / "fixtures" / "crop" / "manifest.json").write_text(
        json.dumps(manifest, indent=1)
    )
    print(f"built {len(picks)} fixtures in {OUT}")
    return 0


def cmd_finish() -> int:
    """Assign the train / held-out split and write the EXIF-orientation copies (after `build`)."""
    import cv2
    import piexif

    R = ROOT / "tests" / "fixtures" / "crop"
    real = R / "real"
    man = json.loads((R / "manifest.json").read_text())
    cats: dict[str, list[str]] = {}
    for n in sorted(man):
        cats.setdefault(man[n]["category"], []).append(n)
    for names in cats.values():
        for i, n in enumerate(names):
            man[n]["split"] = "held-out" if (i % 5 in (1, 3) or i == 5) else "train"
    inv = {3: lambda a: cv2.rotate(a, cv2.ROTATE_180),
           6: lambda a: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE),
           8: lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)}  # fmt: skip
    for base, o in (("portrait_01", 6), ("portrait_05", 8), ("action_01", 3), ("group_05", 6)):
        im = cv2.imread(str(real / f"{base}.jpg"))
        out = real / f"orient{o}_{base}.jpg"
        cv2.imwrite(str(out), inv[o](im), [cv2.IMWRITE_JPEG_QUALITY, 62])
        piexif.insert(piexif.dump({"0th": {piexif.ImageIFD.Orientation: o}}), str(out))
        man[out.name] = {"category": "orientation", "copy_of": f"{base}.jpg", "orientation": o,
                         "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "split": "train"}  # fmt: skip
    (R / "manifest.json").write_text(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(
        {"candidates": cmd_candidates, "build": cmd_build, "finish": cmd_finish}.get(
            sys.argv[1] if len(sys.argv) > 1 else "", lambda: 2
        )()
    )
