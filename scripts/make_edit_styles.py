"""Synthetic photographer styles for the editing-profile evaluation (T15, docs/08_PHASE3.md).

Builds `tests/fixtures/edit_styles/`: base images (the benchmark shoot's clean frames + the CC0
fixtures, exposure- and white-balance-normalised so "undo the perturbation" is well defined),
5 perturbed JPEGs per base (known exposure gain, mired cast and tint cast), and one Lightroom
catalog per style with the ground-truth develop settings in real Lightroom Lua text format:

  A  "bright & airy"   B  "moody"   C  "random" (control: iid sliders, nothing to learn)

    uv run python scripts/make_edit_styles.py [--out DIR] [--base-shoot DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_OUT = ROOT / "tests" / "fixtures" / "edit_styles"
VERSION = 3
N_PERTURB = 5
EDGE = 384
K_CLUSTERS = 6
MIRED_GAIN = 0.0025  # ln-gain of R (and -B) per mired of cast
TINT_GAIN = 6.0  # ln-gain of G per unit of delta-uv
TINT_PER_DUV = 1500.0  # Lightroom tint units per unit of delta-uv
STYLES = ("A", "B", "C")
SLIDERS = ("Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
           "Vibrance", "Saturation")  # fmt: skip
# keys the user's import preset applies to every image (the carry set)
PRESET = {
    "Sharpness": 40, "SharpenRadius": 1.0, "LuminanceSmoothing": 10, "ColorNoiseReduction": 25,
    "LensProfileEnable": 1, "AutoLateralCA": 1, "CameraProfile": "Adobe Standard",
    "ProcessVersion": "11.0", "ToneCurveName2012": "Linear",
}  # fmt: skip
_LIN = (np.arange(256, dtype=np.float32) / 255.0) ** 2.2


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def lua_text(settings: dict) -> str:
    """Lightroom's develop-settings Lua text (tabs, sorted keys, a nested tone curve)."""
    lines = ["s = {"]
    for k in sorted(settings):
        v = settings[k]
        if isinstance(v, str):
            lines.append(f'\t{k} = "{v}",')
        elif isinstance(v, bool):
            lines.append(f"\t{k} = {'true' if v else 'false'},")
        else:
            lines.append(f"\t{k} = {v},")
        if k == "ToneCurveName2012":
            lines.append("\tToneCurvePV2012 = {\n\t\t0,\n\t\t0,\n\t\t255,\n\t\t255,\n\t},")
    lines.append("}")
    return "\n".join(lines)


def build_catalog(path: Path, rows: list[dict]) -> None:
    """A Lightroom-shaped catalog with develop settings. rows: {root, rel, base, ext, develop}."""
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE AgLibraryRootFolder (
            id_local INTEGER PRIMARY KEY, absolutePath TEXT, name TEXT);
        CREATE TABLE AgLibraryFolder (
            id_local INTEGER PRIMARY KEY, rootFolder INTEGER, pathFromRoot TEXT);
        CREATE TABLE AgLibraryFile (
            id_local INTEGER PRIMARY KEY, folder INTEGER, baseName TEXT, extension TEXT);
        CREATE TABLE Adobe_images (id_local INTEGER PRIMARY KEY, rootFile INTEGER, pick REAL,
                                   rating REAL, colorLabels TEXT, masterImage INTEGER);
        CREATE TABLE Adobe_imageDevelopSettings (
            id_local INTEGER PRIMARY KEY, image INTEGER, text TEXT, processVersion TEXT);
        """
    )
    roots: dict[str, int] = {}
    folders: dict[tuple[int, str], int] = {}
    for n, e in enumerate(rows, start=1):
        root = e["root"].rstrip("/") + "/"
        if root not in roots:
            roots[root] = len(roots) + 1
            con.execute(
                "INSERT INTO AgLibraryRootFolder VALUES (?,?,?)",
                (roots[root], root, Path(root).name),
            )
        key = (roots[root], e["rel"])
        if key not in folders:
            folders[key] = len(folders) + 1
            con.execute(
                "INSERT INTO AgLibraryFolder VALUES (?,?,?)", (folders[key], key[0], key[1])
            )
        con.execute(
            "INSERT INTO AgLibraryFile VALUES (?,?,?,?)", (n, folders[key], e["base"], e["ext"])
        )
        con.execute("INSERT INTO Adobe_images VALUES (?,?,?,?,?,NULL)", (n, n, 0, None, ""))
        con.execute(
            "INSERT INTO Adobe_imageDevelopSettings VALUES (?,?,?,?)",
            (n, n, lua_text(e["develop"]), "11.0"),
        )
    con.commit()
    con.close()


def base_images(base_shoot: Path | None) -> list[tuple[str, np.ndarray]]:
    """(name, RGB uint8 at EDGE px) of the clean benchmark frames and the CC0 fixtures."""
    out: list[tuple[str, np.ndarray]] = []
    shoot = base_shoot or ROOT / "data" / "bench" / "shoot_300_7"
    bench = _load("bench_for_styles", ROOT / "scripts" / "benchmark.py")
    if not (shoot / "ground_truth.json").exists():
        shoot = DEFAULT_OUT / "_bench_shoot"
        bench.synthesize(300, shoot)
    truth = json.loads((shoot / "ground_truth.json").read_text())
    files = [shoot / t["file"] for t in truth if not t["degraded"]]
    extra = sorted((ROOT / "tests" / "fixtures" / "real").glob("cc0_*.jpg"))
    extra += sorted((ROOT / "tests" / "fixtures" / "generated" / "src").glob("*.jpg"))
    for f in [*files, *extra]:
        bgr = cv2.imread(str(f), cv2.IMREAD_REDUCED_COLOR_2)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        s = EDGE / max(h, w)
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        out.append((f.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return out


def _to_lin(img: np.ndarray) -> np.ndarray:
    return _LIN[img]


def _from_lin(lin: np.ndarray) -> np.ndarray:
    return np.clip(np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def normalise(img: np.ndarray) -> np.ndarray:
    """Grey-world white balance and mid-grey mean exposure in linear light."""
    lin = _to_lin(img).astype(np.float64)
    m = lin.reshape(-1, 3).mean(axis=0) + 1e-6
    lin = lin * (m[1] / m)  # grey-world: equal channel means
    target = (118.0 / 255.0) ** 2.2
    lin = lin * (target / (lin.mean() + 1e-6))
    return _from_lin(lin)


def perturb(img: np.ndarray, dev: float, cast: float, duv: float) -> np.ndarray:
    """Exposure gain (EV), mired cast and tint cast, applied to linear RGB (von Kries)."""
    lin = _to_lin(img).astype(np.float64)
    gains = np.exp(
        np.array([MIRED_GAIN * cast, TINT_GAIN * duv * -1.0, -MIRED_GAIN * cast])
    )  # R, G, B
    lin = lin * gains * 2.0**dev
    return _from_lin(lin)


def scene_stats(img: np.ndarray) -> tuple[float, float]:
    """(shadow_frac, clip_hi_norm) of the image the photographer sees (the perturbed one)."""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float((y < 56).mean()), float(min(1.0, (y >= 253).mean() / 0.05))


def style_truth(
    style: str, dev: float, cast: float, duv: float, shadow_frac: float, clip_hi: float,
    cluster: int, indoor: int, bright: int, rng: np.random.Generator,
) -> dict:  # fmt: skip
    """Ground-truth develop settings (before rounding) the photographer applies."""
    s_noise, e_noise, wb_noise, t_noise = 3.0, 0.08, 4.0, 2.0
    tint_cast = duv * TINT_PER_DUV
    z = {k: 0.0 for k in SLIDERS}
    if style == "A":
        exp = -dev + 0.4 + 0.3 * (cluster == indoor)
        temp = -cast - 8.0
        z.update(Shadows2012=30 + 20 * shadow_frac, Highlights2012=-40 * clip_hi,
                 Contrast2012=-10.0, Vibrance=15.0, Saturation=-5.0)  # fmt: skip
    elif style == "B":
        exp = -dev - 0.3
        temp = -cast + 10.0
        z.update(Contrast2012=25.0, Blacks2012=-20.0, Whites2012=10.0,
                 Saturation=5.0 if cluster == bright else -15.0)  # fmt: skip
    else:  # C: nothing to learn
        exp = rng.uniform(-1.5, 1.5)
        temp = rng.uniform(-60, 60)
        tint_cast = -rng.uniform(-15, 15)
        z = {k: rng.uniform(-40, 40) for k in SLIDERS}
    out = {"Exposure2012": exp + rng.normal(0, e_noise), "temp": temp + rng.normal(0, wb_noise),
           "tint": -tint_cast + rng.normal(0, t_noise)}  # fmt: skip
    for k, v in z.items():
        out[k] = v + rng.normal(0, s_noise)
    return out


def develop_settings(t: dict) -> dict:
    """The truth as Lightroom develop settings for a JPEG (rounded to Lightroom's precision)."""
    d = dict(PRESET)
    d["Exposure2012"] = round(float(np.clip(t["Exposure2012"], -5, 5)), 2)
    for k in SLIDERS:
        d[k] = int(round(float(np.clip(t[k], -100, 100))))
    d["WhiteBalance"] = "Custom"
    d["IncrementalTemperature"] = int(round(float(np.clip(t["temp"], -100, 100))))
    d["IncrementalTint"] = int(round(float(np.clip(t["tint"], -100, 100))))
    return d


def build(out: Path = DEFAULT_OUT, base_shoot: Path | None = None, seed: int = 0) -> Path:
    """(Re)build the fixture folder; returns it. Images are kept when already present."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    cur = json.loads(manifest.read_text()) if manifest.exists() else {}
    img_dir = out / "images"
    if cur.get("version") == VERSION and cur.get("root") == str(out) and all(
        (out / f"style{s}.lrcat").exists() for s in STYLES
    ):
        return out
    truth_path = out / "truth.json"
    # a fresh checkout has truth.json + images (committed) but no manifest / catalogs (their
    # absolute paths are machine specific, so they are generated, not committed)
    if cur.get("version", VERSION) == VERSION and truth_path.exists() and img_dir.is_dir():
        truth = json.loads(truth_path.read_text())
    else:
        truth = _generate(out, base_shoot, seed)
    rows: dict[str, list[dict]] = {s: [] for s in STYLES}
    for rec in truth:
        base, ext = Path(rec["file"]).stem, "jpg"
        rel = str(Path(rec["file"]).parent) + "/"
        for s in STYLES:
            rows[s].append({"root": str(img_dir), "rel": rel, "base": base, "ext": ext,
                            "develop": develop_settings(rec["truth"][s])})  # fmt: skip
    for s in STYLES:
        build_catalog(out / f"style{s}.lrcat", rows[s])
    manifest.write_text(json.dumps({"version": VERSION, "root": str(out), "n": len(truth)}))
    return out


def _generate(out: Path, base_shoot: Path | None, seed: int) -> list[dict]:
    from sklearn.cluster import KMeans

    from shotsieve.models.image_embed import get_image_embedder

    bases = base_images(base_shoot)
    pre_luma = [float(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean()) for _, im in bases]
    norm = [normalise(im) for _, im in bases]
    emb = get_image_embedder().embed_batch(norm)
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=seed).fit(emb)
    labels = km.labels_
    cl_luma = [
        np.mean([pre_luma[i] for i in np.nonzero(labels == c)[0]]) for c in range(K_CLUSTERS)
    ]
    indoor, bright = int(np.argmin(cl_luma)), int(np.argmax(cl_luma))
    img_dir = out / "images"
    truth: list[dict] = []
    for bi, ((name, _), im) in enumerate(zip(bases, norm, strict=True)):
        rng = np.random.default_rng([seed, bi])
        d = img_dir / f"b{bi:03d}"
        d.mkdir(parents=True, exist_ok=True)
        for k in range(N_PERTURB):
            dev = float(rng.uniform(-1.5, 1.5))
            cast = float(rng.uniform(-60, 60))
            duv = float(rng.uniform(-0.01, 0.01))
            rel = f"b{bi:03d}/p{k}.jpg"
            pim = perturb(im, dev, cast, duv)
            shadow, clip = scene_stats(pim)
            cv2.imwrite(str(img_dir / rel), cv2.cvtColor(pim, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])  # fmt: skip
            t = {
                s: style_truth(s, dev, cast, duv, shadow, clip, int(labels[bi]), indoor, bright,
                               np.random.default_rng([seed, bi, k, ord(s)]))
                for s in STYLES
            }  # fmt: skip
            truth.append({
                "file": rel, "base": bi, "name": name, "cluster": int(labels[bi]), "dev": dev,
                "cast": cast, "duv": duv, "shadow_frac": shadow, "clip_hi": clip, "truth": t,
            })  # fmt: skip
    (out / "truth.json").write_text(json.dumps(truth))
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base-shoot", type=Path)
    args = ap.parse_args(argv)
    out = build(args.out, args.base_shoot)
    n = len(json.loads((out / "truth.json").read_text()))
    print(f"{n} images, styles {', '.join(STYLES)} in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
