"""Adobe Camera Raw settings (`crs:`) in XMP: read, format and merge-safe write.

Values use Lightroom's own formats and ranges. Writes go to an XMP sidecar (RAW) or into an
embedded-XMP *copy* of a JPEG/DNG; originals are never touched. ShotSieve's own writes carry
`shotsieve:EditProfile` so they are told apart from the photographer's edits.
"""

from __future__ import annotations

import hashlib
import io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from shotsieve.io import dng_xmp, jpeg_xmp
from shotsieve.io import xmp as _xmp
from shotsieve.io.formats import RAW_EXTS

CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
PROCESS_VERSION = "11.0"
_CRS = "{" + CRS + "}"

# Lightroom slider ranges: (lo, hi, kind)
RANGES: dict[str, tuple[float, float]] = {
    "Exposure2012": (-5.0, 5.0),
    "Temperature": (2000.0, 50000.0),
    "Tint": (-150.0, 150.0),
    "IncrementalTemperature": (-100.0, 100.0),
    "IncrementalTint": (-100.0, 100.0),
}
DEFAULT_RANGE = (-100.0, 100.0)
# written by ShotSieve itself (the provenance attributes live in the shotsieve namespace)
_OWN_ATTRS = (
    "shotsieve:EditProfile", "shotsieve:EditConfidence", "shotsieve:EditKeys",
    "shotsieve:EditDigest", "shotsieve:DigestV",
)
_PLAIN_KEYS = {"HasSettings", "ProcessVersion", "WhiteBalance", "Version"}

# T22 crop (docs/11_CROP.md section 5). Crop provenance is separate from edit provenance:
# `shotsieve:CropDigest` hashes the canonical CROP_KEYS lines only; the edit digest (v2) hashes the
# crs set WITHOUT them, so adding a crop never makes ShotSieve's own edit look like the user's.
CROP_KEYS = (
    "HasCrop", "CropTop", "CropLeft", "CropBottom", "CropRight", "CropAngle",
    "CropConstrainToWarp",
)  # fmt: skip
CROP_DEFAULTS = {
    "HasCrop": "False", "CropTop": "0", "CropLeft": "0", "CropBottom": "1", "CropRight": "1",
    "CropAngle": "0", "CropConstrainToWarp": "0",
}  # Lightroom's no-crop values, written explicitly on Reset
_DIGEST_V = "2"
_CROP_PROV = ("shotsieve:CropDigest", "shotsieve:CropVersion", "shotsieve:CropCarryKeys",
              "shotsieve:CropCarryDigest")  # fmt: skip


def clamp_value(key: str, v: float) -> float:
    lo, hi = RANGES.get(key, DEFAULT_RANGE)
    return min(max(float(v), lo), hi)


def format_value(key: str, v: float) -> str:
    """Lightroom's text form of a slider value, clamped to its range.

    Exposure2012 `+0.35` (two decimals); Temperature an int in Kelvin (2000-50000); Tint and
    the other sliders signed ints (`+12`, `-40`).
    """
    v = clamp_value(key, v)
    if key == "Exposure2012":
        return f"{v:+.2f}"
    if key == "Temperature":
        return str(int(round(v)))
    return f"{int(round(v)):+d}"


def format_carry(v: float | str) -> str:
    """A carried-over value verbatim (ints without a trailing .0)."""
    if isinstance(v, str):
        return v
    f = float(v)
    return str(int(f)) if f == int(f) and abs(f) < 1e12 else repr(round(f, 6))


@dataclass
class DevelopXmp:
    settings: dict[str, float | str] = field(default_factory=dict)
    has_settings: bool = False
    # ours: ShotSieve wrote it AND nothing changed since. The `crs:` set must still hash to the
    # `shotsieve:EditDigest` we stored; a tag whose digest no longer matches (or is missing)
    # means the photographer edited the file afterwards, so it is theirs.
    ours: bool = False
    tagged: bool = False  # carries shotsieve:EditProfile (whether or not it is still ours)
    profile: str | None = None
    raw: dict[str, str] = field(default_factory=dict)  # every crs key -> text
    edit_keys: list[str] = field(default_factory=list)  # the keys ShotSieve wrote
    # crs child subtrees (curves, masks), canonical form
    struct: dict[str, str] = field(default_factory=dict)
    # T22 crop provenance
    digest_v: str = "1"  # "2": the edit digest excludes CROP_KEYS
    has_crop: bool = False  # crs:HasCrop is True
    crop: dict[str, str] = field(default_factory=dict)  # the raw CROP_KEYS values
    crop_ours: bool = False  # CropDigest present and matching the current CROP_KEYS
    crop_tagged: bool = False
    only_ours: bool = False  # every crs key except HasSettings/ProcessVersion/CROP_KEYS is ours
    made_by_us: bool = False  # an unmodified ShotSieve edit and/or crop write, nothing else
    orientation: int | None = None  # tiff:Orientation of the sidecar (a Lightroom rotation)
    crop_version: int | None = None  # shotsieve:CropVersion (the cull.db crop version written)


@dataclass
class WriteResult:
    path: Path | None
    # written | replaced_own | skipped_user_settings | skipped_exists | skipped_catalog
    # | skipped_unreadable | skipped_user_crop | unchanged | error
    status: str
    reason: str = ""

    @property
    def written(self) -> bool:
        return self.status in ("written", "replaced_own")


def _scalar(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


def _parse_packet(packet: bytes) -> ET.Element | None:
    try:
        return ET.fromstring(packet.strip())
    except ET.ParseError:
        return None


def _canon(el: ET.Element) -> str:
    """Whitespace-normalised serialisation of an element subtree (tag, sorted attributes, text)."""
    attrs = "".join(f" {k}={v!r}" for k, v in sorted(el.attrib.items()))
    body = (el.text or "").strip() + "".join(_canon(c) + (c.tail or "").strip() for c in el)
    return f"<{el.tag}{attrs}>{body}</>"


def crs_digest(
    raw: dict[str, str], struct: dict[str, str] | None = None, exclude: tuple[str, ...] = ()
) -> str:
    """sha256 of the canonical (sorted) `key=value` lines of a crs set, plus its structured
    children (tone curves, masks: Review #7 n1) so a curve-only edit changes the digest.
    `exclude` drops keys (digest v2 excludes the crop keys)."""
    text = "\n".join(f"{k}={raw[k]}" for k in sorted(raw) if k not in exclude)
    if struct:
        text += "\n" + "\n".join(f"{k}#{struct[k]}" for k in sorted(struct))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def crop_digest(raw: dict[str, str]) -> str:
    """sha256 of the canonical CROP_KEYS lines (sorted `key=value`) present in `raw`."""
    text = "\n".join(f"{k}={raw[k]}" for k in sorted(CROP_KEYS) if k in raw)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _describe(root: ET.Element) -> DevelopXmp:
    out = DevelopXmp()
    sv = "{" + _xmp.NS["shotsieve"] + "}"
    digest: str | None = None
    cdigest: str | None = None
    carry_keys: list[str] = []
    carry_digest: str | None = None
    for desc in root.iter(_xmp.RDF + "Description"):
        for name, text in list(desc.attrib.items()):
            if name.startswith(_CRS):
                key = name[len(_CRS) :]
                out.raw[key] = text
                if key == "HasSettings":
                    out.has_settings = text.strip().lower() == "true"
                    continue
                out.settings[key] = _scalar(text)
            elif name == sv + "EditProfile":
                out.tagged, out.profile = True, text
            elif name == sv + "EditDigest":
                digest = text
            elif name == sv + "EditKeys":
                out.edit_keys = [k for k in text.split(",") if k]
            elif name == sv + "DigestV":
                out.digest_v = text.strip() or "1"
            elif name == sv + "CropDigest":
                cdigest = text
            elif name == sv + "CropVersion":
                try:
                    out.crop_version = int(text)
                except ValueError:
                    pass
            elif name == sv + "CropCarryKeys":
                carry_keys = [k for k in text.split(",") if k]
            elif name == sv + "CropCarryDigest":
                carry_digest = text
            elif name == "{" + _xmp.NS["tiff"] + "}Orientation":
                try:
                    out.orientation = int(text)
                except ValueError:
                    pass
        for child in desc:
            if child.tag.startswith(_CRS) and len(child) > 0:
                out.struct[child.tag[len(_CRS) :]] = _canon(child)
            if child.tag.startswith(_CRS) and len(child) == 0 and (child.text or "").strip():
                key = child.tag[len(_CRS) :]
                out.raw[key] = child.text.strip()
                if key == "HasSettings":
                    out.has_settings = child.text.strip().lower() == "true"
                else:
                    out.settings[key] = _scalar(child.text.strip())
    exclude = CROP_KEYS if out.digest_v == _DIGEST_V else ()
    out.ours = (
        out.tagged and digest is not None
        and digest == crs_digest(out.raw, out.struct, exclude)
    )
    out.crop = {k: out.raw[k] for k in CROP_KEYS if k in out.raw}
    out.has_crop = out.raw.get("HasCrop", "").strip().lower() == "true"
    out.crop_tagged = cdigest is not None
    out.crop_ours = cdigest is not None and cdigest == crop_digest(out.raw)
    allowed: set[str] = set(out.edit_keys) if out.ours else set()
    if carry_keys and carry_digest == _carry_digest(out.raw, carry_keys):
        allowed |= set(carry_keys)
    rest = set(out.raw) - {"HasSettings", "ProcessVersion"} - set(CROP_KEYS)
    out.only_ours = not out.struct and rest <= allowed
    out.made_by_us = out.ours or (out.only_ours and out.crop_ours)
    return out


def _carry_digest(raw: dict[str, str], keys: list[str]) -> str:
    text = "\n".join(f"{k}={raw.get(k, '')}" for k in sorted(keys))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _packet_of(master: Path) -> tuple[bytes | None, str]:
    """(XMP packet, where): the sidecar for RAW, the embedded packet for JPEG/DNG."""
    ext = master.suffix.lower()
    if ext == ".xmp":
        try:
            return master.read_bytes(), "sidecar"
        except OSError:
            return None, "sidecar"
    if ext in RAW_EXTS and ext != ".dng":
        try:
            return _xmp.sidecar_path(master).read_bytes(), "sidecar"
        except OSError:
            return None, "sidecar"
    emb = dng_xmp.read_packet(master) if ext == ".dng" else jpeg_xmp.read_packet(master)
    if emb is not None:
        return emb, "embedded"
    try:
        return _xmp.sidecar_path(master).read_bytes(), "sidecar"
    except OSError:
        return None, "sidecar"


def read_develop_xmp(master: str | Path) -> DevelopXmp | None:
    """The `crs:` settings of a master's sidecar (RAW) or embedded packet (JPEG/DNG), with
    provenance. None when there is no readable packet."""
    packet, _ = _packet_of(Path(master))
    if packet is None:
        return None
    root = _parse_packet(packet)
    if root is None:
        return None
    return _describe(root)


# ---- writing ---------------------------------------------------------------------------------


def packet_bytes(tree: ET.ElementTree) -> bytes:
    ET.indent(tree, space="  ")
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=False)
    return (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + buf.getvalue()
        + b'\n<?xpacket end="w"?>\n'
    )


def _strip_own(descs: list[ET.Element], keys: list[str]) -> None:
    """Remove our provenance and the `crs:` keys of ShotSieve's earlier write (and only those:
    foreign keys such as CameraProfile stay)."""
    own = {"{" + _xmp.NS["shotsieve"] + "}" + a.split(":")[1] for a in _OWN_ATTRS}
    drop = {_CRS + k for k in keys}
    for d in descs:
        for name in [n for n in d.attrib if n in drop or n in own]:
            del d.attrib[name]
        for child in [c for c in d if c.tag in drop]:
            d.remove(child)


def _all_crs(root: ET.Element) -> dict[str, str]:
    return _describe(root).raw


def _raw_struct(root: ET.Element) -> tuple[dict[str, str], dict[str, str]]:
    d = _describe(root)
    return d.raw, d.struct


def _process_version(pv: float | str | None) -> str:
    """The PV to write: a reference's / the user's, never lower than ShotSieve's 11.0."""
    try:
        v = float(pv) if pv is not None else 0.0
    except ValueError:
        v = 0.0
    return PROCESS_VERSION if v <= float(PROCESS_VERSION) else f"{v:.1f}"


def confidence_text(confidence: dict[str, float] | None) -> str:
    return ";".join(f"{k}={v:.2f}" for k, v in (confidence or {}).items())


def apply_to_tree(
    tree: ET.ElementTree,
    values: dict[str, float],
    carry: dict[str, float | str] | None,
    profile_hash: str,
    confidence: dict[str, float] | None,
    wb: bool,
    replace_own: bool = False,
    process_version: float | str | None = None,
) -> None:
    """Set the `crs:` attributes on the first rdf:Description of `tree` (foreign content kept).

    `replace_own`: the tree holds an unmodified earlier ShotSieve write, whose keys are removed
    first. Otherwise nothing is removed: foreign `crs:` keys stay and ours are written on top.
    Records `shotsieve:EditKeys` and `shotsieve:EditDigest` (hash of the final crs set)."""
    root = tree.getroot()
    descs = _xmp._descriptions(root)
    if not descs:
        rdf = root.find(_xmp.RDF + "RDF")
        if rdf is None:
            rdf = ET.SubElement(root, _xmp.RDF + "RDF")
        descs = [ET.SubElement(rdf, _xmp.RDF + "Description", {_xmp.RDF + "about": ""})]
    own_keys = _describe(root).edit_keys if replace_own else []
    _strip_own(descs, own_keys)
    items: dict[str, str] = {
        "HasSettings": "True", "ProcessVersion": _process_version(process_version),
    }
    if wb:
        items["WhiteBalance"] = "Custom"
    for k, v in (carry or {}).items():
        if k not in CROP_KEYS:  # a crop is per image: never carried
            items.setdefault(k, format_carry(v))
    for k, v in values.items():
        items[k] = format_value(k, v)
    for k, text in items.items():
        q = _xmp._q("crs:" + k)
        for d in descs:  # a foreign element-form value of the same key would duplicate it
            for child in [c for c in d if c.tag == q]:
                d.remove(child)
            if q in d.attrib and d is not descs[0]:
                del d.attrib[q]
        descs[0].set(q, text)
    descs[0].set(_xmp._q("shotsieve:EditProfile"), profile_hash)
    descs[0].set(_xmp._q("shotsieve:EditConfidence"), confidence_text(confidence))
    descs[0].set(_xmp._q("shotsieve:EditKeys"), ",".join(sorted(items)))
    raw, struct = _raw_struct(root)
    descs[0].set(_xmp._q("shotsieve:DigestV"), _DIGEST_V)
    descs[0].set(_xmp._q("shotsieve:EditDigest"), crs_digest(raw, struct, CROP_KEYS))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _embedded(path: Path) -> bytes | None:
    reader = dng_xmp.read_packet if path.suffix.lower() == ".dng" else jpeg_xmp.read_packet
    return reader(path)


def _check_copy(dst: Path) -> tuple[str, str] | None:
    """(status, reason) when an existing `dst` must not be overwritten, else None. An existing
    file is replaced only when it is an unmodified ShotSieve copy (our tag + matching digest)."""
    if not dst.exists():
        return None
    try:
        packet = _embedded(dst)
    except OSError:
        packet = None
    root = _parse_packet(packet) if packet is not None else None
    if root is None:
        return "skipped_exists", "a file that is not a ShotSieve copy is already there"
    cur = _describe(root)
    if cur.made_by_us:
        return None
    if cur.tagged or cur.has_settings:
        return "skipped_user_settings", "the copy has been edited since ShotSieve wrote it"
    return "skipped_exists", "a file that is not a ShotSieve copy is already there"


def writable(
    master: str | Path, mode: str, catalog_has: bool = False, dst: str | Path | None = None
) -> tuple[bool, str]:
    """Would `write_develop` write this master? (ok, reason when not): the never-overwrite rules
    (catalog already has develop settings; a sidecar / embedded packet with develop settings
    that ShotSieve did not write or that were changed since; an unparseable XMP; for embed mode
    an existing `dst` that is not an unmodified ShotSieve copy)."""
    master = Path(master)
    if catalog_has:
        return False, "skipped_catalog"
    if mode == "sidecar":
        sp = _xmp.sidecar_path(master)
        if sp.exists():
            cur = read_develop_xmp(sp)
            if cur is None:
                return False, "skipped_unreadable"
            if cur.has_settings and not cur.ours and not cur.only_ours:
                return False, "skipped_user_settings"
        return True, ""
    packet = _embedded(master)
    if packet is not None:
        root = _parse_packet(packet)
        if root is None:
            return False, "skipped_unreadable"
        cur = _describe(root)
        if cur.has_settings and not cur.ours and not cur.only_ours:
            return False, "skipped_user_settings"
    if dst is not None:
        bad = _check_copy(Path(dst))
        if bad:
            return False, bad[0]
    return True, ""


def write_develop(
    master: str | Path,
    values: dict[str, float],
    carry: dict[str, float | str] | None = None,
    *,
    profile_hash: str,
    confidence: dict[str, float] | None = None,
    wb: bool = False,
    mode: str = "sidecar",
    out_dir: str | Path | None = None,
    catalog_has: bool = False,
    process_version: float | str | None = None,
) -> WriteResult:
    """Write predicted develop settings, merge-safe.

    mode "sidecar" (RAW): create/merge `<stem>.xmp`. mode "embed" (JPEG/DNG): write an
    embedded-XMP *copy* of the file into `out_dir`; the original's sha256 is asserted unchanged.

    Never overwrites: skipped when the XMP has `crs:HasSettings=True` that ShotSieve did not write
    or that changed since (digest mismatch), or when the catalog already holds develop settings
    for the image. An existing copy is replaced only when it is an unmodified ShotSieve copy.
    ShotSieve's own unmodified earlier write is replaced (its keys only; foreign `crs:` stays).
    The read-modify-write holds the sidecar's lock.
    """
    master = Path(master)
    if catalog_has:
        return WriteResult(None, "skipped_catalog", "the catalog has develop settings")
    _xmp._register(None)
    if mode == "sidecar":
        sp = _xmp.sidecar_path(master)
        with _xmp.sidecar_lock(sp):
            _xmp._register(sp)
            cur = read_develop_xmp(sp) if sp.exists() else None
            if cur is not None and cur.has_settings and not cur.ours and not cur.only_ours:
                return WriteResult(sp, "skipped_user_settings", "sidecar has your develop settings")
            outcome = _xmp.WriteOutcome()
            tree = _xmp._load_tree(sp, outcome)
            if tree is None:
                return WriteResult(sp, "skipped_unreadable", "sidecar could not be parsed")
            replaced = cur is not None and cur.ours
            if cur is not None and not cur.ours:
                _xmp.backup_by_hash(sp)
            apply_to_tree(
                tree, values, carry, profile_hash, confidence, wb, replaced, process_version
            )
            payload = packet_bytes(tree)
            tmp = _xmp._tmp_for(sp)
            tmp.write_bytes(payload)
            os.replace(tmp, sp)
        return WriteResult(sp, "replaced_own" if replaced else "written")

    if mode != "embed" or out_dir is None:
        raise ValueError("mode must be 'sidecar' or 'embed' (with out_dir)")
    ext = master.suffix.lower()
    if ext not in (".jpg", ".jpeg", ".dng"):
        return WriteResult(None, "error", f"cannot embed XMP in {ext} files")
    dst = Path(out_dir) / master.name
    if dst.resolve() == master.resolve():
        return WriteResult(None, "error", "output folder is the original's folder")
    before = _sha256(master)
    packet = _embedded(master)
    replaced = False
    if packet is None:
        tree = _xmp._new_tree()
    else:
        root = _parse_packet(packet)
        if root is None:
            return WriteResult(None, "skipped_unreadable", "embedded XMP could not be parsed")
        cur = _describe(root)
        if cur.has_settings and not cur.ours and not cur.only_ours:
            return WriteResult(None, "skipped_user_settings", "file has your develop settings")
        replaced = cur.ours
        for prefix, uri in _xmp.NS.items():
            ET.register_namespace(prefix, uri)
        tree = ET.ElementTree(root)
    with _xmp.sidecar_lock(dst):
        bad = _check_copy(dst)
        if bad:
            return WriteResult(dst, bad[0], bad[1])
        apply_to_tree(
                tree, values, carry, profile_hash, confidence, wb, replaced, process_version
            )
        payload = packet_bytes(tree)
        if ext == ".dng":
            dng_xmp.write_copy(master, dst, payload)
        else:
            jpeg_xmp.write_copy(master, dst, payload)
    if _sha256(master) != before:  # pragma: no cover - the source is only ever read
        raise RuntimeError(f"original {master} changed during write")
    return WriteResult(dst, "replaced_own" if replaced else "written")


# ---- crop (T22) ------------------------------------------------------------------------------


@dataclass
class CropXmp:
    """The crop a master's XMP carries. `rect_crs` is (left, top, right, bottom) in the stored
    frame; `rect_display` the same crop in display-normalised coordinates (when an orientation
    was given)."""

    has_crop: bool = False
    ours: bool = False
    source: str = "none"  # shotsieve | lightroom | none
    rect_crs: tuple[float, float, float, float] | None = None
    rect_display: tuple[float, float, float, float] | None = None
    orientation: int | None = None  # tiff:Orientation found in the sidecar
    version: int | None = None


def _crop_rect(raw: dict[str, str]) -> tuple[float, float, float, float] | None:
    try:
        r = (
            float(raw["CropLeft"]), float(raw["CropTop"]), float(raw["CropRight"]),
            float(raw["CropBottom"]),
        )
    except (KeyError, ValueError):
        return None
    return r if r[0] < r[2] and r[1] < r[3] else None


def read_crop(master: str | Path, orientation: int | None = None) -> CropXmp:
    """The crop in a master's sidecar (RAW) or embedded packet (JPEG/DNG). `orientation` is the
    image's EXIF orientation; a `tiff:Orientation` in the XMP (a Lightroom rotation) wins."""
    cur = read_develop_xmp(master)
    if cur is None or not cur.has_crop:
        return CropXmp(orientation=None if cur is None else cur.orientation)
    rect = _crop_rect(cur.raw)
    out = CropXmp(
        True, cur.crop_ours, "shotsieve" if cur.crop_ours else "lightroom", rect,
        None, cur.orientation, cur.crop_version,
    )
    o = cur.orientation or orientation
    if rect is not None and o:
        from shotsieve.pipeline import crop_geom as g

        out.rect_display = g.from_crs(g.Rect(*rect), o).as_tuple()
    return out


def _crop_items(rect_crs: tuple[float, float, float, float] | None) -> dict[str, str]:
    if rect_crs is None:
        return dict(CROP_DEFAULTS)
    left, top, right, bottom = (min(max(float(v), 0.0), 1.0) for v in rect_crs)
    return {
        "HasCrop": "True", "CropTop": f"{top:.6f}", "CropLeft": f"{left:.6f}",
        "CropBottom": f"{bottom:.6f}", "CropRight": f"{right:.6f}", "CropAngle": "0",
        "CropConstrainToWarp": "0",
    }  # fmt: skip


def _apply_crop_to_tree(
    tree: ET.ElementTree,
    rect_crs: tuple[float, float, float, float] | None,
    before: DevelopXmp | None,
    carry: dict[str, float | str] | None,
    process_version: float | str | None,
    version: int,
) -> None:
    """Set only the crop keys (+ crop provenance) on the first rdf:Description; every other
    attribute and child stays as it is. A first-ever write (no develop settings) also carries the
    edit-profile preset and `HasSettings`/`ProcessVersion`. Re-signs a ShotSieve edit (v2)."""
    root = tree.getroot()
    descs = _xmp._descriptions(root)
    if not descs:
        rdf = root.find(_xmp.RDF + "RDF")
        if rdf is None:
            rdf = ET.SubElement(root, _xmp.RDF + "RDF")
        descs = [ET.SubElement(rdf, _xmp.RDF + "Description", {_xmp.RDF + "about": ""})]
    d0 = descs[0]
    items = _crop_items(rect_crs)
    had_settings = before is not None and before.has_settings
    carried: list[str] = []
    if not had_settings:
        items["HasSettings"] = "True"
        items["ProcessVersion"] = _process_version(process_version)
        for k, v in (carry or {}).items():
            if k not in CROP_KEYS and k not in ("HasSettings", "ProcessVersion"):
                items[k] = format_carry(v)
                carried.append(k)
    elif "ProcessVersion" not in (before.raw if before else {}):
        items["ProcessVersion"] = _process_version(process_version)
    for name in [n for n in d0.attrib if n.split("}")[-1] in _CROP_PROV_LOCAL
                 and n.startswith("{" + _xmp.NS["shotsieve"] + "}")]:
        del d0.attrib[name]
    for k, text in items.items():
        q = _xmp._q("crs:" + k)
        for d in descs:  # a foreign element-form value of the same key would duplicate it
            for child in [c for c in d if c.tag == q]:
                d.remove(child)
            if q in d.attrib and d is not d0:
                del d.attrib[q]
        d0.set(q, text)
    raw, struct = _raw_struct(root)
    d0.set(_xmp._q("shotsieve:CropDigest"), crop_digest(raw))
    d0.set(_xmp._q("shotsieve:CropVersion"), str(int(version)))
    if carried:
        d0.set(_xmp._q("shotsieve:CropCarryKeys"), ",".join(sorted(carried)))
        d0.set(_xmp._q("shotsieve:CropCarryDigest"), _carry_digest(raw, carried))
    if before is not None and before.ours:  # ShotSieve's own edit: re-sign as v2
        d0.set(_xmp._q("shotsieve:DigestV"), _DIGEST_V)
        d0.set(_xmp._q("shotsieve:EditDigest"), crs_digest(raw, struct, CROP_KEYS))


_CROP_PROV_LOCAL = frozenset(a.split(":")[1] for a in _CROP_PROV)


def _crop_blocked(
    cur: DevelopXmp | None, rect_crs: object, catalog_has: bool, catalog_has_crop: bool
) -> WriteResult | None:
    if catalog_has_crop or (cur is not None and cur.has_crop and not cur.crop_ours):
        return WriteResult(None, "skipped_user_crop", "a Lightroom crop is kept")
    if catalog_has and (cur is None or not cur.has_settings or cur.only_ours):
        return WriteResult(
            None, "skipped_catalog",
            "Lightroom holds edits for this photo; writing the sidecar could replace them on "
            "'Read Metadata'. Crop saved in ShotSieve only.",
        )
    if rect_crs is None and not (cur is not None and cur.crop_ours):
        return WriteResult(None, "unchanged", "no ShotSieve crop to reset")
    return None


def write_crop(
    master: str | Path,
    rect_crs: tuple[float, float, float, float] | None,
    *,
    mode: str = "sidecar",
    out_dir: str | Path | None = None,
    catalog_has: bool = False,
    catalog_has_crop: bool = False,
    carry: dict[str, float | str] | None = None,
    process_version: float | str | None = None,
    version: int = 0,
) -> WriteResult:
    """Write (or, with `rect_crs=None`, reset) `crs:Crop*`, merge-safe.

    `rect_crs` = (left, top, right, bottom) in the STORED frame (`crop_geom.to_crs`). Only the crop
    keys and our crop provenance change; a photographer's develop settings, foreign keys and XMP
    content stay as they are (one `.bak` per content hash). A crop the photographer made
    (`HasCrop=True` without our digest, or a crop in the catalog) is never overwritten:
    `skipped_user_crop`. Reset writes Lightroom's no-crop defaults explicitly. mode "embed" writes
    an embedded-XMP copy (JPEG/DNG) into `out_dir`; the original's sha256 is asserted unchanged.
    """
    master = Path(master)
    _xmp._register(None)
    if mode == "sidecar":
        sp = _xmp.sidecar_path(master)
        with _xmp.sidecar_lock(sp):
            _xmp._register(sp)
            cur = read_develop_xmp(sp) if sp.exists() else None
            if sp.exists() and cur is None:
                return WriteResult(sp, "skipped_unreadable", "sidecar could not be parsed")
            bad = _crop_blocked(cur, rect_crs, catalog_has, catalog_has_crop)
            if bad is not None:
                bad.path = sp
                return bad
            outcome = _xmp.WriteOutcome()
            tree = _xmp._load_tree(sp, outcome)
            if tree is None:
                return WriteResult(sp, "skipped_unreadable", "sidecar could not be parsed")
            if cur is not None:
                _xmp.backup_by_hash(sp)
            _apply_crop_to_tree(tree, rect_crs, cur, carry, process_version, version)
            payload = packet_bytes(tree)
            tmp = _xmp._tmp_for(sp)
            tmp.write_bytes(payload)
            os.replace(tmp, sp)
        return WriteResult(sp, "replaced_own" if cur is not None and cur.crop_ours else "written")

    if mode != "embed" or out_dir is None:
        raise ValueError("mode must be 'sidecar' or 'embed' (with out_dir)")
    ext = master.suffix.lower()
    if ext not in (".jpg", ".jpeg", ".dng"):
        return WriteResult(None, "error", f"cannot embed XMP in {ext} files")
    dst = Path(out_dir) / master.name
    if dst.resolve() == master.resolve():
        return WriteResult(None, "error", "output folder is the original's folder")
    before = _sha256(master)
    with _xmp.sidecar_lock(dst):
        bad_copy = _check_copy(dst)
        if bad_copy:
            return WriteResult(dst, bad_copy[0], bad_copy[1])
        # an existing ShotSieve copy (edit and/or crop) is merged into; else start from the master
        packet = _embedded(dst) if dst.exists() else _embedded(master)
        if packet is None:
            tree, cur = _xmp._new_tree(), None
        else:
            root = _parse_packet(packet)
            if root is None:
                return WriteResult(None, "skipped_unreadable", "embedded XMP could not be parsed")
            cur = _describe(root)
            for prefix, uri in _xmp.NS.items():
                ET.register_namespace(prefix, uri)
            tree = ET.ElementTree(root)
        bad = _crop_blocked(cur, rect_crs, catalog_has, catalog_has_crop)
        if bad is not None:
            bad.path = dst
            return bad
        _apply_crop_to_tree(tree, rect_crs, cur, carry, process_version, version)
        payload = packet_bytes(tree)
        if ext == ".dng":
            dng_xmp.write_copy(master, dst, payload)
        else:
            jpeg_xmp.write_copy(master, dst, payload)
    if _sha256(master) != before:  # pragma: no cover - the source is only ever read
        raise RuntimeError(f"original {master} changed during write")
    return WriteResult(dst, "replaced_own" if cur is not None and cur.crop_ours else "written")
