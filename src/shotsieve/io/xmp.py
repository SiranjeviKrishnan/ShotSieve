"""XMP sidecar read/write (merging, never touches the original)."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import threading
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shotsieve.log import get_logger

log = get_logger("xmp")

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "shotsieve": "http://ns.shotsieve.app/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "Iptc4xmpExt": "http://iptc.org/std/Iptc4xmpExt/2008-02-08/",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
}
RDF = "{" + NS["rdf"] + "}"
_SCALARS = {
    "rating": "xmp:Rating",
    "label": "xmp:Label",
    "decision": "shotsieve:Decision",
    "score": "shotsieve:Score",
    # provenance: the rating / label ShotSieve itself last wrote (so they are never mistaken
    # for the photographer's own choices, and are not overwritten blindly)
    "written_rating": "shotsieve:WrittenRating",
    "written_label": "shotsieve:WrittenLabel",
    "date_created": "photoshop:DateCreated",  # clock-offset-corrected capture time (optional)
}

# Colour labels (T23e): ShotSieve key <-> Lightroom's `xmp:Label` names ("" = no label).
LABEL_XMP = {"red": "Red", "yellow": "Yellow", "green": "Green", "blue": "Blue",
             "purple": "Purple", "none": ""}  # fmt: skip
_LABEL_KEY = {v.lower(): k for k, v in LABEL_XMP.items() if v}


def label_key(text: str | None) -> str:
    """The ShotSieve key of an `xmp:Label` value ("" for none or a name that is not a colour)."""
    return _LABEL_KEY.get((text or "").strip().lower(), "")


# Lightroom Classic reads XMP *sidecars* only for proprietary RAW files. For these formats it
# reads the metadata embedded in the file itself, so a sidecar next to them is ignored there
# (Bridge, Capture One, darktable and others do read it). ShotSieve never edits originals.
SIDECAR_BLIND_EXTS = frozenset(
    {".jpg", ".jpeg", ".dng", ".tif", ".tiff", ".png", ".psd", ".heic", ".heif", ".webp"}
)
LIGHTROOM_NOTE = (
    "Lightroom Classic ignores XMP sidecars for JPEG/TIFF/PNG/HEIC and DNG files (it reads the "
    "metadata embedded in the file, which ShotSieve never modifies). Sidecars were still "
    "written for other tools; in Lightroom use 'Metadata > Read Metadata from Files' on "
    "proprietary RAW only, or import ShotSieve's decisions another way."
)


def lightroom_warning(masters: Iterable[str | Path]) -> str | None:
    """A warning when any master is a format whose sidecar Lightroom does not read."""
    n = sum(Path(m).suffix.lower() in SIDECAR_BLIND_EXTS for m in masters)
    if not n:
        return None
    return f"{n} file(s) are JPEG/DNG-class formats. {LIGHTROOM_NOTE}"


@dataclass
class XmpData:
    rating: int | None = None
    label: str | None = None
    decision: str | None = None
    reasons: list[str] = field(default_factory=list)
    score: float | None = None
    written_rating: int | None = None
    written_label: str | None = None


@dataclass
class WriteOutcome:
    """What a write_sidecar call did with the photographer's own rating / label."""

    rating_kept: bool = False  # an existing foreign rating was left alone
    label_kept: bool = False
    lr_kept: bool = False  # rating / label left alone: Lightroom owns them (lrcat override)
    written_rating: int | None = None  # what this call actually wrote (None: nothing)
    written_label: str | None = None
    skipped_unreadable: bool = False  # the sidecar could not be parsed: backed up, not written


class _SidecarLock:
    """Re-entrant in-process lock plus an advisory `fcntl.flock` on a lock file under the temp dir
    (never inside the shoot), so a CLI `edit apply` and the server cannot race either (n3)."""

    def __init__(self, key: str) -> None:
        self._rl = threading.RLock()
        self._depth = 0
        self._fd: int | None = None
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        self._path = Path(tempfile.gettempdir()) / "shotsieve-locks" / f"{digest}.lock"

    def __enter__(self) -> _SidecarLock:
        self._rl.acquire()
        if self._depth == 0:
            try:
                import fcntl

                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(self._fd, fcntl.LOCK_EX)
            except (ImportError, OSError):  # no flock available: the in-process lock still holds
                if self._fd is not None:
                    os.close(self._fd)
                self._fd = None
        self._depth += 1
        return self

    def __exit__(self, *exc: object) -> None:
        self._depth -= 1
        if self._depth == 0 and self._fd is not None:
            try:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
        self._rl.release()


_LOCKS: dict[str, _SidecarLock] = {}
_LOCKS_GUARD = threading.Lock()


def sidecar_lock(path: str | Path) -> _SidecarLock:
    """The lock of one sidecar (or copy) file: per-process RLock + cross-process flock. Every writer
    that loads, mutates and replaces it (write_sidecar, merge_keywords, crs.write_develop) holds it,
    so the last writer can no longer drop another's change. Re-entrant."""
    try:
        key = str(Path(path).resolve())
    except OSError:
        key = str(path)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, _SidecarLock(key))


def sidecar_path(master: str | Path) -> Path:
    """`<stem>.xmp` next to the master file (path already ending in .xmp is returned as-is)."""
    p = Path(master)
    return p if p.suffix.lower() == ".xmp" else p.with_suffix(".xmp")


def _q(name: str) -> str:
    prefix, local = name.split(":")
    return "{" + NS[prefix] + "}" + local


def _register(path: Path | None) -> None:
    for prefix, uri in NS.items():
        ET.register_namespace(prefix, uri)
    if path and path.exists():
        try:
            for _, (prefix, uri) in ET.iterparse(path, events=("start-ns",)):
                if prefix:
                    try:
                        ET.register_namespace(prefix, uri)
                    except ValueError:
                        # reserved names such as "ns1": ElementTree picks its own prefix for
                        # that URI on output, which is the same namespace to any XMP reader
                        pass
        except ET.ParseError:
            pass


def backup_by_hash(sp: Path) -> Path | None:
    """Copy `sp` to `<name>.bak-<sha256[:12]>` unless that file exists: one backup per distinct
    content, never one per write. Returns the backup path (None when `sp` is unreadable)."""
    try:
        data = sp.read_bytes()
    except OSError:
        return None
    bak = sp.with_name(f"{sp.name}.bak-{hashlib.sha256(data).hexdigest()[:12]}")
    if not bak.exists():
        # temp + replace: a crash mid-write must never leave a truncated file under the hash name
        tmp = bak.with_name(bak.name + ".tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, bak)
        except OSError:
            tmp.unlink(missing_ok=True)
            return None
    return bak


def _load_tree(sp: Path, outcome: WriteOutcome | None = None) -> ET.ElementTree | None:
    """The sidecar's tree, a fresh one when there is none, or None when it exists but cannot be
    parsed: it is then copied once per content hash (`backup_by_hash`) and must be skipped,
    never overwritten (it may hold Lightroom develop settings)."""
    try:
        return ET.parse(sp)
    except OSError:
        return _new_tree()
    except ET.ParseError:
        bak = backup_by_hash(sp)
        log.warning(
            "unreadable XMP sidecar %s: left untouched, backup at %s", sp, bak.name if bak else "-"
        )
        if outcome is not None:
            outcome.skipped_unreadable = True
        return None


def _tmp_for(sp: Path) -> Path:
    return sp.with_name(f"{sp.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")


def _descriptions(root: ET.Element) -> list[ET.Element]:
    return list(root.iter(RDF + "Description"))


def _new_tree() -> ET.ElementTree:
    root = ET.Element(_q("x:xmpmeta"))
    rdf = ET.SubElement(root, RDF + "RDF")
    ET.SubElement(rdf, RDF + "Description", {RDF + "about": ""})
    return ET.ElementTree(root)


def read_sidecar(path: str | Path) -> XmpData | None:
    """Parse our fields from a sidecar (or a master's sidecar). None if absent/unparseable."""
    sp = sidecar_path(path)
    try:
        root = ET.parse(sp).getroot()
    except (OSError, ET.ParseError):
        return None
    data = XmpData()
    for desc in _descriptions(root):
        for key, qname in _SCALARS.items():
            q = _q(qname)
            raw = desc.get(q)
            child = desc.find(q)
            if raw is None and child is not None:
                raw = (child.text or "").strip()
            if raw is None:
                continue
            try:
                value = (
                    int(raw) if key in ("rating", "written_rating")
                    else float(raw) if key == "score" else raw
                )  # fmt: skip
            except ValueError:
                continue
            setattr(data, key, value)
        bag = desc.find(_q("shotsieve:Reasons"))
        if bag is not None:
            data.reasons = [(li.text or "") for li in bag.iter(RDF + "li")]
    return data


def _current(descs: list[ET.Element], qname: str) -> str | None:
    q = _q(qname)
    for d in descs:
        child = d.find(q)
        if child is not None:
            return (child.text or "").strip()
        if q in d.attrib:
            return d.attrib[q].strip()
    return None


def _set(descs: list[ET.Element], qname: str, text: str) -> None:
    q = _q(qname)
    for d in descs:
        if d.find(q) is not None:
            d.find(q).text = text  # type: ignore[union-attr]
            return
        if q in d.attrib:
            d.set(q, text)
            return
    descs[0].set(q, text)


def write_sidecar(master: str | Path, *args: Any, **kwargs: Any) -> Path:
    """Create or merge `<stem>.xmp` (see `_write_sidecar`), holding the sidecar's lock."""
    with sidecar_lock(sidecar_path(master)):
        return _write_sidecar(master, *args, **kwargs)


def _write_sidecar(
    master: str | Path,
    rating: int | None = None,
    label: str | None = None,
    decision: str | None = None,
    reasons: list[str] | None = None,
    score: float | None = None,
    *,
    override: bool = False,
    force: bool = False,
    record: bool = True,
    outcome: WriteOutcome | None = None,
    date_created: str | None = None,
    label_force: bool = False,
) -> Path:
    """Create or merge `<stem>.xmp`; only the given fields change, foreign content is kept.

    The photographer's own `xmp:Rating` / `xmp:Label` are never replaced: a non-zero rating or a
    label that ShotSieve did not itself write (tracked in `shotsieve:WrittenRating/Label`) is
    left alone unless the user explicitly overrode this image in ShotSieve (`override=True`) or
    `force=True`. `record=False` skips the provenance fields (emulates a foreign tool).
    """
    sp = sidecar_path(master)
    _register(sp)
    tree = _load_tree(sp, outcome)
    if tree is None:
        return sp
    root = tree.getroot()
    descs = _descriptions(root)
    if not descs:
        rdf = root.find(RDF + "RDF")
        if rdf is None:
            rdf = ET.SubElement(root, RDF + "RDF")
        descs = [ET.SubElement(rdf, RDF + "Description", {RDF + "about": ""})]
    out = outcome if outcome is not None else WriteOutcome()

    allowed = override or force
    if rating is not None:
        cur, mine = _current(descs, "xmp:Rating"), _current(descs, "shotsieve:WrittenRating")
        foreign = cur not in (None, "", "0") and cur != mine
        if foreign and not allowed:
            out.rating_kept = True
            rating = None
    if label is not None:
        cur, mine = _current(descs, "xmp:Label"), _current(descs, "shotsieve:WrittenLabel")
        foreign = cur not in (None, "") and cur != mine
        if foreign and not (allowed or label_force):
            out.label_kept = True
            label = None

    out.written_rating, out.written_label = rating, label
    values = {
        "rating": rating, "label": label, "decision": decision, "score": score,
        "date_created": date_created,
    }  # fmt: skip
    if record:
        values["written_rating"] = rating
        # a label the user chose in ShotSieve is *theirs*: not recorded as ShotSieve's own, so
        # it reads back as a user label (user_values) from the sidecar alone
        values["written_label"] = None if label_force else label
    for key, value in values.items():
        if value is None:
            continue
        text = f"{value:.4f}" if key == "score" else str(value)
        _set(descs, _SCALARS[key], text)
    if reasons is not None:
        q = _q("shotsieve:Reasons")
        for d in descs:
            for old in d.findall(q):
                d.remove(old)
        bag = ET.SubElement(ET.SubElement(descs[0], q), RDF + "Bag")
        for r in reasons:
            ET.SubElement(bag, RDF + "li").text = r

    ET.indent(tree, space="  ")
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=False)
    payload = (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + buf.getvalue()
        + b'\n<?xpacket end="w"?>\n'
    )
    tmp = _tmp_for(sp)
    tmp.write_bytes(payload)
    os.replace(tmp, sp)
    return sp


def _merge_bag(descs: list[ET.Element], qname: str, values: Iterable[str]) -> int:
    """Append `values` to the rdf:Bag/Seq/Alt of `qname`, never removing or editing foreign items
    (case-insensitive exact de-dup, so re-running adds nothing). Returns the number added."""
    q = _q(qname)
    holder = next((d.find(q) for d in descs if d.find(q) is not None), None)
    if holder is None:
        holder = ET.SubElement(descs[0], q)
    bag = next((c for c in holder if c.tag in (RDF + "Bag", RDF + "Seq", RDF + "Alt")), None)
    if bag is None:
        bag = ET.SubElement(holder, RDF + "Bag")
    have = {(li.text or "").strip().lower() for li in bag.iter(RDF + "li")}
    added = 0
    for v in values:
        v = v.strip()
        if v and v.lower() not in have:
            ET.SubElement(bag, RDF + "li").text = v
            have.add(v.lower())
            added += 1
    return added


def merge_keywords(
    master: str | Path, keywords: Iterable[str] = (), persons: Iterable[str] = ()
) -> int:
    """Merge keywords / people into the sidecar of `master` under the sidecar's lock (see
    `_merge_keywords`)."""
    with sidecar_lock(sidecar_path(master)):
        return _merge_keywords(master, keywords, persons)


def _merge_keywords(
    master: str | Path, keywords: Iterable[str], persons: Iterable[str]
) -> int:
    """Merge `dc:subject` keywords and IPTC `Iptc4xmpExt:PersonInImage` names into the sidecar of
    `master` (created if absent). Existing keywords / people are never removed or overwritten and
    the call is idempotent. Returns the number of items added (0 = the file is not rewritten)."""
    sp = sidecar_path(master)
    _register(sp)
    tree = _load_tree(sp)
    if tree is None:
        return 0
    root = tree.getroot()
    descs = _descriptions(root)
    if not descs:
        rdf = root.find(RDF + "RDF")
        if rdf is None:
            rdf = ET.SubElement(root, RDF + "RDF")
        descs = [ET.SubElement(rdf, RDF + "Description", {RDF + "about": ""})]
    added = _merge_bag(descs, "dc:subject", keywords)
    added += _merge_bag(descs, "Iptc4xmpExt:PersonInImage", persons)
    if added == 0 and sp.exists():
        return 0
    ET.indent(tree, space="  ")
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=False)
    payload = (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + buf.getvalue()
        + b'\n<?xpacket end="w"?>\n'
    )
    tmp = _tmp_for(sp)
    tmp.write_bytes(payload)
    os.replace(tmp, sp)
    return added


def read_keywords(master: str | Path) -> tuple[list[str], list[str]]:
    """(dc:subject keywords, PersonInImage names) of a sidecar; empty lists if absent."""
    try:
        root = ET.parse(sidecar_path(master)).getroot()
    except (OSError, ET.ParseError):
        return [], []

    def items(qname: str) -> list[str]:
        out: list[str] = []
        for d in _descriptions(root):
            for node in d.findall(_q(qname)):
                out += [(li.text or "").strip() for li in node.iter(RDF + "li")]
        return out

    return items("dc:subject"), items("Iptc4xmpExt:PersonInImage")


def user_values(data: XmpData, settings: Any) -> tuple[int | None, str | None]:
    """The (rating, label) of a sidecar that belong to the photographer.

    ShotSieve's own output (what `shotsieve:WrittenRating/Label` recorded, or - for sidecars
    from older versions without provenance - what its `shotsieve:Decision` maps to) is removed,
    so it can never be mistaken for ground truth.
    """
    wr, wl = data.written_rating, data.written_label
    if wr is None and wl is None and data.decision in ("keep", "review", "reject"):
        wr = getattr(settings, f"rating_{data.decision}")
        wl = getattr(settings, f"label_{data.decision}")
    rating = None if data.rating is None or data.rating == wr else data.rating
    label = None if not data.label or data.label == wl else data.label
    return rating, label


def write_decision(
    master: str | Path,
    settings: Any,
    decision: str,
    reasons: list[str],
    score: float,
    *,
    override: dict | None = None,
    force: bool = False,
    outcome: WriteOutcome | None = None,
    user_label: str | None = None,
) -> Path:
    """Write one image's final decision with the settings' rating / label scheme.

    `user_label` is the colour label the user set in ShotSieve (Lightroom name, "" = cleared);
    it replaces the decision's label and the photographer's existing one (user-wins).
    `override` is the user's override row (or None): an explicit override may carry its own
    star rating and is allowed to replace the photographer's existing rating / label.
    """
    rating = {
        "keep": settings.rating_keep, "review": settings.rating_review,
        "reject": settings.rating_reject,
    }[decision]  # fmt: skip
    label = {
        "keep": settings.label_keep, "review": settings.label_review,
        "reject": settings.label_reject,
    }[decision]  # fmt: skip
    lr_owned = bool(override and override.get("note") == "lrcat")
    if override and override.get("rating") is not None:
        rating = override["rating"]
    if lr_owned:  # Lightroom owns this image's rating / label (T20): never fight it
        rating = label = None
        if outcome is not None:
            outcome.lr_kept = True
    if user_label is not None and not lr_owned:  # T23e: the user's colour label wins ("" = none)
        label = user_label
    return write_sidecar(
        master, rating=rating, label=label, decision=decision, reasons=reasons, score=score,
        override=override is not None, force=force, outcome=outcome,
        label_force=user_label is not None and not lr_owned,
    )  # fmt: skip
