"""Copy a DNG with a new XMP packet (TIFF tag 700) and read tags from IFD0.

The packet is appended at the end of the copy and IFD0's tag 700 entry is patched in place
(offset/count). If tag 700 is absent, a rewritten IFD0 with the extra entry is appended and the
header offset is patched. Strip/tile offsets and every other byte are never moved. The source is
only read. Classic TIFF only (BigTIFF is refused).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

TAG_XMP = 700
TAG_AS_SHOT_NEUTRAL = 50728
TAG_PREVIEW_APP = 50966  # PreviewApplicationName (the spec's 50971 is PreviewDateTime)
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


class DngError(ValueError):
    """Not a classic-TIFF DNG this module can rewrite."""


@dataclass
class Entry:
    tag: int
    typ: int
    count: int
    raw: bytes  # the 4 value/offset bytes, as stored
    pos: int  # file offset of this 12-byte entry


def _endian(data: bytes) -> str:
    if data[:2] == b"II":
        e = "<"
    elif data[:2] == b"MM":
        e = ">"
    else:
        raise DngError("not a TIFF file")
    magic = struct.unpack(e + "H", data[2:4])[0]
    if magic == 43:
        raise DngError("BigTIFF is not supported")
    if magic != 42:
        raise DngError("bad TIFF magic")
    return e


def _read_ifd(data: bytes, e: str, off: int) -> tuple[list[Entry], int]:
    if off <= 0 or off + 2 > len(data):
        raise DngError("bad IFD offset")
    n = struct.unpack(e + "H", data[off : off + 2])[0]
    end = off + 2 + 12 * n
    if end + 4 > len(data):
        raise DngError("truncated IFD")
    entries = []
    for i in range(n):
        p = off + 2 + 12 * i
        tag, typ, count = struct.unpack(e + "HHI", data[p : p + 8])
        entries.append(Entry(tag, typ, count, data[p + 8 : p + 12], p))
    nxt = struct.unpack(e + "I", data[end : end + 4])[0]
    return entries, nxt


def _value_bytes(data: bytes, e: str, ent: Entry) -> bytes:
    size = _TYPE_SIZE.get(ent.typ, 1) * ent.count
    if size <= 4:
        return ent.raw[:size]
    off = struct.unpack(e + "I", ent.raw)[0]
    return data[off : off + size]


def ifd0_tags(path: str | Path) -> dict[int, Entry]:
    data = Path(path).read_bytes()
    e = _endian(data)
    entries, _ = _read_ifd(data, e, struct.unpack(e + "I", data[4:8])[0])
    return {x.tag: x for x in entries}


def read_packet(path: str | Path) -> bytes | None:
    try:
        data = Path(path).read_bytes()
        e = _endian(data)
        entries, _ = _read_ifd(data, e, struct.unpack(e + "I", data[4:8])[0])
        for ent in entries:
            if ent.tag == TAG_XMP:
                return _value_bytes(data, e, ent)
    except (OSError, DngError, struct.error):
        return None
    return None


def as_shot_neutral(path: str | Path) -> tuple[float, float, float] | None:
    """The DNG AsShotNeutral (camera RGB of the as-shot neutral), when IFD0 has it."""
    try:
        data = Path(path).read_bytes()
        e = _endian(data)
        entries, _ = _read_ifd(data, e, struct.unpack(e + "I", data[4:8])[0])
        for ent in entries:
            if ent.tag == TAG_AS_SHOT_NEUTRAL and ent.count >= 3:
                raw = _value_bytes(data, e, ent)
                if ent.typ == 5:  # RATIONAL
                    v = struct.unpack(e + "6I", raw[:24])
                    vals = [v[i] / v[i + 1] if v[i + 1] else 0.0 for i in (0, 2, 4)]
                elif ent.typ == 12:  # DOUBLE
                    vals = list(struct.unpack(e + "3d", raw[:24]))
                else:
                    return None
                return (float(vals[0]), float(vals[1]), float(vals[2]))
    except (OSError, DngError, struct.error):
        return None
    return None


def preview_application(path: str | Path) -> str | None:
    """IFD0 PreviewApplicationName (50971), as text. Present when Adobe software rendered the
    preview (the edit is then baked into it)."""
    try:
        data = Path(path).read_bytes()
        e = _endian(data)
        entries, nxt = _read_ifd(data, e, struct.unpack(e + "I", data[4:8])[0])
        for ent in entries:
            if ent.tag == TAG_PREVIEW_APP:
                return _value_bytes(data, e, ent).split(b"\x00")[0].decode("utf-8", "replace")
        # previews usually live in SubIFDs / the preview IFD; look in IFD1.. as well
        seen = 0
        while nxt and seen < 8:
            entries, nxt = _read_ifd(data, e, nxt)
            seen += 1
            for ent in entries:
                if ent.tag == TAG_PREVIEW_APP:
                    return _value_bytes(data, e, ent).split(b"\x00")[0].decode("utf-8", "replace")
        sub = next((x for x in ifd0_tags(path).values() if x.tag == 330), None)
        if sub is not None:
            offs = struct.unpack(
                e + f"{sub.count}I", _value_bytes(data, e, sub)[: 4 * sub.count]
            )
            for off in offs:
                try:
                    sub_entries, _ = _read_ifd(data, e, off)
                except DngError:
                    continue
                for ent in sub_entries:
                    if ent.tag == TAG_PREVIEW_APP:
                        return (
                            _value_bytes(data, e, ent).split(b"\x00")[0].decode("utf-8", "replace")
                        )
    except (OSError, DngError, struct.error):
        return None
    return None


def write_copy(src: str | Path, dst: str | Path, packet: bytes) -> Path:
    """`dst` = `src` with `packet` as its XMP. Only IFD0's entry for tag 700 (or, when absent, a
    rewritten IFD0 appended at the end plus the header offset) changes."""
    data = bytearray(Path(src).read_bytes())
    e = _endian(bytes(data))
    ifd0 = struct.unpack(e + "I", data[4:8])[0]
    entries, nxt = _read_ifd(bytes(data), e, ifd0)
    if len(data) % 2:
        data.append(0)
    pkt_off = len(data)
    data += packet
    if len(data) % 2:
        data.append(0)
    new_entry = struct.pack(e + "HHII", TAG_XMP, 1, len(packet), pkt_off)
    existing = next((x for x in entries if x.tag == TAG_XMP), None)
    if existing is not None:
        data[existing.pos : existing.pos + 12] = new_entry
    else:
        new_ifd_off = len(data)
        rows = [bytes(data[x.pos : x.pos + 12]) for x in entries] + [new_entry]
        rows.sort(key=lambda r: struct.unpack(e + "H", r[:2])[0])
        data += struct.pack(e + "H", len(rows)) + b"".join(rows) + struct.pack(e + "I", nxt)
        data[4:8] = struct.pack(e + "I", new_ifd_off)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(bytes(data))
    tmp.replace(dst)
    return dst
