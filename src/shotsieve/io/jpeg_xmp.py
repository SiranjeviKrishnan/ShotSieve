"""Copy a JPEG with its XMP packet (APP1 `http://ns.adobe.com/xap/1.0/\\0`) replaced or inserted.

Every other byte of the file (all other segments and the entropy-coded data) is copied verbatim.
The source is only read. Extended XMP is not written (a packet must fit one segment).
"""

from __future__ import annotations

from pathlib import Path

XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
MAX_PACKET = 65_502  # 65,535 - 2 (length field) - 31 (header)


class JpegError(ValueError):
    """Not a JPEG this module can rewrite."""


def _segments(data: bytes) -> list[tuple[int, int, int]]:
    """(marker, start, end) of each segment before the entropy-coded data (start of scan)."""
    if data[:2] != b"\xff\xd8":
        raise JpegError("not a JPEG (no SOI)")
    out: list[tuple[int, int, int]] = []
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            raise JpegError(f"bad marker at {i}")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # standalone markers
            out.append((marker, i, i + 2))
            i += 2
            continue
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        end = i + 2 + length
        if length < 2 or end > n:
            raise JpegError("truncated segment")
        out.append((marker, i, end))
        if marker == 0xDA:  # start of scan: the rest is entropy-coded data up to EOI
            break
        i = end
    return out


def read_packet(path: str | Path) -> bytes | None:
    """The embedded XMP packet of a JPEG (None when absent or the file is not a JPEG)."""
    try:
        data = Path(path).read_bytes()
        for marker, s, e in _segments(data):
            if marker == 0xE1 and data[s + 4 : s + 4 + len(XMP_HEADER)] == XMP_HEADER:
                return data[s + 4 + len(XMP_HEADER) : e]
    except (OSError, JpegError):
        return None
    return None


def write_copy(src: str | Path, dst: str | Path, packet: bytes) -> Path:
    """Write `dst` = `src` with `packet` as its XMP (replacing an existing one)."""
    if len(packet) > MAX_PACKET:
        raise JpegError(f"XMP packet is {len(packet)} B, the APP1 limit is {MAX_PACKET} B")
    data = Path(src).read_bytes()
    segs = _segments(data)
    seg = b"\xff\xe1" + (len(packet) + len(XMP_HEADER) + 2).to_bytes(2, "big") + XMP_HEADER + packet
    out = bytearray(data[:2])
    inserted = False
    pos = 2
    for marker, s, e in segs:
        if marker == 0xD8:
            continue
        is_xmp = marker == 0xE1 and data[s + 4 : s + 4 + len(XMP_HEADER)] == XMP_HEADER
        # insert after the leading APP0 / Exif APP1 segments, before anything else
        is_head = marker == 0xE0 or (marker == 0xE1 and data[s + 4 : s + 8] == b"Exif")
        if not inserted and not is_head and not is_xmp:
            out += seg
            inserted = True
        if not is_xmp:
            out += data[s:e]
        pos = e
    if not inserted:
        out += seg
    out += data[pos:]
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(bytes(out))
    tmp.replace(dst)
    return dst
