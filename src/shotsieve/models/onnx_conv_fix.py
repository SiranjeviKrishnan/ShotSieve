"""Make a tf2onnx graph loadable by the CoreML execution provider, without the `onnx` package.

The published NAFNet-SIDD-width32 ONNX (see registry `nafnet_sidd`) was converted from TFLite by
tf2onnx: every `Conv` carries an EMPTY `group` attribute and the 1x1 / 2x2 convolutions have no
`pads`. ONNX Runtime's CPU provider tolerates that; the CoreML provider refuses the graph
("Required param 'pad' is missing", "KernelChannels (1) != InputChannels (64) / Group (1)"). The fix
is purely structural: every Conv gets `pads=[0,0,0,0]` when it has none and `group` = its true
group count (the output-channel count for depthwise weights `[C,1,k,k]`, else 1). The weights and
the maths are untouched (the patched graph is bit-identical on the CPU provider).

A minimal protobuf wire-format rewriter: only the NodeProto / AttributeProto bytes of Conv nodes
are re-encoded, everything else is copied verbatim. Field numbers: ModelProto.graph=7,
GraphProto.node=1 / initializer=5, NodeProto.input=1 / op_type=4 / attribute=5, TensorProto.dims=1
/ name=8, AttributeProto.name=1 / i=3 / ints=8 / type=20 (INT=2, INTS=7)."""

from __future__ import annotations

import contextlib
import mmap
import os
from collections.abc import Iterator
from pathlib import Path


def _read_varint(buf: bytes | memoryview, i: int) -> tuple[int, int]:
    v = shift = 0
    while True:
        b = buf[i]
        i += 1
        v |= (b & 0x7F) << shift
        if not b & 0x80:
            return v, i
        shift += 7


def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _fields(buf: bytes | memoryview) -> Iterator[tuple[int, int, int, int]]:
    """(field number, wire type, start of the value, end of the field) for every field of `buf`."""
    i, n = 0, len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        fno, wt = tag >> 3, tag & 7
        if wt == 0:
            _v, j = _read_varint(buf, i)
            yield fno, wt, i, j
            i = j
        elif wt == 2:
            ln, j = _read_varint(buf, i)
            yield fno, wt, j, j + ln
            i = j + ln
        elif wt == 1:
            yield fno, wt, i, i + 8
            i += 8
        elif wt == 5:
            yield fno, wt, i, i + 4
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")


def _len_field(fno: int, payload: bytes) -> bytes:
    return _varint(fno << 3 | 2) + _varint(len(payload)) + payload


def _attr_ints(name: bytes, values: list[int]) -> bytes:
    body = _len_field(1, name) + b"".join(_varint(8 << 3) + _varint(v) for v in values)
    return _len_field(5, body + _varint(20 << 3) + _varint(7))


def _attr_int(name: bytes, value: int) -> bytes:
    body = _len_field(1, name) + _varint(3 << 3) + _varint(value) + _varint(20 << 3) + _varint(2)
    return _len_field(5, body)


def _weight_shapes(graph: bytes | memoryview) -> dict[bytes, list[int]]:
    shapes: dict[bytes, list[int]] = {}
    for fno, wt, s, e in _fields(graph):
        if fno != 5 or wt != 2:
            continue
        t = graph[s:e]
        name, dims = b"", []
        for f2, w2, s2, e2 in _fields(t):
            if f2 == 8 and w2 == 2:
                name = bytes(t[s2:e2])
            elif f2 == 1 and w2 == 0:
                dims.append(_read_varint(t, s2)[0])
            elif f2 == 1 and w2 == 2:  # packed
                j = s2
                while j < e2:
                    v, j = _read_varint(t, j)
                    dims.append(v)
        shapes[name] = dims
    return shapes


def _fix_conv(node: bytes | memoryview, shapes: dict[bytes, list[int]]) -> bytes:
    kept = bytearray()
    names: set[bytes] = set()
    inputs: list[bytes] = []
    for fno, wt, s, e in _fields(node):
        start = s - (len(_varint(fno << 3 | wt)) + (len(_varint(e - s)) if wt == 2 else 0))
        raw = bytes(node[start:e])
        if fno == 5 and wt == 2:
            attr = node[s:e]
            name = next(
                (bytes(attr[a:b]) for f, w, a, b in _fields(attr) if f == 1 and w == 2), b""
            )
            names.add(name)
            if name != b"group":
                kept += raw
            continue
        if fno == 1 and wt == 2:
            inputs.append(bytes(node[s:e]))
        kept += raw
    if b"pads" not in names:
        kept += _attr_ints(b"pads", [0, 0, 0, 0])
    w = shapes.get(inputs[1]) if len(inputs) > 1 else None
    group = w[0] if w and len(w) == 4 and w[1] == 1 and w[0] > 1 else 1
    kept += _attr_int(b"group", group)
    return bytes(kept)


def _is_conv(node: bytes | memoryview) -> bool:
    return any(f == 4 and w == 2 and node[a:b] == b"Conv" for f, w, a, b in _fields(node))


def fix_conv_file(src: str | Path, dst: str | Path) -> None:
    """Write the repaired copy of the ONNX file `src` to `dst` (atomically). The 124 MB model is
    memory-mapped and streamed: only the few hundred Conv nodes are rebuilt in memory, so the
    process does not keep a transient copy of the weights resident."""
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        model = memoryview(mm)
        try:
            with open(tmp, "wb") as out:
                for fno, wt, s, e in _fields(model):
                    head = len(_varint(fno << 3 | wt)) + (len(_varint(e - s)) if wt == 2 else 0)
                    if fno != 7 or wt != 2:
                        out.write(model[s - head : e])
                        continue
                    graph = model[s:e]
                    shapes = _weight_shapes(graph)
                    parts: list[bytes | memoryview] = []
                    for f2, w2, s2, e2 in _fields(graph):
                        head2 = len(_varint(f2 << 3 | w2))
                        head2 += len(_varint(e2 - s2)) if w2 == 2 else 0
                        node = graph[s2:e2]
                        if f2 == 1 and w2 == 2 and _is_conv(node):
                            parts.append(_len_field(1, _fix_conv(node, shapes)))
                        else:
                            parts.append(graph[s2 - head2 : e2])
                    total = sum(len(p) for p in parts)
                    out.write(_varint(7 << 3 | 2) + _varint(total))
                    for p in parts:
                        out.write(p)
                    del parts, graph
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
            model.release()
            with contextlib.suppress(BufferError):  # slices still referenced: freed with the frame
                mm.close()


def fix_conv_attributes(model: bytes) -> bytes:
    """The model bytes with every Conv given explicit `pads` and a true `group` (see module doc)."""
    out = bytearray()
    for fno, wt, s, e in _fields(model):
        head = len(_varint(fno << 3 | wt)) + (len(_varint(e - s)) if wt == 2 else 0)
        if fno != 7 or wt != 2:
            out += model[s - head : e]
            continue
        graph = model[s:e]
        shapes = _weight_shapes(graph)
        g = bytearray()
        for f2, w2, s2, e2 in _fields(graph):
            head2 = len(_varint(f2 << 3 | w2)) + (len(_varint(e2 - s2)) if w2 == 2 else 0)
            if f2 == 1 and w2 == 2 and _is_conv(graph[s2:e2]):
                g += _len_field(1, _fix_conv(graph[s2:e2], shapes))
                continue
            g += graph[s2 - head2 : e2]
        out += _len_field(7, bytes(g))
    return bytes(out)
