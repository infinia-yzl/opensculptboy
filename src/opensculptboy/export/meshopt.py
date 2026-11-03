# OpenSculptBoy
# Apache License, Version 2.0
"""
Buffers compressed as glTF's ``EXT_meshopt_compression`` reads them, in NumPy.

The formats are those of meshoptimizer (Arseny Kapoulkine, MIT licence), as its decoders and
the extension's specification describe them; this module writes its own encoders from that
description:

- :func:`encode_vertex_buffer` writes the vertex codec, version 0 (mode ``ATTRIBUTES``): the
  elements in blocks of up to 256, each byte of an element as the zigzag delta from the same
  byte of the element before, and each run of 16 deltas packed with 0, 2, 4 or 8 bits per
  delta (a delta that does not fit takes the run's largest code and follows as a whole
  byte). The first element closes the stream, so that the decoder starts from it.
- :func:`encode_index_buffer` writes the index codec, version 1 (mode ``TRIANGLES``): each
  triangle as one code byte that names an edge of a recent triangle (a FIFO of 16 edges) and
  its third vertex (the next new vertex, one of the 16 recent vertices, the last vertex plus
  or minus one, or a zigzag delta in LEB128), or three vertices of their own. The encoder
  numbers the vertices in the order the decoder first meets them (``remap``), so that most
  new vertices are "the next one".
- :func:`order_triangles` puts the triangles in an order that keeps neighbours together (a
  walk over the triangles' shared edges), as meshoptimizer's vertex cache optimisation does
  for the same end.

``viewer/src/meshopt_ref.js`` (the page's decoder) reads both, and ``test_ghost`` checks the
round trip with it and with meshoptimizer's own decoder.
"""

from __future__ import annotations

import numpy as np

VERTEX_HEADER = 0xA0
INDEX_HEADER = 0xE1
# the codes of the index codec's table for triangles whose first vertex is new (meshoptimizer's
# table): each byte holds the FIFO places (1-based, 0 for a new vertex) of the second and third
CODE_AUX = bytes(
    [
        0x00,
        0x76,
        0x87,
        0x56,
        0x67,
        0x78,
        0xA9,
        0x86,
        0x65,
        0x89,
        0x68,
        0x98,
        0x01,
        0x69,
        0,
        0,
    ]
)
_BITS = (0, 2, 4, 8)


def _zigzag8(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.int16)
    s = ((d + 128) % 256) - 128
    return (((s << 1) ^ (s >> 7)) & 0xFF).astype(np.uint8)


def _encode_bytes(values: np.ndarray, out: list) -> None:
    """One byte channel of a block (length a multiple of 16): the header bits, then each run."""
    groups = values.reshape(-1, 16)
    g = len(groups)
    sizes = np.empty((g, 4), np.int64)
    sizes[:, 0] = np.where(groups.any(1), 1 << 30, 0)
    packed = {}
    for k, bits in enumerate(_BITS[1:3], start=1):
        sentinel = (1 << bits) - 1
        escaped = groups >= sentinel
        sizes[:, k] = 16 * bits // 8 + escaped.sum(1)
        codes = np.minimum(groups, sentinel).astype(np.uint8)
        per = 8 // bits
        shaped = codes.reshape(g, -1, per)
        byte = np.zeros(shaped.shape[:2], np.uint8)
        for j in range(per):
            byte |= shaped[:, :, j] << (8 - bits * (j + 1))
        packed[k] = (byte, escaped)
    sizes[:, 3] = 16
    mode = np.argmin(sizes, 1)
    header = np.zeros((g + 3) // 4, np.uint8)
    np.bitwise_or.at(
        header, np.arange(g) // 4, (mode << ((np.arange(g) % 4) * 2)).astype(np.uint8)
    )
    out.append(header.tobytes())
    for i in range(g):
        m = mode[i]
        if m == 0:
            continue
        if m == 3:
            out.append(groups[i].tobytes())
            continue
        byte, escaped = packed[m]
        out.append(byte[i].tobytes())
        if escaped[i].any():
            out.append(groups[i][escaped[i]].tobytes())


def encode_vertex_buffer(data: np.ndarray) -> bytes:
    """The vertex codec (version 0) of ``data`` (count, stride) uint8, stride a multiple of 4
    and at most 256."""
    data = np.ascontiguousarray(data, np.uint8)
    count, stride = data.shape
    if stride % 4 or stride > 256 or count == 0:
        raise ValueError("A vertex buffer needs elements of 4 to 256 bytes, by fours.")
    block = min((8192 // stride) & ~15, 256)
    out = [bytes([VERTEX_HEADER])]
    last = data[0].astype(np.int16)
    for base in range(0, count, block):
        chunk = data[base : base + block].astype(np.int16)
        prev = np.vstack([last[None], chunk[:-1]])
        deltas = _zigzag8(chunk - prev)
        n = len(chunk)
        padded = np.zeros(((n + 15) & ~15, stride), np.uint8)
        padded[:n] = deltas
        for k in range(stride):
            _encode_bytes(padded[:, k], out)
        last = chunk[-1]
    if stride < 32:
        out.append(bytes(32 - stride))
    out.append(data[0].tobytes())
    return b"".join(out)


def _leb128(v: int, out: bytearray) -> None:
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def _zigzag(d: int) -> int:
    return ((d << 1) ^ (d >> 31)) & 0xFFFFFFFF


def order_triangles(
    triangles: np.ndarray, vertices: np.ndarray | None = None
) -> np.ndarray:
    """An order of the triangles (T,) that walks across shared edges: each next triangle is an
    unvisited neighbour of the most recent triangles, else the first unvisited triangle in a
    spatial (Morton) order of their centres, so that most triangles share an edge with one
    just before them."""
    T = np.asarray(triangles, np.int64)
    n = len(T)
    if n == 0:
        return np.zeros(0, np.int64)
    if vertices is not None:
        c = np.asarray(vertices, np.float64)[T].mean(1)
        q = ((c - c.min(0)) / max(np.ptp(c, 0).max(), 1e-12) * 1023).astype(np.int64)
        code = np.zeros(n, np.int64)
        for bit in range(10):
            for axis in range(3):
                code |= ((q[:, axis] >> bit) & 1) << (3 * bit + axis)
        seeds = np.argsort(code, kind="stable")
    else:
        seeds = np.arange(n)
    # the triangles' neighbours across edges
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    owner = np.tile(np.arange(n), 3)
    key = np.minimum(e[:, 0], e[:, 1]) * (int(T.max()) + 1) + np.maximum(
        e[:, 0], e[:, 1]
    )
    order = np.argsort(key, kind="stable")
    key, owner = key[order], owner[order]
    same = key[1:] == key[:-1]
    a, b = owner[:-1][same], owner[1:][same]
    pairs = np.concatenate([np.stack([a, b], 1), np.stack([b, a], 1)])
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    starts = np.searchsorted(pairs[:, 0], np.arange(n + 1))
    nbrs = pairs[:, 1].tolist()
    starts = starts.tolist()
    seen = bytearray(n)
    out = []
    recent = []
    seed_iter = iter(seeds.tolist())
    while len(out) < n:
        nxt = -1
        for t in reversed(recent):
            for j in range(starts[t], starts[t + 1]):
                u = nbrs[j]
                if not seen[u]:
                    nxt = u
                    break
            if nxt >= 0:
                break
        if nxt < 0:
            for s in seed_iter:
                if not seen[s]:
                    nxt = s
                    break
        seen[nxt] = 1
        out.append(nxt)
        recent.append(nxt)
        if len(recent) > 8:
            recent.pop(0)
    return np.asarray(out, np.int64)


def encode_index_buffer(triangles: np.ndarray) -> tuple[bytes, np.ndarray, np.ndarray]:
    """The index codec (version 1) of a triangle list (T, 3), with the vertices numbered in
    the order the decoder first meets them. Returns the bytes, ``remap`` (V,): the new number
    of each old vertex (-1 for one no triangle uses), and the triangles as they decode (T, 3)
    in the new numbers (each a rotation of its input, so the winding holds)."""
    T = np.asarray(triangles, np.int64)
    V = int(T.max()) + 1 if len(T) else 0
    remap = [-1] * V
    codes = bytearray()
    data = bytearray()
    decoded = []
    next_ = 0
    last = 0
    # numbers from the top, for a new vertex that has to be written out in full
    top = len(np.unique(T)) - 1 if len(T) else -1
    # the FIFOs as sequence numbers: an edge (a, b) or a vertex pushed as the k-th entry
    edge_seq: dict = {}
    edges = 0
    vert_seq: dict = {}
    verts = 0

    def vslot(v):
        s = vert_seq.get(v)
        return -1 if s is None else verts - 1 - s

    def eslot(a, b):
        s = edge_seq.get((a, b))
        return -1 if s is None else edges - 1 - s

    aux = {CODE_AUX[i]: i for i in range(14)}
    for tri in T.tolist():
        best = None
        for r in range(3):
            a, b, c = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
            if remap[a] < 0 or remap[b] < 0:
                continue
            s = eslot(remap[a], remap[b])
            if 0 <= s <= 14 and (best is None or s < best[0]):
                best = (s, a, b, c)
        if best is not None:
            s, a, b, c = best
            na, nb = remap[a], remap[b]
            if remap[c] < 0:
                remap[c] = next_
            nc = remap[c]
            vs = vslot(nc)
            if nc == next_:
                code = 0
                next_ += 1
                push = True
            elif 1 <= vs <= 12:
                code = vs
                push = False
            elif nc == last - 1:
                code, last, push = 0x0D, nc, True
            elif nc == last + 1:
                code, last, push = 0x0E, nc, True
            else:
                code = 0x0F
                _leb128(_zigzag(nc - last), data)
                last, push = nc, True
            codes.append((s << 4) | code)
            if push:
                vert_seq[nc] = verts
                verts += 1
            for edge in ((nc, nb), (na, nc)):
                edge_seq[edge] = edges
                edges += 1
            decoded.append((na, nb, nc))
            continue
        # no recent edge: three vertices of their own, the first a new one where one is new
        r = 0
        for k in range(3):
            if remap[tri[k]] < 0:
                r = k
                break
        a, b, c = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
        new_a = remap[a] < 0
        if new_a:
            remap[a] = next_
        na = remap[a]
        a_next = na == next_
        if a_next:
            next_ += 1

        def place(v):
            nonlocal next_
            if remap[v] < 0:
                remap[v] = next_
            nv = remap[v]
            if nv == next_:
                next_ += 1
                return nv, 0
            s = vslot(nv)
            if 0 <= s <= 13:
                return nv, s + 1
            return nv, 0x0F

        nb, z = place(b)
        nc, w = place(c)
        e = (z << 4) | w
        if a_next and e in aux:
            codes.append(0xF0 | aux[e])
        else:
            if e == 0:
                # a zero byte resets the decoder's next vertex: write the third vertex in
                # full, with a number from the top
                next_ -= 1
                remap[c] = top
                top -= 1
                nc = remap[c]
                w = 0x0F
                e = (z << 4) | w
            codes.append(0xFE if a_next else 0xFF)
            data.append(e)
            if not a_next:
                _leb128(_zigzag(na - last), data)
                last = na
            if z == 0x0F:
                _leb128(_zigzag(nb - last), data)
                last = nb
            if w == 0x0F:
                _leb128(_zigzag(nc - last), data)
                last = nc
        vert_seq[na] = verts
        verts += 1
        if z in (0, 0x0F):
            vert_seq[nb] = verts
            verts += 1
        if w in (0, 0x0F):
            vert_seq[nc] = verts
            verts += 1
        for edge in ((nb, na), (nc, nb), (na, nc)):
            edge_seq[edge] = edges
            edges += 1
        decoded.append((na, nb, nc))
    out = bytes([INDEX_HEADER]) + bytes(codes) + bytes(data) + CODE_AUX
    return (
        out,
        np.asarray(remap, np.int64),
        np.asarray(decoded, np.int64).reshape(-1, 3),
    )


def _unzigzag8(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int16)
    return (v >> 1) ^ -(v & 1)


def decode_vertex_buffer(data: bytes, count: int, stride: int) -> np.ndarray:
    """The elements (count, stride) uint8 of a vertex codec stream of version 0, as the
    reference decoder reads it: the check of :func:`encode_vertex_buffer`."""
    src = np.frombuffer(data, np.uint8)
    if src[0] != VERTEX_HEADER:
        raise ValueError("Only the vertex codec's version 0 is read here.")
    block = min((8192 // stride) & ~15, 256)
    last = src[len(src) - stride :].astype(np.int16)
    out = np.zeros((count, stride), np.uint8)
    o = 1
    for base in range(0, count, block):
        n = min(count - base, block)
        groups = (n + 15) // 16
        head = (groups + 3) // 4
        deltas = np.zeros((stride, groups * 16), np.uint8)
        for k in range(stride):
            header = src[o : o + head]
            o += head
            for g in range(groups):
                mode = (int(header[g // 4]) >> ((g % 4) * 2)) & 3
                bits = _BITS[mode]
                seg = deltas[k, 16 * g : 16 * g + 16]
                if bits == 0:
                    continue
                if bits == 8:
                    seg[:] = src[o : o + 16]
                    o += 16
                    continue
                per = 8 // bits
                packed = src[o : o + 16 // per]
                o += 16 // per
                sentinel = (1 << bits) - 1
                for m in range(16):
                    v = (int(packed[m // per]) >> (8 - bits * (m % per + 1))) & sentinel
                    if v == sentinel:
                        v = int(src[o])
                        o += 1
                    seg[m] = v
        d = _unzigzag8(deltas[:, :n].T)
        values = (last[None] + np.cumsum(d, 0)) & 0xFF
        out[base : base + n] = values.astype(np.uint8)
        last = values[-1].astype(np.int16)
    if o != len(src) - max(stride, 32):
        raise ValueError("The vertex stream's length does not match its blocks.")
    return out


def decode_index_buffer(data: bytes, count: int) -> np.ndarray:
    """The indices (count,) of an index codec stream of version 1, as the reference decoder
    reads it: the check of :func:`encode_index_buffer`."""
    src = data
    if src[0] != INDEX_HEADER:
        raise ValueError("Only the index codec's version 1 is read here.")
    tris = count // 3
    code_o, data_o, aux_o = 1, 1 + tris, len(src) - 16
    edges = [0] * 32
    eo = 0
    verts = [0] * 16
    vo = 0
    nxt = last = 0
    out = []

    def leb():
        nonlocal data_o
        n, shift = 0, 0
        while True:
            b = src[data_o]
            data_o += 1
            n |= (b & 0x7F) << shift
            if b < 0x80:
                return n
            shift += 7

    def dez(v):
        return (v >> 1) ^ -(v & 1)

    for _ in range(tris):
        code = src[code_o]
        code_o += 1
        b0, b1 = code >> 4, code & 15

        def efifo(n):
            return edges[(eo - 1 - n) & 31]

        def vfifo(n):
            return verts[(vo - 1 - n) & 15]

        def epush(*xs):
            nonlocal eo
            for x in xs:
                edges[eo] = x
                eo = (eo + 1) & 31

        def vpush(x):
            nonlocal vo
            verts[vo] = x
            vo = (vo + 1) & 15

        if b0 < 15:
            a, b = efifo(2 * b0), efifo(2 * b0 + 1)
            if b1 == 0:
                c = nxt
                nxt += 1
                vpush(c)
            elif b1 < 13:
                c = vfifo(b1)
            elif b1 == 13:
                last -= 1
                c = last
                vpush(c)
            elif b1 == 14:
                last += 1
                c = last
                vpush(c)
            else:
                last += dez(leb())
                c = last
                vpush(c)
            epush(b, c, c, a)
        else:
            if b1 < 14:
                e = src[aux_o + b1]
                z, w = e >> 4, e & 15
                a = nxt
                nxt += 1
                if z == 0:
                    b = nxt
                    nxt += 1
                else:
                    b = vfifo(z - 1)
                if w == 0:
                    c = nxt
                    nxt += 1
                else:
                    c = vfifo(w - 1)
                vpush(a)
                if z == 0:
                    vpush(b)
                if w == 0:
                    vpush(c)
            else:
                e = src[data_o]
                data_o += 1
                if e == 0:
                    nxt = 0
                z, w = e >> 4, e & 15
                if b1 == 14:
                    a = nxt
                    nxt += 1
                else:
                    last += dez(leb())
                    a = last
                if z == 0:
                    b = nxt
                    nxt += 1
                elif z == 15:
                    last += dez(leb())
                    b = last
                else:
                    b = vfifo(z - 1)
                if w == 0:
                    c = nxt
                    nxt += 1
                elif w == 15:
                    last += dez(leb())
                    c = last
                else:
                    c = vfifo(w - 1)
                vpush(a)
                if z in (0, 15):
                    vpush(b)
                if w in (0, 15):
                    vpush(c)
            epush(a, b, b, c, c, a)
        out.extend((a, b, c))
    return np.asarray(out, np.int64)
