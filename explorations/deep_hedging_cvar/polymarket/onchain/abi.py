"""Pure-python keccak-256 and a minimal Solidity ABI decoder (no eth_abi / pycryptodome in the venv).

Only what the extractor needs: uint<N>, address, bytes32, bytes, static/dynamic tuples, T[].
"""
from __future__ import annotations

_RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
       0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
       0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
       0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
       0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
       0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
_ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
_M = (1 << 64) - 1


def _rol(v, n):
    return ((v << n) | (v >> (64 - n))) & _M if n else v


def _keccak_f(A):
    for rc in _RC:
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rol(C[(x + 1) % 5], 1) for x in range(5)]
        A = [[A[x][y] ^ D[x] for y in range(5)] for x in range(5)]
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rol(A[x][y], _ROT[x][y])
        A = [[B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        A[0][0] ^= rc
    return A


def keccak256(data: bytes) -> bytes:
    rate = 136
    p = bytearray(data)
    p.append(0x01)
    while len(p) % rate:
        p.append(0)
    p[-1] |= 0x80
    A = [[0] * 5 for _ in range(5)]
    for off in range(0, len(p), rate):
        for i in range(rate // 8):
            A[i % 5][i // 5] ^= int.from_bytes(p[off + 8 * i: off + 8 * i + 8], "little")
        A = _keccak_f(A)
    return b"".join(A[i % 5][i // 5].to_bytes(8, "little") for i in range(4))


def selector(sig: str) -> str:
    return "0x" + keccak256(sig.encode()).hex()[:8]


def topic(sig: str) -> str:
    return "0x" + keccak256(sig.encode()).hex()


# ------------------------------------------------------------------------------------------ ABI decoding
def _split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def parse_type(s: str):
    s = s.strip()
    if s.endswith("[]"):
        return ("array", parse_type(s[:-2]))
    if s.startswith("("):
        return ("tuple", [parse_type(x) for x in _split_top(s[1:-1])])
    return ("base", s)


def _dynamic(t) -> bool:
    k = t[0]
    if k == "base":
        return t[1] in ("bytes", "string")
    if k == "array":
        return True
    return any(_dynamic(c) for c in t[1])


def _head(t) -> int:
    if _dynamic(t):
        return 32
    if t[0] == "tuple":
        return sum(_head(c) for c in t[1])
    return 32


def _word(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos:pos + 32], "big")


def _static(t, data, pos):
    if t[0] == "tuple":
        return _seq(t[1], data, pos)
    b = t[1]
    if b == "address":
        return "0x" + data[pos + 12:pos + 32].hex()
    if b.startswith("bytes"):          # bytes1..bytes32
        return "0x" + data[pos:pos + 32].hex()
    if b == "bool":
        return bool(_word(data, pos))
    if b.startswith("int"):
        v = _word(data, pos)
        return v - (1 << 256) if v >> 255 else v
    return _word(data, pos)            # uintN


def _at(t, data, loc):
    """Decode a dynamic value whose encoding starts at absolute offset loc."""
    if t[0] == "base":                 # bytes / string
        n = _word(data, loc)
        raw = data[loc + 32: loc + 32 + n]
        return raw.decode() if t[1] == "string" else raw
    if t[0] == "array":
        n = _word(data, loc)
        return _seq([t[1]] * n, data, loc + 32)
    return _seq(t[1], data, loc)       # dynamic tuple


def _seq(types, data, base):
    out, pos = [], base
    for t in types:
        if _dynamic(t):
            out.append(_at(t, data, base + _word(data, pos)))
        else:
            out.append(_static(t, data, pos))
        pos += _head(t)
    return out


def decode(types: list[str], data: bytes) -> list:
    return _seq([parse_type(t) for t in types], data, 0)


def selftest() -> None:
    import common as C
    assert keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    for name, sig in C.EVENT_SIGS.items():
        assert topic(sig) == C.TOPICS[name], (name, topic(sig), C.TOPICS[name])


if __name__ == "__main__":
    selftest()
    print("keccak + event topics OK")
