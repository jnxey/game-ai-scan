"""
自定义对称加解密（与 gv-web src/tools/custom-cipher.js 算法一致）

非标准密码学，适用于本地混淆/缓存：
- 密钥扩展 keystream（FNV-1a + xorshift32）
- 密钥派生 S-box / 逆 S-box
- 6 轮字节变换 + CBC 链式扩散
- 帧头 GV + version + XOR 校验

用法:
    from custom_cipher import encode, decode
    hex_cipher = encode("hello", "my-key")
    plain = decode(hex_cipher, "my-key")
"""

from __future__ import annotations

import re
from typing import Callable

DEFAULT_KEY = "gv"
ROUNDS = 6
HEADER_MAGIC = (0x47, 0x56)  # 'GV'
VERSION = 1

_U32 = 0xFFFFFFFF


def _normalize_key_bytes(key: str | None) -> bytes:
    raw = (key or "").strip() or DEFAULT_KEY
    data = raw.encode("utf-8")
    return data if data else DEFAULT_KEY.encode("utf-8")


def _hash_bytes_to_seed(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & _U32
    return h or 1


def _create_prng(seed: int) -> Callable[[], int]:
    state = (seed & _U32) or 1

    def next_uint() -> int:
        nonlocal state
        state ^= (state << 13) & _U32
        state ^= (state >> 17) & _U32
        state ^= (state << 5) & _U32
        return state & _U32

    return next_uint


def _build_s_boxes(key_bytes: bytes, salt: int) -> tuple[list[int], list[int]]:
    sbox = list(range(256))
    prng = _create_prng(_hash_bytes_to_seed(key_bytes) ^ (salt & _U32))

    for i in range(255, 0, -1):
        j = prng() % (i + 1)
        sbox[i], sbox[j] = sbox[j], sbox[i]

    inv = [0] * 256
    for i, v in enumerate(sbox):
        inv[v] = i
    return sbox, inv


def _expand_keystream(key_bytes: bytes, length: int, salt: int) -> list[int]:
    stream = [0] * length
    prng = _create_prng(_hash_bytes_to_seed(key_bytes) ^ (salt & _U32) ^ 0x4B455953)
    for i in range(length):
        stream[i] = prng() & 0xFF
    return stream


def _rot_left(n: int, bits: int) -> int:
    b = bits & 7
    return ((n << b) | (n >> (8 - b))) & 0xFF


def _rot_right(n: int, bits: int) -> int:
    b = bits & 7
    return ((n >> b) | (n << (8 - b))) & 0xFF


def _forward_round_byte(b: int, i: int, round_: int, sbox: list[int], ks: list[int], ks_len: int) -> int:
    b = sbox[b]
    b ^= ks[(i + round_ * 31) % ks_len]
    b = _rot_left(b, ((i + round_) % 7) + 1)
    b = (b + ks[(i * 3 + round_ * 17) % ks_len]) & 0xFF
    b ^= ks[(i + round_ * 13) % ks_len]
    return b


def _backward_round_byte(b: int, i: int, round_: int, inv: list[int], ks: list[int], ks_len: int) -> int:
    b ^= ks[(i + round_ * 13) % ks_len]
    b = (b - ks[(i * 3 + round_ * 17) % ks_len]) & 0xFF
    b = _rot_right(b, ((i + round_) % 7) + 1)
    b ^= ks[(i + round_ * 31) % ks_len]
    b = inv[b]
    return b


def _checksum_byte(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
    return c & 0xFF


def _pack_payload(plain_bytes: bytes) -> bytes:
    body = bytearray(4 + len(plain_bytes))
    body[0] = HEADER_MAGIC[0]
    body[1] = HEADER_MAGIC[1]
    body[2] = VERSION
    body[3] = _checksum_byte(plain_bytes)
    body[4:] = plain_bytes
    return bytes(body)


def _unpack_payload(frame: bytes) -> bytes:
    if len(frame) < 4:
        raise ValueError("custom_cipher_invalid_payload")
    if frame[0] != HEADER_MAGIC[0] or frame[1] != HEADER_MAGIC[1]:
        raise ValueError("custom_cipher_invalid_magic")
    if frame[2] != VERSION:
        raise ValueError("custom_cipher_unsupported_version")
    plain = frame[4:]
    if _checksum_byte(plain) != frame[3]:
        raise ValueError("custom_cipher_checksum_failed")
    return plain


def _chain_forward(data: bytes, ks: list[int]) -> bytes:
    out = bytearray(len(data))
    prev = ks[0]
    ks_len = len(ks)
    for i, b in enumerate(data):
        out[i] = (b ^ prev ^ ks[i % ks_len]) & 0xFF
        prev = out[i]
    return bytes(out)


def _chain_backward(data: bytes, ks: list[int]) -> bytes:
    out = bytearray(len(data))
    prev = ks[0]
    ks_len = len(ks)
    for i, b in enumerate(data):
        out[i] = (b ^ prev ^ ks[i % ks_len]) & 0xFF
        prev = b
    return bytes(out)


def _transform_forward(data: bytes, key_bytes: bytes) -> bytes:
    length = len(data)
    ks = _expand_keystream(key_bytes, length, 0x53545231)
    sbox, _ = _build_s_boxes(key_bytes, 0x534F5801)
    work = bytearray(data)

    for round_ in range(ROUNDS):
        for i in range(length):
            work[i] = _forward_round_byte(work[i], i, round_, sbox, ks, length)

    return _chain_forward(bytes(work), ks)


def _transform_backward(data: bytes, key_bytes: bytes) -> bytes:
    length = len(data)
    ks = _expand_keystream(key_bytes, length, 0x53545231)
    _, inv = _build_s_boxes(key_bytes, 0x534F5801)
    work = bytearray(_chain_backward(data, ks))

    for round_ in range(ROUNDS - 1, -1, -1):
        for i in range(length):
            work[i] = _backward_round_byte(work[i], i, round_, inv, ks, length)

    return bytes(work)


def _bytes_to_hex(data: bytes) -> str:
    return data.hex()


def _hex_to_bytes(hex_text: str) -> bytes:
    text = (hex_text or "").strip()
    if not text:
        return b""
    if not re.fullmatch(r"[0-9a-fA-F]*", text) or len(text) % 2 != 0:
        raise ValueError("custom_cipher_invalid_hex")
    return bytes.fromhex(text)


def encode(plain_text: str, key: str | None = None) -> str:
    """明文 → 十六进制密文"""
    plain_bytes = (plain_text or "").encode("utf-8")
    key_bytes = _normalize_key_bytes(key)
    frame = _pack_payload(plain_bytes)
    cipher = _transform_forward(frame, key_bytes)
    return _bytes_to_hex(cipher)


def decode(cipher_hex: str, key: str | None = None) -> str:
    """十六进制密文 → 明文"""
    cipher = _hex_to_bytes(cipher_hex)
    key_bytes = _normalize_key_bytes(key)
    frame = _transform_backward(cipher, key_bytes)
    plain_bytes = _unpack_payload(frame)
    return plain_bytes.decode("utf-8")


class CustomCipher:
    """与 JS `customCipher` 对象同名接口"""

    encode = staticmethod(encode)
    decode = staticmethod(decode)


custom_cipher = CustomCipher()


if __name__ == "__main__":
    samples = ["", "hello", "你好abc123"]
    secret = "secret-key"
    for sample in samples:
        encrypted = encode(sample, secret)
        decrypted = decode(encrypted, secret)
        ok = decrypted == sample
        print(f"{sample!r} -> {encrypted[:32]}... -> {decrypted!r} [{ok}]")
