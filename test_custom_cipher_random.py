"""
500 条随机长度字符串 roundtrip 测试 custom_cipher
运行: python test_custom_cipher_random.py
"""
from __future__ import annotations

import random
import string
import sys

from custom_cipher import decode, encode

CHARSET = (
    string.ascii_letters
    + string.digits
    + "\u4f60\u597d\u4e16\u754c!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
)
KEYS = ["gv", "secret", "test-key-123", "", "long-key-" + "x" * 20]


def main() -> int:
    passed = 0
    failed = 0
    failures: list[tuple] = []

    for n in range(500):
        length = random.randint(0, 256)
        plain = "".join(random.choice(CHARSET) for _ in range(length))
        key = random.choice(KEYS)
        try:
            enc = encode(plain, key)
            dec = decode(enc, key)
            if dec != plain:
                failed += 1
                failures.append((n, "mismatch", length, key, plain[:40], dec[:40]))
            else:
                passed += 1
        except Exception as exc:
            failed += 1
            failures.append((n, str(exc), length, key))

    print({"total": 500, "passed": passed, "failed": failed, "sampleFailures": failures[:5]})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
