from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class SDKLike(Protocol):
    def get_path_to_vocab_file(self) -> str: ...


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def build_id_to_str(sdk: SDKLike) -> dict[int, str]:
    vocab_path = Path(sdk.get_path_to_vocab_file())
    with vocab_path.open("r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)

    byte_encoder = _bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    id_to_str: dict[int, str] = {}
    for encoded_token, token_id in raw_vocab.items():
        raw_bytes = bytearray(byte_decoder[ch] for ch in encoded_token)
        id_to_str[token_id] = raw_bytes.decode("utf-8", errors="replace")

    return id_to_str
