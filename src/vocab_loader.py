"""Construye el mapeo token_id -> texto real a partir del vocab.json del SDK.

El SDK solo expone `get_path_to_vocab_file()`, que descarga el vocab.json
original de HuggingFace. Ese archivo mapea "token string" -> id, pero el
"token string" no es texto legible directamente: los tokenizadores estilo
GPT-2/BPE (que Qwen usa) codifican cada BYTE crudo como un carácter
imprimible, para poder representar cualquier byte (incluidos espacios,
saltos de línea, bytes no imprimibles) dentro de un JSON de texto.

Por ejemplo, el token " hola" (con espacio delante) no aparece como
" hola" en vocab.json, sino como "Ġhola", donde "Ġ" es el sustituto
imprimible del byte 0x20 (espacio).

Esta función revierte esa codificación para obtener el texto real que
produce cada token, usando el mapeo estándar bytes<->unicode que definió
el propio GPT-2 (es un algoritmo público y determinista, no depende del
modelo concreto).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class SDKLike(Protocol):
    def get_path_to_vocab_file(self) -> str: ...


def _bytes_to_unicode() -> dict[int, str]:
    """Mapeo estándar byte (0-255) -> carácter unicode imprimible.

    Es el mismo algoritmo que usa el tokenizador original de GPT-2
    (y que reutilizan casi todos los BPE modernos, incluido Qwen).
    Los bytes que ya son "imprimibles" en ASCII/Latin-1 se mapean a
    sí mismos; el resto (espacios, control chars, etc.) se reasignan
    a un rango de caracteres unicode que no colisiona con nada.
    """
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
    """Construye el mapeo completo token_id -> texto real.

    Args:
        sdk: instancia de Small_LLM_Model (o cualquier objeto con
            get_path_to_vocab_file()).

    Returns:
        Diccionario {token_id: texto_decodificado}.
    """
    vocab_path = Path(sdk.get_path_to_vocab_file())
    with vocab_path.open("r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)

    byte_encoder = _bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    id_to_str: dict[int, str] = {}
    for encoded_token, token_id in raw_vocab.items():
        raw_bytes = bytearray(byte_decoder[ch] for ch in encoded_token)
        # errors="replace": algunos tokens BPE son "medio carácter"
        # multibyte (solo forman UTF-8 válido al combinarse con el
        # token vecino). Para nuestro propósito -comparar prefijos de
        # texto ASCII como llaves, comas, dígitos, nombres de función-
        # esto es seguro; esos tokens raros nunca serán "válidos" en
        # nuestra máscara de todos modos.
        id_to_str[token_id] = raw_bytes.decode("utf-8", errors="replace")

    return id_to_str
