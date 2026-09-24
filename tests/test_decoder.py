"""Tests del decodificador restringido con un modelo "guionizado".

El modelo falso devuelve una secuencia de tokens fijada de antemano, así
que los tests son deterministas y no necesitan descargar Qwen.

Uso:  uv run pytest tests/test_decoder.py -v
"""

from __future__ import annotations

import pytest

from src.constrained_decoder import (
    choose_from_candidates,
    generate_number,
    generate_string,
)


class ScriptedLLM:
    """Imita la interfaz del cliente real (id_to_str, encode, logits).

    Como el modelo real, el siguiente token depende solo de la posición
    (len(ids)): pedir un token "por adelantado" no consume nada. Cuando
    se acaba la secuencia, devuelve siempre "}".
    """

    def __init__(self, tokens: list[str]) -> None:
        """Guarda la secuencia y construye un vocabulario que la cubre."""
        self.seq = tokens
        vocab = sorted(set(tokens) | {"}"} | set("0123456789-."))
        self.id_to_str: dict[int, str] = dict(enumerate(vocab))
        self._ids = {t: i for i, t in self.id_to_str.items()}

    def encode(self, text: str) -> list[int]:
        """Un único id de relleno: el contenido del prompt no importa."""
        return [0]

    def next_token_logits(self, input_ids: list[int]) -> list[float]:
        """Da la puntuación máxima al token que toca en esta posición."""
        k = len(input_ids) - 1
        token = self.seq[k] if k < len(self.seq) else "}"
        logits = [0.0] * len(self.id_to_str)
        logits[self._ids[token]] = 1.0
        return logits


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["shre", "k", '",'], "shrek"),
        (['"'], ""),
        (["abc", '"'], "abc"),
        (["Zoë", '"'], "Zoë"),
        (["O'Brien", '"}'], "O'Brien"),
        (["[aeiou", 'AEIOU]"'], "[aeiouAEIOU]"),
        # escapes JSON correctos e incorrectos
        (["\\\\d", "+", '"'], "\\d+"),
        (["\\", "d", "+", '"}'], "\\d+"),
        (['say \\"hi\\"', '"'], 'say "hi"'),
        # comillas internas sin escapar
        (["He said ", '"hi', '"', '"', "}"], 'He said "hi"'),
        (["He said ", '"hi', '""}'], 'He said "hi"'),
        (["He said ", '"hi"', '",'], 'He said "hi"'),
        (['a"b', '"', ","], 'a"b'),
    ],
)
def test_generate_string(tokens: list[str], expected: str) -> None:
    """El contenido generado es el esperado y ya viene desescapado."""
    assert generate_string(ScriptedLLM(tokens), [0]) == expected


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ([" -", "2", ","], -2.0),
        ([" 3", ".", "5", ","], 3.5),
        (["1", "2", "3", "}"], 123.0),
        ([" -", "0", ".", "5", "}"], -0.5),
        (["9"] * 20 + [","], float("9" * 20)),
    ],
)
def test_generate_number(tokens: list[str], expected: float) -> None:
    """Números con signo, decimales, espacio inicial y muchos dígitos."""
    assert generate_number(ScriptedLLM(tokens), [0]) == expected


def test_choose_from_candidates_only_returns_a_candidate() -> None:
    """Aunque el modelo prefiera otra cosa, solo sale un candidato."""
    llm = ScriptedLLM(["fn_", "gr", "eet", "zzz"])
    result = choose_from_candidates(
        llm, [0], candidates=["fn_add_numbers", "fn_greet"]
    )
    assert result in ("fn_add_numbers", "fn_greet")
