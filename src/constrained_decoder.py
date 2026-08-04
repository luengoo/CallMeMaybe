from __future__ import annotations

import re
from typing import Protocol

import numpy as np


class LLMClientLike(Protocol):

    id_to_str: dict[int, str]

    def encode(self, text: str) -> list[int]: ...
    def next_token_logits(self, input_ids: list[int]) -> list[float]: ...


def choose_from_candidates(
    llm: LLMClientLike,
    prompt_ids: list[int],
    candidates: list[str],
    max_extra_tokens: int = 20,
) -> str:
    if not candidates:
        raise ValueError("La lista de candidatos no puede estar vacía")

    generated = ""
    current_ids = list(prompt_ids)

    for _ in range(max_extra_tokens):
        if generated in candidates:
            if not any(
                c != generated and c.startswith(generated) for c in candidates
            ):
                return generated

        logits = np.asarray(llm.next_token_logits(current_ids))

        allowed_mask = np.full(logits.shape, False)
        for token_id, token_str in llm.id_to_str.items():
            if token_id >= len(logits):
                continue
            candidate_text = generated + token_str
            if any(c.startswith(candidate_text) for c in candidates):
                allowed_mask[token_id] = True

        if not allowed_mask.any():
            raise ValueError(
                f"Ningún token del vocabulario permite continuar "
                f"'{generated}' hacia ninguno de los candidatos {candidates}"
            )

        masked_logits = np.where(allowed_mask, logits, -np.inf)
        next_token_id = int(np.argmax(masked_logits))

        current_ids.append(next_token_id)
        generated += llm.id_to_str[next_token_id]

    raise ValueError(
        f"Se superó max_extra_tokens sin converger a un candidato exacto. "
        f"Texto generado hasta ahora: '{generated}'"
    )


# --------------------------------------------------------------------------
# Generación de valores libres: number y string.
#
# A diferencia de choose_from_candidates, aquí no hay una lista cerrada
# de opciones válidas. Usamos el mismo patrón (pedir logits -> decidir
# qué tokens son válidos -> elegir -> repetir) pero con una gramática
# character-level en vez de una lista de strings completos.
# --------------------------------------------------------------------------

_DIGITS = set("0123456789")

_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _number_state(text: str) -> tuple[set[str], bool]:
    """Dado el texto de un número generado hasta ahora, devuelve
    (conjunto de próximos caracteres válidos, si el texto ya es un
    número JSON completo y válido por sí mismo)."""
    if text == "":
        return ({"-"} | _DIGITS, False)
    if text == "-":
        return (_DIGITS, False)

    body = text[1:] if text.startswith("-") else text

    if "." not in body:
        if body == "0":
            # evitamos ceros a la izquierda tipo "007": tras un "0"
            # solo se permite pasar a la parte decimal.
            return ({"."}, True)
        return (_DIGITS | {"."}, True)

    int_part, frac_part = body.split(".", 1)
    if frac_part == "":
        # acabamos de poner el punto, hace falta al menos un dígito
        return (_DIGITS, False)
    return (_DIGITS, True)


def _is_valid_number_continuation(text: str, addition: str) -> bool:
    """¿Sigue siendo un prefijo válido de número si añadimos `addition`
    (posiblemente varios caracteres, como haría un token BPE) a `text`?"""
    if addition == "":
        return False
    current = text
    for ch in addition:
        allowed_chars, _ = _number_state(current)
        if ch not in allowed_chars:
            return False
        current += ch
    return True


def generate_number(
    llm: LLMClientLike,
    prompt_ids: list[int],
    max_extra_tokens: int = 15,
) -> float:
    """Genera un número JSON (int o decimal) token a token.

    Mientras el texto acumulado todavía no es un número válido por sí
    mismo (p. ej. "-", o "3."), se fuerza una continuación válida
    (enmascarado duro, igual que choose_from_candidates). En cuanto el
    texto ya ES un número válido, se deja que el modelo decida
    libremente si quiere seguir extendiéndolo o parar: si su elección
    natural (sin restringir) ya no encaja en la gramática de números,
    interpretamos eso como la señal de que ha terminado.

    Returns:
        El número generado, ya convertido a float.

    Raises:
        ValueError: si se alcanza max_extra_tokens sin producir un
            número válido completo.
    """
    generated = ""
    current_ids = list(prompt_ids)

    for _ in range(max_extra_tokens):
        logits = np.asarray(llm.next_token_logits(current_ids))
        _, terminal = _number_state(generated)

        if terminal:
            best_id = int(np.argmax(logits))
            best_text = llm.id_to_str.get(best_id, "")
            if _is_valid_number_continuation(generated, best_text):
                current_ids.append(best_id)
                generated += best_text
                continue
            break  # el modelo prefiere algo fuera de la gramática -> paramos

        allowed_mask = np.full(logits.shape, False)
        for token_id, token_str in llm.id_to_str.items():
            if token_id >= len(logits):
                continue
            if _is_valid_number_continuation(generated, token_str):
                allowed_mask[token_id] = True

        if not allowed_mask.any():
            raise ValueError(
                f"Ningún token permite continuar el número '{generated}'"
            )

        masked_logits = np.where(allowed_mask, logits, -np.inf)
        next_token_id = int(np.argmax(masked_logits))
        current_ids.append(next_token_id)
        generated += llm.id_to_str[next_token_id]

    if not _NUMBER_RE.fullmatch(generated):
        raise ValueError(
            f"No se generó un número JSON válido tras {max_extra_tokens} "
            f"tokens. Texto acumulado: '{generated}'"
        )

    return float(generated)


_UNSAFE_STRING_CHARS = {'"', "\\"}


def _is_valid_string_token(token_str: str) -> bool:
    """Un token es válido como contenido de string si no puede romper
    el JSON: sin comillas sin escapar, sin backslash, sin caracteres
    de control (saltos de línea, tabs, etc.)."""
    if token_str == "":
        return False
    if any(ch in _UNSAFE_STRING_CHARS for ch in token_str):
        return False
    if any(ord(ch) < 0x20 for ch in token_str):
        return False
    return True


def generate_string(
    llm: LLMClientLike,
    prompt_ids: list[int],
    max_extra_tokens: int = 40,
) -> str:
    """Genera el CONTENIDO de un string JSON (sin las comillas, que
    las pone quien orqueste el JSON completo alrededor).

    Un string puede terminar en cualquier momento (incluso vacío), así
    que no hay un "estado no terminal" que forzar como en los números:
    en cada paso dejamos que el modelo elija libremente (sin máscara).
    Si su elección natural es seguro para JSON (no contiene comillas,
    backslash ni caracteres de control), la aceptamos y seguimos. En
    cuanto el modelo "quiere" producir algo inseguro -típicamente la
    comilla de cierre-, lo tomamos como señal de que el string ha
    terminado y paramos ahí, sin consumir ese token.

    Returns:
        El contenido del string (sin comillas).
    """
    generated = ""
    current_ids = list(prompt_ids)

    for _ in range(max_extra_tokens):
        logits = np.asarray(llm.next_token_logits(current_ids))
        best_id = int(np.argmax(logits))
        best_text = llm.id_to_str.get(best_id, "")

        if not _is_valid_string_token(best_text):
            break

        current_ids.append(best_id)
        generated += best_text

    return generated
