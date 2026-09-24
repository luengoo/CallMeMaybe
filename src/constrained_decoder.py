from __future__ import annotations

import json
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


def _number_piece(generated: str, token: str) -> str | None:
    """Devuelve los caracteres que `token` aporta al número, o None si
    no es una continuación válida.

    El prompt termina en '":' (sin espacio), igual que en un JSON
    normal, así que el primer token del valor suele traer el espacio
    pegado: ' 3', ' -'. Solo en esa primera posición aceptamos un
    espacio inicial y lo descartamos.
    """
    if generated == "" and token.startswith(" "):
        token = token[1:]
    if _is_valid_number_continuation(generated, token):
        return token
    return None


def generate_number(
    llm: LLMClientLike,
    prompt_ids: list[int],
    max_extra_tokens: int = 50,
) -> float:
    """Genera un número JSON (int o decimal) token a token.

    Mientras el texto acumulado todavía no es un número válido por sí
    mismo (p. ej. "", "-", o "3."), se fuerza una continuación válida
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
            piece = _number_piece(generated, llm.id_to_str.get(best_id, ""))
            if piece is None:
                break  # el modelo prefiere salir de la gramática: fin
            current_ids.append(best_id)
            generated += piece
            continue

        allowed_mask = np.full(logits.shape, False)
        for token_id, token_str in llm.id_to_str.items():
            if token_id >= len(logits):
                continue
            if _number_piece(generated, token_str) is not None:
                allowed_mask[token_id] = True

        if not allowed_mask.any():
            raise ValueError(
                f"Ningún token permite continuar el número '{generated}'"
            )

        masked_logits = np.where(allowed_mask, logits, -np.inf)
        next_token_id = int(np.argmax(masked_logits))
        piece = _number_piece(generated, llm.id_to_str[next_token_id])
        assert piece is not None  # garantizado por la máscara
        current_ids.append(next_token_id)
        generated += piece

    if not _NUMBER_RE.fullmatch(generated):
        raise ValueError(
            f"No se generó un número JSON válido tras {max_extra_tokens} "
            f"tokens. Texto acumulado: '{generated}'"
        )

    return float(generated)


# Caracteres que pueden seguir a "\" dentro de un string JSON.
# (Dejamos fuera \uXXXX para no tener que validar 4 dígitos hex.)
_SIMPLE_ESCAPES = set('"\\/bfnrt')


def _quote_closes_string(
    llm: LLMClientLike, ids: list[int], token_id: int, rest: str
) -> bool:
    """Decide si una comilla sin escapar cierra el string o es parte
    del contenido (los modelos pequeños no siempre escapan las comillas
    internas, p. ej. 'He said "hi"').

    La regla es la de la propia gramática JSON: tras la comilla de
    cierre de un valor solo puede venir "," o "}" (con espacios o
    saltos de línea opcionales).
      - Si el token trae texto después de la comilla ('",', '"}',
        '"hi'), lo miramos directamente.
      - Si la comilla es lo último del token, pedimos al modelo un
        token más (sin consumirlo) y miramos qué quiere escribir. Como
        la generación es greedy, ese token es justo el que saldría en
        el paso siguiente, así que la decisión es coherente.
    """
    after = rest
    if not after.strip():
        logits = np.asarray(llm.next_token_logits(ids + [token_id]))
        after = llm.id_to_str.get(int(np.argmax(logits)), "")
    after = after.lstrip(" \t\r\n")
    return after == "" or after[0] in ",}"


def generate_string(
    llm: LLMClientLike,
    prompt_ids: list[int],
    max_extra_tokens: int = 200,
) -> str:
    """Genera el CONTENIDO de un string JSON.

    El prompt debe terminar YA con la comilla de apertura `"`: así el
    modelo empieza directamente por el contenido.

    En cada paso tomamos el token más probable y lo recorremos carácter
    a carácter respetando la gramática de strings JSON:
      - Escapes válidos (`\\\\`, `\\"`, `\\n`...) se aceptan, así que
        valores como la regex `\\d+` se pueden generar. Un escape
        inválido como `\\d` se interpreta como una barra literal.
      - Una comilla sin escapar puede ser el cierre o una comilla
        interna que el modelo no escapó: lo decide
        _quote_closes_string. Si es interna, la guardamos escapada.
      - Un carácter de control termina el string.
    Al acabar se decodifica con json.loads, así que el valor devuelto
    ya está "desescapado" y listo para meter en el resultado.

    Returns:
        El contenido del string (sin comillas, sin escapes JSON).
    """
    raw = ""
    escaping = False
    finished = False
    current_ids = list(prompt_ids)

    for _ in range(max_extra_tokens):
        logits = np.asarray(llm.next_token_logits(current_ids))
        best_id = int(np.argmax(logits))
        token = llm.id_to_str.get(best_id, "")
        if token == "":
            break

        for pos, ch in enumerate(token):
            if escaping:
                if ch not in _SIMPLE_ESCAPES:
                    raw += "\\"  # "\d" -> barra literal + "d"
                raw += ch
                escaping = False
            elif ch == '"':
                if _quote_closes_string(
                    llm, current_ids, best_id, token[pos + 1:]
                ):
                    finished = True
                    break
                raw += '\\"'
            elif ord(ch) < 0x20:
                finished = True
                break
            else:
                raw += ch
                escaping = ch == "\\"

        if finished:
            break
        current_ids.append(best_id)

    if escaping:  # se acabaron los tokens a mitad de un escape
        raw = raw[:-1]

    result = json.loads(f'"{raw}"')
    assert isinstance(result, str)
    return result
