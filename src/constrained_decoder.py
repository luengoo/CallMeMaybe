
from __future__ import annotations

from typing import Protocol

import numpy as np


class LLMClientLike(Protocol):
    """Lo mínimo que necesitamos de LLMClient (real o falso) para decodificar."""

    id_to_str: dict[int, str]

    def next_token_logits(self, input_ids: list[int]) -> list[float]: ...


def choose_from_candidates(
    llm: LLMClientLike,
    prompt_ids: list[int],
    candidates: list[str],
    max_extra_tokens: int = 20,
) -> str:
    """Fuerza al modelo a generar exactamente una de las `candidates`.

    En cada paso:
      1. Pide logits al modelo dado el prompt + lo generado hasta ahora.
      2. Para cada token del vocabulario, comprueba si añadirlo mantiene
         el texto acumulado como PREFIJO de al menos un candidato.
      3. Enmascara (-inf) todos los tokens que no cumplen eso.
      4. Elige el token con mayor logit entre los que quedan.
      5. Repite hasta que el texto acumulado coincide EXACTAMENTE con
         uno de los candidatos (y ese candidato no es prefijo estricto
         de ningún otro, para evitar ambigüedad).

    Args:
        llm: LLMClient (o compatible) con next_token_logits() e id_to_str.
        prompt_ids: input_ids del prompt ya tokenizado.
        candidates: lista cerrada de strings válidos (p. ej. nombres
            de función).
        max_extra_tokens: límite de seguridad para evitar bucles
            infinitos si algo en el vocabulario está mal definido.

    Returns:
        El candidato exacto elegido por el modelo.

    Raises:
        ValueError: si candidates está vacío, o si en algún paso no
            queda ningún token válido (vocabulario incompatible con
            los candidatos, lo cual indica un bug, no un caso normal).
    """
    if not candidates:
        raise ValueError("La lista de candidatos no puede estar vacía")

    generated = ""
    current_ids = list(prompt_ids)

    for _ in range(max_extra_tokens):
        # ¿Ya coincide exactamente con un único candidato posible?
        if generated in candidates:
            # Si además ningún otro candidato lo tiene como prefijo
            # estricto, es una elección inequívoca.
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
