"""Wrapper fino sobre llm_sdk.Small_LLM_Model.

El resto del código (constrained_decoder.py) no debería tener que
saber nada de torch.Tensor, ni de la dimensión de batch, ni de cómo
se construye el vocabulario. Esta clase aísla esos detalles del SDK
real para que el resto del proyecto trabaje solo con list[int] /
list[float] / str.

No se usa ningún atributo ni método privado (con guion bajo) de
llm_sdk, solo la interfaz pública documentada.
"""

from __future__ import annotations

from src.vocab_loader import build_id_to_str

try:
    from llm_sdk import Small_LLM_Model
except ImportError as exc:  # pragma: no cover - mensaje de ayuda
    raise ImportError(
        "No se encontró el paquete llm_sdk. Debe estar copiado en la "
        "raíz del proyecto (junto a src/) e instalado vía uv "
        "(ver pyproject.toml)."
    ) from exc


class LLMClient:
    """Interfaz simplificada sobre el SDK real, pensada para generación
    token a token con decodificación restringida."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self._sdk = Small_LLM_Model(model_name=model_name)
        self.id_to_str: dict[int, str] = build_id_to_str(self._sdk)
        self.vocab_size = len(self.id_to_str)

    def encode(self, text: str) -> list[int]:
        """Tokeniza texto a una lista plana de ids (sin dimensión de batch)."""
        ids_tensor = self._sdk.encode(text)
        # encode() del SDK real devuelve shape (1, seq_len)
        return ids_tensor[0].tolist()

    def decode(self, token_ids: list[int]) -> str:
        return self._sdk.decode(token_ids)

    def next_token_logits(self, input_ids: list[int]) -> list[float]:
        """Logits (sin softmax) para el siguiente token, dado el prefijo."""
        return self._sdk.get_logits_from_input_ids(input_ids)
