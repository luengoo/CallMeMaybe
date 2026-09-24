"""Envoltorio tipado sobre llm_sdk.Small_LLM_Model."""

from __future__ import annotations

from src.vocab_loader import build_id_to_str

try:
    from llm_sdk import Small_LLM_Model
except ImportError as exc:
    raise ImportError(
        "No se encontró el paquete llm_sdk. Debe estar copiado en la "
        "raíz del proyecto (junto a src/) e instalado vía uv "
        "(ver pyproject.toml)."
    ) from exc


class LLMClient:
    """Expone solo lo que necesita el decodificador, con tipos Python."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Carga el modelo y construye el mapa id -> texto del vocabulario."""
        self._sdk = Small_LLM_Model(model_name=model_name)
        self.id_to_str: dict[int, str] = build_id_to_str(self._sdk)
        self.vocab_size = len(self.id_to_str)

    def encode(self, text: str) -> list[int]:
        """Tokeniza `text` y devuelve la lista de ids."""
        ids_tensor = self._sdk.encode(text)
        # tolist() de torch está tipado como Any: convertimos explícitamente.
        return [int(t) for t in ids_tensor[0].tolist()]

    def decode(self, token_ids: list[int]) -> str:
        """Convierte una lista de ids de vuelta a texto."""
        return str(self._sdk.decode(token_ids))

    def next_token_logits(self, input_ids: list[int]) -> list[float]:
        """Devuelve los logits del siguiente token tras `input_ids`."""
        logits = self._sdk.get_logits_from_input_ids(input_ids)
        return [float(x) for x in logits]
