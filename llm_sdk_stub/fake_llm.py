from __future__ import annotations
import json
import random
from pathlib import Path
from src.vocab_loader import _bytes_to_unicode

_STUB_DIR = Path(__file__).parent
_VOCAB_PATH = _STUB_DIR / "fake_vocab.json"
_TOKEN_TEXTS = [
    "{", "}", '"', ":", ",", " ",
    "fn_name", "fn_add_numbers", "fn_reverse_string", "args",
    "a", "b", "s", ".",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "true", "false",
    " numbers",  # con espacio inicial, a propósito
]


def _build_fake_vocab_file() -> None:
    """Genera fake_vocab.json con el mismo formato byte-level que el real."""
    byte_encoder = _bytes_to_unicode()
    vocab: dict[str, int] = {}
    for i, text in enumerate(_TOKEN_TEXTS):
        encoded = "".join(byte_encoder[b] for b in text.encode("utf-8"))
        vocab[encoded] = i
    with _VOCAB_PATH.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


class _FakeTensor:
    def __init__(self, data: list) -> None:
        self._data = data

    def __getitem__(self, idx: int) -> "_FakeTensor":
        return _FakeTensor(self._data[idx])

    def tolist(self) -> list:
        return self._data


class Small_LLM_Model:
    """Imitación de llm_sdk.Small_LLM_Model para pruebas locales."""

    def __init__(self, model_name: str = "fake-model", seed: int = 42) -> None:
        if not _VOCAB_PATH.exists():
            _build_fake_vocab_file()
        with _VOCAB_PATH.open("r", encoding="utf-8") as f:
            self._raw_vocab: dict[str, int] = json.load(f)
        self._id_to_encoded: dict[int, str] = {
            v: k for k, v in self._raw_vocab.items()
        }
        self._vocab_size = len(self._raw_vocab)
        self._rng = random.Random(seed)

        self._text_to_id = {text: i for i, text in enumerate(_TOKEN_TEXTS)}

    def get_path_to_vocab_file(self) -> str:
        return str(_VOCAB_PATH)

    def get_path_to_merges_file(self) -> str:
        raise NotImplementedError("El stub de juguete no simula merges.txt")

    def get_path_to_tokenizer_file(self) -> str:
        raise NotImplementedError(
            "El stub de juguete no simula tokenizer.json")

    def encode(self, text: str) -> _FakeTensor:
        ids: list[int] = []
        i = 0
        pieces = sorted(self._text_to_id.keys(), key=len, reverse=True)
        while i < len(text):
            for piece in pieces:
                if text.startswith(piece, i):
                    ids.append(self._text_to_id[piece])
                    i += len(piece)
                    break
            else:
                raise ValueError(f"No se pudo tokenizar en la posición {i}")
        return _FakeTensor([ids])

    def decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(_TOKEN_TEXTS[i] for i in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Logits ruidosos para el siguiente token (solo, como el SDK real)."""
        return [self._rng.uniform(-1.0, 1.0) for _ in range(self._vocab_size)]
