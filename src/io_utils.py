"""Lectura y escritura robusta de los archivos JSON de entrada/salida.

Los archivos de entrada pueden no existir o contener JSON inválido
(así lo advierte el enunciado). Aquí centralizamos ese manejo de
errores para que el resto del programa no tenga que preocuparse por
excepciones de bajo nivel.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionDefinition


class InputLoadError(Exception):
    """Error controlado al cargar un archivo de entrada."""


def load_prompts(path: Path) -> list[str]:
    """Carga la lista de prompts en lenguaje natural.

    Args:
        path: Ruta al archivo function_calling_tests.json.

    Returns:
        Lista de strings con los prompts a procesar.

    Raises:
        InputLoadError: si el archivo no existe, no es JSON válido,
            o no tiene la forma esperada (array de strings).
    """
    data = _read_json(path)

    if not isinstance(data, list):
        raise InputLoadError(
            f"{path}: se esperaba un array JSON de prompts, "
            f"se encontró {type(data).__name__}"
        )

    prompts: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, str):
            raise InputLoadError(
                f"{path}: el elemento en la posición {i} no es un string"
            )
        prompts.append(item)

    return prompts


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Carga y valida las definiciones de funciones disponibles.

    Args:
        path: Ruta al archivo function_definitions.json.

    Returns:
        Lista de FunctionDefinition validadas.

    Raises:
        InputLoadError: si el archivo no existe, no es JSON válido,
            o no cumple el esquema esperado.
    """
    data = _read_json(path)

    if not isinstance(data, list):
        raise InputLoadError(
            f"{path}: se esperaba un array JSON de definiciones, "
            f"se encontró {type(data).__name__}"
        )

    definitions: list[FunctionDefinition] = []
    for i, item in enumerate(data):
        try:
            definitions.append(FunctionDefinition.model_validate(item))
        except ValidationError as exc:
            raise InputLoadError(
                f"{path}: la definición en la posición {i} no es válida: {exc}"
            ) from exc

    if not definitions:
        raise InputLoadError(f"{path}: no se encontró ninguna función definida")

    return definitions


def write_results(path: Path, results: list[dict[str, object]]) -> None:
    """Escribe la lista de resultados como JSON válido.

    Crea los directorios intermedios si no existen.

    Args:
        path: Ruta de salida, p. ej. data/output/function_calling_results.json.
        results: Lista de diccionarios ya serializables a JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> object:
    """Lee y parsea un archivo JSON, envolviendo los errores comunes."""
    if not path.exists():
        raise InputLoadError(f"{path}: el archivo no existe")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise InputLoadError(f"{path}: JSON inválido ({exc})") from exc
    except OSError as exc:
        raise InputLoadError(f"{path}: error al leer el archivo ({exc})") from exc
