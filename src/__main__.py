from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.function_call_generator import (
    FunctionCallGenerationError,
    generate_function_call,
)
from src.io_utils import (
    InputLoadError,
    load_function_definitions,
    load_prompts,
    write_results,
)
from src.llm_client import LLMClient
from src.models import FunctionCallResult

DEFAULT_TESTS_PATH = Path("data/input/function_calling_tests.json")
DEFAULT_DEFINITIONS_PATH = Path("data/input/functions_definition.json")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Traduce prompts en lenguaje natural a llamadas a función "
            "estructuradas usando decodificación restringida."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_DEFINITIONS_PATH,
        help=(
            "Ruta al archivo JSON de definiciones de función "
            f"(por defecto: {DEFAULT_DEFINITIONS_PATH})"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TESTS_PATH,
        help=(
            "Ruta al archivo JSON de prompts a procesar "
            f"(por defecto: {DEFAULT_TESTS_PATH})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Ruta del archivo JSON de salida "
            f"(por defecto: {DEFAULT_OUTPUT_PATH})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        prompts = load_prompts(args.input)
        definitions = load_function_definitions(args.functions_definition)
    except InputLoadError as exc:
        print(
            f"Error al cargar los archivos de entrada: {exc}",
            file=sys.stderr,
        )
        return 1

    if not prompts:
        print(
            "No hay prompts que procesar. Nada que hacer.", file=sys.stderr
        )
        return 0

    print(
        "Cargando el modelo LLM (esto puede tardar un poco)...",
        file=sys.stderr,
    )
    try:
        llm = LLMClient()
    except Exception as exc:  # noqa: BLE001 - fallo de carga = fatal
        print(f"Error al cargar el modelo LLM: {exc}", file=sys.stderr)
        return 1

    results: list[FunctionCallResult] = []
    failed = 0
    start = time.monotonic()

    for i, prompt in enumerate(prompts, start=1):
        try:
            result = generate_function_call(llm, prompt, definitions)
            results.append(result)
            print(
                f"[{i}/{len(prompts)}] OK: {prompt!r} -> {result.name}",
                file=sys.stderr,
            )
        except FunctionCallGenerationError as exc:
            failed += 1
            print(
                f"[{i}/{len(prompts)}] FALLO en {prompt!r}: {exc}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001 - nunca debe crashear
            failed += 1
            print(
                f"[{i}/{len(prompts)}] ERROR INESPERADO en {prompt!r}: {exc}",
                file=sys.stderr,
            )

    elapsed = time.monotonic() - start

    write_results(args.output, [r.model_dump() for r in results])

    print(
        f"\nCompletado en {elapsed:.1f}s. "
        f"{len(results)} OK, {failed} fallidos, de {len(prompts)} prompts. "
        f"Resultado en {args.output}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
