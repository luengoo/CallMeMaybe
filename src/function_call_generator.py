"""Orquesta una llamada a función completa para un prompt dado.

Ata las piezas de constrained_decoder.py:
  - choose_from_candidates para fn_name y para valores boolean/literales
  - generate_number / generate_string para valores libres

Decisión de diseño importante: mantenemos el TEXTO acumulado (no los
input_ids) como fuente de verdad del estado de generación. Antes de
cada sub-generación (elegir fn_name, generar cada argumento) volvemos
a tokenizar ese texto con llm.encode(). Es menos eficiente que ir
arrastrando los mismos input_ids sin volver a tokenizar, pero evita
un bug sutil: cuando insertamos literales JSON nosotros mismos (comas,
llaves, comillas) entre dos generaciones del modelo, concatenar ids
"a pelo" puede no coincidir con cómo el tokenizador BPE fusionaría
esos mismos caracteres si los viera todos juntos desde el principio.
Retokenizar todo el texto acumulado es la forma robusta de evitarlo.
"""

from __future__ import annotations

import json

from src.constrained_decoder import (
    LLMClientLike,
    choose_from_candidates,
    generate_number,
    generate_string,
)
from src.models import FunctionCallResult, FunctionDefinition

_BOOLEAN_CANDIDATES = ["true", "false"]


class FunctionCallGenerationError(Exception):
    """Error controlado al generar una llamada a función para un prompt."""


def _build_context_prompt(
    user_prompt: str, definitions: list[FunctionDefinition]
) -> str:
    """Construye el texto que le da al modelo el contexto necesario:
    qué funciones existen y qué se le pide. Termina justo antes de
    donde el modelo debe empezar a rellenar el JSON."""
    lines = ["Available functions:"]
    for d in definitions:
        params_desc = ", ".join(
            f"{name}: {spec.type}" for name, spec in d.parameters.items()
        )
        lines.append(f"- {d.name}({params_desc}): {d.description}")
    functions_block = "\n".join(lines)

    return (
        f"{functions_block}\n\n"
        f'User request: "{user_prompt}"\n\n'
        "Respond with a single JSON object with keys fn_name and args, "
        "choosing the correct function and arguments for the request.\n"
    )


def generate_function_call(
    llm: LLMClientLike,
    prompt: str,
    definitions: list[FunctionDefinition],
) -> FunctionCallResult:
    """Genera una llamada a función completa para un prompt.

    Args:
        llm: cliente LLM (real o compatible) con encode/next_token_logits/
            id_to_str.
        prompt: el prompt en lenguaje natural del usuario.
        definitions: las funciones disponibles entre las que elegir.

    Returns:
        Un FunctionCallResult con fn_name y args ya rellenos.

    Raises:
        FunctionCallGenerationError: si algo falla durante la
            generación (tipo de parámetro no soportado, la
            decodificación restringida no converge, etc.).
    """
    if not definitions:
        raise FunctionCallGenerationError(
            "No hay definiciones de función disponibles"
        )

    text = _build_context_prompt(prompt, definitions) + '{"fn_name": "'

    try:
        fn_name_ids = llm.encode(text)
        fn_name = choose_from_candidates(
            llm, fn_name_ids, candidates=[d.name for d in definitions]
        )
    except ValueError as exc:
        raise FunctionCallGenerationError(
            f"No se pudo elegir fn_name para el prompt {prompt!r}: {exc}"
        ) from exc

    fn_def = next((d for d in definitions if d.name == fn_name), None)
    if fn_def is None:  # pragma: no cover - no debería pasar nunca
        raise FunctionCallGenerationError(
            f"El modelo eligió fn_name={fn_name!r}, que no existe en "
            f"las definiciones disponibles"
        )

    text += fn_name + '", "args": {'

    args: dict[str, object] = {}
    param_items = list(fn_def.parameters.items())

    for i, (param_name, param_spec) in enumerate(param_items):
        text += f'"{param_name}": '

        try:
            value_ids = llm.encode(text)

            if param_spec.type == "number":
                value: object = generate_number(llm, value_ids)
            elif param_spec.type == "string":
                value = generate_string(llm, value_ids)
            elif param_spec.type == "boolean":
                bool_str = choose_from_candidates(
                    llm, value_ids, candidates=_BOOLEAN_CANDIDATES
                )
                value = bool_str == "true"
            else:  # pragma: no cover - protegido por pydantic Literal
                raise FunctionCallGenerationError(
                    f"Tipo de parámetro no soportado: {param_spec.type!r}"
                )
        except ValueError as exc:
            raise FunctionCallGenerationError(
                f"No se pudo generar el argumento '{param_name}' "
                f"({param_spec.type}) para el prompt {prompt!r}: {exc}"
            ) from exc

        args[param_name] = value
        # json.dumps nos da la representación JSON correcta del valor
        # (comillas para strings, true/false para booleanos, etc.) sin
        # tener que reconstruirla a mano.
        text += json.dumps(value)
        if i < len(param_items) - 1:
            text += ", "

    text += "}}"

    return FunctionCallResult(prompt=prompt, fn_name=fn_name, args=args)
