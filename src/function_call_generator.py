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


def _tool_schema(d: FunctionDefinition) -> str:
    """Serializa una definición al formato JSON Schema de "tools" que
    usan los modelos de chat con function calling (incluido Qwen3)."""
    schema = {
        "type": "function",
        "function": {
            "name": d.name,
            "description": d.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {"type": spec.type}
                    for name, spec in d.parameters.items()
                },
                "required": list(d.parameters),
            },
        },
    }
    return json.dumps(schema, ensure_ascii=False)


def _build_context_prompt(
    user_prompt: str, definitions: list[FunctionDefinition]
) -> str:
    """Construye el prompt con la plantilla de chat de Qwen3 para tools.

    Qwen3 es un modelo de chat entrenado para hacer function calling con
    un formato concreto: las funciones van como JSON Schema dentro de
    <tools>...</tools> en el mensaje de sistema, y la respuesta del
    asistente es un objeto {"name": ..., "arguments": ...} dentro de
    <tool_call>...</tool_call>. Presentarle la tarea en el mismo formato
    que vio en su entrenamiento le ayuda a interpretar mejor cualquier
    petición. El bloque vacío <think></think> equivale a desactivar el
    modo de razonamiento (enable_thinking=False en la plantilla oficial).

    La decodificación restringida sigue siendo la que garantiza la
    estructura: esto solo mejora la calidad de las elecciones del
    modelo dentro de lo que la gramática permite.

    El texto devuelto termina justo donde el modelo debe escribir el
    nombre de la función (tras '{"name": "').
    """
    tools = "\n".join(_tool_schema(d) for d in definitions)
    return (
        "<|im_start|>system\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user "
        "query.\n\n"
        "You are provided with function signatures within <tools></tools> "
        "XML tags:\n"
        f"<tools>\n{tools}\n</tools>\n\n"
        "For each function call, return a json object with function name "
        "and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        "<tool_call>\n"
        '{"name": "'
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
        Un FunctionCallResult con name y parameters ya rellenos.

    Raises:
        FunctionCallGenerationError: si algo falla durante la
            generación (tipo de parámetro no soportado, la
            decodificación restringida no converge, etc.).
    """
    if not definitions:
        raise FunctionCallGenerationError(
            "No hay definiciones de función disponibles"
        )

    text = _build_context_prompt(prompt, definitions)

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

    # En el prompt usamos "arguments" (la clave del formato de Qwen3);
    # en el fichero de salida la clave será "parameters".
    text += fn_name + '", "arguments": {'

    args: dict[str, object] = {}
    param_items = list(fn_def.parameters.items())

    for i, (param_name, param_spec) in enumerate(param_items):
        text += f'"{param_name}": '
        if param_spec.type == "string":
            # La comilla de apertura la ponemos nosotros: así el modelo
            # empieza directamente por el contenido del string.
            text += '"'

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
        encoded = json.dumps(value)
        # Para strings la comilla de apertura ya está en `text`.
        text += encoded[1:] if param_spec.type == "string" else encoded
        if i < len(param_items) - 1:
            text += ", "

    text += "}}"

    return FunctionCallResult(prompt=prompt, name=fn_name, parameters=args)
