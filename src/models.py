"""Modelos pydantic que representan el esquema del proyecto.

Estos modelos cumplen dos papeles:
1. Validar los archivos de entrada (function_definitions.json).
2. Servir de "fuente de verdad" para construir la máquina de estados
   de decodificación restringida (sabemos qué claves y tipos esperar
   sin tener que volver a parsear JSON crudo en cada paso).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Tipos de parámetro soportados por el esquema de funciones.
ParamType = Literal["number", "string", "boolean"]


class ParameterSpec(BaseModel):
    """Especificación de un único parámetro de una función."""

    type: ParamType


class ReturnSpec(BaseModel):
    """Especificación del tipo de retorno de una función."""

    type: ParamType


class FunctionDefinition(BaseModel):
    """Una función que el sistema puede elegir llamar."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("El nombre de la función no puede estar vacío")
        return v


class FunctionCallResult(BaseModel):
    """Una entrada del archivo de salida function_calling_results.json."""

    prompt: str
    fn_name: str
    args: dict[str, object]
