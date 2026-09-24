#!/usr/bin/env bash
# Casos de error y casos límite para `python -m src`.
# Uso (desde la raíz del proyecto):  bash tests/error_cases.sh
#
# Parte 1: entradas malformadas. Cada caso debe terminar con un mensaje
#          claro y SIN "Traceback".
# Parte 2: prompts límite con el modelo real (tarda ~1 min). Revisa a ojo
#          los valores generados.

PY=${PY:-uv run python}
T=$(mktemp -d)
trap 'chmod -R u+w "$T"; rm -rf "$T"' EXIT
FAILS=0

run_case() {
    local name=$1; shift
    local out
    out=$($PY -m src "$@" 2>&1)
    local code=$?
    if grep -q "Traceback" <<<"$out"; then
        echo "[CRASH] $name (exit $code)"
        echo "$out" | tail -5 | sed 's/^/        /'
        FAILS=$((FAILS + 1))
    else
        echo "[ok]    $name (exit $code)"
        echo "$out" | tail -1 | sed 's/^/        /'
    fi
}

echo '[{"prompt": '                  > "$T/roto.json"
echo '[]'                            > "$T/vacio.json"
: > "$T/cero_bytes.json"
echo '{"prompt": "hola"}'            > "$T/objeto.json"
echo '[{"texto": "hola"}]'           > "$T/sin_clave.json"
echo '[123]'                         > "$T/numero.json"
printf '\xff\xfe['                   > "$T/binario.json"
mkdir "$T/directorio.json"
echo '[{"name": "f", "description": "d", "parameters": {"a": {"type": "array"}}, "returns": {"type": "number"}}]' > "$T/tipo_raro.json"
echo '[{"name": "f"}]'               > "$T/def_incompleta.json"
echo '[{"prompt": "Greet john"}]'    > "$T/uno.json"
mkdir "$T/solo_lectura" && chmod 555 "$T/solo_lectura"

echo "== Parte 1: entradas malformadas =="
run_case "input inexistente"        --input "$T/no_existe.json"
run_case "input JSON roto"          --input "$T/roto.json"
run_case "input lista vacía"        --input "$T/vacio.json"
run_case "input 0 bytes"            --input "$T/cero_bytes.json"
run_case "input objeto, no array"   --input "$T/objeto.json"
run_case "input sin clave prompt"   --input "$T/sin_clave.json"
run_case "input prompt no string"   --input "$T/numero.json"
run_case "input no UTF-8"           --input "$T/binario.json"
run_case "input es un directorio"   --input "$T/directorio.json"
run_case "defs inexistente"         --functions_definition "$T/no_existe.json"
run_case "defs tipo no soportado"   --functions_definition "$T/tipo_raro.json"
run_case "defs incompleta"          --functions_definition "$T/def_incompleta.json"
run_case "defs lista vacía"         --functions_definition "$T/vacio.json"
run_case "argumento desconocido"    --foo
run_case "output es un directorio"  --input "$T/uno.json" --output "$T/directorio.json"
run_case "output sin permisos"      --input "$T/uno.json" --output "$T/solo_lectura/sub/out.json"

echo
echo "== Parte 2: prompts límite (modelo real) =="
cat > "$T/limite.json" <<'EOF'
[
  {"prompt": "What is the sum of 123456789 and 987654321?"},
  {"prompt": "Add 3.5 and -2"},
  {"prompt": "What is the square root of 0?"},
  {"prompt": "Greet Zoë"},
  {"prompt": "Reverse the string 'He said \"hi\"'"},
  {"prompt": "Reverse the string ''"},
  {"prompt": "Reverse the string 'The quick brown fox jumps over the lazy dog while the cat watches from the old wooden fence'"},
  {"prompt": "Replace all digits in 'a1b2c3' with X"},
  {"prompt": "Say hello to Maria"},
  {"prompt": "add numbers 7 8"}
]
EOF
if $PY -m src --input "$T/limite.json" --output "$T/limite_out.json"; then
    cat "$T/limite_out.json"
else
    echo "[FALLO] la parte 2 terminó con error"
    FAILS=$((FAILS + 1))
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "Todo correcto: ningún caso terminó en traceback."
else
    echo "$FAILS caso(s) con problemas (ver arriba)."
fi
exit "$FAILS"
