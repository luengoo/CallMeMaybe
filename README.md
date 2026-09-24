*This project has been created as part of the 42 curriculum by alluengo.*

# call me maybe

Turning natural-language requests into structured, schema-valid function calls
with a 0.6B-parameter language model and constrained decoding.

## Description

Given a list of available functions (name, description, typed parameters) and a
list of natural-language prompts, the program asks a small language model
(`Qwen/Qwen3-0.6B`) which function should be called for each prompt and with
which arguments, and writes the result as JSON:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The model is never trusted to produce JSON on its own. Instead, the program
builds the JSON itself and only lets the model fill the gaps (the function name
and each argument value), restricting at every step which tokens the model is
allowed to pick. The output is therefore always parseable and always matches
the schema in `functions_definition.json`, whatever the model "wants" to say.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- ~10 GB of free disk space for the virtual environment (PyTorch) and the model

### Installation

```bash
make install        # runs `uv sync`
```

The Makefile checks whether `/sgoinfre/students/$USER` exists. If it does, the
virtual environment, the uv cache and the Hugging Face model cache are placed
there instead of the home directory (42 home directories are small). Otherwise
uv behaves as usual and creates `./.venv`. A plain `uv sync` also works.

To use the same environment outside of `make`:

```bash
eval "$(make -s env)"
```

### Running

```bash
make run
```

which is equivalent to:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All three arguments are optional; the values above are the defaults. The first
run downloads the model (~1.5 GB).

### Other Makefile rules

| Rule               | Description                                              |
|--------------------|----------------------------------------------------------|
| `make debug`       | Run the program under `pdb`                              |
| `make lint`        | `flake8 .` and `mypy .` with the required flags          |
| `make lint-strict` | `flake8 .` and `mypy . --strict`                         |
| `make clean`       | Remove `__pycache__` and `.mypy_cache`                   |
| `make fclean`      | `clean` + remove the virtual environment and caches      |

## Example usage

Input prompts (`function_calling_tests.json`), either as objects or plain strings:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  "Reverse the string 'hello'"
]
```

Run with custom paths:

```bash
uv run python -m src --input my_prompts.json --output /tmp/calls.json
```

Progress is printed to stderr:

```
Cargando el modelo LLM (esto puede tardar un poco)...
[1/2] OK: 'What is the sum of 2 and 3?' -> fn_add_numbers
[2/2] OK: "Reverse the string 'hello'" -> fn_reverse_string

Completado en <seconds>s. 2 OK, 0 fallidos, de 2 prompts. Resultado en /tmp/calls.json
```

Output (`/tmp/calls.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers",
   "parameters": {"a": 2.0, "b": 3.0}},
  {"prompt": "Reverse the string 'hello'", "name": "fn_reverse_string",
   "parameters": {"s": "hello"}}
]
```

Errors in the input files produce a clear message and a non-zero exit code
instead of a traceback:

```
$ uv run python -m src --input missing.json
Error al cargar los archivos de entrada: missing.json: el archivo no existe
```

## Algorithm explanation

### Overview

Language models generate text one token at a time: for a given sequence of
token ids, the model returns one score (logit) per token of its vocabulary, and
the next token is usually the one with the highest score. Constrained decoding
inserts a step between "get the logits" and "pick a token": every token that
would break the expected structure gets its logit set to `-inf`, so it can
never be chosen.

The program does not constrain one big JSON grammar. It writes the fixed parts
of the JSON itself and only asks the model for the variable parts:

```
<context prompt>{"name": "  ← model picks one of the function names
fn_add_numbers", "parameters": {"a":  ← model generates a number
 2, "b":                              ← model generates a number
 3}}
```

After each generated piece the whole text is re-tokenized (see
[Design decisions](#design-decisions)) and the next piece is generated.

### Token-to-text mapping

To know whether a token is allowed, the decoder has to know which text each
token id represents. `vocab_loader.py` reads the model's `vocab.json` (through
the SDK's public `get_path_to_vocab_file()`). Qwen uses byte-level BPE, so
entries are stored in an encoded alphabet (e.g. a space is stored as `Ġ`). The
loader reverses the GPT-2 byte-to-unicode table to recover the real text of
each of the ~150k tokens, producing `id_to_str: dict[int, str]`.

### Choosing from a closed set (function names, booleans)

`choose_from_candidates` generates a string that must be exactly one of a list
of candidates. At each step, a token is allowed only if the text generated so
far plus that token is still a prefix of at least one candidate. The allowed
token with the highest logit is appended, and generation stops as soon as the
text equals a candidate that is not a prefix of another one. With function
names `fn_add_numbers`, `fn_greet`... the model can only ever produce one of
those names: the choice of *which* one is entirely the model's.

### Numbers

`generate_number` follows a small state machine for JSON numbers
(`-?digits(.digits)?`, no leading zeros):

- While the text is not yet a complete number (`""`, `"-"`, `"3."`), only
  tokens that keep it a valid prefix are allowed (hard masking).
- Once the text is a complete number, the model's unconstrained top choice is
  checked: if it still extends a valid number it is accepted, otherwise the
  model "wants" to write something else (a comma, a brace), which is taken as
  the end of the number.

The context ends in `":` without a trailing space, so the first token of the
value naturally carries the space (` 3`, ` -`), exactly as the tokenizer splits
normal JSON. That leading space is accepted and discarded only on the first
token.

### Strings

`generate_string` is called with the opening quote already written. At each
step the model's top token is processed character by character following the
JSON string grammar:

- Normal characters are appended.
- Valid escapes (`\\`, `\"`, `\n`...) are accepted, so values such as the regex
  `\d+` can be produced. An invalid escape such as `\d` is interpreted as a
  literal backslash followed by `d`.
- A control character ends the string.
- An unescaped `"` is either the closing quote or a quote the model forgot to
  escape (`He said "hi"`). The JSON grammar decides: after the closing quote of
  a value only `,` or `}` can follow. If the token contains text after the quote,
  that text is checked; if the quote is the last character of the token, the
  model is asked for one more token (without consuming it) and that token is
  checked. Since decoding is greedy, that look-ahead token is exactly the one
  the next step would produce. Inner quotes are stored escaped.

The raw escaped content is finally decoded with `json.loads`, and the output
file is written with `json.dump`, so the result is valid JSON by construction.

## Design decisions

- **Fixed JSON skeleton, model only fills values.** Keys, braces, commas and
  argument order come from the function definition, not from the model. This
  guarantees that all required arguments are present, that there are no extra
  keys, and that every value has the declared type.
- **Re-tokenize the full text before each generated piece.** Concatenating the
  ids of text inserted by the program with the ids generated by the model can
  produce token sequences that the tokenizer would never produce for the same
  text (BPE merges across the boundary). Re-encoding the whole text is slower
  but keeps the model's input in-distribution.
- **Greedy decoding (argmax).** The task needs the single most likely correct
  answer, not diversity. Greedy decoding is deterministic, which makes the
  results reproducible and makes the string look-ahead exact.
- **A plain-text context prompt.** The prompt lists each function as
  `name(param: type): description`, then the request. The native Qwen3
  tool-calling chat template was also tried; it did not improve accuracy on the
  test prompts and produced a less general regex (`34|233` instead of
  `([0-9]+)`), so the simpler prompt was kept.
- **pydantic models** (`FunctionDefinition`, `ParameterSpec`,
  `FunctionCallResult`) validate the input files and define the output schema.
  Unsupported parameter types are rejected when loading, not during generation.
- **Controlled errors.** Input problems raise `InputLoadError`, output problems
  `OutputWriteError`, generation problems `FunctionCallGenerationError`; `main`
  turns them into a message and an exit code. A failure on one prompt is
  reported and the remaining prompts are still processed.
- **The model choice uses only the LLM.** No keyword matching or heuristics: the
  function is chosen by constrained decoding over the model's logits.

## Performance analysis

Measured on a 42 campus workstation (CPU):

| Test set                               | Correct function | Correct arguments | Valid JSON | Time   |
|----------------------------------------|------------------|-------------------|------------|--------|
| `function_calling_tests.json` (11)     | 11/11            | 10/11             | 100%       | ~97 s  |
| Edge-case prompts (10, see Testing)    | 10/10            | 8/10              | 100%       | ~85 s  |

- **Validity:** 100% of the outputs are parseable and schema-compliant. This is
  guaranteed by construction, not measured luck.
- **Accuracy:** the only error on the provided tests is
  `"Replace all vowels in 'Programming is fun' with asterisks"`, where the model
  produces the correct function, source string and regex but uses `"*****"` as
  replacement (one asterisk per vowel of the sentence) instead of `"*"`. The
  prompt is ambiguous for a 0.6B model and fixing it would require a
  prompt-specific hack.
- **Speed:** well under the 5-minute limit. The main costs are one forward pass
  per generated token (the SDK exposes no KV cache, so each pass processes the
  whole context) and a Python loop over the vocabulary to build masks for
  function names and numbers.

## Challenges faced

- **Empty strings.** In the first version the context ended in `"name": `
  without an opening quote. The model's first choice was the opening `"`, which
  the string generator interpreted as the closing quote, so every string was
  empty. Fixed by writing the opening quote before generating.
- **Regular expressions.** Strings originally stopped at any backslash, so
  `\d+` could never be generated. Fixed by implementing JSON escape handling.
- **Negative numbers.** With the context ending in `": ` (trailing space), the
  model has to continue from a lone space token, something it rarely sees during
  training, and it dropped the minus sign (`-2` became `2.0`). Fixed by ending
  the context in `":` and letting the value's first token carry the space.
- **Unescaped inner quotes.** `'He said "hi"'` was cut at the first inner quote.
  Fixed with the JSON-grammar rule and one-token look-ahead described above.
  When the model itself only writes one quote for both the inner and closing
  quote (`He said "hi"}`), the output is `He said "hi`: the decoder cannot
  invent the missing character.
- **Silent truncation.** Qwen tokenizes digits one by one, and the original
  limit of 15 tokens silently truncated long numbers; strings were limited to
  40 tokens. Limits were raised to 50 and 200; generation still stops as soon as
  the value ends, so this costs nothing in the normal case.
- **Output keys.** The output initially used `fn_name`/`args` instead of the
  required `name`/`parameters`.
- **mypy and the SDK layout.** `mypy .` resolved `llm_sdk` to the outer folder
  (a namespace package without the class). Fixed with `mypy_path = "llm_sdk"` in
  `pyproject.toml`, excluding the SDK from checks, and converting the SDK's
  `Any`-typed return values explicitly.
- **Disk space.** PyTorch and the model do not fit in a 42 home directory; the
  Makefile moves the environment and caches to sgoinfre when available.

## Testing strategy

- **Output validation:** `tests/validate_output.py` checks that the output is a
  JSON array, that each entry has exactly `prompt`, `name` and `parameters`,
  that the function exists, that all arguments are present and that their types
  match the definition.
- **Decoder unit tests with a scripted model** (`uv run pytest tests/`): a fake
  client with the same interface as the SDK returns a predetermined token
  sequence, so the decoder can be tested deterministically without downloading
  the model. The 19 tests cover tokens that carry the closing quote plus a
  comma, valid and invalid escapes, unescaped inner quotes, empty and non-ASCII
  strings, negative and decimal numbers, a 20-digit number, and a function-name
  choice where the model "prefers" a token outside the candidates.
- **Toy SDK:** `llm_sdk_stub/fake_llm.py` imitates the SDK (including a
  byte-level-encoded `vocab.json`) with random logits. It was used to check the
  vocabulary loader and the pipeline end to end before downloading the model.
- **Error cases:** `tests/error_cases.sh` runs 16 malformed inputs (missing
  file, broken JSON, empty list, empty file, non-UTF-8 file, directory instead of
  file, wrong structure, unsupported parameter type, incomplete or empty
  definitions, unknown CLI argument, unwritable output) and checks that none of
  them ends in a traceback. It then runs 10 edge-case prompts with the real
  model: large and negative numbers, decimals, zero, non-ASCII names, quotes
  inside strings, an empty string, a long string, a regex and paraphrased
  requests.
- **Static checks:** `make lint` and `make lint-strict` pass.

## Resources

- [Hugging Face — Generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Hugging Face — Summary of the tokenizers (BPE, byte-level BPE)](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- Willard & Louf, *Efficient Guided Generation for Large Language Models*
  (2023) — [arXiv:2307.09702](https://arxiv.org/abs/2307.09702)
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

### Use of AI

An AI assistant (Claude) was used during the project for:

- **Environment setup:** configuring the Makefile to place the virtual
  environment and caches in sgoinfre, and understanding how uv creates and uses
  virtual environments.
- **Debugging:** diagnosing the empty-string bug, the negative-number
  tokenization issue, the output key names and the mypy module-resolution error.
- **Code:** the JSON escape and inner-quote handling in `generate_string`, the
  leading-space handling in `generate_number`, error handling for non-UTF-8
  input and unwritable output, and the explicit type conversions in
  `llm_client.py`.
- **Testing:** the output validator, the decoder unit tests and the
  error-case script.
- **Documentation:** drafting this README.

Every change was run, tested against the real model and reviewed before being
kept. One AI suggestion (the Qwen3 tool-calling prompt) was discarded after
testing because it did not improve the results.
