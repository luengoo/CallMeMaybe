UV = uv
FUNCTIONS_PATH = data/input/functions_definition.json
INPUT_PATH = data/input/function_calling_tests.json
OUTPUT_PATH = data/output/function_calling_results.json

SGOINFRE ?= /sgoinfre/students/$(USER)
STORE    := $(SGOINFRE)/call_me_maybe
ENV_VARS := UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_PYTHON_INSTALL_DIR \
            HF_HOME UV_LINK_MODE

ifneq ($(wildcard $(SGOINFRE)),)
export UV_PROJECT_ENVIRONMENT := $(STORE)/.venv
export UV_CACHE_DIR           := $(STORE)/uv-cache
export UV_PYTHON_INSTALL_DIR  := $(STORE)/uv-python
export HF_HOME                := $(STORE)/huggingface
export UV_LINK_MODE           := copy
endif

.PHONY: all install run debug clean fclean lint lint-strict env

all: run

install:
	@[ ! -d "$(SGOINFRE)" ] || mkdir -p $(STORE)
	$(UV) sync

run:
	$(UV) run python -m src --functions_definition $(FUNCTIONS_PATH) --input $(INPUT_PATH) --output $(OUTPUT_PATH)

debug:
	$(UV) run python -m pdb -m src --functions_definition $(FUNCTIONS_PATH) --input $(INPUT_PATH) --output $(OUTPUT_PATH)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

fclean: clean
	rm -rf .venv $(STORE)

lint:
	$(UV) run flake8 . && \
	$(UV) run mypy . --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 . && $(UV) run mypy . --strict

# Uso: eval "$$(make -s env)"
env:
	@$(foreach v,$(ENV_VARS),echo 'export $(v)=$($(v))';)