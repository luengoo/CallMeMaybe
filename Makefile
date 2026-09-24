UV = uv
INPUT_PATH = data/input/example.json
OUTPUT_PATH = data/output/function_calling_results.json
SGOINFRE ?= ~/sgoinfre/students/alluengo
STORE := $(SGOINFRE)/call_me_maybe

export UV_PROJECT_ENVIROMENT := $(STORE)/.venv
export UV_CACHE_DIR :=$(STORE)/uv-chache
export UV_PYTHON_INSTALL_DIR := $(STORE)/uv-python
export HF_HOME := $(STORE)/huggingface
export UV_LINK_MODE := copy

.PHONY: all install run debug clean fclean lint lint-strict env

all: run

install:
	@mkdir -p $(STORE)
	$(UV) sync

run:
	$(UV) run python -m src --input $(INPUT_PATH) --output $(OUTPUT_PATH)

debug:
	$(UV) run python -m pdb -m src --input $(INPUT_PATH) --output $(OUTPUT_PATH)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

fclean: clean
	rm -rf $(STORE)

lint:
	$(UV) run flake8 . && \
	$(UV) run mypy . --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 . && $(UV) run mypy . --strict

env:
	@echo 'export UV_PROJECT_ENVIROMENT=$(UV_PROJECT_ENVIROMENT)'
	@echo 'export UV_CACHE_DIR=$(UV_CACHE_DIR)'
	@echo 'export UV_PYTHON_INSTALL_DIR$(UV_PYTHON_INSTALL_DIR)'
	@echo 'export HF_HOME=$(HF_HOME)'
	@echo 'export UV_LINK_MODE=$(UV_LINK_MODE)'