UV = uv
INPUT_PATH = data/input/example.json
OUTPUT_PATH = data/output/function_calling_results.json

.PHONY: all install run debug clean lint lint-strict

all: run

install:
	$(UV) sync

run:
	$(UV) run python -m src --input $(INPUT_PATH) --output $(OUTPUT_PATH)

debug:
	$(UV) run python -m pdb -m src --input $(INPUT_PATH) --output $(OUTPUT_PATH)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	$(UV) run flake8 . && \
	$(UV) run mypy . --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 . && $(UV) run mypy . --strict