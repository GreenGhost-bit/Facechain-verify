.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-dev lint type test test-all cov demo demo-live chain-demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

install: ## Install runtime deps + package
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

install-dev: ## Install dev extras (ruff, mypy, pytest, web3)
	$(PY) -m pip install -e ".[dev,evm]"

lint: ## ruff
	$(PY) -m ruff check .

type: ## mypy --strict
	$(PY) -m mypy

test: ## pytest, no network
	$(PY) -m pytest -m "not network"

test-all: ## pytest incl. network tests
	$(PY) -m pytest

cov: ## pytest with coverage report
	$(PY) -m pytest -m "not network" --cov=facechain --cov-report=term-missing

demo: ## Full OFFLINE pipeline: seed corpus -> run -> verify -> tamper demo
	$(PY) -m facechain fetch-corpus --seed-demo
	$(PY) -m facechain run samples/probe_repost.jpg --providers local --anchor local
	@echo "--- independent re-verification of the newest run ---"
	$(PY) -m facechain verify "$$(ls -dt runs/*/ | head -1)" --no-network
	@echo "--- tamper-evidence ---"
	$(PY) -m facechain chain tamper
	-$(PY) -m facechain chain verify
	$(PY) -m facechain fetch-corpus --seed-demo >/dev/null   # (no-op; corpus already there)

demo-live: ## LIVE pipeline via keyless Wikimedia reverse-image search
	$(PY) -m facechain run samples/probe_repost.jpg \
	  --providers wikimedia \
	  --hint "Dwight D. Eisenhower official photo portrait 1959" \
	  --anchor local
	$(PY) -m facechain verify "$$(ls -dt runs/*/ | head -1)"

chain-demo: ## Show the local ledger
	$(PY) -m facechain chain show

clean: ## Remove generated artifacts
	rm -rf runs chaindata data/corpus/*.jpg data/corpus/*.json .demo \
	       .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
