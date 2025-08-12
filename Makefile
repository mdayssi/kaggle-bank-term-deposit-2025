.PHONY: setup lint format

setup:
	conda env create -f environment.yml || true
	conda activate bank-auc && pre-commit install

lint:
	ruff src
	black --check src
	isort --check-only src

format:
	black src
	isort src
