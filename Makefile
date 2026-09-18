.PHONY: install test lint format clean docs

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=sigdiscover --cov-report=html

lint:
	ruff check sigdiscover/
	mypy sigdiscover/

format:
	black sigdiscover/ tests/
	isort sigdiscover/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

docs:
	mkdocs build

run-synthetic:
	python -m sigdiscover run --synthetic --output results/synthetic/

run-tcga:
	python -m sigdiscover run --maf data/tcga/TCGA-BRCA.maf --output results/tcga/
