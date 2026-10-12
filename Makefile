# SlideShift Makefile
# Usage: make <target>

PYTHON = python
PYTEST = python -m pytest
SLIDE_DIR = data/benchmark_cohort
FEATURE_DIR = outputs/features
ROBUSTNESS_DIR = outputs/robustness

.PHONY: help install test test-verbose lint clean \
        synthetic-data extract train robustness demo \
        exp01 exp02 exp03 full-pipeline

help:
	@echo ""
	@echo "SlideShift — Pathology WSI MIL Pipeline"
	@echo ""
	@echo "Development:"
	@echo "  make install        Install package in editable mode"
	@echo "  make test           Run all tests"
	@echo "  make test-verbose   Run tests with verbose output"
	@echo "  make lint           Lint with ruff"
	@echo ""
	@echo "Pipeline:"
	@echo "  make synthetic-data Generate synthetic H&E cohort"
	@echo "  make extract        Extract features (mock encoder)"
	@echo "  make train          Train ABMIL aggregator"
	@echo "  make robustness     Run robustness benchmark"
	@echo "  make demo           Launch Streamlit demo"
	@echo ""
	@echo "Experiments:"
	@echo "  make exp01          Foundation model comparison"
	@echo "  make exp02          MIL aggregator ablation"
	@echo "  make exp03          Robustness benchmark"
	@echo "  make full-pipeline  End-to-end run (all steps)"
	@echo ""

# ─── Development ─────────────────────────────────────

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) tests/ --tb=short -q

test-verbose:
	$(PYTEST) tests/ -v --tb=short

lint:
	$(PYTHON) -m ruff check slideshift/ tests/ --select=E,W,F --ignore=E501

# ─── Pipeline ────────────────────────────────────────

synthetic-data:
	$(PYTHON) -m slideshift.cli synthetic-data \
		--output-dir $(SLIDE_DIR) \
		--num-slides 50

extract:
	$(PYTHON) -m slideshift.cli extract \
		--slide-dir $(SLIDE_DIR) \
		--output-dir $(FEATURE_DIR) \
		--model mock

train:
	$(PYTHON) -m slideshift.cli train \
		--feature-dir $(FEATURE_DIR) \
		--arch abmil \
		--epochs 30

robustness:
	$(PYTHON) -m slideshift.cli robustness \
		--feature-dir $(FEATURE_DIR) \
		--output-dir $(ROBUSTNESS_DIR)

demo:
	$(PYTHON) -m slideshift.cli demo

# ─── Experiments ─────────────────────────────────────

exp01:
	$(PYTHON) experiments/01_foundation_model_comparison.py

exp02:
	$(PYTHON) experiments/02_mil_aggregator_ablation.py

exp03:
	$(PYTHON) experiments/03_robustness_benchmark.py

full-pipeline: synthetic-data extract train robustness exp01 exp02 exp03
	@echo ""
	@echo "Full pipeline complete. Check outputs/ for results."

# ─── Utility ─────────────────────────────────────────

clean:
	@echo "Removing generated outputs..."
	-rd /s /q data outputs 2>nul || rm -rf data outputs
	-rd /s /q .pytest_cache __pycache__ 2>nul || rm -rf .pytest_cache
	@echo "Done."

