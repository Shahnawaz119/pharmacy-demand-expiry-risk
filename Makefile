# Makefile for Pharmacy Medicine Demand & Expiry-Risk Forecasting System

PYTHON ?= python

.PHONY: help setup data train evaluate test api app clean

help:
	@echo "Available targets:"
	@echo "  setup     - Install dependencies and install package in editable mode"
	@echo "  data      - Generate synthetic pharmacy data and execute data quality cleaning"
	@echo "  train     - Run reproducible training pipeline and save artifacts"
	@echo "  evaluate  - Run evaluation suite on held-out test set and export metrics/plots"
	@echo "  test      - Run unit and integration tests with pytest"
	@echo "  api       - Launch FastAPI inference service"
	@echo "  app       - Launch Streamlit decision-support dashboard"
	@echo "  clean     - Remove temporary Python caches and build artifacts"

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) -m src.data.generate_data
	$(PYTHON) -m src.data.clean_data

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluation.evaluate

test:
	$(PYTHON) -m pytest tests/

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

app:
	$(PYTHON) -m streamlit run src/app/main.py

clean:
	@rmdir /s /q build 2>nul || true
	@rmdir /s /q dist 2>nul || true
	@rmdir /s /q src\*.egg-info 2>nul || true
	@del /s /q *.pyc 2>nul || true
