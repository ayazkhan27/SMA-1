PYTHON ?= python3

.PHONY: test gates report ui api datasets model

test:
	$(PYTHON) -m pytest

gates:
	$(PYTHON) -m pytest -m "gate_G0 or gate_G1 or gate_G2 or gate_G3 or gate_G4 or gate_G5 or gate_G6"

report:
	$(PYTHON) -u -m sma.eval.report --out reports/report.html

ui:
	$(PYTHON) -m sma.ui.app

api:
	uvicorn sma.agent.api:app --host 127.0.0.1 --port 8000

datasets:
	$(PYTHON) scripts/fetch_datasets.py --manifest data/manifests/datasets.json

model:
	$(PYTHON) scripts/fetch_model.py
