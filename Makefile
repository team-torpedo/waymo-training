PY ?= venv/bin/python

.PHONY: setup setup-gpu download preprocess train-detection train-lane evaluate all test lint

setup:
	bash scripts/setup_venv.sh

setup-gpu:
	bash scripts/setup_venv.sh --gpu

download:
	$(PY) stages/download.py

preprocess:
	$(PY) stages/preprocess.py

train-detection:
	$(PY) stages/train_detection.py

train-lane:
	$(PY) stages/train_lane.py

evaluate:
	$(PY) stages/evaluate.py

all:
	$(PY) main.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src stages tests main.py
