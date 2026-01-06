.PHONY: setup download-data clean

setup:
	python -m venv .venv && \
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

download-data:
	python scripts/download-setup-unique-dataset.py
	mkdir -p data && mv -n student_dataset_*.pkl data/ 2>/dev/null || true

clean:
	rm -rf .venv data/*
