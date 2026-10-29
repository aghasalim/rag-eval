PY=.venv/bin/python
setup:
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements.txt pandas tabulate
data:
	bash scripts/download_data.sh
test:
	$(PY) -m pytest -q
experiments:
	$(PY) scripts/run_experiments.py && $(PY) scripts/make_report.py
faithfulness:  # needs a valid ANTHROPIC_API_KEY
	$(PY) scripts/faithfulness.py
