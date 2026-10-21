# Shortcuts for the common tasks. Run `make help` for the list.
# Works on macOS and Linux. On Windows, run the commands shown below directly.

PY ?= python
JOBS ?= 5

.PHONY: help setup check test demo demo-quick reproduce fixes poromechanics figures writeup clean

help:
	@echo "make setup          create .venv and install everything (PyTorch included)"
	@echo "make check          environment check: autograd, seeding, speed"
	@echo "make test           run the unit tests"
	@echo "make demo           live demo: paper method vs our fix on all four problems"
	@echo "make demo-quick     same with half the training budget"
	@echo "make reproduce      rerun the paper's three experiments over 5 seeds (slow)"
	@echo "make fixes          rerun our improvements (slow)"
	@echo "make poromechanics  rerun the coupled poromechanics study (section 5)"
	@echo "make figures        redraw the summary figure from the tables"
	@echo "make writeup        rebuild report/writeup/writeup.pdf (needs LaTeX)"
	@echo "Set JOBS=N to change how many runs train in parallel (default $(JOBS))."

setup:
	$(PY) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch
	.venv/bin/pip install -r requirements.txt
	@echo "Done. Activate with: source .venv/bin/activate"

check:
	$(PY) scripts/check_env.py

test:
	$(PY) -m pytest -q

demo:
	$(PY) demo.py

demo-quick:
	$(PY) demo.py --quick

reproduce:
	$(PY) experiments/oscillator_reproduce.py --jobs $(JOBS)
	$(PY) experiments/consolidation_reproduce.py --jobs $(JOBS)
	$(PY) experiments/heat_reproduce.py --jobs $(JOBS)

fixes:
	$(PY) experiments/oscillator_two_stage.py --jobs $(JOBS)
	$(PY) experiments/oscillator_noise.py --jobs $(JOBS)
	$(PY) experiments/consolidation_enrichment.py --jobs $(JOBS)
	$(PY) experiments/heat_corners.py --jobs $(JOBS)

poromechanics:
	$(PY) experiments/poromechanics.py --jobs $(JOBS)

figures:
	$(PY) report/make_summary_figure.py

writeup:
	cd report/writeup && pdflatex -interaction=nonstopmode writeup.tex >/dev/null && pdflatex -interaction=nonstopmode writeup.tex >/dev/null
	@echo "report/writeup/writeup.pdf"

clean:
	rm -rf results/runs results/demo .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
