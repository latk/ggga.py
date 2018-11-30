.PHONY: all qa test lint install-dev examples-no-interactive

PYTHON ?= python

PYTEST = $(PYTHON) -m pytest
FLAKE8 = $(PYTHON) -m flake8
MYPY   = $(PYTHON) -m mypy
PYLINT = $(PYTHON) -m pylint
PIP    = $(PYTHON) -m pip
GGGA_EXAMPLE = $(PYTHON) -m ggga.examples

all:
	# nothing

qa: lint test examples-no-interactive

test:
	$(PYTEST) ggga --doctest-modules $(PYTEST_ARGS)

lint:
	$(FLAKE8) ggga
	$(MYPY) ggga
	$(PYLINT) $(PYLINT_FLAGS) ggga || $(PYTHON) -m pylint_exit $$?

install-dev:
	$(PIP) install -e .[dev]

examples-no-interactive: GGGA_EXAMPLE += --quiet --no-interactive
examples-no-interactive:
	time $(GGGA_EXAMPLE) goldstein-price
	time $(GGGA_EXAMPLE) goldstein-price --model=knn
	time $(GGGA_EXAMPLE) goldstein-price --logy
	time $(GGGA_EXAMPLE) easom
	time $(GGGA_EXAMPLE) himmelblau
	time $(GGGA_EXAMPLE) rastrigin2
	time $(GGGA_EXAMPLE) rosenbrock2 --logy --samples 80
	time $(GGGA_EXAMPLE) sphere2 --samples 80
	time $(GGGA_EXAMPLE) sphere2 --samples 80 --noise 1.5
	time $(GGGA_EXAMPLE) onemax4log --samples 80

README_example.png: SHELL = bash
README_example.png: README.md
	$(PYTHON) <(awk '/^```/ {if (found){nextfile}; found=!found; next} found {print}' $<)
