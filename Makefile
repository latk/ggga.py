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

examples-no-interactive:
	time $(GGGA_EXAMPLE) goldstein-price --quiet --no-interactive
	time $(GGGA_EXAMPLE) goldstein-price --quiet --no-interactive --model=knn
	time $(GGGA_EXAMPLE) goldstein-price --quiet --no-interactive --logy
	time $(GGGA_EXAMPLE) easom --quiet --no-interactive
	time $(GGGA_EXAMPLE) himmelblau --quiet --no-interactive
	time $(GGGA_EXAMPLE) rastrigin2 --quiet --no-interactive
	time $(GGGA_EXAMPLE) rosenbrock2 --quiet --no-interactive --logy --samples 80
	time $(GGGA_EXAMPLE) sphere2 --quiet --no-interactive --samples 80
