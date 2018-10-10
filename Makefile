.PHONY: all qa test lint install-dev examples-no-interactive

PYTHON ?= python

PYTEST = $(PYTHON) -m pytest
FLAKE8 = $(PYTHON) -m flake8
MYPY   = $(PYTHON) -m mypy
PYLINT = $(PYTHON) -m pylint
PIP    = $(PYTHON) -m pip

all:
	# nothing

qa: lint test examples-no-interactive

test:
	$(PYTEST) ggga --doctest-modules

lint:
	$(FLAKE8) ggga
	$(MYPY) ggga
	$(PYLINT) ggga $(PYLINT_FLAGS) || true

install-dev:
	$(PIP) install -e .[dev]

examples-no-interactive:
	time $(PYTHON) -m ggga.examples.goldstein_price --quiet --no-interactive
	time $(PYTHON) -m ggga.examples.goldstein_price --quiet --no-interactive \
		--model=knn
