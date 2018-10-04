.PHONY: all test lint install-dev examples-no-interactive

PYTHON ?= python

all:
	# nothing

test: lint
	$(PYTHON) -m pytest ggga --doctest-modules

lint:
	$(PYTHON) -m flake8 ggga
	$(PYTHON) -m mypy ggga
	$(PYTHON) -m pylint ggga $(PYLINT_FLAGS) || true

install-dev:
	$(PYTHON) -m pip install -e .[dev]

examples-no-interactive:
	time $(PYTHON) ./examples/goldstein_price.py --no-interactive >/dev/null
