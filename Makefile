.PHONY: all qa test lint install-dev examples-no-interactive

PYTHON ?= python

all:
	# nothing

qa: lint test examples-no-interactive

test:
	$(PYTHON) -m pytest ggga --doctest-modules

lint:
	$(PYTHON) -m flake8 ggga
	$(PYTHON) -m mypy ggga
	$(PYTHON) -m pylint ggga $(PYLINT_FLAGS) || true

install-dev:
	$(PYTHON) -m pip install -e .[dev]

examples-no-interactive:
	time $(PYTHON) -m ggga.examples.goldstein_price --quiet --no-interactive
	time $(PYTHON) -m ggga.examples.goldstein_price --quiet --no-interactive \
		--model=knn
