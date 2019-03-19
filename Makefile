.PHONY: all qa test lint install-dev examples-no-interactive doc

PYTHON ?= python

PYTEST = $(PYTHON) -m pytest
FLAKE8 = $(PYTHON) -m flake8
MYPY   = $(PYTHON) -m mypy
PYLINT = $(PYTHON) -m pylint
PIP    = $(PYTHON) -m pip
GGGA_EXAMPLE = $(PYTHON) -m ggga

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
	$(PIP) install -r ./doc/requirements.txt

examples-no-interactive: GGGA_EXAMPLE += --quiet --no-interactive
examples-no-interactive:
	time $(GGGA_EXAMPLE) goldstein-price
	time $(GGGA_EXAMPLE) goldstein-price --model=knn
	time $(GGGA_EXAMPLE) goldstein-price --logy
	time $(GGGA_EXAMPLE) easom
	time $(GGGA_EXAMPLE) himmelblau
	time $(GGGA_EXAMPLE) rastrigin
	time $(GGGA_EXAMPLE) rosenbrock --logy --samples 80
	time $(GGGA_EXAMPLE) rosenbrock -D3 --logy --samples 80
	time $(GGGA_EXAMPLE) sphere --samples 80
	time $(GGGA_EXAMPLE) sphere --samples 80 --noise 1.5
	time $(GGGA_EXAMPLE) onemax-log --samples 80

doc: $(glob doc/source/*) ./README.rst ./README_example.png
	@ test -d ./doc/build || mkdir ./doc/build
	sphinx-build -b html -Wan --keep-going ./doc/source ./doc/build

README_example.png: SHELL = bash
README_example.png: README.rst
	$(PYTHON) <(awk '/^```/ {if (found){nextfile}; found=!found; next} found {print}' $<)
