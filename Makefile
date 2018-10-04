.PHONY: all test lint

all:
	# nothing

test: lint
	python -m pytest ggga

lint:
	python -m flake8 ggga
	python -m mypy ggga
	python -m pylint ggga $(PYLINT_FLAGS)
