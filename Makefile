SHELL := /bin/bash

.PHONY: all
all: lint test

.PHONY: format
format:
	ruff format vect_gan examples

.PHONY: lint
lint:
	ruff check vect_gan examples
	mypy vect_gan examples

.PHONY: test
test:
	pytest --maxfail=1 --disable-warnings -q
