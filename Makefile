.PHONY: check-ruff install-lint lint lint-fix format

PYTHON ?= python
RUFF_CONFIG := pyproject.toml
PYTHON_DIRS := moons mnist rgm spfm

check-ruff:
	@$(PYTHON) -c "import ruff" >/dev/null 2>&1 || { \
		echo "ruff is not installed in this Python environment."; \
		echo "Install it with: make install-lint"; \
		exit 127; \
	}

install-lint:
	$(PYTHON) -m pip install ruff

lint: check-ruff
	$(PYTHON) -m ruff check --config $(RUFF_CONFIG) $(PYTHON_DIRS)

lint-fix: check-ruff
	$(PYTHON) -m ruff check --config $(RUFF_CONFIG) --fix $(PYTHON_DIRS)

format: check-ruff
	$(PYTHON) -m ruff format --config $(RUFF_CONFIG) $(PYTHON_DIRS)
