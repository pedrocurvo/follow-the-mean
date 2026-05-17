.PHONY: check-ruff install-lint lint lint-fix format download rgm-references

PYTHON ?= python
RUFF_CONFIG := pyproject.toml
PYTHON_DIRS := moons mnist rgm spfm
RGM_REFERENCES_DIR := rgm/references
RGM_REFERENCES_URL ?= https://drive.google.com/uc?export=download&id=1nqe0cSa1ptFj1eQCditixQrrfKsmJZ41

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

download: rgm-references

rgm-references:
	@test -n "$(RGM_REFERENCES_URL)" || { \
		echo "Set RGM_REFERENCES_URL to a direct-download zip URL."; \
		echo "Example: make rgm-references RGM_REFERENCES_URL=https://drive.google.com/uc?export=download&id=FILE_ID"; \
		exit 2; \
	}
	@tmp_dir=$$(mktemp -d); \
	archive="$$tmp_dir/rgm-references.zip"; \
	unpack_dir="$$tmp_dir/unpack"; \
	mkdir -p "$$unpack_dir" "$(RGM_REFERENCES_DIR)"; \
	if command -v curl >/dev/null 2>&1; then \
		curl -L --fail "$(RGM_REFERENCES_URL)" -o "$$archive"; \
	elif command -v wget >/dev/null 2>&1; then \
		wget -O "$$archive" "$(RGM_REFERENCES_URL)"; \
	else \
		echo "Install curl or wget to download RGM references."; \
		rm -rf "$$tmp_dir"; \
		exit 127; \
	fi; \
	$(PYTHON) -m zipfile -e "$$archive" "$$unpack_dir"; \
	if [ -d "$$unpack_dir/rgm/references" ]; then \
		cp -R "$$unpack_dir/rgm/references/." "$(RGM_REFERENCES_DIR)/"; \
	elif [ -d "$$unpack_dir/references" ]; then \
		cp -R "$$unpack_dir/references/." "$(RGM_REFERENCES_DIR)/"; \
	else \
		cp -R "$$unpack_dir/." "$(RGM_REFERENCES_DIR)/"; \
	fi; \
	rm -rf "$$tmp_dir"; \
	echo "RGM references are available in $(RGM_REFERENCES_DIR)."
