.PHONY: checks, docs-serve, docs-build, fmt, config, type, tests

pre-commit-checks:
	uv run pre-commit run --hook-stage manual --all-files

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

fmt:
	uv run ruff format fluowkit tests

lint:
	uv run ruff check --fix fluowkit tests

type:
	uv run mypy fluowkit --non-interactive --show-traceback

tests:
	uv run pytest --ignore=histalign/tests

config: ## Configure .netrc with codeartifacts
	@if [ -d /home/sagemaker-user ]; then \
		$(MAKE) config-sagemaker; \
	else \
		$(MAKE) config-local; \
	fi

config-local: ## Configure .netrc with codeartifacts through SSO
	@export AWS_PROFILE=codeartifact && \
	aws sso login --use-device-code && \
	export AWS_DOMAIN="abstra" && \
	export AWS_ACCOUNT_ID="058264397262" && \
	export AWS_REGION="eu-west-1" && \
	export AWS_CODEARTIFACT_REPOSITORY="owkin-pypi" && \
	export AWS_CODEARTIFACT_TOKEN="$$(aws codeartifact get-authorization-token --domain $$AWS_DOMAIN --domain-owner $$AWS_ACCOUNT_ID --query authorizationToken --output text)" && \
    { \
    if [ ! -f ~/.netrc ]; then touch ~/.netrc; fi; \
    chmod 600 ~/.netrc; \
    if grep -q "machine abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com" ~/.netrc; then \
        sed -i.bak "/machine abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com/,+2d" ~/.netrc && rm -f ~/.netrc.bak; \
    fi; \
    echo "machine abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com" >> ~/.netrc; \
    echo "login aws" >> ~/.netrc; \
    echo "password $${AWS_CODEARTIFACT_TOKEN}" >> ~/.netrc; \
    }

config-sagemaker: ## Configure .netrc with Owkin's PyPi credentials
	@if [ ! -f ~/.config/pip/pip.conf ]; then \
		echo "The pip.conf file does not exist at ~/.config/pip/pip.conf"; \
	else \
	    PIP_INDEX_URL=$$(grep -E '^index-url' ~/.config/pip/pip.conf | cut -d ' ' -f 3); \
	    if [ -z "$$PIP_INDEX_URL" ]; then \
	        echo "No index-url found in ~/.config/pip/pip.conf"; \
	        exit 1; \
	    fi; \
	    PASSWORD=$$(echo $$PIP_INDEX_URL | sed -n 's|.*://aws:\([^@]*\)@.*|\1|p'); \
	    if [ -z "$$PASSWORD" ]; then \
	        echo "No password found in the index-url"; \
	        exit 1; \
	    fi; \
	    if [ ! -f ~/.netrc ]; then touch ~/.netrc; fi; \
	    chmod 600 ~/.netrc; \
	    if grep -q "machine abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com" ~/.netrc; then \
	        sed -i "/machine abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com/,+2d" ~/.netrc; \
	    fi; \
	    echo "machine abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com" >> ~/.netrc; \
	    echo "login aws" >> ~/.netrc; \
	    echo "password $$PASSWORD" >> ~/.netrc; \
	    echo "Updated .netrc with credentials for abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com"; \
	fi; \

install: clean  ## Install all package and development dependencies for testing to the active Python's site-packages
	uv sync --all-extras
	@for rc in ~/.bashrc ~/.zshrc; do \
		if [ -f "$$rc" ] && ! grep -q 'export PATH=$$HOME/.local/bin:$$PATH' "$$rc"; then \
			echo 'export PATH=$$HOME/.local/bin:$$PATH' >> "$$rc"; \
		fi \
	done

clean: clean-build clean-pyc clean-test clean-docs ## Remove all build, test, coverage and Python artifacts

clean-build: ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -path ./.venv -prune -false -o -name '*.egg-info' -exec rm -fr {} +
	find . -path ./.venv -prune -false -o -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	find . -path ./.venv -prune -false -o -name '*.pyc' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*.pyo' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*~' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '__pycache__' -exec rm -rf {} +

clean-test: ## Remove test and coverage artifacts
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr prof/
	rm -fr .ruff_cache

clean-docs: ## Remove docs artifacts
	rm -fr docs/_build
	rm -fr docs/api
