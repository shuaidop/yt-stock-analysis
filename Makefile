.PHONY: install lint fmt typecheck test check run brief docker

install:
	uv sync --all-groups

lint:
	uv run ruff check .

fmt:
	uv run ruff format . && uv run ruff check --fix .

typecheck:
	uv run mypy src

test:
	uv run pytest --cov=ytstock --cov-report=term-missing

check: lint test

run:
	uv run ytstock run $(ARGS)

brief:
	uv run ytstock brief $(URLS)

docker:
	docker build -t yt-stock-analysis .
