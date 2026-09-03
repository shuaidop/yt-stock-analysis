FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
COPY configs ./configs
ENV PATH="/app/.venv/bin:$PATH" LOG_FORMAT=json DATABASE_URL=sqlite:////data/ytstock.db REPORTS_DIR=/data/reports
VOLUME ["/data"]
ENTRYPOINT ["ytstock"]
CMD ["--help"]
