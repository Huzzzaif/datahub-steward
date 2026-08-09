# Slim image for the hosted demo.
#
# Installs the base dependencies only — no acryl-datahub, no anthropic. The
# demo runs the in-memory catalog against Groq, so those extras would be ~95 MB
# of wheels that never get imported, paid for on every cold start of a free
# dyno. `uv sync --extra datahub` locally when you want the real thing.
FROM python:3.12-slim AS build

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies resolve in their own layer, so editing source doesn't re-install.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    STEWARD_CATALOG=fake \
    STEWARD_PROVIDER=groq \
    PORT=8000

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src

# Non-root: the container runs untrusted-ish model output and has no reason to
# hold root.
RUN useradd -m -u 10001 steward && chown -R steward:steward /app
USER steward

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/api/health', timeout=4).status==200 else 1)"

# Render and Fly inject $PORT; honour it rather than hardcoding.
CMD ["sh", "-c", "uvicorn steward.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
