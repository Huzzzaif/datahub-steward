# Hosted demo image. Runs the FastAPI UI against the in-memory catalog, because
# a free web dyno cannot host DataHub — see web.py for why that is honest rather
# than a shortcut.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first so code edits don't re-resolve the whole tree.
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    STEWARD_CATALOG=fake \
    STEWARD_PROVIDER=groq \
    PORT=8000

EXPOSE 8000

# Render and Fly both inject $PORT; honour it rather than hardcoding.
CMD ["sh", "-c", "uvicorn steward.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
