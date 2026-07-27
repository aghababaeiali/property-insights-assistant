FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# README.md is needed here too — pyproject.toml's readme field requires it
# to exist for the build backend (hatchling) to build the package metadata
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY agent/ agent/
COPY ml/ ml/
COPY data/ data/

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000"]