# API service.
#
# Linux base matters here beyond convention: the sandbox's memory cap uses POSIX
# rlimits, which are a no-op on Windows. The cap is real in this image.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build deps for the scientific stack, removed in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY sample_data/ ./sample_data/

# The sandbox subprocess inherits this user. Generated code should never run as
# root, whatever the policy says it can import.
RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /app/runs /app/data \
    && chown -R agent:agent /app
USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
