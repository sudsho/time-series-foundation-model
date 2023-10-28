FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# install torch CPU wheel first to keep the image slim, then the rest
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.0.1 \
    && grep -v '^torch==' requirements.txt > requirements.rest.txt \
    && pip install -r requirements.rest.txt \
    && rm requirements.rest.txt

# app code
COPY src/ ./src/
COPY configs/ ./configs/

RUN mkdir -p /app/artifacts/finetune

# non-root for safety
RUN useradd --create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV MODEL_CKPT=/app/artifacts/finetune/best.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
