FROM python:3.10-slim

WORKDIR /app

# system deps for torch + numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

# placeholder for finetuned weights
RUN mkdir -p /app/artifacts/finetune

ENV PYTHONUNBUFFERED=1
ENV MODEL_CKPT=/app/artifacts/finetune/best.pt

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
