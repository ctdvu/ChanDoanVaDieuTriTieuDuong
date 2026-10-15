# Diabetes Diagnosis + RAG Treatment — Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (saves ~5-10GB of unused CUDA packages)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements & install backend dependencies
COPY web/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 8000

ENV PYTHONPATH=/app
WORKDIR /app/web/backend

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
