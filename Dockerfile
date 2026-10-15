# Diabetes Diagnosis + RAG Treatment — Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY web/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Pre-download embedding model (optional, speeds up startup)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

EXPOSE 8000

ENV PYTHONPATH=/app
WORKDIR /app/web/backend

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
