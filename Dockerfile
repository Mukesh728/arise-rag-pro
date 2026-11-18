FROM python:3.11-slim

WORKDIR /app

# Install system deps for chromadb/onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Chroma's default embedding model at build time
# so the first query does not trigger a download
RUN python -c "\
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; \
ef = DefaultEmbeddingFunction(); \
ef(['warm up']); \
print('[Docker Build] Default embedding model cached.')"

# Copy application code
COPY backend/ backend/
COPY frontend/ frontend/
COPY scripts/ scripts/

# Bundle the cleaned ChromaDB as initial data
COPY chroma_db/ chroma_db/

# Create uploads directory
RUN mkdir -p uploads

# Expose the port (default 8000, overridable via $PORT)
EXPOSE 8000

# Single worker: BM25 index lives in process memory, multiple workers would desync
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
