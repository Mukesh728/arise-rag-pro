FROM python:3.11-slim

WORKDIR /app

# Install system deps for chromadb/onnxruntime and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
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

# Create non-root system user and group (uid/gid 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser

# Copy application code
COPY backend/ backend/
COPY frontend/ frontend/
COPY scripts/ scripts/

# Bundle the cleaned ChromaDB as initial data
COPY chroma_db/ chroma_db/

# Create persistent directories and set ownership for non-root appuser
RUN mkdir -p /data uploads chroma_db && \
    chown -R appuser:appgroup /app /data uploads chroma_db

# Set persistent SQLite user database path on /data volume
ENV USERS_DB_PATH=/data/users.db

# Reverse-proxy trust configuration:
# TRUSTED_PROXY_COUNT specifies how many reverse proxies (e.g. Render, Railway, Nginx) sit in front.
# get_client_ip takes the Nth entry from the right (-N) of X-Forwarded-For.
# A count set too high allows client IP spoofing (unsafe).
# A count set too low causes multiple users behind a proxy to share a rate-limit bucket (safe).
ENV TRUSTED_PROXY_COUNT=1

# Expose the port (default 8000, overridable via $PORT)
EXPOSE 8000

# Switch to non-root user
USER appuser

# Container healthcheck using public /api/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Single worker: BM25 index lives in process memory, multiple workers would desync.
# Use --no-proxy-headers so uvicorn does not blindly trust client-supplied headers.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-proxy-headers"]
