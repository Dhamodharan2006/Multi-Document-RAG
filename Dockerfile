FROM python:3.11-slim

# Metadata
LABEL maintainer="multi-doc-rag"
LABEL description="Multi-Document RAG System - Slim build, no GPU/torch"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies (NO torch, NO transformers)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y torch torchvision torchaudio transformers sentence-transformers 2>/dev/null || true

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p /app/data/uploads /app/evaluation/reports

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
