# Multi-stage build to reduce final image size
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgl1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --target /app/dependencies

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder stage
COPY --from=builder /app/dependencies /usr/local/lib/python3.10/site-packages

# Copy scripts (e.g., uvicorn) to PATH
COPY --from=builder /app/dependencies/bin/* /usr/local/bin/

# Ensure scripts are executable
RUN chmod +x /usr/local/bin/*

# Copy application code and weights
COPY app/ ./app/
COPY weights/ ./weights/

# Create necessary directories for volumes
RUN mkdir -p /app/temp_uploads /app/static_results

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]