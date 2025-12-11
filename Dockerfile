FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies first (better caching)
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy only necessary source code
COPY api/ ./api/
COPY inference/ ./inference/
COPY monitoring/ ./monitoring/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Use PORT env variable for Cloud Run compatibility
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
