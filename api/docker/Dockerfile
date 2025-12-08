FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and video support
# libgl1 replaces deprecated libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libgomp1 \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
