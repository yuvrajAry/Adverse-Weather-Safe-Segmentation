# Backend Dockerfile
# Builds a small image for the Flask + Waitress backend.
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gcc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY project/requirements_backend.txt /app/requirements_backend.txt
RUN pip install --no-cache-dir -r /app/requirements_backend.txt

# Copy project sources
COPY project/ /app

# Create runtime folders
RUN mkdir -p /app/uploads /app/results /app/ckpts

# Expose port
EXPOSE ${PORT}

# Default command uses start_backend.py which should read PORT env var
CMD ["python", "start_backend.py"]
