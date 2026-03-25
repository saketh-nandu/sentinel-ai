FROM python:3.11-slim

# Set working directory to the backend package
WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only backend requirements and install for better cache usage
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code into /app/backend
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 sentinel && \
    chown -R sentinel:sentinel /app && \
    mkdir -p /tmp/uploads && \
    chown -R sentinel:sentinel /tmp/uploads

USER sentinel

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
