FROM python:3.11-slim

WORKDIR /app

# Install minimal system tools & geospatial libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libexpat1 \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install unified dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy unified application source (Alignment + DTM + Greenfield Composite)
COPY . .

# Expose port
EXPOSE 8001

# Start unified FastAPI service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
