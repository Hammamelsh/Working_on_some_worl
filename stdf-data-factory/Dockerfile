# Simple Dockerfile for Data Hunting Agent
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN crawl4ai-setup && playwright install

# Copy application files
COPY *.py ./
COPY utils/ ./utils/
COPY validation/ ./validation/

# Create necessary directories
RUN mkdir -p uploads extracted_data

# Expose port
EXPOSE 8501

# Run the streamlit app directly
CMD ["streamlit", "run", "unified_app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"]