FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed by any Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY src/ ./src/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit correctly
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
