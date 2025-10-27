FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for Ollama models
RUN mkdir -p /root/.ollama

# Expose ports
# 7860 - Streamlit
# 11434 - Ollama
EXPOSE 7860 11434

# Make startup script executable
RUN chmod +x start.sh

# Start both Ollama and Streamlit
CMD ["./start.sh"]
