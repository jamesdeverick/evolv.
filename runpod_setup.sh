#!/bin/bash
# Quick setup script for RunPod container restarts

set -e  # Exit on error

echo "🚀 Starting RunPod setup..."

# 1. Pull latest code
echo "📥 Pulling latest code..."
git fetch origin
git checkout claude/advanced-seo-assistant-011CUXXN6J98eQv5MBW1kFpj
git pull origin claude/advanced-seo-assistant-011CUXXN6J98eQv5MBW1kFpj

# 2. Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# 3. Set environment variables
echo "🔑 Setting environment variables..."
export SCRAPINGDOG_API_KEY="68da907d6dcca4eb91ea8469"
export LLM_PROVIDER="ollama"
export OLLAMA_API_BASE="http://127.0.0.1:11434"
export OLLAMA_MODEL="mistral:latest"

# Set Anthropic API key (paste your key here)
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# Add to .bashrc so they persist in new shells
if ! grep -q "SCRAPINGDOG_API_KEY" ~/.bashrc; then
    echo 'export SCRAPINGDOG_API_KEY="68da907d6dcca4eb91ea8469"' >> ~/.bashrc
    echo 'export LLM_PROVIDER="ollama"' >> ~/.bashrc
    echo 'export OLLAMA_API_BASE="http://127.0.0.1:11434"' >> ~/.bashrc
    echo 'export OLLAMA_MODEL="mistral:latest"' >> ~/.bashrc
    echo 'export ANTHROPIC_API_KEY="your_anthropic_api_key_here"' >> ~/.bashrc
    echo "" >> ~/.bashrc
    echo "# Update the ANTHROPIC_API_KEY above with your actual key" >> ~/.bashrc
fi

# 4. Start Ollama in background if not running
echo "🤖 Checking Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# 5. Start Streamlit
echo "✅ Setup complete! Starting Streamlit on port 7860..."
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
