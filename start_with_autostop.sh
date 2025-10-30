#!/bin/bash
# Start Streamlit app with auto-stop protection
# This will automatically stop the pod after 30 minutes of inactivity

set -e

echo "🚀 Starting Evolv SEO Assistant with auto-stop protection..."

# 1. Pull latest code
cd /workspace/evolv.
git pull origin claude/advanced-seo-assistant-011CUXXN6J98eQv5MBW1kFpj

# 2. Set environment variables
export LLM_PROVIDER=ollama
export SCRAPINGDOG_API_KEY="68da907d6dcca4eb91ea8469"
# Add your Anthropic API key here or set in pod environment
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-your_anthropic_api_key_here}"

# 3. Start Ollama in background
echo "🤖 Starting Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
    ollama pull mistral
fi

# 4. Start auto-stop monitor in background
echo "⏱️  Starting auto-stop monitor (30 min idle timeout)..."
nohup bash auto_stop_on_idle.sh > /tmp/autostop.log 2>&1 &

# 5. Start Streamlit (foreground)
echo "✅ Starting Streamlit on port 7860..."
echo "🛡️  Pod will auto-stop after 30 minutes of no connections"
echo "📊 Monitor auto-stop status: tail -f /tmp/autostop.log"
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
