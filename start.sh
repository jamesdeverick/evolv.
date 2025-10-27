#!/bin/bash
set -e

echo "🚀 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "⏳ Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Ollama failed to start"
        exit 1
    fi
    sleep 2
done

echo "📥 Checking if Mistral model is available..."
if ! ollama list | grep -q "mistral:latest"; then
    echo "📦 Pulling Mistral model (this may take 5-10 minutes on first run)..."
    ollama pull mistral:latest
    echo "✅ Mistral model downloaded successfully!"
else
    echo "✅ Mistral model already available!"
fi

echo "🌟 Starting Streamlit app..."
streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true

# Keep script running
wait $OLLAMA_PID
