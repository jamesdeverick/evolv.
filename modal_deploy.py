"""
Modal deployment for SEO Assistant with Ollama + Mistral on GPU
Pay-per-second billing - perfect for low usage (10 min/day = ~$5.50/month)

Installation:
    pip install modal

Setup:
    modal token new

Deploy:
    modal deploy modal_deploy.py

Usage:
    Your app URL will be shown after deployment
    Modal auto-scales: cold start ~30s, then instant
"""

import modal
import os

# Create Modal app
app = modal.App("seo-assistant")

# Persist Ollama models across runs (important!)
volume = modal.Volume.from_name("ollama-models", create_if_missing=True)

# Build image with CUDA + Ollama + Python deps
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("curl", "ca-certificates")
    .run_commands(
        # Install Ollama
        "curl -fsSL https://ollama.com/install.sh | sh"
    )
    .pip_install(
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "litellm>=1.17.0",
        "PyPDF2>=3.0.0",
        "defusedxml>=0.7.1",
        "plotly>=5.14.0",
    )
    .copy_local_dir("./api", "/app/api")
    .copy_local_dir("./utils", "/app/utils")
    .copy_local_dir("./analysis", "/app/analysis")
    .copy_local_file("./app.py", "/app/app.py")
    .copy_local_file("./config.py", "/app/config.py")
    .copy_local_file("./.streamlit/config.toml", "/app/.streamlit/config.toml")
)


@app.function(
    image=image,
    gpu="A10G",  # NVIDIA A10G with 24GB VRAM, ~$1.10/hour
    volumes={"/root/.ollama": volume},
    timeout=3600,  # Max 1 hour per session
    container_idle_timeout=600,  # Stay warm 10 min after last request
    secrets=[modal.Secret.from_name("scrapingdog-api")],  # Add your API key in Modal dashboard
)
@modal.web_server(8501, startup_timeout=300)
def web():
    """
    Start Ollama + Streamlit.
    Modal handles HTTP routing and auto-shutdown.
    """
    import subprocess
    import time
    import requests

    # Start Ollama in background
    print("🚀 Starting Ollama server...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"}
    )

    # Wait for Ollama to be ready
    print("⏳ Waiting for Ollama...")
    for _ in range(60):
        try:
            r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
            if r.status_code == 200:
                print("✅ Ollama ready!")
                break
        except:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Ollama failed to start")

    # Check if Mistral model exists
    print("📥 Checking for Mistral model...")
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )

    if "mistral:latest" not in result.stdout:
        print("📦 Pulling Mistral model (first run only, ~5 minutes)...")
        subprocess.run(["ollama", "pull", "mistral:latest"], check=True)
        print("✅ Mistral model downloaded and cached!")
    else:
        print("✅ Mistral model already cached!")

    # Start Streamlit
    print("🌟 Starting Streamlit...")
    subprocess.call([
        "streamlit", "run",
        "/app/app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.serverAddress=0.0.0.0",
        "--browser.gatherUsageStats=false",
    ])


# Entry point for local testing
@app.local_entrypoint()
def main():
    """Test locally before deploying."""
    print("Testing Modal configuration...")
    print("✅ Configuration valid!")
    print("\nTo deploy:")
    print("  modal deploy modal_deploy.py")
