#!/bin/bash
# ============================================
# Google Cloud VPS Setup Script
# Advanced SEO Assistant with T4 GPU
# ============================================

set -e  # Exit on error

echo "=========================================="
echo "Google Cloud VPS Setup for SEO Assistant"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log function
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================
# 1. System Update & Basic Tools
# ============================================
log_info "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

log_info "Installing essential tools..."
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    htop \
    vim \
    tmux \
    python3-pip \
    python3-venv

# ============================================
# 2. NVIDIA Driver & CUDA Setup (for T4 GPU)
# ============================================
log_info "Checking for NVIDIA GPU..."
if lspci | grep -i nvidia > /dev/null; then
    log_info "NVIDIA GPU detected. Setting up CUDA..."

    # Install NVIDIA drivers if not present
    if ! command -v nvidia-smi &> /dev/null; then
        log_info "Installing NVIDIA drivers..."
        sudo apt-get install -y ubuntu-drivers-common
        sudo ubuntu-drivers autoinstall

        log_warn "NVIDIA drivers installed. System reboot required!"
        log_warn "After reboot, run this script again."
        echo "Reboot now? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            sudo reboot
        else
            exit 0
        fi
    else
        log_info "NVIDIA drivers already installed."
        nvidia-smi
    fi

    # Install CUDA toolkit
    if ! command -v nvcc &> /dev/null; then
        log_info "Installing CUDA toolkit..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit-12-3

        # Add CUDA to PATH
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        source ~/.bashrc
    else
        log_info "CUDA already installed."
    fi
else
    log_warn "No NVIDIA GPU detected. Running in CPU mode."
fi

# ============================================
# 3. Python 3.11 Installation
# ============================================
log_info "Setting up Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Set Python 3.11 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --set python3 /usr/bin/python3.11

log_info "Python version: $(python3 --version)"

# ============================================
# 4. Ollama Installation
# ============================================
log_info "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh

    # Start Ollama service
    sudo systemctl enable ollama
    sudo systemctl start ollama

    log_info "Waiting for Ollama to start..."
    sleep 5
else
    log_info "Ollama already installed."
fi

# Check Ollama status
if systemctl is-active --quiet ollama; then
    log_info "Ollama service is running."
else
    log_warn "Ollama service not running. Starting..."
    sudo systemctl start ollama
fi

# Pull Mistral model
log_info "Pulling Mistral model (this may take a few minutes)..."
ollama pull mistral:latest

# Verify GPU usage
log_info "Verifying Ollama GPU detection..."
ollama list

# ============================================
# 5. Clone Repository
# ============================================
log_info "Setting up application directory..."
APP_DIR="$HOME/seo-assistant"

if [ -d "$APP_DIR" ]; then
    log_warn "Application directory already exists at $APP_DIR"
    echo "Pull latest changes? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cd "$APP_DIR"
        git pull
    fi
else
    log_info "Cloning repository..."
    echo "Enter your Git repository URL (or press Enter to skip):"
    read -r repo_url

    if [ -n "$repo_url" ]; then
        git clone "$repo_url" "$APP_DIR"
    else
        mkdir -p "$APP_DIR"
        log_warn "Skipped git clone. You'll need to upload your files manually."
    fi
fi

cd "$APP_DIR"

# ============================================
# 6. Python Virtual Environment
# ============================================
log_info "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi

log_info "Activating virtual environment..."
source venv/bin/activate

log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ============================================
# 7. Install Python Dependencies
# ============================================
log_info "Installing Python packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --ignore-installed blinker
else
    log_warn "requirements.txt not found. Installing core dependencies..."
    pip install streamlit pandas requests python-dotenv beautifulsoup4 \
                litellm anthropic PyPDF2 defusedxml plotly \
                spacy sentence-transformers scikit-learn
fi

# Download spaCy model
log_info "Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# ============================================
# 8. Environment Configuration
# ============================================
log_info "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=mistral:latest

# Anthropic (for Step 5 content drafting)
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# Scrapingdog API
SCRAPINGDOG_API_KEY=

# GPU Settings
CUDA_VISIBLE_DEVICES=0
EOF

    log_info ".env file created. Please edit it with your API keys:"
    log_info "  nano .env"
else
    log_info ".env file already exists."
fi

# ============================================
# 9. Streamlit Configuration
# ============================================
log_info "Creating Streamlit config..."
mkdir -p .streamlit

cat > .streamlit/config.toml << 'EOF'
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = true

[global]
developmentMode = false

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverPort = 8501
EOF

log_info "Streamlit configured on port 8501"

# ============================================
# 10. Systemd Service Setup
# ============================================
log_info "Creating systemd service for auto-start..."

sudo tee /etc/systemd/system/seo-assistant.service > /dev/null << EOF
[Unit]
Description=SEO Assistant - Generative Search Optimization Tool
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin:/usr/local/cuda/bin:\$PATH"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
ExecStart=$APP_DIR/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable seo-assistant.service

log_info "Systemd service created and enabled."

# ============================================
# 11. Firewall Configuration
# ============================================
log_info "Configuring firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 8501/tcp
    sudo ufw allow 22/tcp
    sudo ufw --force enable
    log_info "Firewall configured. Port 8501 opened."
else
    log_warn "UFW not found. Please configure firewall manually to allow port 8501"
fi

# ============================================
# 12. Google Cloud Firewall (instructions)
# ============================================
log_warn "=========================================="
log_warn "IMPORTANT: Google Cloud Firewall Setup"
log_warn "=========================================="
log_warn "Run these commands in Google Cloud Console (Cloud Shell):"
log_warn ""
log_warn "gcloud compute firewall-rules create allow-seo-assistant \\"
log_warn "  --allow tcp:8501 \\"
log_warn "  --source-ranges 0.0.0.0/0 \\"
log_warn "  --description 'Allow SEO Assistant access'"
log_warn ""
log_warn "Or create via Web Console:"
log_warn "  VPC Network > Firewall > CREATE FIREWALL RULE"
log_warn "  - Name: allow-seo-assistant"
log_warn "  - Targets: All instances in network"
log_warn "  - Source IP ranges: 0.0.0.0/0"
log_warn "  - Protocols/ports: tcp:8501"
log_warn "=========================================="

# ============================================
# 13. Test Installation
# ============================================
log_info "Testing installation..."

# Test Ollama
if curl -s http://localhost:11434/api/tags > /dev/null; then
    log_info "✓ Ollama API accessible"
else
    log_error "✗ Ollama API not accessible"
fi

# Test GPU
if command -v nvidia-smi &> /dev/null; then
    log_info "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    log_warn "✗ GPU not detected (running in CPU mode)"
fi

# Test Python packages
log_info "Testing Python packages..."
python3 << 'EOF'
import sys
packages = ['streamlit', 'pandas', 'spacy', 'sentence_transformers', 'anthropic']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg}")
        missing.append(pkg)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    sys.exit(1)
EOF

# ============================================
# 14. Display Next Steps
# ============================================
echo ""
log_info "=========================================="
log_info "Installation Complete!"
log_info "=========================================="
echo ""
log_info "Next steps:"
echo ""
echo "1. Edit .env file with your API keys:"
echo "   nano $APP_DIR/.env"
echo ""
echo "2. Start the application:"
echo "   sudo systemctl start seo-assistant"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status seo-assistant"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u seo-assistant -f"
echo ""
echo "5. Configure Google Cloud firewall (see instructions above)"
echo ""
echo "6. Access the application:"
echo "   http://$(curl -s ifconfig.me):8501"
echo ""
log_info "To manually start (for testing):"
echo "   cd $APP_DIR"
echo "   source venv/bin/activate"
echo "   streamlit run app.py"
echo ""
log_info "=========================================="
