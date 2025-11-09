#!/bin/bash
# ============================================
# Quick Start - One-Command Setup
# ============================================
# Run this on your Google Cloud VPS:
# wget -O - https://raw.githubusercontent.com/YOUR_REPO/main/quickstart.sh | bash
# Or: curl -fsSL https://raw.githubusercontent.com/YOUR_REPO/main/quickstart.sh | bash

set -e

echo "=========================================="
echo "SEO Assistant - Quick Start"
echo "=========================================="

# Clone repository
if [ ! -d "$HOME/seo-assistant" ]; then
    echo "Cloning repository..."
    git clone https://github.com/jamesdeverick/evolv.git $HOME/seo-assistant
fi

cd $HOME/seo-assistant

# Make setup script executable
chmod +x gcloud_setup.sh

# Run full setup
./gcloud_setup.sh

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next: Edit .env with your API keys:"
echo "  nano ~/seo-assistant/.env"
echo ""
echo "Then start the app:"
echo "  sudo systemctl start seo-assistant"
echo ""
