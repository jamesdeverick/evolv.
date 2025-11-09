# Google Cloud Migration Guide

Complete guide to migrate your SEO Assistant from RunPod to Google Cloud with T4 GPU.

---

## Prerequisites

- ✅ Google Cloud VPS with T4 GPU set up
- ✅ SSH access to the VPS
- ✅ Your Scrapingdog API key
- ✅ Your Anthropic API key (for Step 5 content drafting)

---

## Quick Start (5 Minutes)

### 1. SSH into your Google Cloud VPS

```bash
# From your local machine
gcloud compute ssh YOUR_INSTANCE_NAME --zone=YOUR_ZONE

# Or use the web-based SSH from Google Cloud Console
```

### 2. Upload and run the setup script

**Option A: Clone from Git (recommended)**

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/evolv.git
cd evolv

# Make setup script executable
chmod +x gcloud_setup.sh

# Run setup
./gcloud_setup.sh
```

**Option B: Manual upload**

```bash
# On your local machine, upload the setup script
gcloud compute scp gcloud_setup.sh YOUR_INSTANCE_NAME:~ --zone=YOUR_ZONE

# Then on the VPS
chmod +x ~/gcloud_setup.sh
./gcloud_setup.sh
```

The script will:
- ✅ Install NVIDIA drivers & CUDA (for T4 GPU)
- ✅ Install Ollama and pull Mistral model
- ✅ Set up Python 3.11 with virtual environment
- ✅ Install all dependencies (spaCy, sentence-transformers, etc.)
- ✅ Create systemd service for auto-start
- ✅ Configure firewall

---

## Configuration

### 1. Add Your API Keys

```bash
cd ~/seo-assistant
nano .env
```

Update these values:

```bash
# Scrapingdog API (required for keyword research)
SCRAPINGDOG_API_KEY=your_scrapingdog_key_here

# Anthropic API (required for Step 5 content drafting)
ANTHROPIC_API_KEY=sk-ant-api03-your_key_here

# LLM Configuration (already set for Ollama)
LLM_PROVIDER=ollama
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=mistral:latest
```

Save and exit: `Ctrl+X`, then `Y`, then `Enter`

### 2. Configure Google Cloud Firewall

**Via gcloud CLI:**

```bash
gcloud compute firewall-rules create allow-seo-assistant \
  --allow tcp:8501 \
  --source-ranges 0.0.0.0/0 \
  --description "Allow SEO Assistant access"
```

**Via Web Console:**

1. Go to **VPC Network** > **Firewall**
2. Click **CREATE FIREWALL RULE**
3. Settings:
   - **Name:** `allow-seo-assistant`
   - **Targets:** All instances in the network
   - **Source IP ranges:** `0.0.0.0/0`
   - **Protocols and ports:** `tcp:8501`
4. Click **CREATE**

---

## Starting the Application

### Option 1: Auto-start with Systemd (Recommended)

```bash
# Start the service
sudo systemctl start seo-assistant

# Check status
sudo systemctl status seo-assistant

# View logs
sudo journalctl -u seo-assistant -f
```

The app will automatically start on system reboot.

### Option 2: Manual Start (for testing)

```bash
cd ~/seo-assistant
source venv/bin/activate
streamlit run app.py
```

---

## Accessing the Application

Get your external IP:

```bash
curl ifconfig.me
```

Then open in your browser:

```
http://YOUR_EXTERNAL_IP:8501
```

Or use the Google Cloud Console to find the external IP:
**Compute Engine** > **VM instances** > **External IP**

---

## Verifying GPU Usage

### Check GPU is detected:

```bash
nvidia-smi
```

You should see your T4 GPU with memory usage.

### Check Ollama is using GPU:

```bash
# Run a test prompt
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "Hello"
}'
```

Then check `nvidia-smi` again - you should see GPU memory usage increase.

### Monitor GPU during application use:

```bash
watch -n 1 nvidia-smi
```

This refreshes every second. Press `Ctrl+C` to stop.

---

## Troubleshooting

### Issue: Can't access http://YOUR_IP:8501

**Fix 1: Check firewall**
```bash
# On VPS
sudo ufw status
sudo ufw allow 8501/tcp

# Check Google Cloud firewall rule exists
gcloud compute firewall-rules list | grep 8501
```

**Fix 2: Check service is running**
```bash
sudo systemctl status seo-assistant
```

**Fix 3: Check logs**
```bash
sudo journalctl -u seo-assistant -n 50
```

---

### Issue: Ollama not using GPU

**Fix 1: Verify NVIDIA drivers**
```bash
nvidia-smi
# Should show your T4 GPU
```

**Fix 2: Restart Ollama**
```bash
sudo systemctl restart ollama
sudo systemctl restart seo-assistant
```

**Fix 3: Check CUDA environment**
```bash
echo $CUDA_VISIBLE_DEVICES
# Should output: 0

nvcc --version
# Should show CUDA version
```

---

### Issue: Python package import errors

**Fix: Reinstall packages**
```bash
cd ~/seo-assistant
source venv/bin/activate
pip install -r requirements.txt --ignore-installed blinker --force-reinstall
python -m spacy download en_core_web_sm
```

---

### Issue: Blinker conflicts

**Fix: Force reinstall**
```bash
pip install -r requirements.txt --ignore-installed blinker
```

---

### Issue: Out of memory errors

**Check GPU memory:**
```bash
nvidia-smi
```

**Reduce Mistral model size:**
```bash
# Use a smaller model
ollama pull mistral:7b-instruct-v0.2-q4_0
```

Then update `.env`:
```bash
OLLAMA_MODEL=mistral:7b-instruct-v0.2-q4_0
```

---

## Monitoring & Maintenance

### View real-time logs:

```bash
sudo journalctl -u seo-assistant -f
```

### Restart application:

```bash
sudo systemctl restart seo-assistant
```

### Update code from Git:

```bash
cd ~/seo-assistant
git pull
sudo systemctl restart seo-assistant
```

### Check disk space:

```bash
df -h
```

### Monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

### Check Ollama models:

```bash
ollama list
```

---

## Performance Optimization

### 1. Use faster spaCy model (optional)

For better performance with entity extraction:

```bash
python -m spacy download en_core_web_md  # Medium model (better accuracy)
# or
python -m spacy download en_core_web_lg  # Large model (best accuracy, slower)
```

### 2. Sentence-transformers cache

The first time you run semantic clustering, it downloads a model (~90MB).
This is cached at `~/.cache/torch/sentence_transformers/`

### 3. Pre-warm Ollama

Run a test query to load the model into GPU memory:

```bash
ollama run mistral "test"
```

This keeps the model loaded for faster subsequent requests.

---

## Cost Optimization

### 1. Stop when not in use

```bash
# Stop the VPS from Google Cloud Console when not needed
gcloud compute instances stop YOUR_INSTANCE_NAME --zone=YOUR_ZONE

# Start when needed
gcloud compute instances start YOUR_INSTANCE_NAME --zone=YOUR_ZONE
```

### 2. Use Preemptible/Spot instances

T4 spot instances are ~70% cheaper. Just note they can be terminated.

### 3. Monitor costs

Set up billing alerts in Google Cloud Console:
**Billing** > **Budgets & alerts**

---

## Comparison: RunPod vs Google Cloud

| Feature | RunPod | Google Cloud |
|---------|--------|--------------|
| **GPU** | T4/A4000 | T4 (similar) |
| **Persistence** | Ephemeral (lost on restart) | Persistent disk ✅ |
| **Auto-restart** | Manual | Systemd service ✅ |
| **Networking** | Proxy (WebSocket issues) | Direct access ✅ |
| **Cost** | $0.xx/hr | $0.xx/hr (+ disk) |
| **Setup** | Docker-based | Full control ✅ |

**Key advantages on Google Cloud:**
- ✅ Data persists across restarts
- ✅ Auto-start on system boot
- ✅ No WebSocket proxy issues
- ✅ Full system access (sudo, systemd, etc.)
- ✅ Easier to debug and monitor

---

## Next Steps After Migration

1. **Test Phase 1 NLP features:**
   - Run keyword clustering (should be 10-20x faster with embeddings)
   - Run competitive analysis (check entity extraction)
   - Generate content brief (verify entity guidance appears)

2. **Install additional models (optional):**
   ```bash
   ollama pull llama3.1  # Alternative to Mistral
   ollama pull qwen2     # Another option
   ```

3. **Set up automated backups:**
   Create snapshots of your persistent disk in Google Cloud Console

4. **Monitor GPU utilization:**
   If GPU sits idle, consider using a smaller instance type to save costs

---

## Support

If you run into issues:

1. Check logs: `sudo journalctl -u seo-assistant -n 100`
2. Test Ollama: `curl http://localhost:11434/api/tags`
3. Test GPU: `nvidia-smi`
4. Test Python: `cd ~/seo-assistant && source venv/bin/activate && python -c "import spacy; import sentence_transformers; print('OK')"`

---

## Quick Command Reference

```bash
# Start application
sudo systemctl start seo-assistant

# Stop application
sudo systemctl stop seo-assistant

# Restart application
sudo systemctl restart seo-assistant

# View logs
sudo journalctl -u seo-assistant -f

# Check GPU
nvidia-smi

# Update code
cd ~/seo-assistant && git pull && sudo systemctl restart seo-assistant

# Access Python environment
cd ~/seo-assistant && source venv/bin/activate
```

---

You're all set! Your SEO assistant is now running on Google Cloud with persistent storage and auto-restart. 🚀
