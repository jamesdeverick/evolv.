# RunPod Deployment Guide

## Setting Up Environment Variables in RunPod

### Method 1: Using RunPod Web Interface (Before Deployment)

1. **Create/Edit Your Pod**
   - Go to https://runpod.io/console/pods
   - Click "Deploy" or edit existing pod

2. **Configure Environment Variables**
   - Scroll to "Environment Variables" section
   - Click "+ Add Environment Variable"
   - Add the following:

   ```
   Name: SCRAPINGDOG_API_KEY
   Value: 68da907d6dcca4eb91ea8469
   ```

   Optional (if using Deepseek):
   ```
   Name: DEEPSEEK_API_KEY
   Value: your_deepseek_key_here

   Name: LLM_PROVIDER
   Value: deepseek
   ```

3. **Deploy/Restart Pod**
   - Click "Deploy" or "Restart" for changes to take effect

### Method 2: Using Docker Run Command

If you're starting the container manually:

```bash
docker run -d \
  -e SCRAPINGDOG_API_KEY=68da907d6dcca4eb91ea8469 \
  -e LLM_PROVIDER=ollama \
  -p 7860:7860 \
  -p 11434:11434 \
  your-image-name
```

### Method 3: Enter in App UI (No Restart Required!)

The app has a built-in fallback:

1. Start the app
2. If no API key is found, you'll see a password input field
3. Paste your API key: `68da907d6dcca4eb91ea8469`
4. The app will use it for that session

**Note:** This method requires re-entering the key each time you restart the pod.

## Verifying Setup

Once configured, check the **sidebar** in the app:

✅ **Success:**
- "✓ Connected (HTTP 200)"
- Shows counts for Related, PAA, and Organic results

❌ **Still seeing errors:**
- Click "Refresh Scrapingdog Check" button
- Verify API key at: https://app.scrapingdog.com/dashboard
- Check account credits

## RunPod-Specific Notes

### GPU vs CPU

- **GPU Pods**: Ollama will use GPU automatically for faster inference
- **CPU Pods**: Set `OLLAMA_NUM_GPU=0` (already in Dockerfile)

### Persistent Storage

To keep Ollama models between pod restarts:

1. **Create RunPod Volume**
   - Go to Storage → Create Volume
   - Mount at: `/root/.ollama`

2. **Benefits:**
   - Models persist between restarts
   - No need to re-download Mistral (saves 4GB download)
   - Faster startup time

### Port Access

The app runs on:
- **Port 7860**: Streamlit web interface
- **Port 11434**: Ollama API (internal)

Make sure port 7860 is exposed in RunPod:
- RunPod usually auto-maps ports
- Access via: `https://[pod-id]-7860.proxy.runpod.net`

## Troubleshooting

### "Scrapingdog API key not found"

**Solutions:**
1. Set environment variable in RunPod interface (Method 1 above)
2. Enter key in app UI when prompted
3. Check key is valid: https://app.scrapingdog.com/dashboard

### "LLM Unavailable"

**Causes:**
- Ollama not started
- Mistral model not downloaded
- GPU memory issues

**Solutions:**
```bash
# SSH into pod or use terminal
ollama serve &  # Start Ollama
ollama pull mistral:latest  # Download model
ollama list  # Verify models
```

### Slow Performance

**If using CPU pod:**
- CPU inference is 5-15x slower than GPU
- Consider upgrading to GPU pod (A4000, A5000, or A6000)
- Or switch to cloud LLM: `LLM_PROVIDER=deepseek`

**Expected speeds:**
- **GPU**: 0.5-2 seconds per response
- **CPU**: 5-15 seconds per response

## Cost Optimization

### Using Cloud LLM (Recommended for CPU pods)

Instead of running Ollama, use Deepseek API:

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
```

**Benefits:**
- No GPU needed (use cheapest CPU pod)
- Faster responses than CPU Ollama
- Lower RunPod costs
- Pay per API call instead of 24/7 pod time

**Cost comparison:**
- **GPU Pod**: ~$0.39-0.89/hour (24/7 = $280-640/month)
- **CPU Pod + Deepseek**: ~$0.10/hour + API costs (~$20-50/month total)

## Best Practices

1. **Use persistent storage** for Ollama models
2. **Set environment variables** in RunPod (not .env file)
3. **Monitor credits** on both RunPod and Scrapingdog
4. **Stop pod** when not in use to save credits
5. **Consider cloud LLM** for cost savings on CPU pods

## Common Issue: "Not Ready" on Port 8888

**Problem:** RunPod shows "Port 8888 - Jupyter Lab - Not Ready"

**Solution:** The app runs on **port 7860**, not 8888!

### Quick Fix:

1. **Configure port 7860** in RunPod (not 8888)
2. **Or access directly:** `https://[pod-id]-7860.proxy.runpod.net`
3. **See:** [RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md) for detailed troubleshooting

### First Startup (5-10 minutes)

The first time you start the pod:
- Ollama downloads Mistral model (~4GB)
- This takes 5-10 minutes
- Watch the logs for progress
- Subsequent starts are fast (~30 seconds)

### Check If It's Running

```bash
# SSH into pod or use terminal
curl http://localhost:7860
# Should return HTML if Streamlit is running
```

## Support

- RunPod Docs: https://docs.runpod.io/
- Scrapingdog Dashboard: https://app.scrapingdog.com/dashboard
- Quick troubleshooting: [RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md)
- This repo's issues: https://github.com/jamesdeverick/evolv./issues
