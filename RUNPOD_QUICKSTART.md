# RunPod Quick Fix - "Not Ready" Error

## Problem

RunPod shows: **"Port 8888 - Jupyter Lab - Not Ready"**

This is because RunPod expects Jupyter Lab (port 8888) but your app runs Streamlit (port 7860).

---

## ✅ SOLUTION: Configure RunPod to Use Port 7860

### Method 1: Edit Pod Configuration (Recommended)

1. **Go to RunPod Console:** https://runpod.io/console/pods

2. **Stop your pod** (if running)

3. **Click "Edit" on your pod**

4. **Scroll to "Expose HTTP Ports"** or **"Port Mappings"**

5. **Remove port 8888** and **add port 7860**:
   ```
   7860
   ```

6. **Add environment variable** (while you're here):
   - Name: `SCRAPINGDOG_API_KEY`
   - Value: `68da907d6dcca4eb91ea8469`

7. **Save and Start the pod**

8. **Access your app at:**
   ```
   https://[pod-id]-7860.proxy.runpod.net
   ```

---

### Method 2: Use RunPod's "Connect" Button

1. **Start your pod**

2. **Click "Connect"**

3. **Look for "HTTP Services"**

4. **Click the port dropdown** and select **7860** (if available)

5. **Or manually construct the URL:**
   ```
   https://[your-pod-id]-7860.proxy.runpod.net
   ```

---

### Method 3: Access via Direct URL

Even if RunPod shows "Not Ready" on port 8888, your app might still be running on port 7860!

**Find your Pod ID:**
- It's in your RunPod dashboard (looks like: `abc123xyz456`)

**Access the app directly:**
```
https://[your-pod-id]-7860.proxy.runpod.net
```

**Example:**
```
https://abc123xyz456-7860.proxy.runpod.net
```

---

## 🔍 Troubleshooting - Is My App Actually Running?

### Check Container Logs

1. **In RunPod, click on your pod**

2. **Click "Logs" or "Terminal"**

3. **Look for these messages:**
   ```
   🚀 Starting Ollama server...
   ✅ Ollama is ready!
   ✅ Mistral model already available!
   🌟 Starting Streamlit app...
   ```

4. **You should see:**
   ```
   You can now view your Streamlit app in your browser.

   Network URL: http://0.0.0.0:7860
   ```

### If You Don't See These Messages

**The container might not be starting.** Check for errors:

```bash
# In RunPod terminal:
ps aux | grep streamlit
ps aux | grep ollama
```

### Start Manually (if needed)

```bash
# In RunPod terminal:
cd /app
./start.sh
```

---

## 🚨 Common Issues

### Issue 1: "Connection Refused"
**Cause:** Container not running or startup script failed

**Fix:**
1. Check logs for errors
2. Verify Dockerfile built correctly
3. Manually run: `./start.sh`

### Issue 2: "502 Bad Gateway"
**Cause:** App is starting (Ollama downloading model)

**Fix:**
- Wait 5-10 minutes for first startup (Mistral model download)
- Check logs to see download progress
- Refresh browser periodically

### Issue 3: Port 7860 Not Available
**Cause:** RunPod template using wrong ports

**Fix:**
- Use Custom Dockerfile deployment
- Ensure Dockerfile EXPOSE includes 7860
- Set start command to: `./start.sh`

---

## ⚡ Quick Test

Once you find the correct URL, you should see:

1. **Streamlit app loads** with title: "💡 Advanced LLM SEO Assistant"
2. **Sidebar shows:**
   - LLM status (might say "LLM Unavailable" if Mistral still downloading)
   - Scrapingdog status (needs API key)

3. **If you see a password prompt for Scrapingdog API:**
   - Paste: `68da907d6dcca4eb91ea8469`
   - You should see: ✅ "✓ Connected (HTTP 200)"

---

## 📋 Complete Setup Checklist

- [ ] Pod is running (green status in RunPod)
- [ ] Port 7860 is configured (not 8888)
- [ ] Environment variable `SCRAPINGDOG_API_KEY` is set
- [ ] Logs show "Starting Streamlit app"
- [ ] Can access: `https://[pod-id]-7860.proxy.runpod.net`
- [ ] Scrapingdog shows "✓ Connected"

---

## 🎯 Expected URLs

| Service | Port | URL Pattern |
|---------|------|-------------|
| **Streamlit** (main app) | 7860 | `https://[pod-id]-7860.proxy.runpod.net` |
| Ollama API (internal) | 11434 | Not exposed (internal only) |
| ~~Jupyter Lab~~ | ~~8888~~ | Not used (wrong template) |

---

## 💡 Pro Tips

1. **First startup takes 5-10 minutes** (downloading Mistral model ~4GB)
2. **Use persistent storage** to avoid re-downloading model each restart
3. **Stop pod when not in use** to save credits
4. **Bookmark your unique URL** for easy access

---

## Still Not Working?

### Get Help with These Details:

1. **Pod ID**: `[your-pod-id]`
2. **Error message** from logs
3. **Screenshot** of RunPod dashboard showing ports
4. **Output of:** `curl http://localhost:7860` (from pod terminal)

### Alternative: Use Cloud LLM Instead

If Ollama keeps failing, switch to Deepseek:

1. **Add environment variable:**
   - Name: `LLM_PROVIDER`
   - Value: `deepseek`

2. **Add Deepseek API key:**
   - Name: `DEEPSEEK_API_KEY`
   - Value: `[your-deepseek-key]`

3. **Benefit:** No model download needed, faster startup!

---

**Most Common Fix:** Just access `https://[pod-id]-7860.proxy.runpod.net` directly, ignoring the "Not Ready" message on port 8888! 🚀
