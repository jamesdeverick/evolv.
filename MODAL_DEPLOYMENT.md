# 🚀 Modal.com Deployment - Ollama + Mistral with GPU

Deploy your SEO Assistant with **serverless GPU** - pay only when running!

**Perfect for your use case**: 10 min/day = **~$5.50/month**

---

## 💰 Cost Breakdown

**Your usage**: 10 minutes/day

| Resource | Rate | Monthly Usage | Cost |
|----------|------|---------------|------|
| **A10G GPU** | $1.10/hour | ~5 hours | **$5.50** |
| **Storage** | Free | Model cache | $0 |
| **Network** | Free | < 100GB | $0 |
| **First month** | $30 credit | Free trial | **$0** |

**Total: ~$5.50/month** (first month free with $30 credit!)

**Compare to:**
- Railway CPU (too slow): $0-5/month
- RunPod GPU (24/7): $240/month ❌
- Vast.ai GPU (24/7): $150/month ❌
- Modal GPU (pay-per-use): **$5.50/month** ✅

---

## 📋 Prerequisites

1. **Modal account**: https://modal.com/signup
2. **Scrapingdog API key**: https://www.scrapingdog.com/
3. **Your code** (already set up!)

---

## 🚀 Step-by-Step Deployment

### Step 1: Install Modal CLI

```bash
pip install modal
```

**Verify installation:**
```bash
modal --version
```

### Step 2: Authenticate with Modal

```bash
modal token new
```

This opens your browser to authenticate. Follow the prompts.

**Expected output:**
```
✓ Created token successfully
✓ Saved token to ~/.modal.toml
```

### Step 3: Create Modal Secret for Scrapingdog

Modal needs your Scrapingdog API key as a secret:

```bash
modal secret create scrapingdog-api SCRAPINGDOG_API_KEY=your_api_key_here
```

**Replace** `your_api_key_here` with your actual Scrapingdog key.

**Verify:**
```bash
modal secret list
```

Should show: `scrapingdog-api`

### Step 4: Deploy to Modal

From your project directory:

```bash
cd /path/to/evolv.

# Deploy (this takes 5-10 minutes first time)
modal deploy modal_deploy.py
```

**What happens:**
1. ✅ Uploads your code to Modal
2. ✅ Builds Docker image with CUDA + Ollama
3. ✅ Creates persistent volume for models
4. ✅ Deploys to Modal's infrastructure

**First deploy takes 5-10 minutes** (builds image)

**Expected output:**
```
✓ Initialized. View run at https://modal.com/...
✓ Created objects:
  ├── 🔨 Created mount /Users/...
  ├── 🔨 Created seo-assistant::image
  ├── 🔨 Created seo-assistant::ollama-models volume
  └── 🔨 Created function web
✓ App deployed! 🎉

View app: https://your-username--seo-assistant-web.modal.run
```

### Step 5: Access Your App

**Your app URL** (shown in deploy output):
```
https://your-username--seo-assistant-web.modal.run
```

**First access** (cold start):
- Takes ~30-60 seconds
- Downloads Mistral model (~5 min first time only)
- Then instant!

**Subsequent access**:
- If within 10 min: Instant (warm)
- If after 10 min: ~30s cold start (model cached, no redownload)

---

## ✅ Verify It Works

### 1. Open Your Modal URL

Visit the URL from Step 5

### 2. Check Sidebar

Should show:
- ✅ **"✓ LLM Ready: ollama (mistral:latest)"**
- ✅ **"✓ Scrapingdog: Connected"**

### 3. Test Workflow

1. Enter topic: "content marketing"
2. Click "Proceed to Keyword Research" (Step 2)
3. Verify keywords generated
4. Test "Run LLM Analysis" (Step 3)
5. Generate brief (Step 4)

**If all works**: Success! 🎉

---

## 🔧 Configuration

### Change GPU Type

Edit `modal_deploy.py`:

```python
# Current: A10G ($1.10/hour)
gpu="A10G"

# Options:
gpu="T4"     # $0.60/hour (slower, cheaper)
gpu="A100"   # $4.00/hour (faster, expensive)
```

**Recommendation**: Stick with A10G (best value)

### Adjust Idle Timeout

```python
container_idle_timeout=600  # 10 minutes (default)
```

**Options:**
- `300` (5 min): Saves $ if sporadic use
- `600` (10 min): Better UX for consecutive sessions
- `1200` (20 min): If you do multiple briefs in a row

**For 10 min/day usage**: Keep at 600 (10 min)

### Use Different Model

Edit `modal_deploy.py`:

```python
# In the web() function, change:
subprocess.run(["ollama", "pull", "llama3:8b"], check=True)
```

**Available models:**
- `mistral:latest` (default, 4.1GB, great quality)
- `mistral:7b-instruct` (4.1GB, better at following instructions)
- `llama3:8b` (4.7GB, Meta's model)
- `llama3:70b` (requires more RAM, $$)

---

## 🐛 Troubleshooting

### Problem 1: "Secret not found"

**Error:**
```
modal.exception.NotFoundError: Secret 'scrapingdog-api' not found
```

**Solution:**
```bash
modal secret create scrapingdog-api SCRAPINGDOG_API_KEY=your_actual_key
```

Make sure secret name matches what's in `modal_deploy.py`:
```python
secrets=[modal.Secret.from_name("scrapingdog-api")]
```

### Problem 2: Cold Start Too Slow

**Symptoms**: First request takes 60+ seconds

**This is normal!** Modal needs to:
1. Start container (~20s)
2. Start Ollama (~10s)
3. Load model to GPU (~20s)

**Solutions:**
- Increase `container_idle_timeout` to keep warm longer
- Or just accept the cold start (only happens after 10 min idle)

### Problem 3: Model Download Stuck

**Symptoms**: First deploy shows "Pulling Mistral..." for 10+ minutes

**This is expected!**
- Mistral 7B is 4.1GB
- Downloads once, then cached forever
- Subsequent deploys reuse cached model

**Check progress:**
```bash
modal logs seo-assistant-web
```

### Problem 4: Out of Memory

**Error:**
```
CUDA out of memory
```

**Solutions:**
1. Use smaller model:
   ```python
   subprocess.run(["ollama", "pull", "mistral:7b-q4"], check=True)
   ```

2. Upgrade to A100:
   ```python
   gpu="A100"  # 40GB VRAM instead of 24GB
   ```

### Problem 5: Deployment Fails

**Check logs:**
```bash
modal logs seo-assistant-web
```

**Common issues:**
- Missing files (check copy_local_dir paths)
- Invalid Modal token (run `modal token new`)
- Network timeout (retry deployment)

---

## 📊 Monitoring Usage

### View Live Logs

```bash
modal logs seo-assistant-web --follow
```

Shows real-time output from your app.

### Check Costs

1. Go to https://modal.com/home
2. Click "Usage" tab
3. See GPU hours used

**Expected for 10 min/day:**
- Daily: ~0.17 hours ($0.19)
- Monthly: ~5 hours ($5.50)

### Set Spending Limit

1. Modal dashboard → Settings
2. Set monthly limit (e.g., $10)
3. Get email alerts

---

## 🔄 Updating Your App

### Method 1: Redeploy

```bash
# Make changes locally
vim app.py

# Redeploy
modal deploy modal_deploy.py
```

Modal rebuilds and deploys. Takes 2-5 min.

### Method 2: Live Reload (Development)

```bash
modal serve modal_deploy.py
```

Auto-redeploys on file changes. Great for development!

---

## 🎯 Performance Expectations

### Cold Start (after 10+ min idle)

- Container start: ~20s
- Ollama start: ~10s
- Model load: ~20s
- **Total: ~50-60s**

### Warm (within 10 min)

- **Response time: < 1s** ⚡
- Brief generation: ~2-3s
- Total workflow: ~10-15s

### GPU Performance

**Mistral 7B on A10G:**
- Token generation: ~50-100 tokens/sec
- Brief (500 tokens): ~5-10 seconds
- **Feels instant** compared to CPU!

---

## 💡 Optimization Tips

### 1. Batch Your Work

Instead of generating 1 brief/day:
- Generate 10 briefs once/week
- Uses same GPU time
- Amortizes cold start cost

### 2. Keep Container Warm

If doing multiple briefs:
```python
container_idle_timeout=1200  # 20 min
```

Stays warm between briefs.

### 3. Use Smaller Model for Testing

Development/testing:
```python
subprocess.run(["ollama", "pull", "mistral:7b-q4"])  # 2.5GB, faster
```

Production:
```python
subprocess.run(["ollama", "pull", "mistral:latest"])  # 4.1GB, better quality
```

### 4. Monitor and Adjust

Check usage after first week:
- If over budget → Reduce idle timeout
- If cold starts annoying → Increase idle timeout
- If quality issues → Try llama3:8b

---

## 🔐 Security Best Practices

### 1. Never Commit Secrets

```bash
# .gitignore should have:
.modal.toml
*.env
```

### 2. Rotate API Keys

- Change Scrapingdog key every 3-6 months
- Update Modal secret:
  ```bash
  modal secret create scrapingdog-api SCRAPINGDOG_API_KEY=new_key --force
  ```

### 3. Restrict Access

Modal apps are public by default. To add auth:

```python
# In modal_deploy.py
from modal import web_endpoint

@app.function(...)
@web_endpoint(method="GET", auth={"username": "admin", "password": "your_password"})
def web():
    ...
```

---

## 📈 Scaling

### Current Setup (Low Usage)

- **10 min/day**: Perfect! ~$5.50/month
- **Single user**: No issues
- **Cold starts**: Acceptable

### If Usage Increases

**50 min/day** (~$27.50/month):
- Still cost-effective
- Consider keeping container warm longer

**4+ hours/day** (~$132/month):
- Might be cheaper to use dedicated GPU (RunPod)
- But Modal still easier to manage

**Multiple concurrent users**:
```python
allow_concurrent_inputs=10  # Handle 10 users simultaneously
```

Modal auto-scales containers as needed.

---

## 🆚 Modal vs Railway

| Feature | Modal (GPU) | Railway (CPU) |
|---------|-------------|---------------|
| **GPU/CUDA** | ✅ NVIDIA A10G | ❌ None |
| **Response time** | 0.5-1s | 5-15s |
| **Your cost** | $5.50/month | $0-5/month |
| **Cold start** | 50-60s | 10-20s |
| **Setup** | Medium | Easy |
| **Best for** | Quality + Speed | Budget only |

**Verdict**: Modal is worth the extra $0.50/month for 10x better performance!

---

## 🎉 Success Checklist

After deployment, verify:

- [ ] Modal deploy completed successfully
- [ ] App URL works
- [ ] Sidebar shows Ollama ready
- [ ] Sidebar shows Scrapingdog connected
- [ ] Can generate keywords (Step 2)
- [ ] Can run LLM analysis (Step 3)
- [ ] Can generate content brief (Step 4)
- [ ] Brief quality is good (better than Deepseek!)
- [ ] Response times acceptable (< 3s per brief)
- [ ] Cold start acceptable (~60s)

---

## 🆘 Getting Help

### Modal Support

- **Docs**: https://modal.com/docs
- **Discord**: https://discord.gg/modal (very active!)
- **Examples**: https://modal.com/docs/examples

### Common Commands

```bash
# Deploy
modal deploy modal_deploy.py

# View logs
modal logs seo-assistant-web

# Stop app
modal app stop seo-assistant

# Delete app (removes everything)
modal app delete seo-assistant

# List secrets
modal secret list

# Update secret
modal secret create scrapingdog-api SCRAPINGDOG_API_KEY=new_key --force
```

---

## 📝 Quick Reference

### Deployment Flow

```bash
# 1. Install
pip install modal

# 2. Authenticate
modal token new

# 3. Create secret
modal secret create scrapingdog-api SCRAPINGDOG_API_KEY=your_key

# 4. Deploy
modal deploy modal_deploy.py

# 5. Visit URL shown in output
```

### Cost Formula

```
Cost = GPU_hours × $1.10

Your usage:
- 10 min/day = 0.17 hours/day
- 5 hours/month
- $5.50/month
```

### File Structure

```
evolv/
├── modal_deploy.py       # ⭐ Modal configuration
├── app.py               # Streamlit app
├── config.py            # Settings (LLM_PROVIDER=ollama)
├── api/                 # API clients
├── utils/               # Utilities
└── analysis/            # Analysis modules
```

---

## 🚀 You're Ready!

**Next steps:**

1. Run the deployment commands above
2. Wait 5-10 min for first deploy
3. Visit your Modal URL
4. Test the app
5. Generate your first SEO brief with GPU power! ⚡

**Questions?** Check Modal Discord - they're very helpful!

---

**Enjoy your serverless GPU SEO assistant!** 🎉

Pay $5.50/month for 10x better performance than CPU. Worth it!
