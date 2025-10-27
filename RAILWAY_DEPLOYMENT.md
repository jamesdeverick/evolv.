# 🚂 Railway Deployment Guide - Ollama + Mistral

Complete guide to deploying your SEO Assistant with Ollama and Mistral on Railway.

---

## 📋 Prerequisites

1. **Railway Account**: Sign up at https://railway.app/
2. **GitHub Account**: To connect your repository
3. **Scrapingdog API Key**: Get from https://www.scrapingdog.com/

---

## 💰 Cost Estimate

| Resource | Usage | Cost |
|----------|-------|------|
| **Hobby Plan** | $5 credit/month (free) | $0 |
| **Beyond free tier** | $0.000231/GB-hour RAM | ~$5-20/month |
| **Mistral Model** | ~4.1GB RAM | Included above |
| **Storage** | Models persist | Free (< 100GB) |

**First deployment**: Free with $5 hobby credit (enough for light usage)

---

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Code

Your code is already set up with these files:
- ✅ `Dockerfile` - Container definition with Ollama + Python
- ✅ `start.sh` - Startup script for both services
- ✅ `railway.json` - Railway configuration
- ✅ `.dockerignore` - Build optimization

**Commit and push to GitHub**:

```bash
cd /home/user/evolv.

# Commit Railway files
git add Dockerfile start.sh railway.json .dockerignore config.py
git commit -m "Add Railway deployment support with Ollama + Mistral"
git push
```

### Step 2: Create Railway Project

1. **Go to Railway**: https://railway.app/new

2. **Click "Deploy from GitHub repo"**

3. **Select your repository**: `jamesdeverick/evolv.`

4. **Select branch**: `claude/advanced-seo-assistant-011CUXXN6J98eQv5MBW1kFpj`

5. **Railway will auto-detect** the Dockerfile and start building

### Step 3: Configure Environment Variables

While the deployment is building, set up your environment:

1. **Click on your service** in Railway dashboard

2. **Go to "Variables" tab**

3. **Add these variables**:

   | Variable Name | Value | Notes |
   |--------------|-------|-------|
   | `SCRAPINGDOG_API_KEY` | `your_api_key_here` | Get from Scrapingdog dashboard |
   | `LLM_PROVIDER` | `ollama` | Use Ollama (default) |
   | `OLLAMA_MODEL` | `mistral:latest` | Model to use (default) |
   | `PORT` | `7860` | Streamlit port |

4. **Click "Add" for each variable**

### Step 4: Wait for Initial Build

**First build takes 10-20 minutes** because it:
- ✅ Downloads base Ubuntu image (~500MB)
- ✅ Installs Ollama (~300MB)
- ✅ Installs Python dependencies
- ✅ Pulls Mistral model (~4.1GB) - **This is the slow part**

**Watch the logs**:
- Click "Deployments" tab
- Click on the running deployment
- Monitor progress:
  ```
  🚀 Starting Ollama server...
  ✅ Ollama is ready!
  📦 Pulling Mistral model (this may take 5-10 minutes)...
  ✅ Mistral model downloaded successfully!
  🌟 Starting Streamlit app...
  ```

### Step 5: Generate Public URL

Once deployment succeeds:

1. **Go to "Settings" tab**

2. **Scroll to "Networking"**

3. **Click "Generate Domain"**

4. **Your app will be available at**: `https://your-app-name.up.railway.app`

---

## ✅ Verify Deployment

### 1. Check Health

Visit your Railway URL. You should see:
- ✅ Sidebar shows: **"✓ LLM Ready: ollama (mistral:latest)"**
- ✅ Scrapingdog status: Connected

### 2. Test Workflow

1. Enter a topic: "content marketing"
2. Click "Proceed to Keyword Research"
3. Verify keywords are generated
4. Test LLM analysis in Step 3
5. Generate a content brief in Step 4

If all steps work, **deployment is successful!** 🎉

---

## 🔧 Configuration Options

### Using Different Models

Edit environment variables in Railway:

```bash
# Use Mistral 7B Instruct (better for instructions)
OLLAMA_MODEL=mistral:7b-instruct

# Use Llama 3 (8B)
OLLAMA_MODEL=llama3:8b

# Use Llama 3 (70B) - requires more RAM ($$$)
OLLAMA_MODEL=llama3:70b
```

**Note**: Larger models require more RAM = higher costs

### Adjust Timeouts

If you get timeout errors with large briefs:

```bash
LLM_TIMEOUT=180  # Increase from 120 to 180 seconds
```

---

## 🐛 Troubleshooting

### Problem 1: "Build Failed" During Docker Build

**Symptoms**: Build fails with error messages

**Common Causes**:
```
Error: failed to solve: executor failed running...
```

**Solutions**:
1. Check Dockerfile syntax is correct
2. Verify all files are committed to git
3. Try "Deploy Again" button
4. Check Railway status: https://railway.statuspage.io/

### Problem 2: "Application Failed to Respond"

**Symptoms**: After deploy, app shows error page

**Check Logs**:
1. Go to "Deployments" → Click deployment → View logs
2. Look for errors:

```bash
# Good signs:
✅ Ollama is ready!
✅ Mistral model downloaded successfully!
🌟 Starting Streamlit app...

# Bad signs:
❌ Ollama failed to start
curl: connection refused
Error: model not found
```

**Solutions**:

**If Ollama won't start**:
```bash
# Increase health check timeout
# In railway.json:
"healthcheckTimeout": 600  # Increase to 10 minutes
```

**If model download fails**:
- Check Railway has enough disk space (should be fine < 10GB)
- Try smaller model: `OLLAMA_MODEL=mistral:7b-instruct`
- Redeploy (sometimes network issues)

**If Streamlit won't start**:
```bash
# Check logs for Python errors
# Common: Missing dependency
# Solution: Add to requirements.txt and redeploy
```

### Problem 3: "Out of Memory"

**Symptoms**: App crashes during model loading or inference

**Check Memory Usage**:
1. Railway dashboard shows memory graph
2. Mistral 7B needs ~4-5GB RAM minimum

**Solutions**:

**Option A**: Upgrade Railway plan
```
Hobby: Up to 8GB RAM (included in $5 credit)
Pro: Up to 32GB RAM (pay-as-you-go)
```

**Option B**: Use smaller model
```bash
# Try quantized version (uses less RAM)
OLLAMA_MODEL=mistral:7b-q4  # ~2.5GB RAM
```

**Option C**: Reduce concurrent users
- Railway free tier handles 1-2 concurrent users
- For more, upgrade plan

### Problem 4: Model Download Takes Forever

**Symptoms**: "Pulling Mistral model..." stuck for 20+ minutes

**This is normal for first deploy!**
- Mistral 7B is ~4.1GB
- Railway servers are fast, but still takes 5-10 min
- Model is cached for future deployments

**To speed up**:
1. Use smaller model: `mistral:7b-q4` (~2.5GB)
2. Wait patiently on first deploy
3. Subsequent deploys will be fast (model is cached)

### Problem 5: "Scrapingdog Not OK"

**Symptoms**: Sidebar shows Scrapingdog error

**Solutions**:
1. Verify `SCRAPINGDOG_API_KEY` is set in Railway Variables
2. Check Scrapingdog dashboard for credits
3. Test API key manually:
   ```bash
   curl "https://api.scrapingdog.com/google?api_key=YOUR_KEY&query=test"
   ```
4. Click "Refresh Scrapingdog Check" in app

### Problem 6: High Costs

**Symptoms**: Railway bill higher than expected

**Check Usage**:
1. Railway dashboard → "Usage"
2. Look at:
   - RAM usage over time
   - CPU usage
   - Network egress

**Reduce Costs**:

```bash
# 1. Use smaller model
OLLAMA_MODEL=mistral:7b-q4  # Saves ~50% RAM

# 2. Set sleep/scale to zero
# In Railway: Settings → Enable "Sleep when inactive"

# 3. Limit keyword rows (reduces processing)
# In code: config.py
MAX_KEYWORD_ROWS=20  # Reduce from 50

# 4. Use Hobby plan limits
# Railway will sleep after inactivity (free tier)
```

**Cost Monitoring**:
- Set spending limit in Railway settings
- Enable email alerts for usage

---

## 🔄 Updating Your Deployment

### Method 1: Git Push (Recommended)

```bash
# Make changes locally
cd /home/user/evolv.
vim app.py  # or any file

# Commit and push
git add .
git commit -m "Update: description of changes"
git push

# Railway auto-deploys (takes 2-5 min after push)
```

### Method 2: Railway Dashboard

1. Go to "Deployments" tab
2. Click "Deploy" button
3. Select "Redeploy" to rebuild

### Method 3: Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link project
railway link

# Deploy
railway up
```

---

## 📊 Performance Optimization

### 1. Model Selection by Use Case

| Model | RAM | Speed | Quality | Best For |
|-------|-----|-------|---------|----------|
| `mistral:7b-q4` | 2.5GB | Fast | Good | High traffic, cost-sensitive |
| `mistral:latest` | 4.1GB | Medium | Better | Balanced (recommended) |
| `llama3:8b` | 5GB | Medium | Better | More accurate |
| `llama3:70b` | 40GB | Slow | Best | Quality-first ($$$$) |

### 2. Caching Strategy

App already uses Streamlit caching:
```python
# In api/scrapingdog_client.py
@st.cache_data(ttl=1800)  # 30 min cache
```

**To increase cache** (saves API calls):
```python
# In config.py
CACHE_TTL = 3600  # Increase to 1 hour
```

### 3. Railway Sleep Settings

**Free Tier**: Enable sleep when inactive
- App sleeps after 5 min inactivity
- Wakes on next request (cold start ~10-20s)
- Saves costs

**Pro Tier**: Disable sleep for always-on

---

## 🔐 Security Best Practices

### 1. Environment Variables

✅ **DO**:
- Store all secrets in Railway Variables
- Use Railway's built-in secrets management
- Rotate API keys every 3-6 months

❌ **DON'T**:
- Commit `.env` files to git
- Hardcode API keys in code
- Share Railway project publicly with secrets

### 2. Access Control

Railway allows:
- **Private deployments** (default)
- **Password protection** (Settings → Add password)
- **IP allowlisting** (Pro plan)

**Recommendation**: Add password if using for clients

### 3. Rate Limiting

To prevent abuse:

```python
# Add to app.py
import streamlit as st
from datetime import datetime, timedelta

# Simple rate limiting
if "last_request" not in st.session_state:
    st.session_state.last_request = datetime.now()
else:
    time_diff = datetime.now() - st.session_state.last_request
    if time_diff < timedelta(seconds=5):
        st.error("Please wait a few seconds between requests")
        st.stop()
st.session_state.last_request = datetime.now()
```

---

## 📈 Monitoring

### 1. Built-in Railway Metrics

Railway dashboard shows:
- **CPU usage** over time
- **Memory usage** (watch for OOM)
- **Deployment history** (success/fail)
- **Logs** (real-time and historical)

### 2. Custom Logging

Add to your code:

```python
# In app.py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log important events
logger.info(f"User analyzed keyword: {keyword}")
logger.info(f"Brief generated for: {topic}")
```

View logs in Railway: Deployments → Click deployment → View logs

### 3. Uptime Monitoring

Use external service:
- **UptimeRobot** (free): https://uptimerobot.com/
- **BetterUptime** (free tier): https://betteruptime.com/

Set up HTTP check every 5 minutes to your Railway URL.

---

## 🚀 Advanced Configuration

### 1. Custom Domain

**Connect your own domain**:

1. Go to Settings → Networking
2. Click "Custom Domain"
3. Add domain: `seo.yourdomain.com`
4. Add CNAME record in your DNS:
   ```
   CNAME seo.yourdomain.com → your-app.up.railway.app
   ```
5. SSL automatically provisioned

### 2. Multiple Environments

**Create staging + production**:

```bash
# In Railway, create two services:
1. seo-assistant-staging (your current branch)
2. seo-assistant-production (main branch)

# Each with own variables:
# Staging: Test API keys, smaller model
# Production: Real API keys, full model
```

### 3. Horizontal Scaling

**For high traffic** (Pro plan):

1. Railway → Settings → Scaling
2. Enable "Horizontal Autoscaling"
3. Set replicas: 1-5 (based on traffic)

**Note**: Each replica needs full model in memory = $$

---

## 💡 Tips & Best Practices

### 1. First Deployment

- ✅ Expect 10-20 min for first deploy (model download)
- ✅ Watch logs to see progress
- ✅ Don't cancel if it seems slow - model download takes time
- ✅ Subsequent deploys are faster (5-7 min)

### 2. Development Workflow

```bash
# 1. Test locally first
streamlit run app.py

# 2. Commit when working
git add .
git commit -m "Fix: description"

# 3. Push to trigger Railway deploy
git push

# 4. Monitor logs on Railway
# 5. Test on Railway URL
```

### 3. Cost Management

**Free tier ($5/month credit)**:
- Use `mistral:7b-q4` model (smaller)
- Enable "Sleep when inactive"
- Limit to personal use

**Paid tier (if needed)**:
- Monitor usage weekly
- Set spending limits
- Use appropriate model size

### 4. Model Persistence

**Models are persistent**:
- First deploy downloads Mistral (~10 min)
- Model is saved in container volume
- Subsequent deploys reuse model (fast)
- Only downloads again if you change model

---

## 🆘 Getting Help

### Railway Support

- **Documentation**: https://docs.railway.app/
- **Discord**: https://discord.gg/railway
- **Forum**: https://help.railway.app/

### Common Resources

- **Railway Status**: https://railway.statuspage.io/
- **Ollama Docs**: https://ollama.com/docs
- **Pricing Calculator**: https://railway.app/pricing

---

## 📋 Deployment Checklist

Before going live:

- [ ] Code pushed to GitHub
- [ ] Railway project created
- [ ] Environment variables set:
  - [ ] `SCRAPINGDOG_API_KEY`
  - [ ] `LLM_PROVIDER=ollama`
  - [ ] `OLLAMA_MODEL=mistral:latest`
- [ ] First deployment completed successfully
- [ ] Ollama status shows "Ready"
- [ ] Scrapingdog status shows "OK"
- [ ] Test complete workflow (all 4 steps)
- [ ] Public URL generated
- [ ] (Optional) Custom domain configured
- [ ] (Optional) Password protection enabled
- [ ] Monitoring set up (UptimeRobot)
- [ ] Cost alerts configured

---

## 🎉 Success Criteria

Your deployment is successful when:

✅ Railway URL loads the app
✅ Sidebar shows: "✓ LLM Ready: ollama (mistral:latest)"
✅ Scrapingdog status: "Connected"
✅ Can complete full workflow (topic → keywords → analysis → brief)
✅ App responds within reasonable time (< 30s per step)
✅ No crashes or errors in logs

---

## 📊 Cost Comparison: Railway vs Alternatives

| Platform | Ollama Support | Monthly Cost | Setup Difficulty |
|----------|---------------|--------------|------------------|
| **Railway** | ✅ Yes | $5-20 | Easy |
| HuggingFace Spaces | ❌ No | Free | Easy |
| Modal.com | ✅ Yes | Pay-per-use | Medium |
| DigitalOcean | ✅ Yes | $12-40 | Hard |
| AWS EC2 | ✅ Yes | $20-100 | Hard |

**Railway is the sweet spot**: Easy setup, reasonable cost, full Docker support.

---

## 🔄 Next Steps After Deployment

1. **Share your app**: Send Railway URL to team/clients
2. **Monitor usage**: Check Railway dashboard weekly
3. **Gather feedback**: Improve based on user needs
4. **Optimize costs**: Adjust model size based on usage
5. **Set up staging**: Create separate staging environment for testing

---

**Questions?**
- Check Railway docs: https://docs.railway.app/
- Or ask in Railway Discord: https://discord.gg/railway

🚂 Happy deploying!
