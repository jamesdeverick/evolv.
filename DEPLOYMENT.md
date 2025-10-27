# Deployment Guide: HuggingFace Spaces

This guide explains how to deploy the Advanced LLM SEO Assistant to HuggingFace Spaces.

## Why Your App Wasn't Starting

Your original app was configured to use **Ollama**, which requires a locally running server. HuggingFace Spaces:

- ❌ **Cannot run Ollama** - No persistent background services allowed
- ❌ **T4 GPU doesn't help** - Ollama needs daemon installation, not just GPU access
- ✅ **Solution**: Use cloud-based LLM APIs (Deepseek or OpenAI)

## Step-by-Step Deployment

### 1. Get API Keys

#### Scrapingdog (Required)
1. Go to https://www.scrapingdog.com/
2. Sign up for an account
3. Copy your API key from the dashboard
4. **Pricing**: Free tier available, paid plans start at $25/month

#### Deepseek (Recommended for LLM)
1. Go to https://platform.deepseek.com/
2. Create an account
3. Navigate to API Keys section
4. Generate a new API key
5. **Pricing**: ~$0.14 per million tokens (very affordable)
   - A typical session costs $0.01-0.05
   - Much cheaper than OpenAI GPT-4 ($10/million tokens)

#### Alternative: OpenAI (Optional)
1. Go to https://platform.openai.com/
2. Create an account
3. Add payment method
4. Generate API key
5. **Pricing**: ~$10/million tokens for GPT-4
   - More expensive but potentially higher quality
   - A typical session costs $0.50-2.00

### 2. Create HuggingFace Space

1. **Go to HuggingFace**: https://huggingface.co/new-space

2. **Configure Space**:
   - **Space name**: `seo-assistant` (or your choice)
   - **License**: Choose your preferred license
   - **Select SDK**: **Streamlit**
   - **Space hardware**: **CPU basic** (free tier is sufficient)
     - Note: Don't waste money on GPU - you're using API calls, not local inference

3. **Create Space**

### 3. Upload Code

#### Option A: Via Web Interface

1. Click "Files" tab in your Space
2. Upload these files/folders:
   ```
   app.py
   config.py
   requirements.txt
   .streamlit/config.toml
   api/
   utils/
   analysis/
   ```

#### Option B: Via Git (Recommended)

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy refactored code
cp -r /path/to/evolv/* .

# Commit and push
git add .
git commit -m "Deploy refactored SEO assistant"
git push
```

### 4. Configure Secrets

This is the **most important step** - your app won't work without proper secrets!

1. **Go to Settings**: In your Space, click "Settings" tab

2. **Scroll to "Repository secrets"**

3. **Add the following secrets**:

   | Name | Value | Example |
   |------|-------|---------|
   | `LLM_PROVIDER` | `deepseek` | `deepseek` |
   | `DEEPSEEK_API_KEY` | Your Deepseek API key | `sk-abc123...` |
   | `SCRAPINGDOG_API_KEY` | Your Scrapingdog API key | `64f7e2a1b9c...` |

4. **Click "Save"** after adding each secret

### 5. Verify Deployment

1. **Wait for build**: HuggingFace will automatically rebuild your Space
   - This takes 2-5 minutes
   - Watch the "Logs" tab for progress

2. **Check for errors**:
   ```
   # Good signs in logs:
   ✓ Installing requirements...
   ✓ Starting Streamlit...
   ✓ You can now view your Streamlit app in your browser

   # Bad signs:
   ✗ ModuleNotFoundError
   ✗ API key not found
   ✗ Connection timeout
   ```

3. **Test the app**:
   - Click "App" tab
   - You should see the sidebar with "✓ LLM Ready: deepseek"
   - Try entering a query and proceeding through steps

## Troubleshooting

### Problem: "LLM Unavailable" Error

**Symptoms**: Sidebar shows "✗ LLM Unavailable: DEEPSEEK_API_KEY not configured"

**Solutions**:
1. Verify secret name is exactly `DEEPSEEK_API_KEY` (case-sensitive)
2. Check that you saved the secret
3. Rebuild the Space (Settings → "Factory reboot")
4. Verify API key is valid: https://platform.deepseek.com/api_keys

### Problem: "Scrapingdog Not OK"

**Symptoms**: Sidebar shows "✗ Not OK (HTTP 401)" or "✗ Not OK (HTTP 403)"

**Solutions**:
1. Verify `SCRAPINGDOG_API_KEY` is correct
2. Check your Scrapingdog account has credits
3. Try the "Refresh Scrapingdog Check" button
4. Verify at: https://www.scrapingdog.com/dashboard

### Problem: App Shows "Starting..." Forever

**Symptoms**: Space stuck on loading screen

**Solutions**:
1. Check "Logs" tab for error messages
2. Common issues:
   - Missing `requirements.txt`
   - Import errors (missing dependencies)
   - Syntax errors in code
3. Try "Factory reboot" in Settings

### Problem: "ModuleNotFoundError"

**Symptoms**: Logs show missing module errors

**Solutions**:
1. Verify `requirements.txt` is uploaded
2. Check file contains all dependencies:
   ```
   streamlit>=1.28.0
   pandas>=2.0.0
   requests>=2.31.0
   beautifulsoup4>=4.12.0
   litellm>=1.17.0
   PyPDF2>=3.0.0
   defusedxml>=0.7.1
   plotly>=5.14.0
   ```
3. Force rebuild: push an empty commit
   ```bash
   git commit --allow-empty -m "Force rebuild"
   git push
   ```

### Problem: High API Costs

**Symptoms**: Deepseek/OpenAI bills are higher than expected

**Solutions**:
1. **Use Deepseek** instead of OpenAI (70x cheaper)
2. **Monitor usage**:
   - Deepseek dashboard: https://platform.deepseek.com/usage
   - OpenAI dashboard: https://platform.openai.com/usage
3. **Set limits**:
   - Deepseek: Set monthly budget limit
   - OpenAI: Set hard/soft limits
4. **Cache settings**: App already uses `@st.cache_data` for efficiency

## Cost Estimates

### Deepseek (Recommended)

| Usage | Tokens | Cost |
|-------|--------|------|
| Single complete workflow | ~50K | $0.007 |
| 10 workflows/day × 30 days | ~15M | $2.10/month |
| Heavy use (50 workflows/day) | ~75M | $10.50/month |

### OpenAI GPT-4

| Usage | Tokens | Cost |
|-------|--------|------|
| Single complete workflow | ~50K | $0.50 |
| 10 workflows/day × 30 days | ~15M | $150/month |
| Heavy use (50 workflows/day) | ~75M | $750/month |

**Recommendation**: Start with Deepseek. Only switch to OpenAI if you need higher quality and can justify the 70x cost increase.

## Performance Tips

### 1. Choose Right Hardware

- **CPU basic (free)**: Sufficient for this app
  - All processing is API calls
  - No local model inference
- **Don't upgrade to GPU**: Waste of money for this use case

### 2. Monitor Caching

The app uses Streamlit caching:
- SERP data: 30 min cache
- Scrapingdog status: 5 min cache
- This reduces API calls and costs

### 3. Set Reasonable Limits

In `config.py`, you can adjust:
```python
MAX_KEYWORD_ROWS = 50  # Reduce to 20-30 for faster processing
MAX_COMPETITORS = 10   # Usually 3-5 is enough
```

## Security Best Practices

### 1. Never Commit API Keys

The `.gitignore` file excludes:
```
.env
.streamlit/secrets.toml
```

### 2. Use HuggingFace Secrets

- ✅ Store all API keys in Space Secrets
- ❌ Never hardcode keys in code
- ❌ Never commit `.env` files

### 3. Rotate Keys Regularly

- Change API keys every 3-6 months
- Immediately rotate if key is exposed

## Updating Your Deployment

### Method 1: Web Interface

1. Go to "Files" tab
2. Click on file to edit
3. Make changes
4. Commit

### Method 2: Git Push

```bash
# Make changes locally
cd /path/to/your/space

# Edit files
vim app.py

# Commit and push
git add .
git commit -m "Update: describe your changes"
git push
```

Space will automatically rebuild after push.

## Monitoring

### 1. Check Logs Regularly

- Go to "Logs" tab in your Space
- Look for errors or warnings
- Monitor API call failures

### 2. Test Critical Paths

Weekly testing checklist:
- [ ] LLM status shows "Ready"
- [ ] Scrapingdog status is "OK"
- [ ] Keyword research returns results
- [ ] SERP analysis works
- [ ] Brief generation completes
- [ ] File uploads work (ToV documents)

### 3. Monitor API Usage

- **Deepseek**: https://platform.deepseek.com/usage
- **Scrapingdog**: https://www.scrapingdog.com/dashboard

Set up budget alerts to avoid surprise bills.

## Advanced Configuration

### Using OpenAI Instead

If you prefer OpenAI:

1. **Change secret**:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   ```

2. **Update config.py** (optional):
   ```python
   DEFAULT_LLM_PROVIDER = "openai"
   OPENAI_MODEL = "gpt-4-turbo-preview"  # or "gpt-3.5-turbo" for lower cost
   ```

### Custom Model Settings

Edit `config.py`:
```python
# For faster responses (lower quality)
DEEPSEEK_MODEL = "deepseek-chat"  # Already default

# For more detailed analysis
LLM_TIMEOUT = 180  # Increase timeout for longer responses
```

## Support Resources

- **HuggingFace Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Streamlit Docs**: https://docs.streamlit.io/
- **Deepseek Docs**: https://platform.deepseek.com/docs
- **Scrapingdog Docs**: https://www.scrapingdog.com/documentation

## Next Steps

After successful deployment:

1. **Share your Space**: Make it public or share link with team
2. **Customize branding**: Edit Streamlit theme in `.streamlit/config.toml`
3. **Add analytics**: Track usage with Streamlit analytics
4. **Monitor costs**: Set up budget alerts
5. **Gather feedback**: Improve based on user needs

---

**Questions?** Check the main README.md or open an issue on the repository.
