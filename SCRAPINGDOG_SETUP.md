# Scrapingdog API Setup Guide

## What is Scrapingdog?

Scrapingdog is a SERP (Search Engine Results Page) API that allows you to fetch Google search results programmatically. This app uses it for keyword research and competitive analysis.

## Getting Your API Key

### Step 1: Sign Up
1. Visit https://www.scrapingdog.com/
2. Click "Sign Up" or "Get Started"
3. Create an account (they offer a free trial)

### Step 2: Get Your API Key
1. Log in to your account
2. Go to your dashboard: https://app.scrapingdog.com/dashboard
3. Find your API key (it will look like: `64f7e2a1b9c8d3e4f5a6b7c8d9e0f1a2`)
4. Copy it to your clipboard

## Configuring the App

### Option 1: Using .env File (Recommended for Local Development)

1. Open the `.env` file in the project root
2. Find the line: `SCRAPINGDOG_API_KEY=your_scrapingdog_api_key_here`
3. Replace `your_scrapingdog_api_key_here` with your actual API key
4. Save the file
5. Restart the app

Example:
```bash
SCRAPINGDOG_API_KEY=64f7e2a1b9c8d3e4f5a6b7c8d9e0f1a2
```

### Option 2: Using Environment Variables

```bash
export SCRAPINGDOG_API_KEY=your_actual_api_key_here
```

### Option 3: Using Streamlit Secrets (for Streamlit Cloud)

1. Create `.streamlit/secrets.toml` file
2. Add:
```toml
scrapingdog_api_key = "your_actual_api_key_here"
```

### Option 4: Enter in the App

If the API key is not found in environment variables, the app will prompt you to enter it in the UI.

## Troubleshooting

### HTTP 401 Error

**Error message:** "Unauthorized request, please make sure your API key is valid"

**Causes:**
1. API key is not set (still showing placeholder value)
2. API key is incorrect or has typos
3. API key has expired or been revoked
4. Account has run out of credits

**Solutions:**
1. Double-check your API key is copied correctly (no extra spaces)
2. Verify your API key in the Scrapingdog dashboard
3. Check your account credits/usage limits
4. Generate a new API key if needed

### HTTP 403 Error

**Error message:** "Forbidden"

**Causes:**
1. Account credits exhausted
2. API rate limit exceeded
3. IP address blocked

**Solutions:**
1. Check your Scrapingdog dashboard for credits
2. Upgrade your plan if needed
3. Wait before making more requests

### API Key Not Loading

**Symptoms:**
- App shows "API key not found" despite setting it in .env
- HTTP 401 errors persist

**Solutions:**
1. Restart the Streamlit app after updating .env
2. Check .env file is in the correct directory (project root)
3. Ensure no quotes around the API key value in .env
4. Try using Option 4 (enter in app) to test if the key is valid

## Verifying Setup

Once configured correctly, you should see in the app sidebar:
- ✓ Connected (HTTP 200)
- Related searches count
- People Also Ask count
- Organic results count

## API Limits

Free tier typically includes:
- 1,000 requests per month
- Rate limit: ~10 requests per minute

Check your plan details at: https://app.scrapingdog.com/dashboard

## Need Help?

- Scrapingdog Documentation: https://www.scrapingdog.com/documentation
- Scrapingdog Support: https://www.scrapingdog.com/contact
- Check your dashboard: https://app.scrapingdog.com/dashboard
