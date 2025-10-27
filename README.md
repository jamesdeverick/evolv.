# Advanced LLM SEO Assistant

A powerful SEO content planning tool that combines LLM analysis with real-time keyword research, SERP insights, and content brief generation.

## Features

- 🔍 **Keyword Research**: Automated keyword discovery using Scrapingdog SERP API
- 🤖 **LLM Analysis**: Deep content analysis using local (Ollama) or cloud LLMs (Deepseek, OpenAI)
- 📊 **SERP Insights**: Competitive analysis and gap identification
- 📝 **Content Briefs**: Auto-generated SEO content briefs with tone-of-voice enforcement
- 🎯 **Intent Analysis**: Automatic classification of keyword intent
- 📦 **Keyword Clustering**: Semantic grouping of related keywords
- ⚡ **Cloud-Ready**: Works with both local Ollama and cloud API providers

## Quick Start

### Local Development (with Ollama)

1. **Install Ollama**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Or download from: https://ollama.com/download
   ```

2. **Pull a model**
   ```bash
   ollama pull mistral:latest
   # or
   ollama pull llama3:8b-instruct
   ```

3. **Start Ollama server**
   ```bash
   ollama serve
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

6. **Run the app**
   ```bash
   streamlit run app.py
   ```

### Cloud Deployment (HuggingFace Spaces)

#### Option 1: Using Deepseek API (Recommended)

Deepseek is cost-effective and works well for this use case.

1. **Get Deepseek API Key**
   - Sign up at https://platform.deepseek.com/
   - Generate an API key
   - Pricing: ~$0.14 per million tokens (very affordable)

2. **Deploy to HuggingFace Spaces**
   - Create a new Space at https://huggingface.co/new-space
   - Choose "Streamlit" as SDK
   - Upload your code (or connect to GitHub)

3. **Configure Secrets in HuggingFace Spaces**
   - Go to Settings → Repository secrets
   - Add the following secrets:
     ```
     LLM_PROVIDER=deepseek
     DEEPSEEK_API_KEY=your_deepseek_api_key
     SCRAPINGDOG_API_KEY=your_scrapingdog_api_key
     ```

4. **The app will automatically use Deepseek** instead of local Ollama

#### Option 2: Using OpenAI API

1. **Get OpenAI API Key**
   - Sign up at https://platform.openai.com/
   - Generate an API key
   - Note: More expensive than Deepseek (~$10 per million tokens for GPT-4)

2. **Configure Secrets**
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key
   SCRAPINGDOG_API_KEY=your_scrapingdog_api_key
   ```

### Why HuggingFace Spaces Won't Work with Ollama

Your app wasn't starting on HuggingFace Spaces because:

1. **Ollama requires a local server** - HuggingFace Spaces doesn't allow running background services
2. **T4 GPU is irrelevant** - Ollama needs to be installed and running as a daemon
3. **Solution**: Use cloud API providers (Deepseek or OpenAI) instead

The refactored code now **automatically detects the environment** and switches between:
- Local Ollama (for development)
- Deepseek/OpenAI APIs (for cloud deployment)

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

| Variable | Description | Required |
|----------|-------------|----------|
| `LLM_PROVIDER` | `ollama`, `deepseek`, or `openai` | Yes |
| `SCRAPINGDOG_API_KEY` | Scrapingdog SERP API key | Yes |
| `DEEPSEEK_API_KEY` | Deepseek API key (if using Deepseek) | Conditional |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | Conditional |
| `OLLAMA_API_BASE` | Ollama server URL (default: `http://127.0.0.1:11434`) | Conditional |
| `OLLAMA_MODEL` | Ollama model name (default: `mistral:latest`) | Conditional |

### Streamlit Secrets (for HuggingFace Spaces)

Create a `.streamlit/secrets.toml` file or use HuggingFace Spaces UI:

```toml
scrapingdog_api_key = "your_scrapingdog_key"
```

## Project Structure

```
evolv/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration and constants
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README.md                       # This file
├── api/
│   ├── llm_client.py              # Unified LLM client (Ollama/Deepseek/OpenAI)
│   └── scrapingdog_client.py     # Scrapingdog API client
├── utils/
│   ├── file_processing.py        # File upload handlers (DOCX/PDF/TXT)
│   └── web_scraping.py           # Web scraping utilities
└── analysis/
    ├── keyword_extraction.py      # Keyword extraction algorithms
    ├── keyword_analyzer.py        # LLM-powered keyword analysis
    ├── competitive_analyzer.py    # Competitor analysis
    └── content_brief_creator.py   # Content brief generation
```

## Key Improvements from Original Code

### 1. **Cloud Deployment Ready**
   - ✅ Works with cloud API providers (Deepseek, OpenAI)
   - ✅ Automatic provider detection
   - ✅ No local Ollama server required

### 2. **Security Enhancements**
   - ✅ URL validation to prevent SSRF attacks
   - ✅ File size limits (10MB max)
   - ✅ XML bomb protection using `defusedxml`
   - ✅ Safe URL parsing

### 3. **Code Quality**
   - ✅ Modular architecture (split into 9+ files)
   - ✅ Clear separation of concerns
   - ✅ Better error handling
   - ✅ Type hints and documentation

### 4. **Bug Fixes**
   - ✅ Fixed deprecated `st.experimental_rerun()` → `st.rerun()`
   - ✅ Consistent API usage (litellm throughout)
   - ✅ Proper DataFrame column validation
   - ✅ Improved exception handling

## Usage Guide

### Step 1: Enter Topic & Client Info
- Enter your main topic/keyword
- Optionally provide a client URL (or leave blank for net-new content)
- Upload audit findings (PDF, Markdown, or Text)

### Step 2: Review Keywords
- View auto-generated keywords from SERP analysis and LLM brainstorming
- Select which keywords to target
- Run competitive analysis
- Create keyword clusters

### Step 3: LLM Analysis
- Get automated content gap analysis
- Review proposed outline and structure
- Customize the analysis prompt if needed

### Step 4: Generate Content Brief
- Create a comprehensive SEO content brief
- Edit in-app before exporting
- Download as Markdown

## API Costs (Estimated)

| Provider | Model | Cost per 1M tokens | Typical request cost |
|----------|-------|-------------------|---------------------|
| Deepseek | deepseek-chat | $0.14 | ~$0.01-0.02 |
| OpenAI | GPT-4 | $10.00 | ~$0.50-1.00 |
| Ollama | mistral:latest | Free (local) | $0 |

**Recommendation**: Use Deepseek for cloud deployment - it's 70x cheaper than GPT-4 with similar quality for this use case.

## Troubleshooting

### HuggingFace Spaces Issues

**Problem**: App won't start on HuggingFace Spaces
**Solution**:
1. Make sure `LLM_PROVIDER` is set to `deepseek` or `openai` (not `ollama`)
2. Verify API keys are in Secrets
3. Check logs for specific error messages

**Problem**: "LLM not available" error
**Solution**:
1. Verify your API key is correct
2. Check if you have credits/quota remaining
3. Try the probe endpoint: `https://api.deepseek.com/v1/models`

### Local Development Issues

**Problem**: Ollama timeout errors on Windows
**Solution**: Use `127.0.0.1` instead of `localhost` in `OLLAMA_API_BASE`

**Problem**: Model not found
**Solution**: Run `ollama pull mistral:latest` to download the model

## Contributing

This is a refactored version of the original single-file app. Key improvements:
- Modular architecture
- Cloud deployment support
- Security enhancements
- Bug fixes

## License

[Your license here]

## Support

For issues related to:
- **Scrapingdog**: https://www.scrapingdog.com/support
- **Deepseek**: https://platform.deepseek.com/docs
- **OpenAI**: https://platform.openai.com/docs
- **Ollama**: https://ollama.com/docs
