# Upgrade: PAA fix, structured JSON, Mistral Small 3.2

## Files changed
llm_client.py → api/ · scrapingdog_client.py → api/ · keyword_analyzer.py, audit_parser.py, content_reviser.py → analysis/ (same dirs as before)
app_fixed.py → app.py · benchmark_models.py → project root (new)

## 1. Try Mistral Small 3.2 (VM, no hosting change)
    curl -fsSL https://ollama.com/install.sh | sh     # update Ollama first
    ollama pull mistral-small3.2                      # ~15GB; confirm tag at ollama.com/library
    export OLLAMA_MODEL=mistral-small3.2
    export OLLAMA_NUM_CTX=16384                       # raise to 32768 only if VRAM allows
    export LLM_TIMEOUT=300                            # 24B is slower than 7B
    pkill -f streamlit; streamlit run app.py
Unset OLLAMA_MODEL to fall back to config.py's model. The sidebar/status shows the active model.

## 2. Benchmark before switching
    python benchmark_models.py <audit.md> mistral:latest mistral-small3.2
Compares audit-parse success, requirements found, key-entity recall, keyword JSON validity, time.

## 3. People Also Ask
Scrapingdog's documented key (peopleAlsoAskedFor) was already correct. Causes of "never works":
- Sidebar probe searched "test", which has no PAA box → now "what is kubernetes".
- Google only shows PAA on some queries → added question-style Google Autocomplete as fallback.
- "Is PAA" was a question-word guess → now true only for real PAA/autocomplete questions.
- Parsing is tolerant of other key names; if none found, Step 2 prints the response keys.

## 4. Structured JSON
Audit parsing, revisions, and keyword batches now use Ollama's `format` schema, so malformed JSON should stop. Old parsers remain as backup.
