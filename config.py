# --------------------------------------------
# Configuration and Constants
# --------------------------------------------

import os

# ========== API Configuration ==========
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # "ollama" or "deepseek" or "openai"

# Ollama Configuration
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")

# Deepseek Configuration (for cloud deployment)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# OpenAI Configuration (alternative)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# Scrapingdog Configuration
SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY", "")

# ========== Keyword Extraction Constants ==========
ALLOWED_NUMERIC_WORDS = {"3d", "2fa", "4k", "5g"}
YEAR_MIN, YEAR_MAX = 2010, 2035
MAX_KEYWORD_ROWS = 50
MIN_KEYWORD_LENGTH = 5

# ========== File Upload Limits ==========
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CONTENT_LENGTH = 50000  # characters

# ========== Timeout Settings ==========
SCRAPINGDOG_TIMEOUT = 25
LLM_TIMEOUT = 120
WEB_FETCH_TIMEOUT = 15

# ========== Cache Settings ==========
CACHE_TTL = 1800  # 30 minutes
STATUS_CACHE_TTL = 300  # 5 minutes

# ========== Content Analysis ==========
MAX_COMPETITORS = 10
DEFAULT_COMPETITORS = 3

# ========== Stop Words ==========
COMMON_STOP_WORDS = set("""
a an the and but or for nor on at by to from in out of with about above after again against all am any
are aren't as be because been before being below between both by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's
me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't
she she'd she'll she's should shouldn't so some such than that that's their theirs them themselves then there there's these
they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were
weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll
you're you've your yours yourself yourselves vs versus
""".split())

# ========== Brief Tones ==========
BRIEF_TONES = [
    "Informative and friendly",
    "Executive & concise",
    "Technical & precise",
    "Conversational",
    "Formal and authoritative",
    "Persuasive"
]

# ========== Content Types ==========
CONTENT_TYPES = ["Any", "Informational", "Commercial", "Navigational"]
