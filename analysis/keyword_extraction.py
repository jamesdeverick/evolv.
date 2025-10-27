# --------------------------------------------
# Keyword Extraction and Filtering
# --------------------------------------------

import re
import math
from typing import List, Tuple, Set, Optional, Dict, Any
from config import (
    COMMON_STOP_WORDS,
    ALLOWED_NUMERIC_WORDS,
    YEAR_MIN,
    YEAR_MAX,
    MIN_KEYWORD_LENGTH
)


def is_acceptable_token(tok: str) -> bool:
    """
    Check if a token is acceptable (not overly numeric).

    Args:
        tok: Token to check

    Returns:
        True if token is acceptable
    """
    t = tok.lower()
    if t in ALLOWED_NUMERIC_WORDS:
        return True
    if t.isdigit():
        n = int(t)
        return YEAR_MIN <= n <= YEAR_MAX
    return not any(ch.isdigit() for ch in t)


def phrase_has_bad_numbers(phrase: str) -> bool:
    """
    Check if a phrase contains unacceptable numeric tokens.

    Args:
        phrase: Phrase to check

    Returns:
        True if phrase has bad numbers
    """
    return any(not is_acceptable_token(w) for w in phrase.split())


def extract_and_filter_keywords(
    text_list: List[str],
    initial_query_words: List[str],
    min_len_chars: int = MIN_KEYWORD_LENGTH,
    exclude_words: Optional[Set[str]] = None,
    source_type: str = "general",
    serp_insights_context: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, float]]:
    """
    Extract and score keywords from text with sophisticated filtering.

    Args:
        text_list: List of text snippets to analyze
        initial_query_words: Words from the initial query
        min_len_chars: Minimum keyword length
        exclude_words: Additional words to exclude
        source_type: "llm", "serp", or "general" for scoring weights
        serp_insights_context: Optional SERP insights for boosting

    Returns:
        List of (keyword, score) tuples sorted by score
    """
    if exclude_words is None:
        exclude_words = set()

    all_exclude = exclude_words.union(COMMON_STOP_WORDS)
    initial_q = {w.lower() for w in initial_query_words}

    # Process and clean text
    processed = []
    excluded_snippets = {
        "No snippet available.",
        "No detailed snippet available (could not retrieve full content).",
        "No snippet available for this result."
    }

    for item in text_list:
        if isinstance(item, str) and item and item not in excluded_snippets:
            s = item.strip()

            # Remove URLs
            s = re.sub(r'https?://[^\s/$.?#].[^\s]*', ' ', s, flags=re.I)

            # Remove file extensions
            s = re.sub(
                r'\b[a-zA-Z0-9_-]+\.(?:png|jpg|jpeg|gif|webp|pdf|docx?|xlsx?|pptx?|html|js|css|zip|rar|svg)\b',
                ' ', s, flags=re.I
            )

            # Remove base64 data
            s = re.sub(r'data:[^;]+;base64,[^\s]+', ' ', s, flags=re.I)

            # Remove long hex strings (hashes)
            s = re.sub(r'[0-9a-fA-F]{32,}', ' ', s)

            # Remove ordinal numbers
            s = re.sub(r'\b\d+(st|nd|rd|th)\b', ' ', s, flags=re.I)

            # Remove "No. 123" patterns
            s = re.sub(r'\b(?:no\.?|#)\s*\d+\b', ' ', s, flags=re.I)

            # Remove standalone numbers
            s = re.sub(r'\b\d+\b', ' ', s)

            # Remove web-related words
            s = re.sub(r'\b(?:www|http|https|com|net|org|co|io|ly)\b', ' ', s, flags=re.I)

            # Remove non-word characters (keep hyphens)
            s = re.sub(r'[^\w\s-]', ' ', s)

            # Normalize whitespace
            s = re.sub(r'\s+', ' ', s).strip()

            if s:
                processed.append(s)

    full_text = " ".join(processed).lower()

    # Extract words
    words = [
        w for w in full_text.split()
        if w not in all_exclude and is_acceptable_token(w)
    ]

    # Extract n-grams (phrases)
    phrases = []
    for n in (2, 3, 4):
        for i in range(len(words) - (n - 1)):
            ph = " ".join(words[i:i+n])
            if all(w not in all_exclude for w in ph.split()) and not phrase_has_bad_numbers(ph):
                phrases.append(ph)

    all_terms = words + phrases

    # Prepare SERP insights for boosting
    gaps_text = ""
    unique_text = ""
    if serp_insights_context:
        gaps = serp_insights_context.get("gaps_to_exploit", "")
        if isinstance(gaps, list):
            gaps = " ".join(map(str, gaps))
        gaps_text = str(gaps).lower()

        uniq = serp_insights_context.get("unique_angles", "")
        if isinstance(uniq, list):
            uniq = " ".join(map(str, uniq))
        unique_text = str(uniq).lower()

    # Score keywords
    scored = {}
    for term in all_terms:
        if phrase_has_bad_numbers(term):
            continue

        t = term.lower()
        wc = len(t.split())

        # Base score by word count
        if wc == 1:
            base = 10
        elif wc == 2:
            base = 50
        elif wc == 3:
            base = 150
        else:
            base = 300

        # Boost if contains query words
        if any(q in t for q in initial_q):
            base *= 1.5

        # Boost by source type
        if source_type == "llm":
            base *= 2.0

        # Boost if in SERP insights
        if gaps_text and t in gaps_text:
            base *= 1.8
        if unique_text and t in unique_text:
            base *= 1.5

        # Frequency-based scoring
        raw_freq = full_text.count(t)
        score = base * (1 + math.log(raw_freq + 1)) * 100

        # Apply filtering rules
        if wc == 1:
            if t in initial_q:
                scored[term] = max(scored.get(term, 0), score)
            elif source_type == "serp" and (t in all_exclude or len(t) < 4):
                continue
            elif t not in all_exclude and len(term) >= min_len_chars:
                scored[term] = max(scored.get(term, 0), score)
        else:
            if len(term) >= min_len_chars and not any(w in all_exclude for w in t.split()):
                scored[term] = max(scored.get(term, 0), score)

    # Sort by score and filter by minimum relevance
    sorted_keywords = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    min_relevance = 1500
    return [kv for kv in sorted_keywords if kv[1] >= min_relevance]
