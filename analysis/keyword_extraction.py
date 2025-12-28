# --------------------------------------------
# Enhanced Keyword Extraction with Semantic Deduplication
# analysis/keyword_extraction.py
# --------------------------------------------

import re
import math
from typing import List, Tuple, Set, Optional, Dict, Any
from difflib import SequenceMatcher
from collections import Counter
from config import (
    COMMON_STOP_WORDS,
    ALLOWED_NUMERIC_WORDS,
    YEAR_MIN,
    YEAR_MAX,
    MIN_KEYWORD_LENGTH
)


def semantic_similarity(phrase1: str, phrase2: str) -> float:
    """
    Calculate semantic similarity between two keyword phrases.
    
    Args:
        phrase1: First phrase
        phrase2: Second phrase
        
    Returns:
        Similarity score between 0 and 1
    """
    # Normalize
    p1_words = set(phrase1.lower().split())
    p2_words = set(phrase2.lower().split())
    
    # Jaccard similarity (word overlap)
    if not p1_words or not p2_words:
        return 0.0
    
    intersection = len(p1_words & p2_words)
    union = len(p1_words | p2_words)
    jaccard = intersection / union if union > 0 else 0
    
    # String similarity (handles word order)
    string_sim = SequenceMatcher(None, phrase1.lower(), phrase2.lower()).ratio()
    
    # Combined score (favor Jaccard for keywords since word order matters less)
    return (jaccard * 0.7) + (string_sim * 0.3)


def deduplicate_semantically(
    scored_keywords: List[Tuple[str, float]],
    similarity_threshold: float = 0.75
) -> List[Tuple[str, float]]:
    """
    Remove semantically similar keywords, keeping highest scoring version.
    
    Args:
        scored_keywords: List of (keyword, score) tuples
        similarity_threshold: Similarity threshold (0-1) for considering duplicates
        
    Returns:
        Deduplicated list of (keyword, score) tuples
    """
    if not scored_keywords:
        return []
    
    # Sort by score descending (keep best versions)
    sorted_kw = sorted(scored_keywords, key=lambda x: x[1], reverse=True)
    
    unique = []
    for keyword, score in sorted_kw:
        is_duplicate = False
        
        for existing_kw, existing_score in unique:
            similarity = semantic_similarity(keyword, existing_kw)
            
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append((keyword, score))
    
    return unique


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


def score_keyword_advanced(
    keyword: str,
    frequency: int,
    initial_query_words: List[str],
    source_type: str,
    serp_insights_context: Optional[Dict[str, Any]] = None
) -> float:
    """
    Advanced multi-factor keyword scoring.
    
    Args:
        keyword: The keyword phrase
        frequency: How many times it appeared
        initial_query_words: Words from the main query
        source_type: "llm", "serp", or "general"
        serp_insights_context: SERP insights for boosting
        
    Returns:
        Keyword score (0-10000+)
    """
    keyword_lower = keyword.lower()
    words = keyword_lower.split()
    word_count = len(words)
    initial_q = {w.lower() for w in initial_query_words}
    
    # Score components
    scores = {
        "base_value": 0,        # Base value by word count
        "relevance": 0,          # Relevance to main topic
        "intent_value": 0,       # Search intent signals
        "frequency": 0,          # Appearance frequency
        "serp_opportunity": 0,   # SERP alignment
        "source_boost": 0,       # Source type multiplier
        "penalties": 0           # Quality penalties
    }
    
    # 1. BASE VALUE (by word count - sweet spot is 3-5 words)
    if word_count == 1:
        scores["base_value"] = 10  # Single words are less valuable
    elif word_count == 2:
        scores["base_value"] = 50  # Good
    elif 3 <= word_count <= 5:
        scores["base_value"] = 150  # Best - long-tail sweet spot
    elif 6 <= word_count <= 8:
        scores["base_value"] = 100  # Still good but getting long
    else:
        scores["base_value"] = 50  # Too long, less likely to be searched
    
    # 2. RELEVANCE to main query
    overlap = sum(1 for q in initial_q if q in keyword_lower)
    if overlap > 0:
        scores["relevance"] = overlap * 30  # Strong boost for query words
    
    # 3. INTENT VALUE - identify valuable search patterns
    # Question keywords (high value for PAA and featured snippets)
    question_words = ["what", "how", "why", "when", "where", "who", "which", "can", "should", "is", "are", "do", "does"]
    if any(keyword_lower.startswith(q + " ") for q in question_words):
        scores["intent_value"] += 40
    
    # Commercial intent signals
    commercial = ["best", "top", "review", "compare", "vs", "versus", "alternative"]
    if any(signal in keyword_lower for signal in commercial):
        scores["intent_value"] += 25
    
    # Informational signals
    informational = ["guide", "tutorial", "how to", "tips", "examples", "learn"]
    if any(signal in keyword_lower for signal in informational):
        scores["intent_value"] += 20
    
    # Valuable modifiers
    modifiers = ["best", "top", "complete", "ultimate", "comprehensive", "advanced", "beginner"]
    if any(mod in keyword_lower for mod in modifiers):
        scores["intent_value"] += 15
    
    # 4. FREQUENCY SCORE
    # Use logarithmic scaling to prevent over-weighting common terms
    scores["frequency"] = math.log(frequency + 1) * 30
    
    # 5. SERP OPPORTUNITY
    if serp_insights_context:
        gaps = serp_insights_context.get("gaps_to_exploit", "")
        if isinstance(gaps, list):
            gaps = " ".join(map(str, gaps))
        gaps_text = str(gaps).lower()
        
        unique = serp_insights_context.get("unique_angles", "")
        if isinstance(unique, list):
            unique = " ".join(map(str, unique))
        unique_text = str(unique).lower()
        
        # Boost if keyword addresses SERP gaps
        if gaps_text and any(word in gaps_text for word in words):
            scores["serp_opportunity"] += 50
        
        # Boost if keyword aligns with unique angles
        if unique_text and any(word in unique_text for word in words):
            scores["serp_opportunity"] += 30
    
    # 6. SOURCE TYPE BOOST
    source_multipliers = {
        "llm": 2.0,      # LLM suggestions are curated
        "serp": 1.5,     # SERP data shows actual demand
        "general": 1.0   # Default
    }
    scores["source_boost"] = sum(scores.values()) * (source_multipliers.get(source_type, 1.0) - 1.0)
    
    # 7. PENALTIES
    # Low-quality signals
    low_quality = ["cheap", "free", "hack", "trick", "secret", "scam"]
    if any(lq in keyword_lower for lq in low_quality):
        scores["penalties"] -= 100
    
    # Overly generic single words (unless in main query)
    if word_count == 1 and keyword_lower in ["tips", "guide", "examples", "ideas", "help"]:
        if keyword_lower not in initial_q:
            scores["penalties"] -= 50
    
    # Too many special characters
    special_chars = len(re.findall(r'[^\w\s-]', keyword))
    if special_chars > 2:
        scores["penalties"] -= 30
    
    # Redundant words
    if len(words) != len(set(words)):
        scores["penalties"] -= 80
    
    # Calculate final score
    total_score = sum(scores.values())
    return max(0, total_score)  # Never negative


def extract_and_filter_keywords(
    text_list: List[str],
    initial_query_words: List[str],
    min_len_chars: int = MIN_KEYWORD_LENGTH,
    exclude_words: Optional[Set[str]] = None,
    source_type: str = "general",
    serp_insights_context: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, float]]:
    """
    Extract and score keywords with advanced filtering and deduplication.

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

            # Remove standalone numbers (but keep years)
            s = re.sub(r'\b(?!(?:19|20)\d{2}\b)\d+\b', ' ', s)

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

    # Extract n-grams with frequency counting
    phrase_freq = Counter()
    
    # Add individual words
    for w in words:
        if len(w) >= min_len_chars and w not in all_exclude:
            phrase_freq[w] += 1
    
    # Extract 2-5 word phrases (extended range for better long-tail)
    for n in (2, 3, 4, 5):
        for i in range(len(words) - (n - 1)):
            phrase_words = words[i:i+n]
            
            # Skip if any word is excluded or has bad numbers
            if any(w in all_exclude for w in phrase_words):
                continue
            
            phrase = " ".join(phrase_words)
            
            if not phrase_has_bad_numbers(phrase) and len(phrase) >= min_len_chars:
                phrase_freq[phrase] += 1

    # Score all keywords
    scored = []
    for keyword, freq in phrase_freq.items():
        score = score_keyword_advanced(
            keyword=keyword,
            frequency=freq,
            initial_query_words=initial_query_words,
            source_type=source_type,
            serp_insights_context=serp_insights_context
        )
        
        if score > 0:  # Only keep keywords with positive scores
            scored.append((keyword, score))

    # Sort by score
    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    # Semantic deduplication (removes "corporate budgeting" and "budgeting corporate" variants)
    deduplicated = deduplicate_semantically(scored, similarity_threshold=0.75)

    # Filter by minimum relevance
    min_relevance = 100  # Lowered from 1500 to be less aggressive
    
    return [kv for kv in deduplicated if kv[1] >= min_relevance]


def generate_llm_keyword_variations(
    base_keyword: str,
    main_topic: str,
    llm_client,
    max_variations: int = 30,
    desired_intent: str = "Informational"
) -> List[str]:
    """
    Use LLM to generate strategic keyword variations.
    
    Args:
        base_keyword: Primary keyword to expand from
        main_topic: Overall topic context
        llm_client: LLM client instance
        max_variations: Maximum variations to generate
        desired_intent: Target content intent
        
    Returns:
        List of keyword strings
    """
    if not llm_client.available:
        return []
    
    # Intent-specific guidance
    intent_guidance = {
        "Informational": "Focus on how-to, what is, why, benefits, guides, tutorials, and explanatory queries",
        "Commercial": "Focus on best, top, reviews, comparisons, alternatives, and evaluation queries",
        "Transactional": "Focus on pricing, buying, deals, and purchase-related queries",
        "Navigational": "Focus on brand names, product names, and login/account queries",
        "Any": "Include a balanced mix of informational, commercial, and question-based queries"
    }
    
    guidance = intent_guidance.get(desired_intent, intent_guidance["Any"])
    
    prompt = f"""Generate {max_variations} strategic keyword variations for SEO content.

Main Topic: "{main_topic}"
Base Keyword: "{base_keyword}"
Content Intent: {desired_intent}

{guidance}

Generate keywords that are:
1. Semantically related but NOT duplicates of the base keyword
2. Different user intents and search angles
3. Mix of specificity (broader topics + specific long-tail)
4. Include question-based variants (what/how/why/when/where)
5. 3-6 words ideal (natural language phrases)
6. Include valuable modifiers (best, top, guide, complete) where appropriate
7. Actually searchable phrases people would type into Google

CRITICAL: Return ONLY a comma-separated list.
NO numbering, NO explanations, NO preamble, NO markdown.

Example format:
how to improve corporate budgeting, corporate budget planning best practices, what is zero-based budgeting

Your keywords:"""

    try:
        raw = llm_client.complete(prompt, temperature=0.7, max_tokens=400)
        
        # Extract keywords
        # Remove any markdown, numbering, or formatting
        raw = re.sub(r'```.*?```', '', raw, flags=re.DOTALL)  # Remove code blocks
        raw = re.sub(r'^\d+[\.\)]\s*', '', raw, flags=re.MULTILINE)  # Remove numbering
        raw = re.sub(r'^\-\s*', '', raw, flags=re.MULTILINE)  # Remove bullet points
        
        # Split by commas or newlines
        keywords = re.split(r'[,\n]+', raw)
        
        # Clean and filter
        cleaned = []
        for kw in keywords:
            kw = kw.strip()
            
            # Remove quotes
            kw = kw.strip('"\'')
            
            # Skip if empty or too short
            if not kw or len(kw) < 3:
                continue
            
            # Skip if too long (likely not a real keyword)
            if len(kw.split()) > 10:
                continue
            
            # Skip if looks like instructions or meta-text
            if any(meta in kw.lower() for meta in ["example", "keyword", "variation", "here are", "such as"]):
                continue
            
            cleaned.append(kw)
        
        return cleaned[:max_variations]
    
    except Exception as e:
        print(f"LLM keyword generation error: {e}")
        return []
