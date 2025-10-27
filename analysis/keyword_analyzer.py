# --------------------------------------------
# Keyword Analysis with LLM
# --------------------------------------------

import json
import re
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional
from collections import Counter

from api.llm_client import get_llm_client
from api.scrapingdog_client import ScrapingdogClient
from analysis.keyword_extraction import extract_and_filter_keywords
from config import COMMON_STOP_WORDS, MAX_KEYWORD_ROWS


class KeywordAnalyzer:
    """Analyze keywords using LLM and SERP data."""

    def __init__(self, scrapingdog_client: ScrapingdogClient):
        """
        Initialize keyword analyzer.

        Args:
            scrapingdog_client: Configured Scrapingdog client
        """
        self.scrapingdog = scrapingdog_client
        self.llm = get_llm_client()

    def infer_content_type(self, keyword: str) -> str:
        """
        Infer content type using heuristics.

        Args:
            keyword: Keyword to analyze

        Returns:
            "Informational", "Commercial", or "Navigational"
        """
        k = keyword.lower()

        informational = [
            "how to", "what is", "guide", "tutorial", "explain", "examples",
            "definition", "learn", "why", "who", "when", "meaning", "tips",
            "steps", "best practices"
        ]

        commercial = [
            "buy", "price", "cost", "best", "top", "review", "vs", "comparison",
            "alternatives", "deal", "discount", "services", "agency", "software",
            "tool", "platform", "pricing", "hire", "consultant", "solution"
        ]

        navigational = [
            "login", "dashboard", "account", "careers", "contact", "about us",
            "my account", "sign up", "sign in"
        ]

        if any(t in k for t in navigational):
            return "Navigational"
        if any(t in k for t in commercial):
            return "Commercial"
        return "Informational"

    def analyze_keyword_with_llm(self, keyword: str, all_keywords: List[str]) -> Dict[str, Any]:
        """
        Analyze a single keyword using LLM for deeper insights.

        Args:
            keyword: Target keyword to analyze
            all_keywords: Full list of keywords for context

        Returns:
            Dict with keyword analysis
        """
        if not self.llm.available:
            return {
                "keyword": keyword,
                "inferred_intent": self.infer_content_type(keyword),
                "needs_own_page": False,
                "rationale_for_own_page": "LLM unavailable.",
                "semantically_related_keywords_for_grouping": []
            }

        # Get context keywords (excluding target and stop words)
        context = [
            kw for kw in all_keywords
            if kw.lower() != keyword.lower()
            and kw.lower() not in COMMON_STOP_WORDS
            and len(kw) >= 3
        ][:100]

        prompt = f"""You are an expert SEO content strategist. Analyze the target keyword and determine:
- primary search intent
- whether it warrants a standalone page
- which other keywords (from the supplied list) are semantically related and can be covered on the SAME page.

Target Keyword: "{keyword}"

All brainstormed keywords (for relatedness):
{json.dumps(context, indent=2)}

Return ONLY JSON:
{{
  "keyword": "original",
  "inferred_intent": "Informational|Commercial|Navigational",
  "needs_own_page": true|false,
  "rationale_for_own_page": "brief text",
  "semantically_related_keywords_for_grouping": ["k1","k2", "..."]
}}"""

        try:
            raw = self.llm.complete(prompt, temperature=0.3, max_tokens=800)

            # Extract JSON
            m = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
            s = m.group(1) if m else raw[raw.find("{"):raw.rfind("}")+1]
            parsed = json.loads(s)

            return {
                "keyword": parsed.get("keyword", keyword),
                "inferred_intent": parsed.get("inferred_intent", self.infer_content_type(keyword)),
                "needs_own_page": parsed.get("needs_own_page", False),
                "rationale_for_own_page": parsed.get("rationale_for_own_page", ""),
                "semantically_related_keywords_for_grouping": [
                    x for x in parsed.get("semantically_related_keywords_for_grouping", [])
                    if isinstance(x, str)
                ]
            }
        except Exception:
            return {
                "keyword": keyword,
                "inferred_intent": self.infer_content_type(keyword),
                "needs_own_page": False,
                "rationale_for_own_page": "LLM parsing error.",
                "semantically_related_keywords_for_grouping": []
            }

    def analyze_and_identify_keywords(
        self,
        query_topic: str,
        related_searches: List[str],
        organic_results: List[Dict],
        desired_content_intent: str
    ) -> tuple[pd.DataFrame, str, Dict[str, Any]]:
        """
        Main orchestrator for keyword analysis.

        Args:
            query_topic: Main topic/query
            related_searches: Related searches from SERP
            organic_results: Organic results from SERP
            desired_content_intent: Filter for content type

        Returns:
            Tuple of (keywords_df, selected_keyword, serp_insights)
        """
        # Get SERP results for analysis
        serp_results, serp_raw, _ = self.scrapingdog.analyze_serp(query_topic)

        # Build SERP insights
        serp_insights = {
            "common_themes": "; ".join([
                s.get("title", "") for s in serp_results[:5]
            ]) if serp_results else "",
            "gaps_to_exploit": [],
            "unique_angles": []
        }

        # Brainstorm keywords with LLM
        llm_brainstormed = []
        if self.llm.available:
            with st.spinner(f"Brainstorming keywords with {self.llm.model}..."):
                prompt = (
                    f"List 30 SEO-relevant keyword ideas for topic: {query_topic}. "
                    "Return as comma-separated list."
                )
                raw = self.llm.complete(prompt, temperature=0.2, max_tokens=500)
                llm_brainstormed = [
                    k.strip() for k in re.split(r'[,;\n]+', raw) if k.strip()
                ]
        else:
            st.warning("LLM unavailable; proceeding with SERP-derived keywords only.")

        # Extract and score keywords from different sources
        initial_words = query_topic.split()

        llm_scored = extract_and_filter_keywords(
            llm_brainstormed,
            initial_words,
            source_type="llm",
            serp_insights_context=serp_insights
        )

        serp_related_scored = extract_and_filter_keywords(
            related_searches,
            initial_words,
            source_type="serp",
            serp_insights_context=serp_insights
        )

        serp_snippets = [
            f"{r.get('title','')} {r.get('snippet','')}"
            for r in (organic_results or [])
        ]
        serp_snippets_scored = extract_and_filter_keywords(
            serp_snippets,
            initial_words,
            source_type="serp",
            serp_insights_context=serp_insights
        )

        # Combine and deduplicate
        combined = {}
        for term, score in serp_related_scored + serp_snippets_scored + llm_scored:
            combined[term] = max(combined.get(term, 0), score)

        # Remove main query from supporting keywords
        combined = {
            k: v for k, v in combined.items()
            if k.lower() != query_topic.lower()
        }

        # Sort and limit
        sorted_k = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:MAX_KEYWORD_ROWS]
        all_kw = [query_topic] + [k for k, _ in sorted_k]

        # Build dataframe
        rows = []

        # Add main topic first
        main_intent = self.infer_content_type(query_topic)
        rows.append({
            "Selected": True,
            "Keyword": query_topic,
            "Inferred Potential Score": 999999,
            "Content Type": main_intent,
            "Requires Own Content": True,
            "Rationale for Own Page": "Main target keyword for the brief.",
            "Semantically Related Keywords": ""
        })

        # Analyze top keywords with LLM
        subset = all_kw[1:11]  # Top 10 supporting keywords
        if self.llm.available and subset:
            with st.spinner(f"Performing detailed LLM analysis for top {len(subset)} keywords..."):
                for kw in subset:
                    info = self.analyze_keyword_with_llm(kw, all_kw)
                    rel = [
                        r for r in info.get("semantically_related_keywords_for_grouping", [])
                        if r.lower() != query_topic.lower()
                    ]
                    rows.append({
                        "Selected": True,
                        "Keyword": info.get("keyword", kw),
                        "Inferred Potential Score": combined.get(kw, 0),
                        "Content Type": info.get("inferred_intent", self.infer_content_type(kw)),
                        "Requires Own Content": info.get("needs_own_page", False),
                        "Rationale for Own Page": info.get("rationale_for_own_page", ""),
                        "Semantically Related Keywords": ", ".join(rel)
                    })

        # Add remaining keywords without detailed analysis
        for kw, score in sorted_k[len(subset):]:
            rows.append({
                "Selected": True,
                "Keyword": kw,
                "Inferred Potential Score": score,
                "Content Type": self.infer_content_type(kw),
                "Requires Own Content": False,
                "Rationale for Own Page": "Beyond top-10 detailed analysis.",
                "Semantically Related Keywords": ""
            })

        df = pd.DataFrame(rows)

        # Apply content type filter
        if desired_content_intent != "Any":
            keep = df[
                (df["Keyword"].str.lower() != query_topic.lower()) &
                (df["Content Type"] == desired_content_intent)
            ]
            main = df[df["Keyword"].str.lower() == query_topic.lower()]
            df = pd.concat([main, keep]).reset_index(drop=True)

            if main_intent != desired_content_intent:
                st.warning(
                    f"Main topic inferred as '{main_intent}', but you filtered for "
                    f"'{desired_content_intent}'. Main stays; others filtered."
                )

        return df, query_topic, serp_insights


class KeywordClusterer:
    """Cluster keywords semantically using LLM."""

    def __init__(self):
        """Initialize clusterer."""
        self.llm = get_llm_client()

    def create_clusters(self, keywords_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create semantic clusters from keywords.

        Args:
            keywords_df: DataFrame with keywords

        Returns:
            Dict mapping cluster names to keyword lists
        """
        if not self.llm.available:
            st.warning("LLM disabled; cannot cluster.")
            return {}

        if keywords_df is None or keywords_df.empty or "Keyword" not in keywords_df.columns:
            st.warning("No keywords to cluster.")
            return {}

        # Get top keywords
        top_keywords = [
            str(k) for k in keywords_df.head(20)["Keyword"].tolist()
            if isinstance(k, str) and k.strip()
        ]

        if not top_keywords:
            st.warning("No valid keywords for clustering.")
            return {}

        prompt = (
            "Group these keywords into semantic clusters (distinct topics/user intent):\n\n"
            f"{', '.join(top_keywords)}\n\n"
            'Return JSON: {"Cluster Name": ["kw1","kw2", ...], ...}. '
            'Ensure each keyword appears exactly once.'
        )

        try:
            raw = self.llm.complete(prompt, temperature=0.2, max_tokens=800)

            # Extract JSON
            m = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
            if m:
                s = m.group(1)
            else:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                s = raw[start:end] if start >= 0 and end > start else "{}"

            clusters = json.loads(s)

            # Clean and deduplicate
            seen = set()
            cleaned = {}

            if isinstance(clusters, dict):
                for name, kws in clusters.items():
                    bucket = []
                    if isinstance(kws, (list, tuple)):
                        for kw in kws:
                            if isinstance(kw, str) and kw in top_keywords and kw not in seen:
                                bucket.append(kw)
                                seen.add(kw)
                    if bucket:
                        cleaned[str(name)] = bucket

            # Add missing keywords to "Misc" cluster
            missing = [k for k in top_keywords if k not in seen]
            if missing:
                cleaned["Misc"] = missing

            return cleaned
        except Exception as e:
            st.error(f"Clustering error: {e}")
            return {}


def extract_common_headings(headings_list: List[List[str]], min_count: int = 2) -> List[str]:
    """
    Find common headings across multiple pages.

    Args:
        headings_list: List of heading lists from different pages
        min_count: Minimum occurrences to be considered common

    Returns:
        List of common headings
    """
    counter = Counter()

    for headings in headings_list:
        for h in headings:
            # Normalize heading
            norm = re.sub(r"[^\w\s]", "", h.lower())
            if len(norm.split()) >= 2:
                counter[norm] += 1

    return [h for h, c in counter.most_common(10) if c >= min_count]
