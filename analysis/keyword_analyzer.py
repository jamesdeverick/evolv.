# --------------------------------------------
# Improved Keyword Analysis with Multi-Factor Scoring
# analysis/keyword_analyzer.py
# --------------------------------------------

import json
import re
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from api.llm_client import get_llm_client
from api.scrapingdog_client import ScrapingdogClient
from analysis.keyword_extraction import (
    extract_and_filter_keywords,
    generate_llm_keyword_variations,
    deduplicate_semantically
)
from config import COMMON_STOP_WORDS, MAX_KEYWORD_ROWS


class KeywordAnalyzer:
    """Enhanced keyword analyzer with multi-factor scoring."""

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
        Infer content type using enhanced heuristics.

        Args:
            keyword: Keyword to analyze

        Returns:
            "Informational", "Commercial", "Transactional", or "Navigational"
        """
        k = keyword.lower()

        # Check for questions first (strong informational signal)
        question_words = ["what", "how", "why", "when", "where", "who", "which", "can", "should", "is", "are", "do", "does"]
        if any(k.startswith(q + " ") for q in question_words) or k.endswith("?"):
            return "Informational"

        # Navigational (highest priority after questions)
        navigational = [
            "login", "dashboard", "account", "careers", "contact", "about us",
            "my account", "sign up", "sign in", "portal", "app download"
        ]
        if any(t in k for t in navigational):
            return "Navigational"

        # Transactional (buying intent)
        transactional = [
            "buy", "price", "cost", "pricing", "purchase", "order", "cheap",
            "discount", "deal", "coupon", "sale", "shop", "store"
        ]
        if any(t in k for t in transactional):
            return "Transactional"

        # Commercial (research before buying)
        commercial = [
            "best", "top", "review", "reviews", "vs", "versus", "comparison",
            "compare", "alternative", "alternatives", "option", "options",
            "software", "tool", "service", "solution", "platform"
        ]
        if any(t in k for t in commercial):
            return "Commercial"

        # Informational (learning/understanding)
        informational = [
            "guide", "tutorial", "how to", "explain", "examples", "definition",
            "meaning", "tips", "steps", "best practices", "learn", "understand"
        ]
        if any(t in k for t in informational):
            return "Informational"

        # Default to informational
        return "Informational"

    def calculate_keyword_score(
        self,
        keyword: str,
        base_score: float,
        main_topic: str,
        desired_intent: str,
        serp_insights: Optional[Dict] = None
    ) -> Tuple[float, str, Dict]:
        """
        Calculate comprehensive keyword score with breakdown.
        
        Args:
            keyword: Keyword to score
            base_score: Initial score from extraction
            main_topic: Main topic for relevance
            desired_intent: Target content intent
            serp_insights: SERP analysis data
            
        Returns:
            Tuple of (final_score, grade, score_breakdown)
        """
        k_lower = keyword.lower()
        words = k_lower.split()
        word_count = len(words)
        
        # Initialize score components
        breakdown = {
            "base": base_score,
            "relevance": 0,
            "intent_match": 0,
            "specificity": 0,
            "question_value": 0,
            "serp_alignment": 0,
            "penalties": 0
        }
        
        # 1. RELEVANCE to main topic
        topic_words = set(main_topic.lower().split())
        keyword_words = set(words)
        overlap = len(topic_words & keyword_words)
        
        if overlap > 0:
            breakdown["relevance"] = overlap * 50
            
        # Bonus if contains full topic
        if main_topic.lower() in k_lower:
            breakdown["relevance"] += 100
        
        # 2. INTENT MATCH
        inferred_intent = self.infer_content_type(keyword)
        if desired_intent == "Any" or inferred_intent == desired_intent:
            breakdown["intent_match"] = 100
        else:
            breakdown["intent_match"] = 30  # Partial credit
        
        # 3. SPECIFICITY (long-tail value)
        if word_count == 1:
            breakdown["specificity"] = 10
        elif word_count == 2:
            breakdown["specificity"] = 40
        elif 3 <= word_count <= 5:
            breakdown["specificity"] = 100  # Sweet spot
        elif 6 <= word_count <= 8:
            breakdown["specificity"] = 60
        else:
            breakdown["specificity"] = 20
        
        # 4. QUESTION VALUE
        question_starters = ["what", "how", "why", "when", "where", "who", "which", "can", "should"]
        if any(k_lower.startswith(q + " ") for q in question_starters) or k_lower.endswith("?"):
            breakdown["question_value"] = 80  # High value for PAA
        
        # 5. SERP ALIGNMENT
        if serp_insights:
            themes = str(serp_insights.get("common_themes", "")).lower()
            if themes and any(word in themes for word in words):
                breakdown["serp_alignment"] = 50
        
        # 6. PENALTIES
        # Low quality signals
        low_quality = ["cheap", "free", "hack", "trick", "secret"]
        if any(lq in k_lower for lq in low_quality):
            breakdown["penalties"] -= 100
        
        # Overly generic
        if word_count == 1 and k_lower in ["tips", "guide", "help", "info"]:
            breakdown["penalties"] -= 50
        
        # Calculate final score
        final_score = sum(breakdown.values())
        final_score = max(0, final_score)  # Never negative
        
        # Grade
        if final_score >= 400:
            grade = "A"
        elif final_score >= 300:
            grade = "B"
        elif final_score >= 200:
            grade = "C"
        elif final_score >= 100:
            grade = "D"
        else:
            grade = "F"
        
        return final_score, grade, breakdown

    def analyze_keyword_with_llm(
        self,
        keyword: str,
        all_keywords: List[str],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze a single keyword using LLM for deeper insights.

        Args:
            keyword: Target keyword to analyze
            all_keywords: Full list of keywords for context
            context: Additional context (main topic, etc.)

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

        # Get context keywords
        context_kw = [
            kw for kw in all_keywords[:50]  # Top 50
            if kw.lower() != keyword.lower()
            and kw.lower() not in COMMON_STOP_WORDS
            and len(kw) >= 3
        ][:30]  # Limit to 30

        prompt = f"""Analyze this keyword for SEO content planning.

Target Keyword: "{keyword}"
{f'Main Topic: {context}' if context else ''}

Related Keywords for context:
{', '.join(context_kw[:20])}

Determine:
1. Primary search intent (Informational/Commercial/Transactional/Navigational)
2. Should this have its own dedicated page, or be covered in existing content?
3. Which related keywords (from the list) can be covered on the SAME page?

Return ONLY valid JSON (no markdown, no preamble):
{{
  "keyword": "{keyword}",
  "inferred_intent": "Informational|Commercial|Transactional|Navigational",
  "needs_own_page": true|false,
  "rationale_for_own_page": "1-2 sentence explanation",
  "semantically_related_keywords_for_grouping": ["keyword1", "keyword2"]
}}"""

        try:
            raw = self.llm.complete(prompt, temperature=0.3, max_tokens=600)

            # Extract JSON
            m = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
            if m:
                s = m.group(1)
            else:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                s = raw[start:end] if start >= 0 and end > start else "{}"

            parsed = json.loads(s)

            return {
                "keyword": parsed.get("keyword", keyword),
                "inferred_intent": parsed.get("inferred_intent", self.infer_content_type(keyword)),
                "needs_own_page": parsed.get("needs_own_page", False),
                "rationale_for_own_page": parsed.get("rationale_for_own_page", ""),
                "semantically_related_keywords_for_grouping": [
                    x for x in parsed.get("semantically_related_keywords_for_grouping", [])
                    if isinstance(x, str) and x in all_keywords
                ][:10]
            }
        except Exception as e:
            st.warning(f"LLM analysis failed for '{keyword}': {str(e)[:100]}")
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
        Main orchestrator for keyword analysis with improved scoring.

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
        serp_titles = [s.get("title", "") for s in serp_results[:5]] if serp_results else []
        serp_insights = {
            "common_themes": "; ".join(serp_titles),
            "gaps_to_exploit": [],
            "unique_angles": []
        }

        # 1. BRAINSTORM KEYWORDS WITH LLM
        llm_brainstormed = []
        if self.llm.available:
            with st.spinner(f"Brainstorming keyword variations with {self.llm.model}..."):
                llm_brainstormed = generate_llm_keyword_variations(
                    base_keyword=query_topic,
                    main_topic=query_topic,
                    llm_client=self.llm,
                    max_variations=40,
                    desired_intent=desired_content_intent
                )
                st.info(f"✓ LLM generated {len(llm_brainstormed)} keyword variations")
        else:
            st.warning("LLM unavailable; using SERP-only keywords.")

        # 2. EXTRACT AND SCORE FROM ALL SOURCES
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

        # 3. COMBINE AND DEDUPLICATE
        all_scored = llm_scored + serp_related_scored + serp_snippets_scored
        
        # Semantic deduplication
        deduplicated = deduplicate_semantically(all_scored, similarity_threshold=0.75)
        
        # Remove main query
        deduplicated = [(k, s) for k, s in deduplicated if k.lower() != query_topic.lower()]

        # Sort by score
        sorted_keywords = sorted(deduplicated, key=lambda x: x[1], reverse=True)[:MAX_KEYWORD_ROWS]

        # All keywords for context
        all_kw = [query_topic] + [k for k, _ in sorted_keywords]

        # 4. BUILD DATAFRAME
        rows = []

        # Main topic
        main_intent = self.infer_content_type(query_topic)
        main_score, main_grade, main_breakdown = self.calculate_keyword_score(
            query_topic,
            999999,
            query_topic,
            desired_content_intent,
            serp_insights
        )
        
        rows.append({
            "Selected": True,
            "Keyword": query_topic,
            "Inferred Potential Score": main_score,
            "Grade": main_grade,
            "Content Type": main_intent,
            "Requires Own Content": "Yes",
            "Rationale for Own Page": "Main target keyword.",
            "Semantically Related Keywords": "",
            "Is PAA": "No",
            "Word Count": len(query_topic.split())
        })

        # Analyze top 20 with LLM
        top_for_analysis = min(20, len(sorted_keywords))
        
        if self.llm.available and top_for_analysis > 0:
            with st.spinner(f"Analyzing top {top_for_analysis} keywords..."):
                progress = st.progress(0)
                
                for idx, (kw, base_score) in enumerate(sorted_keywords[:top_for_analysis]):
                    info = self.analyze_keyword_with_llm(kw, all_kw, context=query_topic)
                    
                    final_score, grade, breakdown = self.calculate_keyword_score(
                        kw,
                        base_score,
                        query_topic,
                        desired_content_intent,
                        serp_insights
                    )
                    
                    rel = info.get("semantically_related_keywords_for_grouping", [])
                    rel = [r for r in rel if r.lower() != query_topic.lower()]
                    
                    is_paa = (
                        kw.endswith("?") or 
                        any(kw.lower().startswith(q + " ") for q in ["what", "how", "why", "when", "where", "who"])
                    )
                    
                    rows.append({
                        "Selected": final_score >= 200,
                        "Keyword": info.get("keyword", kw),
                        "Inferred Potential Score": final_score,
                        "Grade": grade,
                        "Content Type": info.get("inferred_intent", self.infer_content_type(kw)),
                        "Requires Own Content": "Yes" if info.get("needs_own_page", False) else "No",
                        "Rationale for Own Page": info.get("rationale_for_own_page", ""),
                        "Semantically Related Keywords": ", ".join(rel[:5]),
                        "Is PAA": "Yes" if is_paa else "No",
                        "Word Count": len(kw.split())
                    })
                    
                    progress.progress((idx + 1) / top_for_analysis)
                
                progress.empty()

        # Add remaining keywords
        for kw, base_score in sorted_keywords[top_for_analysis:]:
            final_score, grade, breakdown = self.calculate_keyword_score(
                kw,
                base_score,
                query_topic,
                desired_content_intent,
                serp_insights
            )
            
            is_paa = (
                kw.endswith("?") or 
                any(kw.lower().startswith(q + " ") for q in ["what", "how", "why"])
            )
            
            rows.append({
                "Selected": final_score >= 250,
                "Keyword": kw,
                "Inferred Potential Score": final_score,
                "Grade": grade,
                "Content Type": self.infer_content_type(kw),
                "Requires Own Content": "No",
                "Rationale for Own Page": "Can be covered in main content.",
                "Semantically Related Keywords": "",
                "Is PAA": "Yes" if is_paa else "No",
                "Word Count": len(kw.split())
            })

        df = pd.DataFrame(rows)

        # Filter by intent
        if desired_content_intent != "Any":
            keep = df[
                (df["Keyword"].str.lower() != query_topic.lower()) &
                (df["Content Type"] == desired_content_intent)
            ]
            main = df[df["Keyword"].str.lower() == query_topic.lower()]
            df = pd.concat([main, keep]).reset_index(drop=True)

            if len(keep) < len(df) - 1:
                st.info(
                    f"Filtered to {len(keep)} {desired_content_intent} keywords"
                )

        # Sort by score
        df = df.sort_values("Inferred Potential Score", ascending=False).reset_index(drop=True)

        return df, query_topic, serp_insights


class KeywordClusterer:
    """Cluster keywords semantically."""

    def __init__(self):
        try:
            from analysis.semantic_analyzer import SemanticAnalyzer
            self.semantic_analyzer = SemanticAnalyzer()
            self.available = self.semantic_analyzer.available
        except:
            self.semantic_analyzer = None
            self.available = False

        if not self.available:
            self.llm = get_llm_client()
            self.use_llm_fallback = self.llm.available
        else:
            self.use_llm_fallback = False

    def create_clusters(self, keywords_df: pd.DataFrame) -> Dict[str, List[str]]:
        if keywords_df is None or keywords_df.empty or "Keyword" not in keywords_df.columns:
            return {}

        top_keywords = [
            str(k) for k in keywords_df.head(25)["Keyword"].tolist()
            if isinstance(k, str) and k.strip()
        ]

        if not top_keywords:
            return {}

        if self.available:
            try:
                st.info("⚡ Using semantic clustering...")
                clusters = self.semantic_analyzer.cluster_keywords_semantically(
                    top_keywords,
                    distance_threshold=0.6
                )
                return clusters
            except:
                pass

        if self.use_llm_fallback:
            st.info("Using LLM clustering...")
            return self._cluster_llm(top_keywords)

        return self._simple_cluster(top_keywords)

    def _simple_cluster(self, keywords: List[str]) -> Dict[str, List[str]]:
        clusters = {}
        used = set()
        
        for i, kw in enumerate(keywords):
            if kw in used:
                continue
                
            cluster_name = f"Cluster {len(clusters) + 1}: {kw[:40]}"
            cluster = [kw]
            used.add(kw)
            
            kw_words = set(kw.lower().split())
            
            for other_kw in keywords[i+1:]:
                if other_kw in used:
                    continue
                
                other_words = set(other_kw.lower().split())
                if len(kw_words & other_words) >= 2:
                    cluster.append(other_kw)
                    used.add(other_kw)
            
            clusters[cluster_name] = cluster
        
        return clusters

    def _cluster_llm(self, top_keywords: List[str]) -> Dict[str, List[str]]:
        prompt = f"""Group into 4-6 semantic clusters:

{', '.join(top_keywords)}

Return ONLY JSON:
{{
  "Cluster 1": ["kw1", "kw2"],
  "Cluster 2": ["kw3"]
}}"""

        try:
            raw = self.llm.complete(prompt, temperature=0.2, max_tokens=800)
            m = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.DOTALL | re.I)
            s = m.group(1) if m else raw[raw.find("{"):raw.rfind("}")+1]
            clusters = json.loads(s)
            
            seen = set()
            cleaned = {}
            for name, kws in clusters.items():
                bucket = [k for k in kws if k in top_keywords and k not in seen]
                for k in bucket:
                    seen.add(k)
                if bucket:
                    cleaned[str(name)] = bucket
            
            missing = [k for k in top_keywords if k not in seen]
            if missing:
                cleaned["Other"] = missing
            
            return cleaned
        except:
            return self._simple_cluster(top_keywords)


def extract_common_headings(headings_list: List[List[str]], min_count: int = 2) -> List[str]:
    counter = Counter()
    for headings in headings_list:
        for h in headings:
            norm = re.sub(r"[^\w\s]", "", h.lower())
            if len(norm.split()) >= 2:
                counter[norm] += 1
    return [h for h, c in counter.most_common(10) if c >= min_count]
