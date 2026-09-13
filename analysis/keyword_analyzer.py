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
            scrapingdog_client: Configured Scrapingdog client - also used for
                                real search interest data (Google Trends) to
                                filter out keywords with no real demand.
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
        serp_insights: Optional[Dict] = None,
        trends_interest: Optional[float] = None
    ) -> Tuple[float, str, Dict]:
        """
        Calculate comprehensive keyword score with breakdown.
        
        Args:
            keyword: Keyword to score
            base_score: Initial score from extraction
            main_topic: Main topic for relevance
            desired_intent: Target content intent
            serp_insights: SERP analysis data
            trends_interest: Real relative search interest from Google Trends
                            (via Scrapingdog), 0-100 scale, if available. When
                            provided, this becomes the DOMINANT scoring factor -
                            a keyword nobody searches for should never outrank
                            one with real demand, regardless of how "clean" it
                            looks structurally. None means no Trends data was
                            returned for this keyword (different from a
                            confirmed 0, which means Trends returned data but
                            interest was negligible across the whole window).
            
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
            "trends_score": 0,
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
        
        # 6. REAL SEARCH INTEREST (Google Trends via Scrapingdog, when
        # available - this is the biggest fix). Without this, keywords that
        # "look" plausible (right length, contains topic words) can outscore
        # keywords nobody actually searches for. Trends gives a 0-100
        # RELATIVE interest score, not an absolute volume number, so
        # thresholds here are scaled 0-100, not raw search counts.
        if trends_interest is not None:
            if trends_interest == 0:
                # Trends returned real data and interest was negligible
                # across the whole 12-month window - heavy penalty
                # regardless of how clean the keyword looks structurally
                breakdown["penalties"] -= 200
            elif trends_interest < 5:
                breakdown["trends_score"] = 10
            elif trends_interest < 15:
                breakdown["trends_score"] = 60
            elif trends_interest < 35:
                breakdown["trends_score"] = 120
            elif trends_interest < 60:
                breakdown["trends_score"] = 180
            else:
                breakdown["trends_score"] = 220  # High relative interest - strong signal

        # 7. PENALTIES
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

    def analyze_keywords_batch(
        self,
        keywords: List[str],
        all_keywords: List[str],
        context: str = "",
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Analyze MULTIPLE keywords in a small number of LLM calls instead of
        one call per keyword. This is the main speed fix: 20 keywords used
        to mean 20 sequential round-trips; this brings it down to ~2-3 calls.

        Args:
            keywords: List of keywords to analyze
            all_keywords: Full list of keywords for relatedness context
            context: Main topic for context
            batch_size: How many keywords per LLM call (8 is a good balance -
                       large enough to cut call count, small enough that the
                       model doesn't lose track of items or truncate output)

        Returns:
            List of analysis dicts, one per input keyword, in the same order
        """
        if not keywords:
            return []

        if not self.llm.available:
            return [self._fallback_keyword_analysis(kw) for kw in keywords]

        results = []
        context_kw_full = [
            kw for kw in all_keywords[:50]
            if kw.lower() not in COMMON_STOP_WORDS and len(kw) >= 3
        ]

        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            batch_results = self._analyze_batch_chunk(batch, context_kw_full, context)
            results.extend(batch_results)

        return results

    def _analyze_batch_chunk(
        self,
        batch: List[str],
        context_kw_full: List[str],
        context: str
    ) -> List[Dict[str, Any]]:
        """Analyze a single batch (chunk) of keywords in ONE LLM call."""
        # Exclude the batch's own keywords from the context list to avoid noise
        batch_lower = {k.lower() for k in batch}
        context_kw = [kw for kw in context_kw_full if kw.lower() not in batch_lower][:25]

        keyword_list = "\n".join(f"{idx + 1}. {kw}" for idx, kw in enumerate(batch))

        prompt = f"""Analyze these {len(batch)} keywords for SEO content planning.

Main Topic: {context}

Keywords to analyze:
{keyword_list}

Related keywords for context (use ONLY these exact strings when suggesting groupings):
{', '.join(context_kw)}

For EACH keyword above, determine:
1. Primary search intent (Informational/Commercial/Transactional/Navigational)
2. Should it have its own dedicated page, or be covered in existing content?
3. Which related keywords (from the context list only) could be covered on the SAME page?

Return ONLY a valid JSON array with EXACTLY {len(batch)} objects, in the SAME ORDER as the numbered list above. No markdown, no preamble, no explanation - just the array:

[
  {{"keyword": "exact keyword text", "inferred_intent": "Informational|Commercial|Transactional|Navigational", "needs_own_page": true|false, "rationale_for_own_page": "1 sentence", "semantically_related_keywords_for_grouping": ["kw1", "kw2"]}}
]"""

        try:
            # Scale max_tokens with batch size - roughly 150-200 tokens per keyword result
            raw = self.llm.complete(prompt, temperature=0.3, max_tokens=max(800, len(batch) * 220))

            parsed_array = self._extract_json_array(raw)

            if not parsed_array or len(parsed_array) == 0:
                st.warning(f"Batch analysis returned no results for {len(batch)} keywords, using fallback scoring.")
                return [self._fallback_keyword_analysis(kw) for kw in batch]

            # Map results back to original keywords by position, with a
            # safety fallback if the LLM returned a different count or
            # reordered things (occasionally happens with smaller models)
            results = []
            for idx, kw in enumerate(batch):
                if idx < len(parsed_array) and isinstance(parsed_array[idx], dict):
                    item = parsed_array[idx]
                    rel = item.get("semantically_related_keywords_for_grouping", [])
                    rel = [r for r in rel if isinstance(r, str) and r in context_kw_full]
                    results.append({
                        "keyword": kw,  # Use original keyword, not LLM's echo (avoids drift)
                        "inferred_intent": item.get("inferred_intent", self.infer_content_type(kw)),
                        "needs_own_page": item.get("needs_own_page", False),
                        "rationale_for_own_page": item.get("rationale_for_own_page", ""),
                        "semantically_related_keywords_for_grouping": rel[:10]
                    })
                else:
                    results.append(self._fallback_keyword_analysis(kw))

            return results

        except Exception as e:
            st.warning(f"Batch keyword analysis failed: {str(e)[:150]}. Using fallback scoring for this batch.")
            return [self._fallback_keyword_analysis(kw) for kw in batch]

    def _extract_json_array(self, text: str) -> List[Dict]:
        """Robustly extract a JSON array from LLM response text."""
        cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)

        try:
            start = cleaned.index('[')
            end = cleaned.rindex(']') + 1
            json_str = cleaned[start:end]
        except ValueError:
            return []

        try:
            result = json.loads(json_str)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            # Try fixing trailing commas
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                result = json.loads(fixed)
                return result if isinstance(result, list) else []
            except json.JSONDecodeError:
                return []

    def _fallback_keyword_analysis(self, keyword: str) -> Dict[str, Any]:
        """Fallback analysis when LLM is unavailable or batch parsing fails."""
        return {
            "keyword": keyword,
            "inferred_intent": self.infer_content_type(keyword),
            "needs_own_page": False,
            "rationale_for_own_page": "Heuristic fallback (LLM analysis unavailable or failed).",
            "semantically_related_keywords_for_grouping": []
        }

    def analyze_keyword_with_llm(
        self,
        keyword: str,
        all_keywords: List[str],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        DEPRECATED for bulk use - kept for single-keyword ad-hoc analysis only
        (e.g. re-analyzing one keyword after manual edit). For processing
        multiple keywords, use analyze_keywords_batch() instead, which does
        the same analysis in a fraction of the LLM calls.
        """
        if not self.llm.available:
            return self._fallback_keyword_analysis(keyword)

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

        Reordered pipeline (vs. original):
          1. Brainstorm + extract candidates (unchanged, cheap)
          2. Dedupe (unchanged)
          3. NEW: Pull real search interest data (Google Trends, via
             Scrapingdog) for ALL candidates and drop confirmed-zero-interest
             keywords BEFORE spending LLM time on them
          4. Batch LLM analysis (2-3 calls instead of 20) on the SURVIVORS
          5. Score everything, real volume data now dominates the score

        This fixes both the speed complaint (20 sequential LLM calls -> 2-3
        batched calls) and the relevance complaint (keywords that "look"
        plausible but nobody searches for get filtered before they ever
        reach the expensive analysis step).

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

        # Sort by heuristic score first (cheap, no API calls yet)
        sorted_keywords = sorted(deduplicated, key=lambda x: x[1], reverse=True)[:MAX_KEYWORD_ROWS]

        # 4. NEW: PULL REAL SEARCH INTEREST DATA (GOOGLE TRENDS) AND FILTER
        # CONFIRMED-DEAD KEYWORDS. This happens BEFORE the expensive LLM
        # analysis step, so we're not wasting LLM calls analyzing keywords
        # nobody searches for. Batched 5-at-a-time via Scrapingdog's Trends
        # TIMESERIES endpoint (5 credits per call of up to 5 keywords).
        trends_lookup: Dict[str, float] = {}
        if sorted_keywords:
            with st.spinner("Checking real search interest (Google Trends)..."):
                candidate_kws = [kw for kw, _ in sorted_keywords]
                try:
                    trends_lookup = self.scrapingdog.get_trends_interest_batched(candidate_kws)
                except Exception as e:
                    st.warning(f"Google Trends lookup failed: {str(e)[:150]}. Falling back to heuristic scoring only.")
                    trends_lookup = {}

                if trends_lookup:
                    found_count = len(trends_lookup)
                    st.info(f"✓ Retrieved real search interest for {found_count}/{len(candidate_kws)} keywords")

                    # Filter: drop keywords with CONFIRMED zero interest
                    # across the whole 12-month window BEFORE spending LLM
                    # time on them. Keep keywords Trends returned no data
                    # for at all (could be too new/niche, or the request
                    # failed for just that chunk) - they simply won't get
                    # the interest score boost and will rank lower naturally
                    # rather than being incorrectly dropped.
                    before_count = len(sorted_keywords)
                    sorted_keywords = [
                        (kw, score) for kw, score in sorted_keywords
                        if trends_lookup.get(kw, 1) != 0
                    ]
                    dropped = before_count - len(sorted_keywords)
                    if dropped > 0:
                        st.info(f"✓ Filtered out {dropped} keywords with confirmed zero search interest - saving LLM analysis time on dead keywords")
                else:
                    st.caption("💡 No Google Trends data returned - proceeding with heuristic scoring only.")

        # All keywords for context (post-filtering)
        all_kw = [query_topic] + [k for k, _ in sorted_keywords]

        # 5. BUILD DATAFRAME
        rows = []

        # Main topic
        main_intent = self.infer_content_type(query_topic)
        main_score, main_grade, main_breakdown = self.calculate_keyword_score(
            query_topic,
            999999,
            query_topic,
            desired_content_intent,
            serp_insights,
            trends_interest=trends_lookup.get(query_topic)
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
            "Search Interest": trends_lookup.get(query_topic, "—"),
            "Word Count": len(query_topic.split())
        })

        # 6. BATCH LLM ANALYSIS (2-3 calls instead of 20 sequential calls)
        top_for_analysis = min(20, len(sorted_keywords))
        analysis_results: Dict[str, Dict] = {}

        if self.llm.available and top_for_analysis > 0:
            top_keywords = [kw for kw, _ in sorted_keywords[:top_for_analysis]]
            with st.spinner(f"Analyzing top {top_for_analysis} keywords (batched)..."):
                batch_analysis = self.analyze_keywords_batch(
                    top_keywords, all_kw, context=query_topic, batch_size=8
                )
                analysis_results = {item["keyword"]: item for item in batch_analysis}

        for idx, (kw, base_score) in enumerate(sorted_keywords[:top_for_analysis]):
            info = analysis_results.get(kw, self._fallback_keyword_analysis(kw))
            interest = trends_lookup.get(kw)

            final_score, grade, breakdown = self.calculate_keyword_score(
                kw,
                base_score,
                query_topic,
                desired_content_intent,
                serp_insights,
                trends_interest=interest
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
                "Search Interest": interest if interest is not None else "—",
                "Word Count": len(kw.split())
            })

        # Add remaining keywords (beyond top_for_analysis) - no LLM call,
        # just heuristic + Trends interest scoring
        for kw, base_score in sorted_keywords[top_for_analysis:]:
            interest = trends_lookup.get(kw)
            final_score, grade, breakdown = self.calculate_keyword_score(
                kw,
                base_score,
                query_topic,
                desired_content_intent,
                serp_insights,
                trends_interest=interest
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
                "Search Interest": interest if interest is not None else "—",
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
