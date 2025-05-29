import streamlit as st
from serpapi import GoogleSearch
import os
import re
from collections import Counter
import math
import json
import pandas as pd
from docxtpl import DocxTemplate # Import docxtpl
from datetime import date # Import date for current date

# --- Configuration & Setup ---

# SerpApi API key
serpapi_api_key = None
try:
    serpapi_api_key = st.secrets["serpapi_api_key"]
except (AttributeError, KeyError):
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if not serpapi_api_key:
    st.error("SerpApi API key not found. Please set it in .streamlit/secrets.toml or as an environment variable (SERPAPI_API_KEY).")
    st.stop()

# --- Agent Definitions ---
# ALL CLASSES ARE DEFINED HERE, BEFORE THEY ARE INSTANTIATED

class KeywordResearcher:
    def __init__(self, api_key):
        self.api_key = api_key

    # IMPORTANT FIX: Changed 'self' to '_self' in the method signature for caching
    @st.cache_data(ttl=3600) # Cache results for 1 hour
    def get_keywords(_self, query, category=None): # <--- HERE: _self is used in the signature
        st.info(f"Making SerpApi call for initial keywords: **'{query}'**")
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": _self.api_key, # <--- AND HERE: _self is used to access instance attributes
                "num": 10
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            keywords = []
            if "organic_results" in results:
                keywords.extend([r.get("title") for r in results["organic_results"] if r.get("title")])

            if "related_searches" in results:
                keywords.extend([s.get("query") for s in results["related_searches"] if s.get("query")])

            if "related_questions" in results:
                keywords.extend([q.get("question") for q in results["related_questions"] if q.get("question")])

            return keywords
        except Exception as e:
            st.error(f"Error in KeywordResearcher.get_keywords with SerpApi: {e}")
            return []

class SERPAnalyst:
    def __init__(self, api_key):
        self.api_key = api_key

    # IMPORTANT FIX: Changed 'self' to '_self' in the method signature for caching
    @st.cache_data(ttl=3600) # Cache results for 1 hour
    def analyze_serp(_self, query, category=None): # <--- HERE: _self is used in the signature
        st.info(f"Making SerpApi call for SERP analysis: **'{query}'**")
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": _self.api_key, # <--- AND HERE: _self is used to access instance attributes
                "num": 10
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            serp_data = []
            if "organic_results" in results:
                for r in results["organic_results"]:
                    serp_data.append({
                        "url": r.get("link", "N/A"),
                        "title": r.get("title", "N/A"),
                        "snippet": r.get("snippet", "No snippet available.")
                    })
            return serp_data
        except Exception as e:
            st.error(f"Error in SERPAnalyst.analyze_serp with SerpApi: {e}")
            return []

class DataAnalyzer:
    def __init__(self, serp_api_key): # Pass api_key to DataAnalyzer
        self.serp_api_key = serp_api_key # Store API key for internal SERP calls

    def brainstorm_keywords_with_llm(self, query_topic, initial_keywords, serp_data, category=None):
        serp_data_str = "\n".join([
            f"  - Title: {d.get('title', 'N/A')}\n    Snippet: {d.get('snippet', 'N/A')}"
            for d in serp_data
        ])

        category_context = ""
        category_for_prompt_sentence = ""
        if category:
            category_display_name = category.replace('_', ' ').title()
            category_context = f" for content categorized as '{category_display_name}'"
            category_for_prompt_sentence = f"Consider the typical language and focus of {category_display_name} content."

        # MODIFIED PROMPT: This prompt is for the *initial keyword brainstorming*
        # It still returns a JSON array of strings as per original design.
        prompt = f"""You are an expert SEO content strategist and keyword brainstormer. Your task is to generate a comprehensive list of potential SEO keywords, long-tail phrases, and related concepts based on the provided initial keywords and SERP data{category_context}.
        The main topic for this analysis is: "{query_topic}".

        **Generate at least 15-25 distinct keyword ideas, focusing on:**
        * Core topic variations: Different ways users might search for the main topic.
        * Long-tail keywords: More specific, multi-word phrases or questions, often indicating specific user intent.
        * User intent: Keywords suggesting informational, commercial, or navigational intent (explicitly state intent if clear, e.g., "best [product] for X", "how to [task]").
        * Related sub-topics: Concepts frequently mentioned in the SERP snippets that expand on the main topic.
        * Problem/Solution oriented keywords: How users might phrase their problems related to the topic or seek solutions.
        * Comparative keywords (e.g., "X vs Y").
        * Location-based keywords (if applicable, e.g., "SaaS SEO agency London").
        {category_for_prompt_sentence}

        **Output Format:**
        Return ONLY a JSON array of strings. Each string in the array should be a distinct, clean keyword idea.
        DO NOT include any conversational text, explanations, intros, outros, or any other characters before or after the JSON array.
        The JSON must be valid and directly parsable.

        Example:
        ["SaaS SEO services for startups", "B2B software content strategy", "How to choose an enterprise SEO agency", "Lead generation for SaaS companies", "Best SEO tools for tech companies"]

        ---

        Initial keywords (from initial broad search related to "{query_topic}"):
        {', '.join(initial_keywords) if initial_keywords else 'None'}

        Comprehensive SERP Data (top 10 search results including Titles and Snippets for "{query_topic}"):
        {serp_data_str}

        ---

        Based on this analysis, generate a JSON array of 15-25 high-potential keyword ideas:
        """

        st.markdown("#### LLM Prompt Details (Keyword Brainstorming):")
        with st.expander("Click to view the full prompt sent to Llama 3 for keyword brainstorming"):
            st.code(prompt, language='markdown')

        try:
            from litellm import completion
            response = completion(
                model='ollama/llama3',
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=700,
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                raw_llm_output = response.choices[0].message.content

                st.markdown("#### Raw Llama 3 Output (Keyword Brainstorming):")
                with st.expander("Click to view the raw (unprocessed) output from Llama 3 for keyword brainstorming"):
                    st.code(raw_llm_output, language='text')

                try:
                    # Pre-process the raw LLM output to extract only the JSON part
                    json_start = raw_llm_output.find('[')
                    json_end = raw_llm_output.rfind(']')

                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_string = raw_llm_output[json_start : json_end + 1]
                        llm_suggested_keywords = json.loads(json_string)

                        if not isinstance(llm_suggested_keywords, list):
                            st.warning("LLaMA 3 returned valid JSON, but it was not a list as expected. Please check the prompt instruction.")
                            return []
                        return llm_suggested_keywords
                    else:
                        st.error(f"Could not find a valid JSON array ([...]) in the LLaMA 3 output. Raw output: `{raw_llm_output}`")
                        return []

                except json.JSONDecodeError as e:
                    st.error(f"LLaMA 3 output was not valid JSON after extraction attempt: {e}. Extracted string: `{json_string}`")
                    return []
            else:
                st.warning("LLaMA 3 did not return any choices in the response.")
                return []
        except Exception as e:
            st.error(f"Error during LLaMA 3 keyword brainstorming: {e}")
            return []

    # New helper method for internal SERP analysis for a single keyword
    # It's good to cache this if it might be called multiple times for the same keyword
    @st.cache_data(ttl=3600)
    def _analyze_single_keyword_serp(_self, keyword, category=None): # Use _self
        st.info(f"Analyzing SERP for individual keyword: **'{keyword}'** (this uses SerpApi credits).")
        try:
            params = {
                "engine": "google",
                "q": keyword,
                "api_key": _self.serp_api_key, # Use _self.serp_api_key
                "num": 5 # Fetch fewer results for individual keyword analysis to save calls
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            serp_data = []
            if "organic_results" in results:
                for r in results["organic_results"]:
                    serp_data.append({
                        "url": r.get("link", "N/A"),
                        "title": r.get("title", "N/A"),
                        "snippet": r.get("snippet", "No snippet available.")
                    })
            return serp_data
        except Exception as e:
            st.warning(f"Warning: Error analyzing single keyword SERP for '{keyword}': {e}. Content type will be inferred heuristically.")
            return []

    # New helper method to infer content type using only keyword string (no API call)
    def _infer_content_type_simple_heuristic(self, keyword):
        keyword_lower = keyword.lower()
        informational_terms = ["how to", "what is", "guide", "tutorial", "explain", "examples", "definition", "learn", "why", "who", "when", "meaning"]
        commercial_terms = ["buy", "price", "cost", "best", "top", "review", "vs", "comparison", "alternatives", "deal", "discount", "services", "agency", "software", "tool", "platform", "pricing", "pricing plan", "hire", "consultant"]
        navigational_terms = ["login", "dashboard", "account", "careers", "contact", "about us", "my account", "sign up", "sign in"]

        # Prioritize navigational, then commercial, then informational
        if any(term in keyword_lower for term in navigational_terms):
            return "Navigational"
        if any(term in keyword_lower for term in commercial_terms):
            return "Commercial"
        # If no strong commercial or navigational signals, default to informational
        return "Informational"


    # New helper method to infer content type using SERP results
    def _infer_content_type(self, keyword, serp_results):
        keyword_lower = keyword.lower()

        # Define keywords associated with different intents
        informational_terms = ["how to", "what is", "guide", "tutorial", "explain", "examples", "definition", "learn", "why", "who", "when", "meaning"]
        commercial_terms = ["buy", "price", "cost", "best", "top", "review", "vs", "comparison", "alternatives", "deal", "discount", "services", "agency", "software", "tool", "platform", "pricing", "pricing plan", "hire", "consultant"]
        navigational_terms = ["login", "dashboard", "account", "careers", "contact", "about us", "my account", "sign up", "sign in"]
        # Add common brand names/company names here if applicable to your domain, e.g., if you frequently search for "HubSpot login"
        # navigational_terms.extend(["hubspot", "semrush", "ahrefs"])

        # Scores for each intent based on keyword presence
        informational_score = sum(1 for term in informational_terms if term in keyword_lower)
        commercial_score = sum(1 for term in commercial_terms if term in keyword_lower)
        navigational_score = sum(1 for term in navigational_terms if term in keyword_lower)

        # Analyze SERP results for stronger signals
        if serp_results:
            for result in serp_results:
                title_lower = result.get("title", "").lower()
                snippet_lower = result.get("snippet", "").lower()
                url_lower = result.get("url", "").lower()

                # Boost scores based on SERP content
                if any(term in title_lower or term in snippet_lower for term in informational_terms):
                    informational_score += 2 # Stronger signal from title/snippet
                if any(term in title_lower or term in snippet_lower for term in commercial_terms):
                    commercial_score += 2
                if any(term in navigational_terms for term in title_lower.split() + url_lower.split('/') + keyword_lower.split()): # Check URL for navigational as well, and if the keyword itself is directly navigational
                    navigational_score += 3 # Navigational is often very direct

                # Specific URL patterns can strongly indicate intent
                if "/blog/" in url_lower or "/guides/" in url_lower or "/learn/" in url_lower:
                    informational_score += 1
                if "/product/" in url_lower or "/service/" in url_lower or "/pricing/" in url_lower or "/buy/" in url_lower:
                    commercial_score += 1
                if any(term in url_lower for term in navigational_terms):
                    navigational_score += 1

                # Check if the URL is just a domain or very short path (often indicative of navigational)
                if len(url_lower.split('/')) <= 3 and keyword_lower.replace(" ", "") in url_lower.replace("www.", "").replace(".com", "").replace(".net", "").replace(".org", ""):
                    navigational_score += 1


        # Determine the highest scoring intent
        # Prioritize navigational, then commercial, then informational as a fallback
        if navigational_score > informational_score and navigational_score > commercial_score:
            return "Navigational"
        elif commercial_score > informational_score: # Check commercial if not navigational
            return "Commercial"
        elif informational_score > 0: # If any informational signals found
            return "Informational"
        else:
            return "Informational" # Default to informational if no strong signals (most content is informational)


    def analyze_data_and_identify_target_keywords_orchestrator(self, query_topic, initial_keywords_from_serpapi, serp_data, selected_category=None):
        st.markdown("### Llama 3's Brainstormed Keyword Ideas (for context):")
        llm_brainstormed_keywords_raw = self.brainstorm_keywords_with_llm(query_topic, initial_keywords_from_serpapi, serp_data, category=selected_category)
        if llm_brainstormed_keywords_raw:
            st.write("---")
            st.write("#### Extracted & Cleaned Llama 3 Keywords (from LLM's direct output):")
            st.write("\n".join([f"- {kw}" for kw in llm_brainstormed_keywords_raw]))
        else:
            st.write("No direct keyword suggestions from Llama 3 for brainstorming.")


        initial_topic_words = set(query_topic.lower().split())
        final_target_keywords_scored = []
        llm_generated_scored_keywords = [] # Store LLM keywords separately with their initial score

        # 2. Process LLM suggestions with a significant score boost
        if llm_brainstormed_keywords_raw:
            llm_processed_keywords = extract_and_filter_keywords(
                llm_brainstormed_keywords_raw,
                initial_topic_words,
                min_len_words=1, # Allow single words from LLM if relevant (e.g., "pricing")
                min_len_chars=3,
                source_type='llm',
                # Minimal exclude list for LLM keywords, as they should be clean by design
                exclude_words={"a", "an", "the", "for", "in", "of", "to", "and", "or", "is", "with", "best", "how", "what", "why"}
            )
            # Add a significant initial boost to LLM keywords
            for kw, score in llm_processed_keywords:
                boosted_score = score + 500 # Substantial boost to prioritize LLM ideas
                final_target_keywords_scored.append((kw, boosted_score))
                llm_generated_scored_keywords.append((kw, boosted_score)) # Keep separate for brief selection


        # 3. Combine initial titles (from SerpApi) and SERP snippets for robust keyword extraction and scoring
        all_raw_text_for_extraction = []
        all_raw_text_for_extraction.append(query_topic)
        all_raw_text_for_extraction.extend(initial_keywords_from_serpapi)

        for d in serp_data:
            snippet_content = d.get('snippet', '')
            if snippet_content and snippet_content not in ["No snippet available.", "No detailed snippet available (could not retrieve full content).", "No snippet available for this result."]:
                all_raw_text_for_extraction.append(snippet_content)

        extracted_from_serp_scored = extract_and_filter_keywords(
            all_raw_text_for_extraction,
            initial_topic_words,
            source_type='serp', # Indicate source for tailored filtering/scoring
            # A much more comprehensive and aggressive exclude list for SERP data
            exclude_words={
                "a", "an", "the", "for", "in", "of", "to", "and", "or", "is", "with", "best", "vs", "how", "what", "why", "guide",
                "companies", "services", "solutions", "systems", "top", "agency", "seo", "saas", "software", "rocket", "labs",
                "rock", "rankings", "media", "lifted", "search", "turn", "growth", "accelerate", "skale", "inc", "ltd", "llc",
                "group", "marketing", "digital", "performance", "b2b", "mrr", "gmb", "sem", "provider", "partners", "partner",
                "dedicated", "business", "company", "firm", "expert", "consultant", "strategy", "strategies", "management",
                "online", "website", "content", "leads", "revenue", "traffic", "rank", "ranking", "get", "read", "click", "find", "more", "learn", "start", "discover", "your", "our", "their",
                "technical",
                "markitors", "ninjapromo", "seoworks", "smartsites", "straightnorth", "pearl", "lemon", "ironpaper", "cleverly", "lycold", "outreac", # Specific noise from screenshots
                "pricing", "cold", "email", "abm", "contentmarketing", "consultancy", "engine", "setup", "audit", "specialist",
                "slyminh", "esweo", "thumbdatimagegifbase64", "lgww3x02002002003evg3e", # Extremely noisy patterns from screenshots
                "www", "http", "https", "com", "net", "org", "co", "io", "ly", "html", "js", "css", "src", "cdn", "assets", "var", "function", "div", "span", "class", "id", # Web/code remnants
                "copyright", "all rights reserved", "privacy policy", "terms of service", "disclaimer", "cookie policy", "contact us", "about us", "careers", "blog", "news", "support", "faq", "resources", "download", "sign up", "login", "subscribe", # Common footer/sidebar items
                "nav-link", "footer", "header", "menu", "sidebar", "widget", "button", "cta", "read more", "learn more", "view all", "article", "page", "post", "section", "chapter", "part", # Generic structural words
                "free", "audithttpsscalabledmediaio", "audit", "image", "file", "skip", "trust", "were", "ctl", "on", "pay", "boost", "tech" # More specific junk
            }
        )
        final_target_keywords_scored.extend(extracted_from_serp_scored)

        # 4. Deduplicate and re-sort based on combined scores
        unique_keywords_map = {}
        for kw, score in final_target_keywords_scored:
            normalized_kw = kw.lower()
            if normalized_kw not in unique_keywords_map or score > unique_keywords_map[normalized_kw][1]:
                unique_keywords_map[normalized_kw] = (kw, score)

        final_target_keywords_scored_deduped = list(unique_keywords_map.values())
        final_target_keywords_scored_deduped.sort(key=lambda x: x[1], reverse=True)

        # 5. Select the primary keyword for the brief
        selected_brief_keyword = ""

        # FIRST PRIORITY: Choose from LLM-generated keywords if possible
        if llm_generated_scored_keywords:
            llm_generated_scored_keywords.sort(key=lambda x: x[1], reverse=True) # Sort LLM keywords by their boosted score
            for kw, score in llm_generated_scored_keywords:
                kw_lower = kw.lower()
                # Ensure the LLM keyword is descriptive and aligns with the topic
                if any(word in kw_lower for word in initial_topic_words) and \
                   len(kw.split()) >= 2 and len(kw) > 8 and \
                   kw_lower not in {"technical", "seo", "consultancy", "professional", "engine", "setup", "audit", "specialist"}:
                    selected_brief_keyword = kw
                    break

            if not selected_brief_keyword and llm_generated_scored_keywords:
                # Fallback within LLM keywords: pick the highest scored one if no perfect match
                selected_brief_keyword = llm_generated_scored_keywords[0][0]

        # SECOND PRIORITY: If no suitable LLM keyword, then pick from the overall top-scored list
        if not selected_brief_keyword and final_target_keywords_scored_deduped:
            for kw, score in final_target_keywords_scored_deduped:
                kw_lower = kw.lower()
                if any(word in kw_lower for word in initial_topic_words) and \
                   len(kw.split()) >= 2 and len(kw) > 8 and \
                   kw_lower not in {"technical", "seo", "consultancy", "professional", "engine", "setup", "audit", "specialist"}:
                    selected_brief_keyword = kw
                    break

            if not selected_brief_keyword and final_target_keywords_scored_deduped: # Absolute fallback
                selected_brief_keyword = final_target_keywords_scored_deduped[0][0]
        else:
            selected_brief_keyword = query_topic # Fallback to original query if nothing suitable found

        # --- NEW LOGIC: DETERMINE CONTENT TYPE AND REQUIRES OWN CONTENT EFFICIENTLY ---
        keywords_with_flags_and_types = []

        # Keep track of keywords for which we've done a full SERP call
        processed_keywords_for_serp_call = set()

        # Define how many top keywords get a full SERP analysis
        TOP_N_FOR_FULL_SERP_ANALYSIS = 5 # Changed to limit to top 5

        st.markdown(f"#### Running detailed SERP analysis for the top {TOP_N_FOR_FULL_SERP_ANALYSIS} keywords and the selected main brief keyword...")
        # Add the selected_brief_keyword to a list to ensure it's processed if not already in top N
        keywords_to_analyze_fully = [item[0] for item in final_target_keywords_scored_deduped[:TOP_N_FOR_FULL_SERP_ANALYSIS]]
        if selected_brief_keyword and selected_brief_keyword not in keywords_to_analyze_fully:
            keywords_to_analyze_fully.append(selected_brief_keyword)

        for kw, score in final_target_keywords_scored_deduped:
            requires_own_content = False
            content_type = "Informational" # Default to informational

            # Only perform full SERP analysis for the top N and the main brief keyword
            if kw in keywords_to_analyze_fully:
                kw_serp_results = self._analyze_single_keyword_serp(kw, category=selected_category)
                content_type = self._infer_content_type(kw, kw_serp_results) # Use the more robust inference
                processed_keywords_for_serp_call.add(kw)

                # Heuristic for 'Requires Own Content' based on full SERP analysis
                if content_type == "Navigational":
                    requires_own_content = True
                elif content_type == "Commercial" and score > 400:
                    requires_own_content = True
                elif content_type == "Informational" and len(kw.split()) <= 2 and score > 450:
                    if kw_serp_results:
                        avg_snippet_length = sum(len(res.get('snippet', '')) for res in kw_serp_results) / max(1, len(kw_serp_results))
                        if avg_snippet_length > 150 and len(kw_serp_results) >= 3:
                            requires_own_content = True
            else:
                # For other keywords, use a simpler, credit-free heuristic
                content_type = self._infer_content_type_simple_heuristic(kw)
                # For non-top keywords, generally assume they don't require own content unless very high score or specific type
                if content_type == "Navigational": # A navigational term often implies its own destination
                     requires_own_content = True
                elif score > 550 and len(kw.split()) > 1: # Very high score for multi-word non-top keyword
                    requires_own_content = True # Could be a key sub-topic

            keywords_with_flags_and_types.append({
                "Keyword": kw,
                "Inferred Potential Score": f"{score:.1f}",
                "Content Type": content_type,
                "Include in Brief": True, # Default to true, user can uncheck
                "Requires Own Content": requires_own_content
            })

        # Sort the final list by score again
        keywords_with_flags_and_types.sort(key=lambda x: float(x["Inferred Potential Score"]), reverse=True)

        df_keywords = pd.DataFrame(keywords_with_flags_and_types)

        return df_keywords, selected_brief_keyword

# --- Helper Function for Keyword Extraction and Enhanced Potential Scoring ---
# This function is used by DataAnalyzer, so it should be defined before DataAnalyzer is instantiated
def extract_and_filter_keywords(text_list, initial_query_words, min_len_words=2, min_len_chars=5, exclude_words=None, source_type='general'):
    if exclude_words is None:
        exclude_words = set()

    initial_query_words_lower = set([word.lower() for word in initial_query_words])

    extracted_keywords_with_counts = Counter()

    processed_texts = []
    for item in text_list:
        if isinstance(item, str):
            if item and item not in ["No snippet available.", "No detailed snippet available (could not retrieve full content).", "No snippet available for this result."]:
                clean_item = item.strip()

                # Step 1: VERY AGGRESSIVE PRE-FILTERING OF KNOWN NOISE PATTERNS
                # URLs, file paths, hashes, base64, long numeric sequences, specific junk
                clean_item = re.sub(r'https?://[^\s/$.?#].[^\s]*', ' ', clean_item, flags=re.IGNORECASE)
                clean_item = re.sub(r'\b[a-zA-Z0-9_-]+\.(?:png|jpg|jpeg|gif|webp|pdf|doc|docx|xls|xlsx|ppt|pptx|html|js|css|zip|rar|webp|svg)\b', ' ', clean_item, flags=re.IGNORECASE)
                clean_item = re.sub(r'data:[^;]+;base64,[^\s]+', ' ', clean_item, flags=re.IGNORECASE) # Base64 strings
                clean_item = re.sub(r'[0-9a-fA-F]{32,}', ' ', clean_item) # Long hex strings (potential hashes/IDs)
                clean_item = re.sub(r'\b\d{5,}\b', ' ', clean_item) # Long numbers
                clean_item = re.sub(r'\b(?:lgww3x02002002003evg3e|thumbdatimagegifbase64|slyminh|esweo|audithttpsscalabledmediaio)\b', ' ', clean_item, flags=re.IGNORECASE) # Specific identified junk

                # Remove non-alphanumeric characters but keep spaces and hyphens for phrases
                clean_item = re.sub(r'[^\w\s-]', '', clean_item)
                clean_item = re.sub(r'\s+', ' ', clean_item).strip() # Normalize whitespace

                if clean_item:
                    processed_texts.append(clean_item)

    full_text = " ".join(processed_texts).lower()

    # Tokenize into words and phrases
    words = full_text.split()
    # Consider bigrams and trigrams
    phrases = []
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if all(word not in exclude_words for word in phrase.split()):
            phrases.append(phrase)
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        if all(word not in exclude_words for word in phrase.split()):
            phrases.append(phrase)

    all_terms = words + phrases

    # Filter out short words and common stopwords/noise
    filtered_terms = [
        term for term in all_terms
        if len(term.split()) >= min_len_words and len(term) >= min_len_chars and term not in exclude_words
        and not any(ex_word in term for ex_word in exclude_words) # Exclude phrases containing exclude words
    ]

    # Score terms based on frequency and relevance to initial query
    for term in filtered_terms:
        score = 1
        # Boost for terms containing words from the initial query
        if any(q_word in term for q_word in initial_query_words_lower):
            score += 2
        # Further boost for longer, more specific phrases
        if len(term.split()) >= 3:
            score += 1.5
        elif len(term.split()) == 2:
            score += 1

        extracted_keywords_with_counts[term] += score

    # Apply a logarithmic decay to scores to reduce dominance of very high frequency terms
    # and normalize them to a more manageable range for "potential score"
    scored_keywords = []
    for term, count in extracted_keywords_with_counts.items():
        # Apply a base score + log of count to give diminishing returns for very high counts
        # and multiply by 100 to make scores more visible (e.g., 100-1000 range)
        potential_score = (count + math.log(count + 1)) * 100
        scored_keywords.append((term, potential_score))

    return scored_keywords

class ContentBriefCreator:
    def __init__(self):
        pass

    # MODIFIED create_content_brief: This function now generates the *full structured Markdown brief*
    # for all sections that the LLM is responsible for.
    def create_content_brief(self, keyword, related_keywords=None, category=None, content_type=None,
                             audience_data=None, serp_insights=None, word_count_range=None):
        category_context = ""
        if category:
            category_context = f" for content categorized as '{category.replace('_', ' ').lower()}'"

        content_type_context = ""
        if content_type:
            content_type_context = f" The primary search intent is determined to be **{content_type}**."

        related_keywords_str = ""
        clean_related_keywords = [
            kw for kw in (related_keywords if related_keywords else [])
            if kw.lower() != keyword.lower() # Exclude main keyword from related
        ]
        if clean_related_keywords:
            related_keywords_str_formatted = ", ".join(clean_related_keywords[:15]) # Take top 15 for prompt
            related_keywords_str = f"\n\n**Additional Important Keywords to Consider:** {related_keywords_str_formatted}"

        # Prepare audience data for the prompt if available
        audience_prompt_section = ""
        if audience_data:
            audience_prompt_section += "\n### 1. Target Audience (Based on Initial Analysis):"
            for key, value in audience_data.items():
                audience_prompt_section += f"\n* **{key.replace('_', ' ').title()}:** {value}"

        # Prepare SERP insights for the prompt if available
        serp_insights_prompt_section = ""
        if serp_insights:
            serp_insights_prompt_section += "\n### Competitor Analysis Insights (Based on SERP):"
            serp_insights_prompt_section += f"\n* **Common Themes:** {serp_insights.get('common_themes', 'Not available.')}"
            serp_insights_prompt_section += f"\n* **Gaps to Exploit:** {serp_insights.get('gaps_to_exploit', 'Not available.')}"
            serp_insights_prompt_section += f"\n* **Unique Angles:** {serp_insights.get('unique_angles', 'Not available.')}"


        # --- MAIN PROMPT FOR GENERATING THE FULL CONTENT BRIEF IN MARKDOWN ---
        # This prompt is designed to output the entire brief in a structured Markdown format
        # that can then be parsed by our Python script for docxtpl.
        prompt = f"""Generate a comprehensive SEO content brief for the target keyword: "{keyword}"{category_context}.
        {content_type_context}

        The brief should be structured to guide a content writer effectively and optimize for both traditional SEO and generative AI search.
        Return the entire brief in **strict Markdown format**, ensuring precise use of `#`, `##`, `###`, `*`, and `**` for formatting.
        Do NOT include any conversational text, explanations, intros, or outros outside of the specified sections.

        ## 1. Target Audience
        * **Demographics:** Describe the ideal reader's demographics (e.g., age range, profession, location).
        * **Psychographics:** Describe their psychographics (e.g., interests, values, attitudes).
        * **Pain Points:** List 3-5 key challenges or problems they face related to this topic.
        * **Needs/Goals:** List 3-5 specific solutions or information they are seeking.

        ## 2. Search Intent
        * **Primary Intent:** Elaborate on the user's likely goal when searching for "{keyword}" (e.g., Informational, Commercial, Navigational, Transactional).
        * **User Goal:** Explain what the user aims to achieve with this search.
        * **Content Role:** Describe how this specific content will fulfill that intent and user goal.

        ## 3. Tone of Voice
        [THIS SECTION IS FOR CLIENT-SPECIFIC TONE OF VOICE. IT WILL BE UPDATED MANUALLY AFTER GENERATION.]
        [Examples: Professional, Authoritative, Empathetic, Conversational, Solution-Oriented, Humorous]

        ## 4. Word Count
        Provide an estimated word count range (e.g., **1,500-2,000 words**).

        ## 5. Keyword Research & Heading Structure
        **Main Keyword:** **{keyword}**

        **Supporting Keywords:**
        * List 10-15 additional relevant keywords, entities, and LSI (Latent Semantic Indexing) terms that should be used naturally throughout the content to enhance topical authority. These are NOT necessarily heading ideas, but terms to weave into paragraphs.
        {related_keywords_str}

        **Potential H2 Headings (based on keywords):**
        * Provide 5-8 logical H2 headings for the content outline, integrating important related keywords and sub-topics naturally.

        ## 6. Content Outline
        ### Introduction
        * Generate a compelling hook or opening statement.
        * Provide a brief overview of what the content will cover.
        * State the main problem or question the content will address.

        ### [LLM GENERATED H2 HEADING IDEA 1 from above section]
        * Generate 3-5 detailed sub-points/topics for this H2.

        ### [LLM GENERATED H2 HEADING IDEA 2 from above section]
        * Generate 3-5 detailed sub-points/topics for this H2.

        ### [LLM GENERATED H2 HEADING IDEA 3 from above section]
        * Generate 3-5 detailed sub-points/topics for this H2.

        ### [LLM GENERATED H2 HEADING IDEA 4 from above section]
        * Generate 3-5 detailed sub-points/topics for this H2.

        ### Conclusion
        * Generate a summary of key takeaways.
        * Generate a call to action (e.g., "Learn more," "Sign up," "Contact us").
        * Generate next steps for the reader.

        ## 7. Call to Action (CTA)
        * **Primary CTA:** Suggest a clear and concise primary call to action.
        * **Secondary CTAs (if any):** Suggest any secondary CTAs.
        * **Placement:** Suggest ideal placement within the content (e.g., "End of article, within relevant sections").

        ## 8. Suggested Title and Meta Description
        * **Recommended Title:** Optimize for SEO and click-through. Must include the main target keyword "{keyword}" and be compelling (under 60 characters).
        * **Meta Description:** Concise, enticing, and keyword-rich summary of the content. Must include the main target keyword "{keyword}" (under 160 characters).

        ---
        """
        st.markdown("#### LLM Prompt Details (Content Brief Generation):")
        with st.expander("Click to view the full prompt sent to Llama 3 for brief generation"):
            st.code(prompt, language='markdown')

        try:
            from litellm import completion
            response = completion(
                model='ollama/llama3',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000 # Increased max_tokens to allow for a more detailed brief
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return "No valid response from LLaMA 3 for content brief creation."
        except Exception as e:
            st.error(f"Error during LLaMA 3 content brief creation: {e}")
            return "Error during LLaMA 3 content brief creation."

# --- Helper Function for Keyword Extraction and Enhanced Potential Scoring ---
# This function is used by DataAnalyzer, so it should be defined before DataAnalyzer is instantiated
def extract_and_filter_keywords(text_list, initial_query_words, min_len_words=2, min_len_chars=5, exclude_words=None, source_type='general'):
    if exclude_words is None:
        exclude_words = set()

    initial_query_words_lower = set([word.lower() for word in initial_query_words])

    extracted_keywords_with_counts = Counter()

    processed_texts = []
    for item in text_list:
        if isinstance(item, str):
            if item and item not in ["No snippet available.", "No detailed snippet available (could not retrieve full content).", "No snippet available for this result."]:
                clean_item = item.strip()

                # Step 1: VERY AGGRESSIVE PRE-FILTERING OF KNOWN NOISE PATTERNS
                # URLs, file paths, hashes, base64, long numeric sequences, specific junk
                clean_item = re.sub(r'https?://[^\s/$.?#].[^\s]*', ' ', clean_item, flags=re.IGNORECASE)
                clean_item = re.sub(r'\b[a-zA-Z0-9_-]+\.(?:png|jpg|jpeg|gif|webp|pdf|doc|docx|xls|xlsx|ppt|pptx|html|js|css|zip|rar|webp|svg)\b', ' ', clean_item, flags=re.IGNORECASE)
                clean_item = re.sub(r'data:[^;]+;base64,[^\s]+', ' ', clean_item, flags=re.IGNORECASE) # Base64 strings
                clean_item = re.sub(r'[0-9a-fA-F]{32,}', ' ', clean_item) # Long hex strings (potential hashes/IDs)
                clean_item = re.sub(r'\b\d{5,}\b', ' ', clean_item) # Long numbers
                clean_item = re.sub(r'\b(?:lgww3x02002002003evg3e|thumbdatimagegifbase64|slyminh|esweo|audithttpsscalabledmediaio)\b', ' ', clean_item, flags=re.IGNORECASE) # Specific identified junk

                # Remove non-alphanumeric characters but keep spaces and hyphens for phrases
                clean_item = re.sub(r'[^\w\s-]', '', clean_item)
                clean_item = re.sub(r'\s+', ' ', clean_item).strip() # Normalize whitespace

                if clean_item:
                    processed_texts.append(clean_item)

    full_text = " ".join(processed_texts).lower()

    # Tokenize into words and phrases
    words = full_text.split()
    # Consider bigrams and trigrams
    phrases = []
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if all(word not in exclude_words for word in phrase.split()):
            phrases.append(phrase)
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        if all(word not in exclude_words for word in phrase.split()):
            phrases.append(phrase)

    all_terms = words + phrases

    # Filter out short words and common stopwords/noise
    filtered_terms = [
        term for term in all_terms
        if len(term.split()) >= min_len_words and len(term) >= min_len_chars and term not in exclude_words
        and not any(ex_word in term for ex_word in exclude_words) # Exclude phrases containing exclude words
    ]

    # Score terms based on frequency and relevance to initial query
    for term, count in Counter(filtered_terms).items(): # Use Counter on filtered_terms
        score = 1
        # Boost for terms containing words from the initial query
        if any(q_word in term for q_word in initial_query_words_lower):
            score += 2
        # Further boost for longer, more specific phrases
        if len(term.split()) >= 3:
            score += 1.5
        elif len(term.split()) == 2:
            score += 1

        extracted_keywords_with_counts[term] += score

    # Apply a logarithmic decay to scores to reduce dominance of very high frequency terms
    # and normalize them to a more manageable range for "potential score"
    scored_keywords = []
    for term, count in extracted_keywords_with_counts.items():
        # Apply a base score + log of count to give diminishing returns for very high counts
        # and multiply by 100 to make scores more visible (e.g., 100-1000 range)
        potential_score = (count + math.log(count + 1)) * 100
        scored_keywords.append((term, potential_score))

    return scored_keywords

# --- Helper Function for Parsing LLM Markdown Output for DOCX ---
# This function extracts specific sections of content from the LLM's Markdown string.
# It assumes the LLM's output includes the section heading (e.g., "## 1. Target Audience")
# and extracts the content *after* that heading.
def extract_section_content_from_llm_markdown(markdown_text, section_heading_in_llm_output):
    # Find the start of the desired section heading in the LLM's Markdown
    start_of_heading_in_llm = markdown_text.find(section_heading_in_llm_output)
    if start_of_heading_in_llm == -1:
        return "" # Section not found in LLM output

    # Content starts immediately after the heading line break
    content_start = markdown_text.find('\n', start_of_heading_in_llm + len(section_heading_in_llm_output))
    if content_start == -1:
        return "" # No content after heading

    # Find the start of the next major section (e.g., "## ") or the next "---"
    # Search from the beginning of the *content* (content_start + 1)
    next_heading_pos = markdown_text.find('\n## ', content_start + 1)
    next_separator_pos = markdown_text.find('\n---', content_start + 1)

    end_index = -1
    if next_heading_pos != -1 and next_separator_pos != -1:
        end_index = min(next_heading_pos, next_separator_pos)
    elif next_heading_pos != -1:
        end_index = next_heading_pos
    elif next_separator_pos != -1:
        end_index = next_separator_pos

    if end_index == -1: # It's the last section in the LLM output
        content = markdown_text[content_start:].strip()
    else:
        content = markdown_text[content_start:end_index].strip()

    return content

# --- Streamlit Application Layout and Logic ---
st.set_page_config(layout="wide", page_title="AI Content Brief Generator")

st.title("Agency Content Brief Generator")

# Initialize agents
keyword_researcher = KeywordResearcher(serpapi_api_key)
serp_analyst = SERPAnalyst(serpapi_api_key)
data_analyzer = DataAnalyzer(serpapi_api_key) # Pass API key to DataAnalyzer
content_brief_creator = ContentBriefCreator()

# User Inputs (Sidebar for cleaner UI)
st.sidebar.header("Brief Details")
query_topic = st.sidebar.text_input("Main Topic / Target Query:", "SaaS SEO agency")
client_name_input = st.sidebar.text_input("Client Name:", "Client X")
project_name_input = st.sidebar.text_input("Project Name:", "SEO Content Project")
page_type_input = st.sidebar.text_input("Page Type (e.g., Blog Post, Landing Page):", "Blog Post")

# Additional Manual Inputs for DOCX placeholders not generated by LLM directly
st.sidebar.subheader("Additional Manual Inputs")
seo_rationale_input = st.sidebar.text_area("SEO Rationale (Manual):", "Based on competitive analysis and target keyword difficulty.")
link_to_spa_input = st.sidebar.text_input("Link to SPA (Manual):", "https://example.com/spa-report")
url_input = st.sidebar.text_input("Page URL (Manual):", "https://youragency.com/blog/your-content-brief-topic")


if st.button("Generate Full Content Brief"):
    if not query_topic:
        st.warning("Please enter a main topic/target query.")
        st.stop()

    with st.spinner("Step 1/3: Researching initial keywords via SerpApi..."):
        initial_keywords = keyword_researcher.get_keywords(query_topic)

    with st.spinner("Step 2/3: Analyzing SERP for initial topic..."):
        serp_data = serp_analyst.analyze_serp(query_topic)

    with st.spinner("Step 3/3: Brainstorming keywords and analyzing data..."):
        df_keywords, selected_brief_keyword = data_analyzer.analyze_data_and_identify_target_keywords_orchestrator(
            query_topic, initial_keywords, serp_data
        )
        if df_keywords.empty:
            st.error("No keywords could be identified for analysis. Please try a different query.")
            st.stop()

        st.subheader("Identified Keywords and Content Types:")
        st.dataframe(df_keywords)

        st.info(f"Selected Primary Keyword for Brief: **{selected_brief_keyword}**")

        # Extract related keywords for the brief generation prompt
        # We'll take the top 15 'Include in Brief' keywords, excluding the main one if present
        related_keywords_for_brief = df_keywords[
            (df_keywords['Include in Brief'] == True) &
            (df_keywords['Keyword'].str.lower() != selected_brief_keyword.lower())
        ]['Keyword'].head(15).tolist()

        # Determine overall content type for the main brief (can be based on the selected_brief_keyword's type)
        main_keyword_row = df_keywords[df_keywords['Keyword'] == selected_brief_keyword]
        inferred_content_type = main_keyword_row['Content Type'].iloc[0] if not main_keyword_row.empty else "Informational"

    with st.spinner("Generating detailed content brief with LLaMA 3..."):
        # Call ContentBriefCreator to get the full structured Markdown brief from LLM
        # Pass relevant data for the LLM to use in its generation.
        full_brief_markdown = content_brief_creator.create_content_brief(
            keyword=selected_brief_keyword,
            related_keywords=related_keywords_for_brief,
            content_type=inferred_content_type,
            # You can pass more structured data here if your LLM prompt can utilize it
            # e.g., audience_data={'demographics': '...', 'pain_points': '...'},
            # serp_insights={'common_themes': '...', 'gaps_to_exploit': '...'}
        )

        if "No valid response from LLaMA 3" in full_brief_markdown or "Error during LLaMA 3" in full_brief_markdown:
            st.error("Failed to generate content brief from LLaMA 3.")
            st.stop()

    st.success("Content brief generated successfully!")
    st.markdown("---")
    st.subheader("Generated Brief Content (Markdown Preview):")
    st.markdown(full_brief_markdown) # Display Markdown preview in Streamlit

    # --- Populate DOCX Template and Provide Download ---
    with st.spinner("Preparing DOCX for download..."):
        try:
            # Ensure your template file is named 'agency_brief_template.docx'
            # and is in the same directory as this script.
            doc = DocxTemplate("agency_brief_template.docx")

            # Prepare the context dictionary for docxtpl
            context = {
                'main_topic_keyword': selected_brief_keyword, # Use the selected primary keyword
                'client_name': client_name_input,
                'current_date': date.today().strftime("%B %d, %Y"),
                'project_name': project_name_input,
                'page_type': page_type_input,
                'seo_rationale_content': seo_rationale_input,
                'link_to_spa_content': link_to_spa_input,
                'url_content': url_input,
            }

            # Extract content for each section from the LLM's Markdown output
            # and map it to the docxtpl context variables.
            context['target_audience_content'] = extract_section_content_from_llm_markdown(full_brief_markdown, "## 1. Target Audience")
            context['search_intent_content'] = extract_section_content_from_llm_markdown(full_brief_markdown, "## 2. Search Intent")
            context['tone_of_voice_content'] = extract_section_content_from_llm_markdown(full_brief_markdown, "## 3. Tone of Voice")

            # For word count, extract just the "X-Y words" part and remove Markdown bolding
            word_count_section_text = extract_section_content_from_llm_markdown(full_brief_markdown, "## 4. Word Count")
            context['target_word_count_range'] = word_count_section_text.replace('**', '')

            context['keyword_research_content'] = extract_section_content_from_llm_markdown(full_brief_markdown, "## 5. Keyword Research & Heading Structure")
            context['content_outline'] = extract_section_content_from_llm_markdown(full_brief_markdown, "## 6. Content Outline")
            context['cta_content'] = extract_section_content_from_llm_markdown(full_brief_markdown, "## 7. Call to Action (CTA)")

            # For Title and Meta Description, parse specific lines from the extracted section
            title_meta_section_text = extract_section_content_from_llm_markdown(full_brief_markdown, "## 8. Suggested Title and Meta Description")
            # Extract Recommended Title using regex
            title_match = re.search(r'\*\*Recommended Title:\*\* (.+)', title_meta_section_text)
            context['recommended_title'] = title_match.group(1).strip() if title_match else ""
            # Extract Meta Description using regex
            meta_desc_match = re.search(r'\*\*Meta Description:\*\* (.+)', title_meta_section_text)
            context['meta_description'] = meta_desc_match.group(1).strip() if meta_desc_match else ""

            # Render the document with the prepared context
            doc.render(context)

            # Define the output filename
            output_filename = f"Content_Brief_{selected_brief_keyword.replace(' ', '_').replace('/', '_')}_{date.today().strftime('%Y%m%d')}.docx"
            doc.save(output_filename) # Save the populated document

            st.success("DOCX brief created and ready for download!")
            # Provide a download button for the generated DOCX
            with open(output_filename, "rb") as file:
                st.download_button(
                    label="Download Content Brief (DOCX)",
                    data=file,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            os.remove(output_filename) # Clean up the temporary DOCX file after download

        except Exception as e:
            st.error(f"An error occurred during DOCX generation: {e}")
            st.info("Please ensure 'agency_brief_template.docx' is in the same directory as your script and has the correct placeholders.")
            st.markdown("---")
            st.subheader("Generated Brief Content (Markdown Preview):")
            st.markdown(full_brief_markdown) # Still show markdown preview even if DOCX fails