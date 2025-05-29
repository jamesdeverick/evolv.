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
import litellm
from litellm import completion, ModelResponse

# --- Streamlit UI Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="AI SEO Content Brief Generator")

st.title("AI SEO Content Brief Generator")
st.markdown("Use AI to quickly generate detailed content briefs for your SEO needs.")

# --- Check for litellm and Ollama setup ---
LITELLM_AVAILABLE = True
try:
    # Test if llama3 model is available by making a dummy call.
    # Increased timeout for initial connection test
    test_response = completion(
        model="ollama/llama3",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        timeout=10,
        api_base="http://localhost:11434" # Explicitly pass api_base for Ollama
    )
    if test_response and test_response.choices and test_response.choices[0].message.content:
        st.success("Ollama Llama 3 is configured and running!")
    else:
        raise Exception("LiteLLM test call to Ollama Llama 3 failed to return expected content.")
except ImportError:
    LITELLM_AVAILABLE = False
    st.warning("Warning: `litellm` library not found. LLM features will be disabled. Please install it with `pip install litellm`.")
except Exception as e:
    LITELLM_AVAILABLE = False
    st.warning(f"Warning: Ollama Llama 3 not properly configured/running. LLM features will be disabled. Error: {e}")
    st.info("Please ensure Ollama is running (`ollama run llama3`) and accessible on `http://localhost:11434`.")


# --- Configuration & Setup ---

# SerpApi API key
serpapi_api_key = None
try:
    serpapi_api_key = st.secrets["serpapi_api_key"]
except (AttributeError, KeyError):
    serpapi_api_key = os.getenv("SERPAPI_KEY") # Changed to SERPAPI_KEY as per common env var name

if not serpapi_api_key:
    st.error("SerpApi API key not found. Please set it in .streamlit/secrets.toml or as an environment variable (SERPAPI_KEY).")
    st.stop()

# Initialize session state for multi-step process
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'query_topic' not in st.session_state:
    st.session_state.query_topic = ""

# Brief Details - now part of session state, not always in sidebar
if 'client_name' not in st.session_state:
    st.session_state.client_name = "Client X"
if 'project_name' not in st.session_state:
    st.session_state.project_name = "" # Will be auto-populated
if 'page_type' not in st.session_state:
    st.session_state.page_type = "" # Will be auto-populated
if 'seo_rationale' not in st.session_state:
    st.session_state.seo_rationale = "Based on competitive analysis and target keyword difficulty."
if 'link_to_spa' not in st.session_state:
    st.session_state.link_to_spa = "https://example.com/spa-report"
if 'url_input' not in st.session_state:
    st.session_state.url_input = "" # Will be auto-populated

if 'initial_keywords_data' not in st.session_state:
    st.session_state.initial_keywords_data = None
if 'serp_results_data' not in st.session_state:
    st.session_state.serp_results_data = None
if 'analyzed_keywords_df' not in st.session_state:
    st.session_state.analyzed_keywords_df = None
if 'generated_brief_content' not in st.session_state:
    st.session_state.generated_brief_content = None
if 'selected_brief_keyword' not in st.session_state:
    st.session_state.selected_brief_keyword = None
if 'related_keywords_for_brief' not in st.session_state:
    st.session_state.related_keywords_for_brief = None
if 'page_title' not in st.session_state:
    st.session_state.page_title = ""
if 'meta_description' not in st.session_state:
    st.session_state.meta_description = ""

# --- Helper Function for Keyword Extraction and Enhanced Potential Scoring ---
# This function will now be called with different exclude_words lists based on source
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

                # Robust cleaning of URLs, image data, and common artifacts
                clean_item = re.sub(r'https?://[^\s/$.?#].[^\s]*', ' ', clean_item, flags=re.IGNORECASE)
                clean_item = re.sub(r'\b[a-zA-Z0-9_-]+\.(?:png|jpg|jpeg|gif|webp|pdf|doc|docx|xls|xlsx|ppt|pptx|html|js|css|zip|rar|webp|svg)\b', ' ', clean_item, flags=re.IGNORECASE)
                clean_item = re.sub(r'data:[^;]+;base64,[^\s]+', ' ', clean_item, flags=re.IGNORECASE)
                clean_item = re.sub(r'[0-9a-fA-F]{32,}', ' ', clean_item) # Remove long hex strings
                clean_item = re.sub(r'\b\d{5,}\b', ' ', clean_item) # Remove long numbers
                clean_item = re.sub(r'\b(?:lgww3x02002002003evg3e|thumbdatimagegifbase64|slyminh|esweo|audithttpsscalabledmediaio|www|http|https|com|net|org|co|io|ly)\b', ' ', clean_item, flags=re.IGNORECASE)

                # Remove non-alphanumeric except spaces and hyphens, then normalize spaces
                clean_item = re.sub(r'[^\w\s-]', '', clean_item)
                clean_item = re.sub(r'\s+', ' ', clean_item).strip()

                if clean_item:
                    processed_texts.append(clean_item)

    full_text = " ".join(processed_texts).lower()

    words = full_text.split()
    phrases = []
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if all(word not in exclude_words for word in phrase.split()):
            phrases.append(phrase)
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        if all(word not in exclude_words for word in phrase.split()):
            phrases.append(phrase)
    for i in range(len(words) - 3): # Add 4-word phrases
        phrase = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
        if all(word not in exclude_words for word in phrase.split()):
            phrases.append(phrase)

    all_terms = words + phrases

    filtered_terms = []
    for term in all_terms:
        term_lower = term.lower()
        
        # --- MODIFIED: Stricter filtering for single words ---
        if len(term.split()) == 1:
            # Only allow single words if they are part of the initial query or are very specific
            if term_lower in initial_query_words_lower:
                filtered_terms.append(term)
            elif source_type == 'llm' and term_lower not in exclude_words: # Trust LLM more for single words if not explicitly excluded
                 filtered_terms.append(term)
            elif source_type == 'serp' and (term_lower in exclude_words or len(term) < 4): # Very strict for generic SERP words
                continue
            else:
                filtered_terms.append(term) # Allow other single words if they pass length/exclude
        else: # For multi-word phrases, apply standard filtering
            if len(term) >= min_len_chars and not any(ex_word == term_lower or (ex_word in term_lower and len(term.split()) == 1) for ex_word in exclude_words):
                filtered_terms.append(term)


    # Score calculation
    for term in filtered_terms:
        score = 1 # Base score for existence
        term_lower = term.lower()
        word_count = len(term.split())

        # Boost for relevance to initial query
        if any(q_word in term_lower for q_word in initial_query_words_lower):
            score += 2 # Higher relevance boost

        # --- MODIFIED: Increased boost for longer phrases ---
        if word_count >= 4:
            score += 5 # Even higher boost for 4+ word phrases
        elif word_count == 3:
            score += 3 # Increased boost for 3-word phrases
        elif word_count == 2:
            score += 1.5 # Increased boost for 2-word phrases

        # Adjust score based on source type - LLM output is trusted more
        if source_type == 'llm':
            score += 500 # LLM keywords get a strong initial boost (increased from 10)

        # Count occurrences
        extracted_keywords_with_counts[term] += score

    # Apply a logarithmic decay to scores to reduce dominance of very high frequency terms
    # and normalize them to a more manageable range for "potential score"
    scored_keywords = []
    for term, base_score in extracted_keywords_with_counts.items():
        # Final score based on accumulated base_score and a logarithmic factor for total occurrences
        final_score = (base_score + math.log(base_score + 1)) * 100
        scored_keywords.append((term, final_score))

    # Sort and filter by a minimum relevance score
    scored_keywords.sort(key=lambda x: x[1], reverse=True)

    # Only return keywords above a certain relevance threshold to eliminate low-quality terms
    min_relevance_threshold = 100 # Adjust this value if needed
    return [kw for kw in scored_keywords if kw[1] >= min_relevance_threshold]


class KeywordResearcher:
    def __init__(self, api_key):
        self.api_key = api_key

    @st.cache_data(ttl=3600)
    def get_keywords(_self, query, category=None):
        st.info(f"Making SerpApi call for initial keywords: **'{query}'**")
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": _self.api_key,
                "num": 10,
                "gl": "uk", # Prioritize UK results
                "hl": "en"
            }
            if category:
                params["tbm"] = "shop" if category == "e_commerce" else "news" if category == "news_and_media" else None # Example, adjust as needed

            search = GoogleSearch(params)
            results = search.get_dict()

            related_searches = []
            if "related_searches" in results:
                related_searches = [s["query"] for s in results["related_searches"]]

            people_also_ask = []
            if "related_questions" in results:
                people_also_ask = [q["question"] for q in results["related_questions"]]

            organic_results_data = []
            if "organic_results" in results:
                for r in results["organic_results"]:
                    organic_results_data.append({
                        "url": r.get("link", "N/A"),
                        "title": r.get("title", "N/A"),
                        "snippet": r.get("snippet", "No snippet available.")
                    })
            
            # Return raw SerpApi data for the LLM to process
            return related_searches, people_also_ask, organic_results_data
        except Exception as e:
            st.error(f"Error during SerpApi call: {e}")
            return [], [], []

class SERPAnalyst:
    def __init__(self, api_key):
        self.api_key = api_key

    @st.cache_data(ttl=3600)
    def analyze_serp(_self, keyword, category=None):
        # --- MODIFIED: DISABLED SERP API CALL TO SAVE CREDITS ---
        st.info(f"Skipping detailed SerpApi analysis for individual keyword: **'{keyword}'** to save credits.")
        return [] # Return empty list to simulate no SERP data

    def _infer_content_type_simple_heuristic(self, keyword):
        keyword_lower = keyword.lower()
        informational_terms = ["how to", "what is", "guide", "tutorial", "explain", "examples", "definition", "learn", "why", "who", "when", "meaning", "tips", "steps", "best practices"]
        commercial_terms = ["buy", "price", "cost", "best", "top", "review", "vs", "comparison", "alternatives", "deal", "discount", "services", "agency", "software", "tool", "platform", "pricing", "pricing plan", "hire", "consultant", "solution"]
        navigational_terms = ["login", "dashboard", "account", "careers", "contact", "about us", "my account", "sign up", "sign in"]

        if any(term in keyword_lower for term in navigational_terms):
            return "Navigational"
        if any(term in keyword_lower for term in commercial_terms):
            return "Commercial"
        return "Informational"

    def _infer_content_type(self, keyword, serp_results):
        # --- MODIFIED: ALWAYS USE SIMPLE HEURISTIC SINCE SERP RESULTS ARE DISABLED ---
        return self._infer_content_type_simple_heuristic(keyword)

class DataAnalyzer: # Moved DataAnalyzer class definition here
    def __init__(self, serp_api_key):
        self.serp_api_key = serp_api_key

    # Re-added the brainstorm_keywords_with_llm method as provided by the user
    def brainstorm_keywords_with_llm(self, query_topic, initial_keywords, serp_data, category=None):
        if not LITELLM_AVAILABLE:
            st.error("LLM features are disabled because `litellm` is not installed or Ollama/Llama3 is not running.")
            return []

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
            response = completion(
                model='ollama/llama3',
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=700,
                api_base="http://localhost:11434" # Explicitly pass api_base for Ollama
            )

            if isinstance(response, ModelResponse) and hasattr(response, 'choices') and len(response.choices) > 0:
                raw_llm_output = response.choices[0].message.content

                st.markdown("#### Raw Llama 3 Output (Keyword Brainstorming):")
                with st.expander("Click to view the raw (unprocessed) output from Llama 3 for keyword brainstorming"):
                    st.code(raw_llm_output, language='text')

                try:
                    json_start = raw_llm_output.find('[')
                    json_end = raw_llm_output.rfind(']')

                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_string = raw_llm_output[json_start : json_end + 1]
                        
                        # --- MODIFIED LOGIC HERE TO HANDLE LLM'S UNEXPECTED JSON FORMAT ---
                        llm_raw_parsed_output = json.loads(json_string)
                        llm_suggested_keywords = []

                        if isinstance(llm_raw_parsed_output, list):
                            for item in llm_raw_parsed_output:
                                if isinstance(item, dict) and len(item) == 1:
                                    # Extract the single key from the dictionary
                                    keyword_from_llm = list(item.keys())[0]
                                    llm_suggested_keywords.append(keyword_from_llm)
                                elif isinstance(item, str):
                                    # Handle cases where it might correctly return a string directly
                                    llm_suggested_keywords.append(item)
                                else:
                                    st.warning(f"Unexpected item format in LLM output: {item}. Skipping.")
                            return llm_suggested_keywords
                        else:
                            st.warning("LLaMA 3 returned valid JSON, but it was not a list as expected. Please check the prompt instruction.")
                            return []
                        # --- END MODIFIED LOGIC ---

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

    @st.cache_data(ttl=3600)
    def _analyze_single_keyword_serp(_self, keyword, category=None):
        # --- MODIFIED: DISABLED SERP API CALL TO SAVE CREDITS ---
        st.info(f"Skipping detailed SerpApi analysis for individual keyword: **'{keyword}'** to save credits.")
        return [] # Return empty list to simulate no SERP data

    def _infer_content_type_simple_heuristic(self, keyword):
        keyword_lower = keyword.lower()
        informational_terms = ["how to", "what is", "guide", "tutorial", "explain", "examples", "definition", "learn", "why", "who", "when", "meaning", "tips", "steps", "best practices"]
        commercial_terms = ["buy", "price", "cost", "best", "top", "review", "vs", "comparison", "alternatives", "deal", "discount", "services", "agency", "software", "tool", "platform", "pricing", "pricing plan", "hire", "consultant", "solution"]
        navigational_terms = ["login", "dashboard", "account", "careers", "contact", "about us", "my account", "sign up", "sign in"]

        if any(term in keyword_lower for term in navigational_terms):
            return "Navigational"
        if any(term in keyword_lower for term in commercial_terms):
            return "Commercial"
        return "Informational"

    def _infer_content_type(self, keyword, serp_results):
        # --- MODIFIED: ALWAYS USE SIMPLE HEURISTIC SINCE SERP RESULTS ARE DISABLED ---
        return self._infer_content_type_simple_heuristic(keyword)

class DataAnalyzer: # Moved DataAnalyzer class definition here
    def __init__(self, serp_api_key):
        self.serp_api_key = serp_api_key

    # Re-added the brainstorm_keywords_with_llm method as provided by the user
    def brainstorm_keywords_with_llm(self, query_topic, initial_keywords, serp_data, category=None):
        if not LITELLM_AVAILABLE:
            st.error("LLM features are disabled because `litellm` is not installed or Ollama/Llama3 is not running.")
            return []

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
            response = completion(
                model='ollama/llama3',
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=700,
                api_base="http://localhost:11434" # Explicitly pass api_base for Ollama
            )

            if isinstance(response, ModelResponse) and hasattr(response, 'choices') and len(response.choices) > 0:
                raw_llm_output = response.choices[0].message.content

                st.markdown("#### Raw Llama 3 Output (Keyword Brainstorming):")
                with st.expander("Click to view the raw (unprocessed) output from Llama 3 for keyword brainstorming"):
                    st.code(raw_llm_output, language='text')

                try:
                    json_start = raw_llm_output.find('[')
                    json_end = raw_llm_output.rfind(']')

                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_string = raw_llm_output[json_start : json_end + 1]
                        
                        # --- MODIFIED LOGIC HERE TO HANDLE LLM'S UNEXPECTED JSON FORMAT ---
                        llm_raw_parsed_output = json.loads(json_string)
                        llm_suggested_keywords = []

                        if isinstance(llm_raw_parsed_output, list):
                            for item in llm_raw_parsed_output:
                                if isinstance(item, dict) and len(item) == 1:
                                    # Extract the single key from the dictionary
                                    keyword_from_llm = list(item.keys())[0]
                                    llm_suggested_keywords.append(keyword_from_llm)
                                elif isinstance(item, str):
                                    # Handle cases where it might correctly return a string directly
                                    llm_suggested_keywords.append(item)
                                else:
                                    st.warning(f"Unexpected item format in LLM output: {item}. Skipping.")
                            return llm_suggested_keywords
                        else:
                            st.warning("LLaMA 3 returned valid JSON, but it was not a list as expected. Please check the prompt instruction.")
                            return []
                        # --- END MODIFIED LOGIC ---

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

    @st.cache_data(ttl=3600)
    def _analyze_single_keyword_serp(_self, keyword, category=None):
        # --- MODIFIED: DISABLED SERP API CALL TO SAVE CREDITS ---
        st.info(f"Skipping detailed SerpApi analysis for individual keyword: **'{keyword}'** to save credits.")
        return [] # Return empty list to simulate no SERP data

    def _infer_content_type_simple_heuristic(self, keyword):
        keyword_lower = keyword.lower()
        informational_terms = ["how to", "what is", "guide", "tutorial", "explain", "examples", "definition", "learn", "why", "who", "when", "meaning", "tips", "steps", "best practices"]
        commercial_terms = ["buy", "price", "cost", "best", "top", "review", "vs", "comparison", "alternatives", "deal", "discount", "services", "agency", "software", "tool", "platform", "pricing", "pricing plan", "hire", "consultant", "solution"]
        navigational_terms = ["login", "dashboard", "account", "careers", "contact", "about us", "my account", "sign up", "sign in"]

        if any(term in keyword_lower for term in navigational_terms):
            return "Navigational"
        if any(term in keyword_lower for term in commercial_terms):
            return "Commercial"
        return "Informational"

    def _infer_content_type(self, keyword, serp_results):
        # --- MODIFIED: ALWAYS USE SIMPLE HEURISTIC SINCE SERP RESULTS ARE DISABLED ---
        return self._infer_content_type_simple_heuristic(keyword)

    def analyze_data_and_identify_target_keywords_orchestrator(self, query_topic, initial_keywords_from_serpapi, serp_data, selected_category=None):
        st.markdown("### Llama 3's Brainstormed Keyword Ideas (for context):")
        # Call the LLM for brainstorming
        llm_brainstormed_keywords_raw = self.brainstorm_keywords_with_llm(query_topic, initial_keywords_from_serpapi, serp_data, category=selected_category)
        if llm_brainstormed_keywords_raw:
            st.write("---")
            st.write("#### Extracted & Cleaned Llama 3 Keywords (from LLM's direct output):")
            st.write("\n".join([f"- {kw}" for kw in llm_brainstormed_keywords_raw]))
        else:
            st.write("No direct keyword suggestions from Llama 3 for brainstorming.")

        initial_topic_words = set(query_topic.lower().split())
        final_target_keywords_scored = []
        llm_generated_scored_keywords = []

        # 1. Process LLM suggestions with a significant score boost
        if llm_brainstormed_keywords_raw:
            # Use less aggressive exclude words for LLM output, as we trust its intelligence
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
                boosted_score = score + 500000 # MASSIVE BOOST TO PRIORITIZE LLM IDEAS
                final_target_keywords_scored.append((kw, boosted_score))
                llm_generated_scored_keywords.append((kw, boosted_score)) # Keep separate for brief selection


        # 2. Combine initial titles (from SerpApi) and SERP snippets for robust keyword extraction and scoring
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
                "technical", "used", "powered", "right", "jobs", "service", "studio", "center", "365", "servicenow", "copilot", # Added more generic words from user's screenshot
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

        # 3. Deduplicate and re-sort based on combined scores
        unique_keywords_map = {}
        for kw, score in final_target_keywords_scored:
            normalized_kw = kw.lower()
            if normalized_kw not in unique_keywords_map or score > unique_keywords_map[normalized_kw][1]:
                unique_keywords_map[normalized_kw] = (kw, score)

        final_target_keywords_scored_deduped = list(unique_keywords_map.values())
        final_target_keywords_scored_deduped.sort(key=lambda x: x[1], reverse=True)

        # 4. Select the primary keyword for the brief
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
        # TOP_N_FOR_FULL_SERP_ANALYSIS = 7 # Adjust this number based on your SerpApi credit usage tolerance
        # --- MODIFIED: DISABLED SERP API CALLS, SO THIS IS NO LONGER USED FOR API CALLS ---
        TOP_N_FOR_FULL_SERP_ANALYSIS = 0 # Set to 0 to effectively disable direct SERP calls in the loop below

        st.markdown(f"#### Running detailed SERP analysis for the top {TOP_N_FOR_FULL_SERP_ANALYSIS} keywords and the selected main brief keyword...")
        # Add the selected_brief_keyword to a list to ensure it's processed if not already in top N
        keywords_to_analyze_fully = [item[0] for item in final_target_keywords_scored_deduped[:TOP_N_FOR_FULL_SERP_ANALYSIS]]
        if selected_brief_keyword and selected_brief_keyword not in keywords_to_analyze_fully:
            keywords_to_analyze_fully.append(selected_brief_keyword)

        for kw, score in final_target_keywords_scored_deduped:
            requires_own_content = False
            content_type = "Informational" # Default

            # --- MODIFIED: ALWAYS USE SIMPLE HEURISTIC FOR CONTENT TYPE ---
            content_type = self._infer_content_type_simple_heuristic(kw)
            
            # Heuristic for 'Requires Own Content' based on score and type (without full SERP analysis)
            if content_type == "Navigational":
                requires_own_content = True
            elif content_type == "Commercial" and score > 400000: # Adjusted threshold due to massive LLM boost
                requires_own_content = True
            elif content_type == "Informational" and len(kw.split()) <= 2 and score > 450000: # Adjusted threshold
                requires_own_content = True

            keywords_with_flags_and_types.append({
                "Keyword": kw,
                "Inferred Potential Score": f"{score:.1f}",
                "Content Type": content_type,
                "Include in Brief": True, # Default to true, user can uncheck
                "Requires Own Content": requires_own_content,
                "Semantically Related Keywords": "" # Initialize this column as empty
            })

        # Sort the final list by score again
        keywords_with_flags_and_types.sort(key=lambda x: float(x["Inferred Potential Score"]), reverse=True)

        df_keywords = pd.DataFrame(keywords_with_flags_and_types)

        return df_keywords, selected_brief_keyword

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
        Do NOT include any conversational text, explanations, intros, outros, or any other characters before or after the specified sections.

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

# --- Streamlit Application Layout and Logic ---

# Initialize agents
keyword_researcher = KeywordResearcher(serpapi_api_key)
serp_analyst = SERPAnalyst(serpapi_api_key)
data_analyzer = DataAnalyzer(serpapi_api_key) # Pass API key to DataAnalyzer
content_brief_creator = ContentBriefCreator()


# Step 1: Input Keyword Query and Brief Details
if st.session_state.current_step == 1:
    st.subheader("Step 1: Enter your primary topic and brief details")

    # Moved inputs from sidebar to main content area
    st.session_state.query_topic = st.text_input("Main Topic / Target Query:", st.session_state.query_topic, key="step1_query_topic")
    st.session_state.client_name = st.text_input("Client Name:", st.session_state.client_name, key="step1_client_name")
    st.session_state.project_name = st.text_input("Project Name:", st.session_state.project_name, key="step1_project_name")
    st.session_state.page_type = st.text_input("Page Type (e.g., Blog Post, Landing Page):", st.session_state.page_type, key="step1_page_type")
    st.session_state.seo_rationale = st.text_area("SEO Rationale (Manual):", st.session_state.seo_rationale, key="step1_seo_rationale")
    st.session_state.link_to_spa = st.text_input("Link to SPA (Manual):", st.session_state.link_to_spa, key="step1_link_to_spa")
    st.session_state.url_input = st.text_input("Page URL (Manual):", st.session_state.url_input, key="step1_url_input")

    if st.button("Generate Initial Keywords & Proceed to Step 2", key="generate_initial_keywords_button_step1"):
        if not st.session_state.query_topic:
            st.warning("Please enter a main topic/target query.")
        else:
            with st.spinner("Researching initial keywords via SerpApi..."):
                # Call get_keywords from KeywordResearcher instance
                related_searches, people_also_ask, organic_results_data = keyword_researcher.get_keywords(st.session_state.query_topic)
                
                # Store all relevant data in session state
                st.session_state.initial_keywords_data = related_searches # These are the related searches
                st.session_state.serp_results_data = organic_results_data # These are the organic results snippets
                
                # --- IMPORTANT CHANGE: Instead of heavily filtering here, we'll prepare data for the LLM ---
                # We'll use the LLM to brainstorm the keywords that go into the table.
                # For now, just present a basic list of keywords from SerpApi for the LLM to process.
                
                # Combine all raw SerpApi keywords for the LLM's input
                all_raw_serp_keywords_for_llm = list(set(related_searches + people_also_ask + [r.get("title") for r in organic_results_data if r.get("title")] + [r.get("snippet") for r in organic_results_data if r.get("snippet")]))
                
                # The actual table for Step 2 will be generated after LLM brainstorming
                # For now, we transition to Step 2, where the LLM brainstorming and table generation will occur.
                
                # We need to call the orchestrator here to get the initial dataframe for Step 2
                # The orchestrator will now handle the LLM brainstorming and initial filtering for the table.
                st.session_state.analyzed_keywords_df, st.session_state.selected_brief_keyword = \
                    data_analyzer.analyze_data_and_identify_target_keywords_orchestrator(
                        st.session_state.query_topic,
                        related_searches, # Pass raw related searches to LLM
                        organic_results_data # Pass raw organic results data to LLM
                    )
                
                if st.session_state.analyzed_keywords_df is not None and not st.session_state.analyzed_keywords_df.empty:
                    st.session_state.current_step = 2
                else:
                    st.error("Initial keyword analysis and LLM brainstorming failed. Please try a different query or check API key.")
                    st.session_state.analyzed_keywords_df = None


# Step 2: Review and Select Keywords for Detailed Analysis
if st.session_state.current_step == 2:
    st.subheader("Step 2: Review and Select Keywords for Detailed Analysis")
    st.info("Review the keywords below. Uncheck any you don't want to analyze further. You can also add new keywords.")

    if st.session_state.analyzed_keywords_df is not None: # Now using analyzed_keywords_df directly
        edited_df = st.data_editor(
            st.session_state.analyzed_keywords_df,
            column_config={
                "Selected": st.column_config.CheckboxColumn(
                    "Select for Analysis?",
                    help="Select keywords to analyze their SERP in detail",
                    default=True,
                ),
                "Inferred Potential Score": st.column_config.NumberColumn(
                    "Potential Score",
                    help="An inferred score of keyword potential (higher is better)",
                    format="%.1f",
                ),
                "Content Type": st.column_config.TextColumn(
                    "Inferred Content Type",
                    help="Predicted primary search intent (Informational, Commercial, Navigational)",
                ),
                "Requires Own Content": st.column_config.CheckboxColumn(
                    "Requires Own Content?",
                    help="Indicates if this keyword likely needs its own dedicated content piece",
                    default=False,
                ),
                "Semantically Related Keywords": st.column_config.TextColumn(
                    "Semantically Related Keywords",
                    help="Keywords closely related to this term (from SERP analysis)",
                ),
                "Include in Brief": st.column_config.CheckboxColumn( # New column for brief inclusion
                    "Include in Brief?",
                    help="Select keywords to be included in the final content brief's supporting keywords list",
                    default=True,
                ),
            },
            hide_index=True,
            num_rows="dynamic",
            key="keyword_data_editor" # Added unique key
        )
        st.session_state.analyzed_keywords_df = edited_df # Update the session state with edited df

        if st.button("Proceed to Step 3", key="proceed_to_step3_button_step2"): # Changed button text
            # No need to re-run data_analyzer.orchestrator here, as it was already run in Step 1
            # We just proceed with the already analyzed_keywords_df
            st.session_state.current_step = 3
    else:
        st.warning("No keywords to display. Go back to Step 1 to generate them.")

# Step 3: Review Analysis and Generate Brief
if st.session_state.current_step == 3:
    st.subheader("Step 3: Review Analysis and Generate Content Brief")

    if st.session_state.analyzed_keywords_df is not None and not st.session_state.analyzed_keywords_df.empty:
        st.write("### Analyzed Keywords & Competitor Data")
        st.dataframe(st.session_state.analyzed_keywords_df, key="analyzed_keywords_dataframe") # Added unique key

        # Allow user to select primary keyword for brief
        keyword_options = st.session_state.analyzed_keywords_df['Keyword'].tolist()
        # Ensure selected_brief_keyword is one of the options, otherwise default to first
        if st.session_state.selected_brief_keyword not in keyword_options:
            st.session_state.selected_brief_keyword = keyword_options[0] if keyword_options else ""

        st.session_state.selected_brief_keyword = st.selectbox(
            "Select the primary keyword for the content brief:",
            options=keyword_options,
            index=keyword_options.index(st.session_state.selected_brief_keyword) if st.session_state.selected_brief_keyword in keyword_options else 0,
            key="primary_brief_keyword_select" # Added unique key
        )
        
        # Display related keywords for the selected brief keyword
        if st.session_state.selected_brief_keyword:
            selected_row = st.session_state.analyzed_keywords_df[st.session_state.analyzed_keywords_df['Keyword'] == st.session_state.selected_brief_keyword]
            if not selected_row.empty:
                # Get semantically related keywords from the analyzed data
                # Ensure the column exists before trying to access it
                related_kws_str = selected_row['Semantically Related Keywords'].iloc[0] if 'Semantically Related Keywords' in selected_row.columns else ""
                related_kws_list = [kw.strip() for kw in related_kws_str.split(',') if kw.strip()] if related_kws_str else []
                
                # Filter related keywords to only include those marked as 'Include in Brief' in the dataframe
                # This ensures consistency with the analysis step
                keywords_for_multiselect = st.session_state.analyzed_keywords_df[
                    st.session_state.analyzed_keywords_df['Include in Brief'] == True
                ]['Keyword'].tolist()
                
                # Filter related_kws_list to only include keywords that are also in keywords_for_multiselect
                # and exclude the main selected brief keyword itself.
                default_related_selection = [
                    kw for kw in keywords_for_multiselect
                    if kw.lower() != st.session_state.selected_brief_keyword.lower()
                ]
                # Further filter out any single-word generic terms from the default selection
                generic_single_words_to_exclude = {"for", "an", "customer", "what", "best", "to", "of", "in", "and", "used", "get", "powered", "right", "jobs", "service", "studio", "center", "365", "servicenow", "copilot", "the"} # Added "the"
                default_related_selection = [
                    kw for kw in default_related_selection
                    if not (len(kw.split()) == 1 and kw.lower() in generic_single_words_to_exclude)
                ]


                st.session_state.related_keywords_for_brief = st.multiselect(
                    "Select additional keywords to include in the brief (from analyzed relevant terms):",
                    options=keywords_for_multiselect, # Provide all relevant keywords as options
                    default=default_related_selection, # Default to the filtered related keywords
                    key="related_keywords_multiselect" # Added unique key
                )
            
            # Allow user to input custom page title and meta description
            current_title_suggestion = selected_row['Keyword'].iloc[0] if not selected_row.empty else st.session_state.query_topic
            st.session_state.page_title = st.text_input("Proposed Page Title (for Meta Title):", st.session_state.page_title if st.session_state.page_title else f"{current_title_suggestion} - [Client Name]", key="page_title_input") # Added unique key
            st.session_state.meta_description = st.text_area("Proposed Meta Description:", st.session_state.meta_description if st.session_state.meta_description else f"Discover comprehensive insights on {current_title_suggestion} with our expert guide.", key="meta_description_input") # Added unique key

        if st.button("Generate Content Brief & Proceed to Step 4", key="generate_content_brief_button_step3"):
            if LITELLM_AVAILABLE:
                with st.spinner("Generating detailed content brief with LLM..."):
                    # Determine overall content type for the main brief (based on the selected_brief_keyword's type)
                    main_keyword_row = st.session_state.analyzed_keywords_df[st.session_state.analyzed_keywords_df['Keyword'] == st.session_state.selected_brief_keyword]
                    inferred_content_type = main_keyword_row['Content Type'].iloc[0] if not main_keyword_row.empty else "Informational"

                    st.session_state.generated_brief_content = content_brief_creator.create_content_brief(
                        keyword=st.session_state.selected_brief_keyword,
                        related_keywords=st.session_state.related_keywords_for_brief,
                        content_type=inferred_content_type
                    )
                    if "No valid response from LLaMA 3" in st.session_state.generated_brief_content or "Error during LLaMA 3" in st.session_state.generated_brief_content:
                        st.error("Failed to generate content brief from LLaMA 3.")
                        st.session_state.generated_brief_content = None # Clear content on failure
                    else:
                        st.session_state.current_step = 4
            else:
                st.error("LLM is not available. Please check the `litellm` setup.")
    else:
        st.warning("No analyzed keywords data. Go back to Step 2.")

# Navigation Buttons
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.session_state.current_step > 1 and st.button("Back", key="back_button"):
        st.session_state.current_step -= 1
        st.rerun()
with col2:
    if st.session_state.current_step < 4 and st.button("Next", key="next_button"):
        st.session_state.current_step += 1
        st.rerun()
