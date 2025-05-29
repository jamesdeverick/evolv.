import streamlit as st
import os
import re
from collections import Counter
import math
import json
import pandas as pd
from docxtpl import DocxTemplate # Import docxpl
from datetime import date # Import date for current date
import litellm
from litellm import completion, ModelResponse
import requests # For Scrapingdog API calls
from bs4 import BeautifulSoup # For parsing HTML from Scrapingdog

# --- Streamlit UI Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="evolv. agency SEO virtual assistant")

st.title("evolv. agency SEO virtual assistant")
# Add a placeholder logo image
st.image("https://placehold.co/150x50/000000/FFFFFF?text=evolv.%20Logo", caption="evolv. agency Logo", use_container_width=True)
st.markdown("Uses AI and Scrapingdog to quickly generate detailed content briefs for your SEO needs.")

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

# Scrapingdog API key
scrapingdog_api_key = None
try:
    scrapingdog_api_key = st.secrets["scrapingdog_api_key"]
except (AttributeError, KeyError):
    scrapingdog_api_key = os.getenv("SCRAPINGDOG_API_KEY")

if not scrapingdog_api_key:
    st.error("Scrapingdog API key not found. Please set it in .streamlit/secrets.toml or as an environment variable (SCRAPINGDOG_API_KEY).")
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
if 'desired_content_intent' not in st.session_state: # New state variable for desired intent
    st.session_state.desired_content_intent = "Any"

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

# Define a set of common English stop words for filtering
COMMON_STOP_WORDS = set([
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "by", "to", "from", "in", "out",
    "of", "with", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "vs", "versus"
])

# --- Helper Function for Keyword Extraction and Enhanced Potential Scoring ---
# This function will now be called with different exclude_words lists based on source
def extract_and_filter_keywords(text_list, initial_query_words, min_len_words=2, min_len_chars=5, exclude_words=None, source_type='general', serp_insights_context=None):
    if exclude_words is None:
        exclude_words = set()

    # Combine provided exclude_words with common stop words
    all_exclude_words = exclude_words.union(COMMON_STOP_WORDS)

    initial_query_words_lower = set([word.lower() for word in initial_query_words])

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
        if all(word not in all_exclude_words for word in phrase.split()): # Use all_exclude_words
            phrases.append(phrase)
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        if all(word not in all_exclude_words for word in phrase.split()): # Use all_exclude_words
            phrases.append(phrase)
    for i in range(len(words) - 3): # Add 4-word phrases
        phrase = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
        if all(word not in all_exclude_words for word in phrase.split()): # Use all_exclude_words
            phrases.append(phrase)

    all_terms = words + phrases

    filtered_terms = []
    for term in all_terms:
        term_lower = term.lower()
        
        # --- MODIFIED: Stricter filtering for single words using COMMON_STOP_WORDS ---
        if len(term.split()) == 1:
            # Only allow single words if they are part of the initial query (main topic)
            # OR if from LLM and not a common stop word AND meets min_len_chars.
            # Otherwise, skip generic single words.
            if term_lower in initial_query_words_lower:
                filtered_terms.append(term)
            elif source_type == 'llm' and term_lower not in all_exclude_words and len(term) >= min_len_chars:
                 filtered_terms.append(term)
            elif source_type == 'serp' and (term_lower in all_exclude_words or len(term) < 4):
                continue # Very strict for generic SERP words
            else:
                # For other single words, only include if not a stop word and meets min_len_chars
                if term_lower not in all_exclude_words and len(term) >= min_len_chars:
                    filtered_terms.append(term)
        else: # For multi-word phrases, apply standard filtering
            # Ensure no individual word in the phrase is a stop word AND meets min_len_chars
            if len(term) >= min_len_chars and not any(word_part in all_exclude_words for word_part in term_lower.split()):
                filtered_terms.append(term)

    # Re-calculate scores based on word count and frequency
    re_scored_keywords = {}
    for term in filtered_terms:
        term_lower = term.lower()
        word_count = len(term.split())
        
        base_value = 0
        if word_count == 1:
            base_value = 10 # Very low base for single words
        elif word_count == 2:
            base_value = 50
        elif word_count == 3:
            base_value = 150
        elif word_count >= 4:
            base_value = 300

        # Initial boost for relevance to initial query
        if any(q_word in term_lower for q_word in initial_query_words_lower):
            base_value *= 1.5 # Multiplicative boost

        # Apply LLM source boost
        if source_type == 'llm':
            base_value *= 2.0 # Strong multiplicative boost for LLM suggestions

        # Apply SERP insights boost (NEW)
        if serp_insights_context:
            gaps_to_exploit = serp_insights_context.get('gaps_to_exploit', '').lower()
            unique_angles = serp_insights_context.get('unique_angles', '').lower()
            
            if gaps_to_exploit and term_lower in gaps_to_exploit:
                base_value *= 1.8 # High boost for keywords found in identified gaps
            if unique_angles and term_lower in unique_angles:
                base_value *= 1.5 # Moderate boost for keywords found in unique angles

        # Calculate raw frequency of the term in the full text
        raw_frequency = full_text.count(term_lower)
        
        # Combine base value with logarithmic frequency
        # This gives more weight to the base value (phrase length, source) and less to raw frequency
        final_score = base_value * (1 + math.log(raw_frequency + 1)) * 100
        
        # Store the maximum score for a term if it appears multiple times from different sources
        re_scored_keywords[term] = max(re_scored_keywords.get(term, 0), final_score)

    # Convert to list of tuples for sorting
    scored_keywords = list(re_scored_keywords.items())

    # Sort and filter by a minimum relevance score
    scored_keywords.sort(key=lambda x: x[1], reverse=True)

    min_relevance_threshold = 1500 # Increased threshold to filter out more generic terms (increased from 1000)
    return [kw for kw in scored_keywords if kw[1] >= min_relevance_threshold]


class KeywordResearcher:
    def __init__(self, api_key):
        self.api_key = api_key

    @st.cache_data(ttl=3600)
    def get_keywords(_self, query, category=None):
        st.info(f"Making Scrapingdog API call for initial keywords: **'{query}'**")
        
        # Scrapingdog API endpoint for Google Search
        # Note: This is a hypothetical endpoint structure. You might need to adjust
        # based on Scrapingdog's actual documentation.
        scrapingdog_url = f"https://api.scrapingdog.com/google?api_key={_self.api_key}&q={query}&gl=uk&hl=en"

        try:
            response = requests.get(scrapingdog_url)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            html_content = response.text

            soup = BeautifulSoup(html_content, 'html.parser')

            organic_results_data = []
            # Attempt to find organic results (common Google SERP selectors)
            # This is a basic attempt and might need refinement based on actual HTML from Scrapingdog
            for g_result in soup.find_all('div', class_='g'): # Common class for Google organic results
                title_tag = g_result.find('h3')
                link_tag = g_result.find('a')
                snippet_tag = g_result.find('span', class_='st') # Common class for snippet

                title = title_tag.get_text() if title_tag else "N/A"
                link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else "N/A"
                snippet = snippet_tag.get_text() if snippet_tag else "No snippet available."
                
                organic_results_data.append({
                    "url": link,
                    "title": title,
                    "snippet": snippet
                })

            # --- Simplified/Simulated Related Searches and People Also Ask ---
            # Extracting these accurately from raw HTML is complex and highly dependent
            # on the exact HTML structure returned by Scrapingdog.
            # For demonstration, we'll generate some based on the query.
            related_searches = [
                f"{query} alternatives",
                f"best {query} tools",
                f"how to use {query}",
                f"{query} pricing",
                f"what is {query}"
            ]
            
            people_also_ask = [
                f"What are the benefits of {query}?",
                f"How does {query} work?",
                f"Is {query} expensive?", # Corrected f-string syntax here
                f"Who uses {query}?"
            ]

            st.info(f"Successfully retrieved data from Scrapingdog for: **'{query}'")
            return related_searches, people_also_ask, organic_results_data

        except requests.exceptions.RequestException as e:
            st.error(f"Error during Scrapingdog API call: {e}")
            return [], [], []
        except Exception as e:
            st.error(f"Error parsing Scrapingdog response or unexpected issue: {e}")
            return [], [], []

class SERPAnalyst:
    def __init__(self, api_key):
        self.api_key = api_key

    @st.cache_data(ttl=3600)
    def analyze_serp(_self, keyword, category=None):
        st.info(f"Making Scrapingdog API call for detailed SERP analysis of: **'{keyword}'**")
        
        # Scrapingdog API endpoint for Google Search
        scrapingdog_url = f"https://api.scrapingdog.com/google?api_key={_self.api_key}&q={keyword}&gl=uk&hl=en"

        try:
            response = requests.get(scrapingdog_url)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            html_content = response.text

            soup = BeautifulSoup(html_content, 'html.parser')

            serp_results = []
            # Attempt to find organic results (common Google SERP selectors)
            for g_result in soup.find_all('div', class_='g'): # Common class for Google organic results
                title_tag = g_result.find('h3')
                link_tag = g_result.find('a')
                snippet_tag = g_result.find('span', class_='st') # Common class for snippet

                title = title_tag.get_text() if title_tag else "N/A"
                link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else "N/A"
                snippet = snippet_tag.get_text() if snippet_tag else "No snippet available."
                
                serp_results.append({
                    "url": link,
                    "title": title,
                    "snippet": snippet
                })
            st.info(f"Successfully retrieved detailed SERP data for: **'{keyword}'**")
            return serp_results

        except requests.exceptions.RequestException as e:
            st.error(f"Error during Scrapingdog API call for SERP analysis: {e}")
            return []
        except Exception as e:
            st.error(f"Error parsing Scrapingdog response or unexpected issue during SERP analysis: {e}")
            return []


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
        # For now, still using simple heuristic. More advanced intent classification
        # would involve analyzing the SERP results (e.g., presence of e-commerce sites,
        # "how-to" articles, etc.).
        return self._infer_content_type_simple_heuristic(keyword)

class DataAnalyzer:
    def __init__(self, api_key): # Changed serp_api_key to api_key for consistency
        self.api_key = api_key
        self.serp_analyst = SERPAnalyst(api_key) # Instantiate SERPAnalyst here

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
        * **Specific, multi-word phrases (3+ words):** Prioritize phrases that indicate clear user intent and are highly relevant to the main topic.
        * **Long-tail keywords:** These are crucial for targeting niche queries and often have lower competition.
        * **User intent:** Clearly identify keywords suggesting informational, commercial, or navigational intent.
        * **Related sub-topics and entities:** Concepts frequently mentioned in the SERP snippets that expand on the main topic.
        * **Problem/Solution oriented keywords:** How users might phrase their problems related to the topic or seek solutions.
        * **Comparative keywords** (e.g., "X vs Y").
        * **Location-based keywords** (if applicable, e.g., "SaaS SEO agency London").
        * **Strictly avoid generic single words or stop words:** Unless a single word is the explicit main topic or a highly specific, unique entity, do not include it. Focus on meaningful phrases.
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
                                elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], str): # NEW: Handle list containing a single string
                                    llm_suggested_keywords.append(item[0])
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
        # For now, still using simple heuristic. More advanced intent classification
        # would involve analyzing the SERP results (e.g., presence of e-commerce sites,
        # "how-to" articles, etc.).
        return self._infer_content_type_simple_heuristic(keyword)

    def analyze_data_and_identify_target_keywords_orchestrator(self, query_topic, related_searches, organic_results_data, desired_content_intent="Any"):
        # Step 1: Perform detailed SERP analysis for the main query topic (NEW)
        serp_insights = None
        if LITELLM_AVAILABLE:
            with st.spinner(f"Performing detailed SERP analysis for '{query_topic}' to extract insights..."):
                detailed_serp_results = self.serp_analyst.analyze_serp(query_topic)
                if detailed_serp_results:
                    serp_snippet_texts = [r.get("snippet", "") for r in detailed_serp_results if r.get("snippet")]
                    serp_title_texts = [r.get("title", "") for r in detailed_serp_results if r.get("title")]
                    
                    serp_analysis_prompt = f"""Analyze the following top 10 SERP titles and snippets for the keyword "{query_topic}".
                    Identify:
                    1.  **Common Themes:** What are the recurring topics, angles, or sub-topics across these results?
                    2.  **Gaps to Exploit:** What information or angles seem to be missing or under-addressed by competitors?
                    3.  **Unique Angles:** What unique perspectives or approaches could differentiate new content?

                    Provide the output as a JSON object with keys: "common_themes", "gaps_to_exploit", "unique_angles".
                    Each value should be a concise string summary.

                    SERP Titles:
                    {json.dumps(serp_title_texts, indent=2)}

                    SERP Snippets:
                    {json.dumps(serp_snippet_texts, indent=2)}
                    """
                    try:
                        llm_serp_response = completion(
                            model='ollama/llama3',
                            messages=[{"role": "user", "content": serp_analysis_prompt}],
                            temperature=0.5,
                            max_tokens=500,
                            api_base="http://localhost:11434"
                        )
                        raw_serp_llm_output = llm_serp_response.choices[0].message.content
                        serp_insights = json.loads(raw_serp_llm_output)
                        if not isinstance(serp_insights, dict) or not all(k in serp_insights for k in ["common_themes", "gaps_to_exploit", "unique_angles"]):
                            st.warning("LLM returned SERP insights in an unexpected format. Using default insights.")
                            serp_insights = None # Reset if format is wrong
                    except Exception as e:
                        st.error(f"Error generating SERP insights with Llama 3: {e}. Proceeding without detailed SERP insights.")
                        serp_insights = None
                else:
                    st.info("No detailed SERP results to analyze for insights. Proceeding without detailed SERP insights.")
        else:
            st.info("LLM unavailable for SERP insights. Proceeding without detailed SERP insights.")

        # Step 2: LLM brainstorms keywords based on initial Scrapingdog data and now, SERP insights
        with st.spinner("Brainstorming keywords with Llama 3... This may take a moment."):
            llm_brainstormed_keywords = self.brainstorm_keywords_with_llm(query_topic, related_searches, organic_results_data)
            if not llm_brainstormed_keywords:
                st.error("LLM brainstorming failed to produce keywords.")
                return pd.DataFrame(), None

        # Step 3: Extract and score keywords from all sources, now incorporating SERP insights
        initial_query_words = query_topic.split()
        
        # Keywords from LLM (trusted more)
        llm_scored_keywords = extract_and_filter_keywords(llm_brainstormed_keywords, initial_query_words, source_type='llm', serp_insights_context=serp_insights)

        # Keywords from Scrapingdog related searches (simulated for now)
        scrapingdog_related_scored_keywords = extract_and_filter_keywords(related_searches, initial_query_words, source_type='serp', serp_insights_context=serp_insights)

        # Keywords from Scrapingdog organic snippets (titles and snippets)
        scrapingdog_snippets_to_extract = [r.get("title", "") + " " + r.get("snippet", "") for r in organic_results_data]
        scrapingdog_snippets_scored_keywords = extract_and_filter_keywords(scrapingdog_snippets_to_extract, initial_query_words, source_type='serp', serp_insights_context=serp_insights)

        # Combine and deduplicate scored keywords, prioritizing LLM scores
        combined_keywords = {}
        for term, score in scrapingdog_related_scored_keywords:
            combined_keywords[term] = max(combined_keywords.get(term, 0), score) # Take max score if already exists
        for term, score in scrapingdog_snippets_scored_keywords:
            combined_keywords[term] = max(combined_keywords.get(term, 0), score) # Take max score if already exists
        for term, score in llm_scored_keywords: # LLM scores override or significantly boost
            combined_keywords[term] = max(combined_keywords.get(term, 0), score) # LLM source boost is now handled within extract_and_filter_keywords

        # Sort by final score
        sorted_combined_keywords = sorted(combined_keywords.items(), key=lambda x: x[1], reverse=True)

        # Prepare DataFrame
        analyzed_data = []
        for keyword, score in sorted_combined_keywords:
            # Infer content type using the simple heuristic
            content_type = self._infer_content_type_simple_heuristic(keyword)
            
            # Determine if it requires its own content (simple heuristic for now)
            requires_own_content = False
            # Increased score thresholds for requiring own content to be more selective
            if content_type == "Informational" and len(keyword.split()) >= 3 and score > 1500: # Adjusted threshold
                requires_own_content = True
            elif content_type == "Commercial" and len(keyword.split()) >= 2 and score > 1800: # Adjusted threshold
                requires_own_content = True

            analyzed_data.append({
                "Selected": True, # Default to selected
                "Keyword": keyword,
                "Inferred Potential Score": score,
                "Content Type": content_type,
                "Requires Own Content": requires_own_content,
                "Semantically Related Keywords": "" # This will be populated later if needed
            })

        df = pd.DataFrame(analyzed_data)

        # --- MODIFIED: Ensure query_topic is the selected_brief_keyword and apply intent filter ---
        # Add query_topic to the DataFrame if it's not already there, with high priority
        query_topic_lower = query_topic.lower()
        query_topic_inferred_type = self._infer_content_type_simple_heuristic(query_topic)

        if query_topic_lower not in df['Keyword'].str.lower().values:
            new_row = pd.DataFrame([{
                "Selected": True,
                "Keyword": query_topic,
                "Inferred Potential Score": 99999,  # Give it a very high score
                "Content Type": query_topic_inferred_type,
                "Requires Own Content": True,
                "Semantically Related Keywords": ""
            }])
            df = pd.concat([new_row, df]).reset_index(drop=True)
        else:
            # If it exists, ensure it's selected and has a high score
            df.loc[df['Keyword'].str.lower() == query_topic_lower, 'Selected'] = True
            df.loc[df['Keyword'].str.lower() == query_topic_lower, 'Inferred Potential Score'] = 99999
            df.loc[df['Keyword'].str.lower() == query_topic_lower, 'Requires Own Content'] = True
            df.loc[df['Keyword'].str.lower() == query_topic_lower, 'Content Type'] = query_topic_inferred_type # Ensure its intent is updated

        # Apply desired content intent filter to the DataFrame for supporting keywords
        if desired_content_intent != "Any":
            # Filter out keywords that do not match the desired intent, but always keep the main query topic
            filtered_df = df[
                (df["Content Type"] == desired_content_intent) |
                (df["Keyword"].str.lower() == query_topic_lower) # Always keep the main query topic
            ].copy() # Use .copy() to avoid SettingWithCopyWarning
            
            # If the main query topic's inferred intent doesn't match the desired intent, warn the user
            if query_topic_inferred_type != desired_content_intent:
                st.warning(f"Note: Your main topic '{query_topic}' is inferred as '{query_topic_inferred_type}', but you selected to filter for '{desired_content_intent}' keywords. The brief will still use '{query_topic}' as the main keyword, but supporting keywords will align with '{desired_content_intent}'.")
            
            df = filtered_df # Update the DataFrame to the filtered version

        # Sort again to bring the query topic to the top if it wasn't already
        df = df.sort_values(by="Inferred Potential Score", ascending=False).reset_index(drop=True)
        
        # The selected brief keyword is now explicitly the query topic
        selected_brief_keyword = query_topic

        # Store SERP insights in session state to be used later in Step 3
        st.session_state.serp_insights = serp_insights

        return df, selected_brief_keyword


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

        ## 5. Keyword Research
        **Main Keyword:** **{keyword}**

        **Supporting Keywords:**
        * List 10-15 additional relevant keywords, entities, and LSI (Latent Semantic Indexing) terms that should be used naturally throughout the content to enhance topical authority. These are NOT necessarily heading ideas, but terms to weave into paragraphs.
        {related_keywords_str}

        ## 6. Content Outline
        ### Introduction
        * Generate a compelling hook or opening statement.
        * Provide a brief overview of what the content will cover.
        * State the main problem or question the content will address.

        **Main Content Sections (Generate 4-6 H2 headings. Under each H2, generate 2-4 relevant H3 headings, and under each H3, provide 2-3 detailed sub-points/topics):**
        ## H2 Heading Example 1: [Relevant H2 Title incorporating keywords]
        ### H3 Sub-heading 1.1: [Relevant H3 Title]
        * Sub-point 1.1.1
        * Sub-point 1.1.2

        ### H3 Sub-heading 1.2: [Relevant H3 Title]
        * Sub-point 1.2.1
        * Sub-point 1.2.2
        * Sub-point 1.2.3

        ## H2 Heading Example 2: [Relevant H2 Title incorporating keywords]
        ### H3 Sub-heading 2.1: [Relevant H3 Title]
        * Sub-point 2.1.1
        * Sub-point 2.1.2

        ### H3 Sub-heading 2.2: [Relevant H3 Title]
        * Sub-point 2.2.1
        * Sub-point 2.2.2
        * Sub-point 2.2.3

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

        ## 9. Frequently Asked Questions (FAQs)
        * Based on the target keyword and related topics, generate 3-5 relevant questions that users might ask.
        * For each question, provide a concise, accurate answer that directly addresses the question.
        * Format each FAQ as a question followed by its answer.
        Example:
        **Q: What is generative AI?**
        **A:** Generative AI refers to artificial intelligence models capable of producing new, original content like text, images, or audio, rather than just analyzing existing data.

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
                max_tokens=2000, # Increased max_tokens to allow for a more detailed brief
                timeout=120 # Increased timeout to 120 seconds
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                generated_content = response.choices[0].message.content
                if not generated_content.strip(): # Check if content is empty after stripping whitespace
                    st.error("LLaMA 3 returned an empty response for content brief creation.")
                    return "No valid response from LLaMA 3 for content brief creation."
                return generated_content
            else:
                st.warning("LLaMA 3 did not return any choices in the response for content brief creation.")
                return "No valid response from LLaMA 3 for content brief creation."
        except litellm.exceptions.Timeout as e:
            st.error(f"LLM call timed out: {e}. The model took too long to respond. Please try again or simplify the prompt.")
            return "Error during LLaMA 3 content brief creation: Timeout."
        except Exception as e:
            st.error(f"An unexpected error occurred during LLaMA 3 content brief creation: {e}")
            return f"Error during LLaMA 3 content brief creation: {e}"

# --- Streamlit Application Layout and Logic ---

# Initialize agents
keyword_researcher = KeywordResearcher(scrapingdog_api_key)
serp_analyst = SERPAnalyst(scrapingdog_api_key)
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
    st.session_state.desired_content_intent = st.selectbox( # New selectbox for content intent
        "Desired Content Intent for Supporting Keywords:",
        ("Any", "Informational", "Commercial"),
        index=("Any", "Informational", "Commercial").index(st.session_state.desired_content_intent),
        key="step1_desired_content_intent"
    )

    if st.button("Generate Initial Keywords & Proceed to Step 2", key="generate_initial_keywords_button_step1"):
        if not st.session_state.query_topic:
            st.warning("Please enter a main topic/target query.")
        else:
            with st.spinner("Researching initial keywords via Scrapingdog..."): # Updated spinner text
                # Call get_keywords from KeywordResearcher instance
                related_searches, people_also_ask, organic_results_data = keyword_researcher.get_keywords(st.session_state.query_topic)
                
                # Store all relevant data in session state
                st.session_state.initial_keywords_data = related_searches # These are the related searches
                st.session_state.serp_results_data = organic_results_data # These are the organic results snippets
                
                # --- IMPORTANT CHANGE: Instead of heavily filtering here, we'll prepare data for the LLM ---
                # We'll use the LLM to brainstorm the keywords that go into the table.
                # For now, just present a basic list of keywords from Scrapingdog for the LLM to process.
                
                # Combine all raw Scrapingdog keywords for the LLM's input
                all_raw_serp_keywords_for_llm = list(set(related_searches + people_also_ask + [r.get("title") for r in organic_results_data if r.get("title")] + [r.get("snippet") for r in organic_results_data if r.get("snippet")]))
                
                # The actual table for Step 2 will be generated after LLM brainstorming
                # For now, we transition to Step 2, where the LLM brainstorming and table generation will occur.
                
                # We need to call the orchestrator here to get the initial dataframe for Step 2
                # The orchestrator will now handle the LLM brainstorming and initial filtering for the table.
                # Create an instance of DataAnalyzer here, as it's needed for the orchestrator
                data_analyzer_instance = DataAnalyzer(scrapingdog_api_key)
                st.session_state.analyzed_keywords_df, st.session_state.selected_brief_keyword = \
                    data_analyzer_instance.analyze_data_and_identify_target_keywords_orchestrator(
                        st.session_state.query_topic,
                        related_searches, # Pass raw related searches to LLM
                        organic_results_data, # Pass raw organic results data to LLM
                        st.session_state.desired_content_intent # Pass the new desired intent
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
                    help="Keywords that are closely related and should be included for topical authority.",
                ),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="keyword_selection_data_editor"
        )
        st.session_state.analyzed_keywords_df = edited_df
        
        # Explicitly cast 'Selected' column to boolean to prevent ValueError
        st.session_state.analyzed_keywords_df['Selected'] = st.session_state.analyzed_keywords_df['Selected'].astype(bool)


        # Update selected_brief_keyword if the user changes the selection in the table
        # This part remains, but the initial setting in orchestrator ensures query_topic is prioritized.
        selected_rows = st.session_state.analyzed_keywords_df[st.session_state.analyzed_keywords_df["Selected"]]
        if not selected_rows.empty:
            # Re-evaluate the "best" keyword based on current selections and potential score,
            # but the orchestrator has already ensured query_topic is top.
            re_evaluated_target = selected_rows.sort_values(by="Inferred Potential Score", ascending=False).iloc[0]["Keyword"]
            st.session_state.selected_brief_keyword = re_evaluated_target
        else:
            st.warning("Please select at least one keyword in the table above to proceed.")
            st.session_state.selected_brief_keyword = None


    st.markdown("---")
    st.subheader("Selected Target Keyword for Brief Generation:")
    if st.session_state.selected_brief_keyword:
        st.write(f"**{st.session_state.selected_brief_keyword}**")
        # Ensure related keywords exclude the main query topic itself
        st.session_state.related_keywords_for_brief = st.session_state.analyzed_keywords_df[
            (st.session_state.analyzed_keywords_df["Selected"] == True) &
            (st.session_state.analyzed_keywords_df["Keyword"].str.lower() != st.session_state.selected_brief_keyword.lower())
        ]["Keyword"].tolist()
        st.write("Related keywords that will be included in the brief:")
        st.write(", ".join(st.session_state.related_keywords_for_brief[:20]) + "...") # Show a few for brevity
    else:
        st.warning("Please select at least one keyword in the table above to proceed.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Step 1", key="back_to_step1_from_step2"):
            st.session_state.current_step = 1
    with col2:
        if st.button("Generate Content Brief (Step 3)", key="generate_brief_button_step2"):
            if st.session_state.selected_brief_keyword:
                st.session_state.current_step = 3
            else:
                st.warning("Please select a target keyword for the brief.")


# Step 3: Generate and Review Content Brief
if st.session_state.current_step == 3:
    st.subheader("Step 3: Generate and Review Content Brief")

    target_keyword = st.session_state.selected_brief_keyword
    if target_keyword:
        st.info(f"Generating brief for: **{target_keyword}**")
        
        # Get content type for the selected keyword
        content_type_for_brief = st.session_state.analyzed_keywords_df[
            st.session_state.analyzed_keywords_df["Keyword"] == target_keyword
        ]["Content Type"].iloc[0] if not st.session_state.analyzed_keywords_df.empty else "Informational"

        # Use SERP insights stored in session state from DataAnalyzer.orchestrator
        serp_insights_for_brief = st.session_state.get('serp_insights', None)

        with st.spinner("Generating detailed content brief with Llama 3... This might take a while."):
            st.session_state.generated_brief_content = content_brief_creator.create_content_brief(
                keyword=target_keyword,
                related_keywords=st.session_state.related_keywords_for_brief,
                category=st.session_state.page_type, # Using page_type as category for now
                content_type=content_type_for_brief,
                # For now, audience_data is static.
                audience_data={
                    "Demographics": "Professionals, marketers, business owners, or individuals interested in SEO and content strategy.",
                    "Psychographics": "Seeking to improve online visibility, understand SEO best practices, or find solutions for content creation challenges.",
                    "Pain Points": "Low organic traffic, difficulty ranking for target keywords, inconsistent content performance, lack of clear content strategy.",
                    "Needs/Goals": "Actionable strategies for SEO, guidance on content planning, tools and techniques for keyword research, understanding search intent."
                },
                serp_insights=serp_insights_for_brief, # Pass the dynamically generated serp_insights
                word_count_range="1,500-2,500 words" # Example static range
            )
        
        # Extract Page Title and Meta Description from the generated brief content
        if st.session_state.generated_brief_content:
            title_match = re.search(r"\*\*Recommended Title:\*\*\s*(.+)", st.session_state.generated_brief_content)
            meta_match = re.search(r"\*\*Meta Description:\*\*\s*(.+)", st.session_state.generated_brief_content)
            
            if title_match:
                st.session_state.page_title = title_match.group(1).strip()
            if meta_match:
                st.session_state.meta_description = meta_match.group(1).strip()

        st.markdown("---")
        st.subheader("Generated Content Brief:")
        st.markdown(st.session_state.generated_brief_content)

        # Download button for the brief
        if st.session_state.generated_brief_content:
            # Prepare data for docxtpl template
            context = {
                'main_topic_keyword': st.session_state.query_topic,
                'page_type': st.session_state.page_type,
                'client_name': st.session_state.client_name,
                'project_name': st.session_state.project_name,
                'seo_rationale_content': st.session_state.seo_rationale,
                'link_to_spa_content': st.session_state.link_to_spa,
                'url_content': st.session_state.url_input,
                'recommended_title': st.session_state.page_title,
                'meta_description': st.session_state.meta_description,
                'current_date': date.today().strftime("%Y-%m-%d"), # Add current date
                'content_outline': st.session_state.generated_brief_content # Pass the full Markdown content for outline
            }

            # Load the template
            try:
                template = DocxTemplate("agency_brief_template.docx")
                template.render(context)
                brief_filename = f"Content_Brief_{st.session_state.query_topic.replace(' ', '_')}_{date.today().strftime('%Y%m%d')}.docx"
                template.save(brief_filename)

                with open(brief_filename, "rb") as f:
                    st.download_button(
                        label="Download Content Brief as .docx",
                        data=f.read(),
                        file_name=brief_filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            except Exception as e:
                st.error(f"Error generating Word document: {e}. Please ensure 'agency_brief_template.docx' exists and is valid.")
                st.info("You can still copy the Markdown content above.")

    else:
        st.warning("No content brief generated yet. Please go back to Step 2 and select a keyword.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Step 2", key="back_to_step2_from_step3"):
            st.session_state.current_step = 2
    with col2:
        if st.button("Reset All and Start Over", key="reset_all_from_step3"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.session_state.current_step = 1
            st.experimental_rerun() # Rerun to clear all inputs and state
