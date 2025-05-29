import crewai
from exa_py import Exa
from litellm import completion
import os
import re

# Setup environment
exasearch_api_key = '0cf89ba2-93d7-4211-b969-a97e1b543c99'

# Ensure Ollama is running and 'llama2' is pulled:
# In your terminal:
# 1. ollama pull llama2
# 2. (Ensure ollama service is running in background)

# Initialize ExaSearch
exa = Exa(api_key=exasearch_api_key)

# Keyword Researcher Agent
class KeywordResearcher:
    def __init__(self, exa_instance):
        self.exa = exa_instance

    def get_keywords(self, query):
        try:
            # We are still getting titles here for initial broad keywords
            results = self.exa.search(query)
            keywords = [r.title for r in results.results if r.title]
            return keywords
        except Exception as e:
            print(f"Error in KeywordResearcher.get_keywords: {e}")
            return []

# SERP Analyst Agent (using ExaSearch)
class SERPAnalyst:
    def __init__(self, exa_instance):
        self.exa = exa_instance

    def analyze_serp(self, query):
        try:
            # This fetches titles and snippets, which is crucial for deeper analysis
            results = self.exa.search(query, num_results=10) # Get top 10 results
            serp_data = []
            for r in results.results:
                serp_data.append({
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.text # This is the valuable content snippet
                })
            return serp_data
        except Exception as e:
            print(f"Error in SERPAnalyst.analyze_serp: {e}")
            return []

# Data Analyzer Agent (using LLaMA 2 to analyze keyword & SERP data)
class DataAnalyzer:
    def __init__(self):
        pass

    def analyze_data_and_identify_target_keywords(self, query_topic, keywords, serp_data):
        # Prepare SERP data including titles AND snippets for LLM analysis
        serp_data_str = "\n".join([
            f"  - Title: {d.get('title', 'N/A')}\n    Snippet: {d.get('snippet', 'N/A')}"
            for d in serp_data
        ])

        # Even more aggressive prompt
        prompt = f"""You are an expert SEO data analyst. Your ONLY task is to identify the most promising SEO keywords based on provided initial keywords and comprehensive SERP data (including content snippets).
        The initial broad topic for this analysis is: "{query_topic}".

        **CRITICAL INSTRUCTION: Your output MUST be ONLY a comma-separated list of 3 to 5 high-potential target keywords.
        DO NOT include ANY introductory phrases, conversational text, explanations, or numbering.
        DO NOT put ANY quotation marks around the entire list or individual keywords.
        Just the keywords, separated by commas. No extra text whatsoever.**

        **How to identify High-Potential Keywords (considering simulated search volume/relevance):**
        * **Deeply Analyze SERP Snippets:** Go beyond just titles. Extract core topics, related concepts, specific phrases, and common questions from the 'Snippet' text of top-ranking pages.
        * **Identify Long-Tail Opportunities:** Focus on multi-word phrases or questions that appear frequently in the snippets and seem to represent specific, detailed user queries.
        * **Synthesize and Refine:** Combine insights from initial keywords and snippets to create refined, more targeted, and specific keyword phrases.
        * **Prioritize User Intent:** Select keywords that suggest various user intents (e.g., informational, commercial, navigational) related to the topic.
        * **Relevance and Specificity:** Ensure keywords are highly relevant to the main topic and are specific enough to target a niche audience.

        **Example of desired output format:**
        SaaS SEO agency best practices, B2B software SEO services, MRR growth strategies for SaaS, expert SaaS SEO consultants, technology startup SEO

        ---

        Initial keywords (from initial broad search related to "{query_topic}"):
        {', '.join(keywords) if keywords else 'None'}

        Comprehensive SERP Data (top 10 search results including Titles and Snippets for "{query_topic}"):
        {serp_data_str}

        ---

        Based on this analysis of initial keywords and comprehensive SERP data, provide the 3-5 high-potential target keywords as a strict comma-separated list:
        """
        try:
            response = completion(
                model='ollama/llama2', # Sticking with Llama 2 as per hardware constraints
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.01, # Extremely low temperature for minimal creativity
                max_tokens=250, # Sufficient tokens for long-tail keywords
                stop=["\n\n", "\n", "."] # Stop at period as well to prevent sentence completion
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                raw_output = response.choices[0].message.content
                # Attempt to extract text that looks like a comma-separated list.
                # More robust regex to handle variations and potential leading/trailing junk.
                match = re.search(r'(?:[\s\S]*?)([\w\s,\-:\'\(\)\[\]]{5,}(?:,\s*[\w\s,\-:\'\(\)\[\]]{5,}){0,})', raw_output, re.IGNORECASE | re.DOTALL)
                if match:
                    cleaned_output = match.group(1).strip().replace('"', '') # Take group 1 (the actual list)
                    # Further defensive cleaning: remove leading/trailing non-keyword characters
                    cleaned_output = re.sub(r'^[^\w\s]*', '', cleaned_output)
                    cleaned_output = re.sub(r'[^\w\s]*$', '', cleaned_output)
                    return cleaned_output
                else:
                    return "No recognizable keyword list from LLaMA 2."
            else:
                return "No valid response from LLaMA 2 for data analysis."
        except Exception as e:
            print(f"Error in DataAnalyzer.analyze_data_and_identify_target_keywords: {e}")
            return "Error during LLaMA 2 data analysis."

# Content Brief Creator Agent (using LLaMA 2)
class ContentBriefCreator:
    def __init__(self):
        pass

    def create_content_brief(self, keyword):
        prompt = f"""Generate a comprehensive SEO content brief for the target keyword: "{keyword}".

        The brief should include:
        1. Target Audience
        2. Search Intent (e.g., informational, transactional, navigational)
        3. Key Topics/Headings to Cover (as a bulleted list)
        4. Important Keywords to Include (LSI keywords, related terms - as a comma-separated list)
        5. Competitor Analysis Insights (e.g., common themes, gaps to exploit)
        6. Desired Word Count Range (e.g., 1500-2000 words)
        7. Call to Action ideas (e.g., Subscribe, Buy Now, Learn More)
        8. A suggested Title and Meta Description.
        """
        try:
            response = completion(
                model='ollama/llama2',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return "No valid response from LLaMA 2 for content brief creation."
        except Exception as e:
            print(f"Error in ContentBriefCreator.create_content_brief: {e}")
            return "Error during LLaMA 2 content brief creation."

# --- Helper Function for Keyword Extraction ---
def extract_and_filter_keywords(text_list, min_len_words=2, min_len_chars=5, exclude_words=None):
    if exclude_words is None:
        exclude_words = {"a", "an", "the", "for", "in", "of", "to", "and", "or", "is", "with", "best", "vs", "how", "what", "why", "guide", "companies", "services", "solutions", "systems", "top", "agency", "seo", "saas", "software"}
    
    extracted_keywords = []
    
    for text_item in text_list:
        if not text_item:
            continue
        
        # Clean text: remove punctuation, convert to lowercase, replace common separators with spaces
        processed_text = re.sub(r'[^\w\s-]', '', text_item).lower().replace('|', ' ').replace('-', ' ').replace(':', ' ').strip()
        words = processed_text.split()
        
        # Generate 2-word, 3-word, and 4-word phrases
        for n in range(2, 5): # phrases of length 2, 3, 4
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                # Filter out phrases containing too many excluded words or being too generic
                phrase_words = phrase.split()
                if not any(word in exclude_words for word in phrase_words) and \
                   len(phrase) >= min_len_chars:
                    extracted_keywords.append(phrase)
        
        # Also include relevant single words if they are long enough and not excluded
        for word in words:
            if word not in exclude_words and len(word) >= min_len_chars and len(word.split()) == 1:
                extracted_keywords.append(word)

    # Deduplicate and sort by length (descending) to prioritize more descriptive phrases
    unique_keywords = list(dict.fromkeys(extracted_keywords))
    unique_keywords.sort(key=len, reverse=True)
    return unique_keywords

# Initialize agents
keyword_researcher = KeywordResearcher(exa)
serp_analyst = SERPAnalyst(exa)
data_analyzer = DataAnalyzer()
content_brief_creator = ContentBriefCreator()

# --- Get initial keyword topic from user ---
initial_topic = input("Please enter the initial keyword topic for SEO analysis: ")
print(f"Starting SEO analysis for topic: '{initial_topic}'\n")

# Use the initial_topic as the query for the workflow
query = initial_topic

# Get high-potential keywords (from titles)
print("\n--- Running Keyword Research ---")
keywords = keyword_researcher.get_keywords(query)
print(f"Keywords found: {keywords}")
if not keywords:
    print("No keywords found. Exiting.")
    exit()

# Analyze SERP for the query using ExaSearch (this fetches titles and snippets)
print("\n--- Analyzing SERP ---")
serp_analysis = serp_analyst.analyze_serp(query)
print(f"SERP Analysis: {serp_analysis}")
if not serp_analysis:
    print("No SERP data found. Exiting.")
    exit()

# Analyze data and identify target keywords using LLaMA 2
# LLaMA will now leverage the 'serp_analysis' data (including snippets)
print("\n--- Analyzing Data and Identifying Target Keywords ---")
target_keywords_raw = data_analyzer.analyze_data_and_identify_target_keywords(query, keywords, serp_analysis)
print(f"LLaMA 2 identified target keywords (raw response): {target_keywords_raw}")

# --- Keyword Selection Logic ---
final_target_keywords = []

# Try to parse LLM's output first
try:
    potential_llm_keywords = [kw.strip() for kw in target_keywords_raw.split(',') if kw.strip()]
    # Apply robust filtering to LLM keywords too
    potential_llm_keywords = extract_and_filter_keywords(
        potential_llm_keywords,
        min_len_words=1, # Allow single words if long enough
        min_len_chars=4, # Reduced min char length for LLM output
        exclude_words={"a", "an", "the", "for", "in", "of", "to", "and", "or", "is", "with", "best", "vs", "how", "what", "why", "guide", "companies", "services", "solutions", "systems", "sure", "top", "agency", "seo", "saas", "software"}
    )
    final_target_keywords.extend(potential_llm_keywords)
except Exception as parse_error:
    print(f"Error parsing LLaMA 2 output as direct list: {parse_error}. Relying on robust extraction.")

# If LLM didn't provide enough good keywords, or for initial extraction
# Combine initial titles and SERP snippets for robust keyword extraction
all_raw_text_for_extraction = [query] + keywords + [d.get('snippet', '') for d in serp_analysis]
extracted_from_serp = extract_and_filter_keywords(all_raw_text_for_extraction)
final_target_keywords.extend(extracted_from_serp)

# Deduplicate and pick the top 5 (or fewer if not enough)
final_target_keywords = list(dict.fromkeys(final_target_keywords))
final_target_keywords.sort(key=len, reverse=True) # Prioritize longer, more specific phrases

# Select the primary keyword for the brief
selected_brief_keyword = ""
if final_target_keywords:
    # Try to pick the most descriptive keyword that isn't too short or generic
    for kw in final_target_keywords:
        if len(kw.split()) > 1 and len(kw) > 8: # Prefer multi-word, longer phrases
            selected_brief_keyword = kw
            break
    if not selected_brief_keyword and final_target_keywords: # If no multi-word, take the best single
        selected_brief_keyword = final_target_keywords[0]
else:
    selected_brief_keyword = query # Fallback to original query if all else fails

print(f"Parsed and filtered target keywords: {final_target_keywords[:5]}") # Show top 5
print(f"Selected keyword for content brief: {selected_brief_keyword}")

# Create content brief for a selected keyword using LLaMA 2
print("\n--- Creating Content Brief ---")
content_brief = content_brief_creator.create_content_brief(selected_brief_keyword)
print("Content Brief:", content_brief)