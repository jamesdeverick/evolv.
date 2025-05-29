from langchain_ollama import OllamaLLM
from crewai import Agent, Task, Crew
from langchain_core.tools import BaseTool, Tool
from typing import List
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET


# Initialize the LLM (Ollama)
llm = OllamaLLM(model="mixtral")

# --- Define Custom Tools ---

class GoogleSuggestTool(BaseTool):
    name: str = "GoogleSuggest"
    description: str = "Fetches keyword suggestions from Google Autocomplete."

    def _run(self, query: str) -> List[str]:
        """Run Google suggest and return a list of suggestions."""
        url = f"http://suggestqueries.google.com/complete/search?output=toolbar&hl=en&q={query}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            suggestions = [
                sq.find('suggestion').get('data') for sq in root.findall('.//CompleteSuggestion/suggestion')
            ]
            return suggestions
        except requests.exceptions.RequestException as e:
            return [f"Error fetching Google suggest: {e}"]

    async def _arun(self, query: str) -> List[str]:
        """Async run not supported for GoogleSuggestTool."""
        raise NotImplementedError("Async not implemented for GoogleSuggestTool")

class GooglePAATool(BaseTool):
    name: str = "GooglePeopleAlsoAsk"
    description: str = "Fetches 'People Also Ask' questions from Google search results."

    def _run(self, query: str) -> List[str]:
        """Run Google search and extract 'People Also Ask' questions."""
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            paa_divs = soup.find_all("div", class_="related-question-pair")
            questions = [
                div.find("div", class_="related-question-pair").text
                for div in paa_divs
                if div.find("div", class_="related-question-pair")
            ]
            return questions
        except requests.exceptions.RequestException as e:
            return [f"Error fetching or parsing Google PAA: {e}"]

    async def _arun(self, query: str) -> List[str]:
        """Async run not supported for GooglePAATool."""
        raise NotImplementedError("Async not implemented for GooglePAATool")

class GoogleRelatedSearchesTool(BaseTool):
    name: str = "GoogleRelatedSearches"
    description: str = "Fetches 'Related searches' keywords from Google search results."

    def _run(self, query: str) -> List[str]:
        """Run Google search and extract 'Related searches' keywords."""
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            related_searches_div = soup.find("div", id="related-searches")
            if related_searches_div:
                links = related_searches_div.find_all("a")
                related_keywords = [link.text for link in links]
                return related_keywords
            return []
        except requests.exceptions.RequestException as e:
            return [f"Error fetching or parsing Google Related Searches: {e}"]

    async def _arun(self, query: str) -> List[str]:
        """Async run not supported for GoogleRelatedSearchesTool."""
        raise NotImplementedError("Async not implemented for GoogleRelatedSearchesTool")

# Initialize the tools using Tool.from_function
google_suggest_tool = Tool.from_function(
    func=GoogleSuggestTool().run,
    name="GoogleSuggest",
    description="Fetches keyword suggestions from Google Autocomplete."
)

google_paa_tool = Tool.from_function(
    func=GooglePAATool().run,
    name="GooglePeopleAlsoAsk",
    description="Fetches 'People Also Ask' questions from Google search results."
)

google_related_searches_tool = Tool.from_function(
    func=GoogleRelatedSearchesTool().run,
    name="GoogleRelatedSearches",
    description="Fetches 'Related searches' keywords from Google search results."
)

# Keyword Research Agent (now uses the tools created with Tool.from_function)
keyword_agent = Agent(
    role="SEO Keyword Researcher",
    goal="Identify relevant and trending keywords for the specified topic using Google's free features.",
    backstory="An SEO specialist skilled in leveraging Google's search engine results page for in-depth keyword research.",
    tools=[google_suggest_tool, google_paa_tool, google_related_searches_tool],
    verbose=True
)

# SERP Analysis Agent
serp_agent = Agent(
    role="SERP Analyst",
    goal="Analyze top SERP results for keyword structure, intent, and gaps based on the keywords provided.",
    backstory="An analyst specialized in evaluating search engine results for optimization opportunities.",
    verbose=True
)

# Content Brief Agent
brief_agent = Agent(
    role="Content Brief Creator",
    goal="Create a structured content brief based on keyword research and SERP analysis.",
    backstory="A content strategist who crafts optimized content outlines.",
    verbose=True
)

# Define Tasks
keyword_task = Task(
    description="Research trending and high-potential keywords for the topic 'AI tools for SEO' by exploring Google Autocomplete suggestions, 'People Also Ask' questions, and 'Related searches'. Provide a comprehensive list of keywords and initial insights into their potential.",
    agent=keyword_agent,
    expected_output="A list of relevant keywords, grouped by themes or intent if possible, with initial observations."
)

serp_task = Task(
    description="Analyze the top search results on Google for the keywords identified by the SEO Keyword Researcher. Identify common content structures, user intent signals, and potential content gaps.",
    agent=serp_agent,
    expected_output="SERP analysis summary, highlighting content patterns and opportunities."
)

brief_task = Task(
    description="Create a detailed content brief for creating content about 'AI tools for SEO', using the keyword research and SERP analysis insights.",
    agent=brief_agent,
    expected_output="SEO-focused content brief in markdown format."
)

# Crew Initialization
crew = Crew(
    agents=[keyword_agent, serp_agent, brief_agent],
    tasks=[keyword_task, serp_task, brief_task],
    verbose=True
)

# Execute the workflow
result = crew.kickoff()
print(result)