from langchain_ollama import OllamaLLM
from langchain_community.tools import TavilySearchResults
from crewai import Agent, Task, Crew
from langchain_core.tools import BaseTool
from pydantic import Field


# Initialize the LLM (Ollama)
llm = OllamaLLM(model="mixtral")

# Initialize Tavily Search
search_tool = TavilySearchResults(
    tavily_api_key="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",
    max_results=5,
    include_answer=True,
    include_raw_content=True,
    include_images=True
)

# Tavily Search Tool
class TavilySearchTool(BaseTool):
    name: str = "TavilySearch"
    description: str = "Fetches real-time search insights for given keywords."
    
    def _run(self, query: str) -> str:
        """Run Tavily search and return results"""
        print(f"Running search for: {query}")
        response = search_tool.invoke(query)
        return response

    async def _arun(self, query: str) -> str:
        """Async search call"""
        raise NotImplementedError("Async not implemented for TavilySearchTool")


# Initialize the Tavily tool
tavily_tool = TavilySearchTool()

# Keyword Research Agent
keyword_agent = Agent(
    role="SEO Keyword Researcher",
    goal="Identify relevant and trending keywords for the specified topic.",
    backstory="An SEO specialist focused on keyword research using real-time search data.",
    tools=[tavily_tool],
    verbose=True
)

# SERP Analysis Agent
serp_agent = Agent(
    role="SERP Analyst",
    goal="Analyze top SERP results for keyword structure, intent, and gaps.",
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
    description="Research trending and high-potential keywords for the topic 'AI tools for SEO'.",
    agent=keyword_agent,
    expected_output="A list of top keywords with annotations."
)

serp_task = Task(
    description="Analyze top search results for 'AI tools for SEO' and identify content structure and opportunities.",
    agent=serp_agent,
    expected_output="SERP analysis summary with gaps and suggestions."
)

brief_task = Task(
    description="Create a detailed content brief using keyword and SERP insights.",
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
