import os
import pandas as pd
from langchain_exa import ExaSearchRetriever

# Set your Exa API key
os.environ["EXA_API_KEY"] = "0cf89ba2-93d7-4211-b969-a97e1b543c99"

# Initialize the retriever
retriever = ExaSearchRetriever()

# Prompt for the main keyword
main_keyword = input("Enter the main keyword for research: ")

# Fetch search results
print(f"🔍 Searching Exa for: {main_keyword}")
results = retriever.invoke(main_keyword)

# Extracting keywords and subtopics from results
print("📌 Extracting keywords and subtopics...")
keywords = []
for result in results:
    content = result["content"]
    title = result["title"]
    keywords.append({"Title": title, "Content": content})

# Convert to DataFrame for better visualization
df = pd.DataFrame(keywords)

# Save to CSV
df.to_csv(f"{main_keyword}_exa_research.csv", index=False)
print(f"✅ Results saved to {main_keyword}_exa_research.csv")

# Display the DataFrame
import ace_tools as tools; tools.display_dataframe_to_user(name=f"{main_keyword} Keyword Research", dataframe=df)
