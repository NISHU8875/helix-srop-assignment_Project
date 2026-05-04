"""
Knowledge Agent — handles documentation questions via RAG.

Calls search_docs and formats results as structured tool calls.
Citations: must reference chunk IDs from retrieved documents.
"""
from google.adk.agents import LlmAgent
from app.agents.tools.search_docs import search_docs
from app.settings import settings


KNOWLEDGE_INSTRUCTION = """
You are the Helix Knowledge Agent — an expert on our product documentation.

When answering questions:
1. Call the search_docs tool with the user's query.
2. Read the returned chunks carefully.
3. Formulate an answer citing specific chunk IDs (e.g. "According to [chunk_abc_0001], ...").
4. If no relevant chunks are found, say "I couldn't find information about that in our docs."

Always cite chunk IDs from the search results — this helps users and helps us track retrieval quality.
"""


# Create the knowledge agent with search_docs tool
knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_docs],  # search_docs is the main tool for this agent
)

