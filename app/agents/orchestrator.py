"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via ADK's AgentTool.
This means the LLM decides which tool to call — you do not parse its output.

Intent → sub-agent:
  knowledge:  "how do I X", "what is X", docs questions
  account:    "show my builds", "my account status", usage questions
  smalltalk:  greetings, thanks — root agent handles inline (no tool call)

See docs/google-adk-guide.md for AgentTool pattern and event extraction.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from app.agents.knowledge import knowledge_agent
from app.agents.account import account_agent
from app.settings import settings

ROOT_INSTRUCTION = """
You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent.

Intent → tool:
- HOW to do something, WHAT something is, docs/feature questions → knowledge_agent
- Their account, builds, status, usage → account_agent
- Greetings or off-topic → respond directly, no tool call

Always call a tool when intent matches. Never answer knowledge or account questions yourself.
User context will be in the system message — use it.
"""


def create_orchestrator(context: str = "") -> LlmAgent:
    """
    Create the root orchestrator with optional context injection.
    
    Args:
        context: Additional context to append to instruction (e.g., user state)
    """
    instruction = ROOT_INSTRUCTION
    if context:
        instruction = f"{instruction}\n\n{context}"

    # Wrap sub-agents as tools
    knowledge_tool = AgentTool(agent=knowledge_agent)
    account_tool = AgentTool(agent=account_agent)

    # Create root agent with sub-agents as tools
    root_agent = LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=instruction,
        tools=[knowledge_tool, account_tool],
    )

    return root_agent

