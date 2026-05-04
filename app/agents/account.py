"""
Account Agent — handles user account and build queries.

Calls account tools (get_recent_builds, get_account_status) to provide
user-specific information.
"""
from google.adk.agents import LlmAgent
from app.agents.tools.account_tools import (
    get_recent_builds,
    get_account_status,
)
from app.settings import settings


ACCOUNT_INSTRUCTION = """
You are the Helix Account Agent — an expert on user accounts and CI/CD builds.

When the user asks about:
- Their builds, pipelines, or build history: call get_recent_builds
- Their account status, usage, limits, or plan: call get_account_status

Return the results in a human-friendly format with clear summaries.
If there are failed builds, highlight them prominently.
"""


# Create the account agent with its tools
account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)

