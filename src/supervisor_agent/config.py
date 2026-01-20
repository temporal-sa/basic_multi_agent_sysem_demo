"""Configuration for the multi-agent personal assistant demo workflow."""

from src.resources.myprompts.provider import LLMProvider

# Temporal connection settings
TASK_QUEUE = "supervisor-task-queue"
ADDRESS = "localhost:7233"

# LLM provider configuration.
#
# The original LangChain example uses OpenAI's ChatGPT. We keep that
# default here so tool-calling works out-of-the-box for users who have
# an OpenAI API key configured. You can switch this to GEMINI or another
# provider supported by the shared ``myprompts`` / ``mytools`` packages
# if desired.
PROVIDER = LLMProvider.OPENAI
MODEL = "gpt-4o"
