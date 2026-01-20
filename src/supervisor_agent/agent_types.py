from __future__ import annotations

"""Shared Pydantic models for the multi-agent demo.

These types define the data contract between workflows and activities
(`AgentStepInput`/`AgentStepOutput`, `ToolCall`, etc.) as well as
the inputs/outputs used by the CLI and workflow APIs.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LlmResponse(BaseModel):
    """Structured JSON response used by Gemini for tool-calling."""

    result: int


class AgentStepInput(BaseModel):
    """Single LLM step input containing provider-formatted messages."""

    messages: List[Dict[str, Any]]


class ToolCall(BaseModel):
    """Represents a model-requested tool invocation."""

    name: str
    arguments: Dict[str, Any]


class AgentStepOutput(BaseModel):
    """Normalized LLM step output used by workflows and activities."""

    # Whether the model finished
    is_final: bool

    # Plain-text final answer
    output_text: Optional[str] = None

    # If tool requested:
    tool_call: Optional[ToolCall] = None

    # Raw model message for history (optional)
    model_message: Dict[str, Any]


class PersonalAssistantInput(BaseModel):
    """Top-level input to the personal assistant workflow.

    Attributes:
        query: Natural language request describing what the assistant
            should do. Typical examples combine scheduling and email
            actions, such as:

            ``"Schedule a team standup for tomorrow at 9am and email the team"``
    """

    query: str


class PersonalAssistantResult(BaseModel):
    """Structured result returned by the personal assistant workflow.

    Attributes:
        final_response: Natural language message summarizing what the
            assistant did, suitable for direct display to an end user.
        tool_calls: The ordered list of tool invocations the workflow
            executed while satisfying the request.
        steps: The number of LLM interaction steps the workflow ran.
    """

    final_response: str
    tool_calls: List[ToolCall] = []
    steps: int


class ChatSessionConfig(BaseModel):
    """Configuration for an interactive chat session."""

    # Reserved for future options (e.g. persona, locale). Included to
    # keep the workflow input as a single Pydantic model, consistent
    # with project conventions.
    system_note: str | None = None


class ChatMessage(BaseModel):
    """User message sent from the CLI to the chat workflow."""

    text: str


class ChatResponse(BaseModel):
    """Assistant response exposed via workflow query."""

    text: str
    turn_index: int
