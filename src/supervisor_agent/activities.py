"""Activities for LLM steps and tool execution in the multi-agent demo.

Workflows build a provider-agnostic prompt history, convert it into an
``AgentStepInput``, and call ``llm_step_activity``. This activity then
talks to the configured provider (OpenAI, Gemini, etc.) and normalizes
the result into an ``AgentStepOutput`` with either assistant text or a
requested ``ToolCall``. Tool calls are executed by ``tool_activity``,
which dispatches to the registered Python functions and returns their
string result back to the workflow.
"""

import inspect
import json
import typing
from typing import Any, Dict

from google import genai
from pydantic import BaseModel
from temporalio import activity

from src.resources.mytools import TOOL_DISPATCH, load_tool_modules
from src.resources.mytools.schemas import build_openai_tools
from .agent_types import AgentStepInput, AgentStepOutput, LlmResponse, ToolCall
from .config import MODEL, PROVIDER

# Register the demo's tool module so that the OpenAI/Gemini clients
# see `schedule_event` and `manage_email` as callable tools.
#
# NOTE: Tools are regular Python functions defined in
# `src/supervisor_agent/tools.py` and decorated with
# `@src.resources.mytools.decorators.tool`. Importing that module here
# ensures those decorators run at worker start-up.
load_tool_modules("src.supervisor_agent.tools")


# Set provider based on config (Supports OpenAI, Gemini; tested on Gemini)
def _provider_name(provider) -> str:
    if hasattr(provider, "value"):
        return str(getattr(provider, "value")).lower()
    return str(provider).lower()


def _is_final_text(text: str) -> bool:
    """Return True if the model text explicitly marks a final answer."""
    normalized = text.strip().lower()
    return normalized.startswith("final summary") or normalized.startswith("final answer")


_PROVIDER_NAME = _provider_name(PROVIDER)

# Launch LLM Client
if _PROVIDER_NAME == "gemini":
    _llm_client = genai.Client()
elif _PROVIDER_NAME == "openai":
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - runtime configuration issue
        raise RuntimeError(
            "OpenAI provider selected but the 'openai' package is not installed."
        ) from exc
    _llm_client = OpenAI()
else:
    raise NotImplementedError(f"Unsupported LLM provider for activities: {PROVIDER!r}")


# This activity sends a message to an LLM and Returns the output. It is tailiored to provide output structure for this specfic X+1 usecase.
@activity.defn
async def llm_step_activity(step: AgentStepInput) -> AgentStepOutput:
    contents = step.messages

    # Gemini Call Handling
    if _PROVIDER_NAME == "gemini":
        # Define output structure
        config = {
            "response_mime_type": "application/json",
            "response_json_schema": LlmResponse.model_json_schema(),
        }

        # Call Gemini model
        resp = _llm_client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )

        # Parse the response
        candidates = getattr(resp, "candidates", None) or []

        # If empty response or no response return whole response as plain text.
        if not candidates or not getattr(candidates[0], "content", None):
            txt = str(resp)
            return AgentStepOutput(
                is_final=False,
                output_text=txt,
                model_message={},
            )

        # Grab the response message and check for multiple parts
        msg = candidates[0].content
        parts = getattr(msg, "parts", None) or []

        # If no parts, return the message
        if not parts:
            txt = str(msg)
            return AgentStepOutput(
                is_final=False,
                output_text=txt,
                model_message={"role": getattr(msg, "role", None)},
            )

        # If parts, parse first part for a potential function call
        part = parts[0]
        func_call = getattr(part, "function_call", None)

        # If a function call is specified, return it
        if func_call:
            return AgentStepOutput(
                is_final=False,
                tool_call=ToolCall(
                    name=func_call.name,
                    arguments=dict(func_call.args),
                ),
                model_message={"role": getattr(msg, "role", None)},
            )

        # Otherwise plain text;
        txt = getattr(part, "text", None)
        if txt is None:
            txt = str(part)

        # Treat as final only if the model explicitly marks it as such.
        is_final = _is_final_text(txt)

        return AgentStepOutput(
            is_final=is_final,
            output_text=txt,
            model_message={"role": getattr(msg, "role", None)},
        )

    # OpenAI Call Handling
    if _PROVIDER_NAME == "openai":
        tools = build_openai_tools()

        # Call the model and get a response
        response = _llm_client.chat.completions.create(
            model=MODEL,
            messages=contents,
            tools=tools,
            tool_choice="auto",
        )

        # Grab the top message choice from the response
        message = response.choices[0].message

        # If the model requested a tool call, surface it.
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            fn = call.function
            try:
                args = json.loads(fn.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            return AgentStepOutput(
                is_final=False,
                tool_call=ToolCall(
                    name=fn.name,
                    arguments=args,
                ),
                model_message={"role": getattr(message, "role", "assistant")},
            )

        # Otherwise treat it as a normal assistant message.
        txt = message.content or ""
        is_final = _is_final_text(txt)

        return AgentStepOutput(
            is_final=is_final,
            output_text=txt,
            model_message={"role": getattr(message, "role", "assistant")},
        )

    raise NotImplementedError(f"llm_step_activity not implemented for provider={PROVIDER!r}")


# Generic helper method for tool invocation -- Not utilized by this agent but here as a starter for extending this example
def _invoke_tool(fn, args: Dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if len(params) == 1:
        param = params[0]
        hints = typing.get_type_hints(fn)  # type: ignore[name-defined]
        annotation = hints.get(param.name)

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            model_cls = annotation
            return fn(model_cls(**args))

    return fn(**args)


# Generic activity method for tool calling -- Not utilized by this agent, but here as a starter for extending this example
@activity.defn
async def tool_activity(tool_call: ToolCall) -> str:
    """Activity that executes a registered tool by name.

    The mapping from tool name → Python function is maintained by the
    shared ``mytools`` package via its global registry. This indirection
    keeps the workflow deterministic: the workflow only sees the logical
    tool name and string result, while any side effects (API calls, DB
    writes, etc.) remain inside this activity.
    """
    tool_fn = TOOL_DISPATCH[tool_call.name]
    result = _invoke_tool(tool_fn, tool_call.arguments)

    # Convert to text for Gemini consumption
    return str(result)
