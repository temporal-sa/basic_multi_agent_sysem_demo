"""Temporal multi-agent personal assistant workflow.

This module contains a Temporal Workflow that refactors the
``langchain_version.py`` multi-agent personal assistant example into a
fully Temporalized design:

- All LLM calls live in Activities (see ``activities.py``)
- Tools are defined and registered via the shared ``mytools`` package
  (see ``tools.py``)
- The workflow itself is deterministic and only orchestrates calls to
  Activities while keeping track of the conversation history.

The high-level behavior matches the LangChain example:

- A "supervisor" agent receives a natural language request
- The supervisor decides when to:
  - schedule events (via the ``schedule_event`` tool)
  - send emails (via the ``manage_email`` tool)
  - or do both in sequence
- The workflow keeps a structured trace of the tools used and the number
  of LLM steps taken.
"""

from __future__ import annotations

from datetime import timedelta
from typing import List

from temporalio import workflow

from src.resources.myprompts.history import PromptHistory
from src.resources.myprompts.models import ModelPrompt, SystemPrompt, TaskPrompt, UserPrompt

from src.resources.custom_types.types import AgentInput as CompanyResearchAgentInput
from src.company_research_agent.workflow import AgentLoopWorkflow

from .agent_types import (
    AgentStepInput,
    AgentStepOutput,
    ChatMessage,
    ChatResponse,
    ChatSessionConfig,
    PersonalAssistantInput,
    PersonalAssistantResult,
    ToolCall,
)
from .config import ADDRESS, PROVIDER, TASK_QUEUE


# ---------------------------------------------------------------------------
# Prompt configuration
# ---------------------------------------------------------------------------

# These prompts are adapted from the original LangChain multi-agent
# personal assistant example. In that example, the calendar and email
# agents are separate LangChain agents. In this Temporal version we keep
# the same role descriptions but expose their capabilities as tools that
# a single supervisor agent can call.

# Based on specialization guidance from recent tool-using LLM work
# (e.g., ReAct: Yao et al., 2022; Toolformer: Schick et al., 2023),
# we give each capability a focused role description to reduce
# ambiguity when the supervisor chooses tools.
CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

WEATHER_AGENT_PROMPT = (
    "You are a weather assistant. "
    "Given a location (and optionally units), call get_weather to "
    "retrieve current conditions and present them in a concise, "
    "user-friendly way."
)

# Supervisor prompt structure inspired by ReAct (Yao et al., 2022),
# chain-of-thought prompting (Wei et al., 2022), and bandit-style tool
# selection work (e.g., Shinn et al., 2023 "Reflexion" style feedback).
# It encourages explicit planning, conservative tool use, and a single,
# clearly marked final summary.
SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant coordinating multiple tools. "
    "Your objective is to maximize the user's utility while minimizing "
    "unnecessary or repeated tool calls.\n\n"
    "Available capabilities:\n\n"
    "Calendar agent capabilities:\n"
    f"{CALENDAR_AGENT_PROMPT}\n\n"
    "Email agent capabilities:\n"
    f"{EMAIL_AGENT_PROMPT}\n\n"
    "Weather agent capabilities:\n"
    f"{WEATHER_AGENT_PROMPT}\n\n"
    "You also have access to a long-running company research agent via "
    "the `company_research` tool, which can perform deep competitive "
    "analysis and return a structured report.\n\n"
    "Decision protocol (ReAct-style):\n"
    "1) For each user request, briefly reason about what you know and "
    "   what information is missing.\n"
    "2) Decide whether to CALL_TOOL (choose exactly one tool and "
    "   arguments) or RESPOND (produce the final summary).\n"
    "3) Prefer tools that are likely to be useful; avoid repeating the "
    "   same tool with the same arguments unless new information makes "
    "   earlier results invalid.\n\n"
    "Tool selection hints:\n"
    "- Use schedule_event only for explicit scheduling/changes to "
    "  meetings.\n"
    "- Use manage_email only when composing or sending an email.\n"
    "- Use get_weather only for current or near-term conditions at a "
    "  specific location.\n"
    "- Use company_research only when the user requests an in-depth "
    "  company or competitive analysis.\n\n"
    "Final response contract:\n"
    "- After you have finished using tools for this request, respond "
    "  with a single message that starts that interprets the results of "
    "  your actions taken and engages in the ongoing conversation. \n"
    "- Do not call more tools after you have produced the FINAL SUMMARY."
)


# THIS WORKFLOW IS NOT PART OF THE DEMO. HOWEVER, IT HAS LESS LOGIC IN IT THAN THE MORE ADVANCED
# AND LONG-RUNNING CHATPERSONALASSISTANTWORKFLOW. IVE LEFT IT IN HERE AS A DEMONSTRATION TOOL ONLY.
@workflow.defn
class PersonalAssistantWorkflow:
    """Workflow that orchestrates a multi-agent personal assistant.

    The workflow is responsible for:

    - Building and maintaining a prompt history using ``myprompts``
    - Calling an LLM step activity that can request tools
    - Calling a tool-execution activity when the LLM requests a tool
    - Looping until the LLM returns a final answer or a safety limit is
      reached

    All non-deterministic work (LLM calls, network I/O, etc.) is handled
    inside activities; the workflow state itself is simple and fully
    replayable.
    """

    def __init__(self) -> None:
        # Prompt history is modeled as a pure data structure so that it
        # can be safely replayed by the Temporal Workflow engine.
        self.history = PromptHistory()
        self.tool_calls: List[ToolCall] = []
        self.steps: int = 0

    @workflow.run
    async def run(self, request: PersonalAssistantInput) -> PersonalAssistantResult:
        """Entry point for the personal assistant workflow.

        Args:
            request: Structured input containing the user's natural
                language query.

        Returns:
            A ``PersonalAssistantResult`` that includes the final answer
            plus a trace of tool calls and LLM steps.
        """

        # Seed the conversation with the supervisor's system instructions
        # and the user's top-level request.
        self.history.add(SystemPrompt(text=SUPERVISOR_PROMPT.strip()))
        # Task prompt follows recent prompt-engineering patterns where the
        # high-level task is separated from the tool policy and finalization
        # rules (see Wei et al., 2022; Schick et al., 2023).
        self.history.add(
            TaskPrompt(
                text=(
                    "The user may ask you to schedule meetings, send emails, "
                    "look up information, or perform all of these in one "
                    "request. Use tools conservatively and only when they "
                    "are likely to improve the answer. Prefer re-using "
                    "information you already have over repeating the same "
                    "tool calls. When you have finished using tools and are "
                    "ready to answer the user, respond with a single message "
                    "that synthesizes your findings and engages the user in natural conversation."
                ),
            )
        )
        self.history.add(UserPrompt(text=request.query))

        provider = PROVIDER  # Pull model provider from config
        max_steps = 100  # safety limit so demo workflows always terminate quickly
        last_text: str = ""
        tool_calls_this_request = 0  # used to guardrail against getting stuck in the 'Act' stage
        last_weather_result: str | None = (
            None  # used to guardrail against getting stuck in 'Act' stage for weather tool
        )

        for step_index in range(max_steps):
            self.steps = step_index + 1

            # Convert the prompt history into provider-specific messages
            messages = self.history.to_messages(provider=provider)
            step_input = AgentStepInput(messages=messages)

            # 1. Ask the LLM what to do next (possibly invoking tools).
            raw_result = await workflow.execute_activity(
                "llm_step_activity",
                step_input,
                schedule_to_close_timeout=timedelta(seconds=60),
            )

            if isinstance(raw_result, dict):
                result = AgentStepOutput(**raw_result)
            else:
                result = raw_result

            # 2. If the model requested a tool, execute it via the
            #    generic tool activity and feed the result back into the
            #    conversation. For the special `company_research` tool,
            #    delegate to the long-running company_research_agent
            #    child workflow instead.
            if result.tool_call:
                self.tool_calls.append(result.tool_call)
                tool_calls_this_request += 1

                tool_name = result.tool_call.name

                if tool_name == "company_research":
                    tool_response = await _run_company_research_subagent(result.tool_call)
                elif tool_name == "get_weather":
                    # Simple bandit-style guardrail: avoid repeating the
                    # same expensive weather call once we already have a
                    # good result for this request.
                    if last_weather_result is not None:
                        tool_response = last_weather_result
                    else:
                        tool_response = await workflow.execute_activity(
                            "tool_activity",
                            result.tool_call,
                            schedule_to_close_timeout=timedelta(seconds=30),
                        )
                        if (
                            isinstance(tool_response, str)
                            and "Current weather for" in tool_response
                        ):
                            last_weather_result = tool_response
                else:
                    tool_response = await workflow.execute_activity(
                        "tool_activity",
                        result.tool_call,
                        schedule_to_close_timeout=timedelta(seconds=30),
                    )

                # Record the tool output in the prompt history in a way
                # that is readable by both the LLM and humans.
                self.history.add(
                    UserPrompt(
                        text=f"[Tool {result.tool_call.name} output]: {tool_response}",
                    ),
                )
                workflow.logger.info(
                    "Tool %s executed with response: %s",
                    result.tool_call.name,
                    tool_response,
                )

                # If we have already made several tool calls and have a
                # high-confidence weather result, synthesize a final
                # summary rather than looping indefinitely on tools.
                if tool_calls_this_request >= 3 and last_weather_result is not None:
                    final_message = (
                        "FINAL SUMMARY: "
                        f"{last_weather_result} I used the get_weather tool "
                        "to retrieve these conditions and stopped calling "
                        "tools once I had a reliable result."
                    )
                    workflow.logger.info(
                        "Tool budget reached with weather result; "
                        "returning synthesized final summary after %s steps",
                        self.steps,
                    )
                    return PersonalAssistantResult(
                        final_response=final_message,
                        tool_calls=self.tool_calls,
                        steps=self.steps,
                    )

                # Continue the loop so the LLM can observe the tool result.
                continue

            # 3. Capture any plain-text model output. Only end the loop
            #    once the model explicitly signals it is finished.
            if result.output_text:
                last_text = result.output_text
                self.history.add(ModelPrompt(text=result.output_text))
                if result.is_final:
                    workflow.logger.info("LLM produced final summary after %s steps", self.steps)
                    break

        final_message = last_text or "The assistant could not produce a response."
        return PersonalAssistantResult(
            final_response=final_message,
            tool_calls=self.tool_calls,
            steps=self.steps,
        )


@workflow.defn
class ChatPersonalAssistantWorkflow:
    """Long-lived chat workflow that talks to the assistant via signals.

    This workflow is designed to be driven by an external CLI. The CLI
    sends user messages using signals and reads the latest assistant
    response using a query. All LLM calls and tool invocations still
    happen inside activities, so the workflow remains deterministic and
    replay-safe.
    """

    def __init__(self) -> None:
        # Maintain a rolling conversation history that is shared across
        # all chat turns in the session.
        self.history = PromptHistory()
        self._pending_messages: List[ChatMessage] = []
        self._latest_response: ChatResponse | None = None
        self._turn_index: int = 0
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Signals and queries
    # ------------------------------------------------------------------

    @workflow.signal
    def submit_user_message(self, message: ChatMessage) -> None:
        """Receive a new user message from the CLI."""
        self._pending_messages.append(message)

    @workflow.signal
    def close(self) -> None:
        """Request a graceful shutdown of the chat session."""
        self._closed = True

    @workflow.query
    def get_latest_response(self) -> ChatResponse | None:
        """Return the most recent assistant response, if any."""
        return self._latest_response

    # ------------------------------------------------------------------
    # Main chat loop
    # ------------------------------------------------------------------

    @workflow.run
    async def run(self, config: ChatSessionConfig) -> None:
        """Run a long-lived chat session with the personal assistant."""

        # Seed the conversation with the same supervisor prompt used by
        # the one-shot workflow so behavior stays consistent.
        self.history.add(SystemPrompt(text=SUPERVISOR_PROMPT.strip()))
        # Chat-specific task prompt mirrors the supervisor prompt but
        # scopes decisions to a single turn, following guidance from
        # conversational tool-use studies (e.g., OpenAI function calling,
        # 2023) to separate per-turn reasoning from global context.
        self.history.add(
            TaskPrompt(
                text=(
                    "You are participating in an ongoing chat session. "
                    "For each user message, decide whether to schedule "
                    "events, send emails, look up information, or invoke "
                    "the company research agent using tools. Keep each "
                    "assistant reply concise and user-friendly. When you "
                    "have finished using tools for a given user message "
                    "and are ready to answer, respond with a single "
                    "message that starts with 'FINAL SUMMARY:' followed "
                    "by your final explanation for that turn."
                ),
            )
        )

        if config.system_note:
            self.history.add(SystemPrompt(text=config.system_note))

        provider = PROVIDER
        max_steps_per_turn = 8

        while True:
            # Wait for either a new message or a close request.
            await workflow.wait_condition(
                lambda: bool(self._pending_messages) or self._closed,
            )

            if self._closed and not self._pending_messages:
                workflow.logger.info("Chat session closed by client signal.")
                return

            # Pull the next user message and append it to the prompt
            # history as a user prompt.
            message = self._pending_messages.pop(0)
            self._turn_index += 1
            self.history.add(UserPrompt(text=message.text))

            last_text: str = ""
            tool_calls_this_turn = 0
            last_weather_result: str | None = None

            for _ in range(max_steps_per_turn):
                # Convert history to provider-specific messages.
                messages = self.history.to_messages(provider=provider)
                step_input = AgentStepInput(messages=messages)

                raw_result = await workflow.execute_activity(
                    "llm_step_activity",
                    step_input,
                    schedule_to_close_timeout=timedelta(seconds=60),
                )

                if isinstance(raw_result, dict):
                    result = AgentStepOutput(**raw_result)
                else:
                    result = raw_result

                # Handle tool calls first, mirroring the one-shot workflow.
                if result.tool_call:
                    tool_calls_this_turn += 1
                    tool_name = result.tool_call.name

                    if tool_name == "company_research":
                        tool_response = await _run_company_research_subagent(
                            result.tool_call,
                        )
                    elif tool_name == "get_weather":
                        if last_weather_result is not None:
                            tool_response = last_weather_result
                        else:
                            tool_response = await workflow.execute_activity(
                                "tool_activity",
                                result.tool_call,
                                schedule_to_close_timeout=timedelta(seconds=30),
                            )
                            if (
                                isinstance(tool_response, str)
                                and "Current weather for" in tool_response
                            ):
                                last_weather_result = tool_response
                    else:
                        tool_response = await workflow.execute_activity(
                            "tool_activity",
                            result.tool_call,
                            schedule_to_close_timeout=timedelta(seconds=30),
                        )

                    self.history.add(
                        UserPrompt(
                            text=f"[Tool {result.tool_call.name} output]: {tool_response}",
                        ),
                    )
                    workflow.logger.info(
                        "Chat turn %s tool %s executed with response: %s",
                        self._turn_index,
                        result.tool_call.name,
                        tool_response,
                    )

                    # Per-turn tool budget: if we've already made several
                    # tool calls and have a solid weather result, produce a
                    # synthesized final summary instead of continuing to
                    # call tools.
                    if tool_calls_this_turn >= 3 and last_weather_result is not None:
                        last_text = (
                            "FINAL SUMMARY: "
                            f"{last_weather_result} I used the get_weather "
                            "tool to retrieve these conditions and stopped "
                            "calling tools once I had a reliable result."
                        )
                        break

                    # Continue this turn so the LLM can see the tool output.
                    continue

                # Capture plain-text assistant output. Finalize this turn
                # once the model explicitly signals it is finished.
                if result.output_text:
                    last_text = result.output_text
                    self.history.add(ModelPrompt(text=result.output_text))
                    if result.is_final:
                        break

            final_text = last_text or "The assistant could not produce a response."
            self._latest_response = ChatResponse(
                text=final_text,
                turn_index=self._turn_index,
            )


async def _run_company_research_subagent(tool_call: ToolCall) -> str:
    """Execute the company research sub-agent as a child workflow.

    The supervisor agent calls this helper when the LLM selects the
    `company_research` tool. The helper starts the
    `company_research_agent.AgentLoopWorkflow` child workflow, waits for
    it to finish, and returns a concise textual summary suitable for
    inclusion in the main conversation history.
    """
    args = tool_call.arguments or {}
    company = args.get("company") or args.get("company_name") or args.get("query") or ""

    if not company:
        return "company_research tool was called without a 'company' argument."

    child_input = CompanyResearchAgentInput(task=company)

    # Let Temporal choose a deterministic child workflow ID; we only
    # specify the workflow function and input.
    result: dict = await workflow.execute_child_workflow(
        AgentLoopWorkflow.run, child_input, task_queue="company-research-task-queue"
    )

    markdown = (result.get("markdown_report") or "").strip()
    if not markdown:
        return (
            "The company research agent completed without producing a "
            "report. No additional details are available."
        )

    # Limit the text we feed back into the supervisor history to keep
    # prompts reasonably sized, while still providing rich signal.
    max_chars = 2000
    if len(markdown) > max_chars:
        snippet = markdown[: max_chars - 3].rstrip() + "..."
    else:
        snippet = markdown

    return f"Company research report (markdown excerpt):\n\n{snippet}"


async def main() -> None:  # pragma: no cover
    """Convenience entry point for running the demo workflow once.

    This function assumes a worker is already running for the
    ``agent-task-queue`` task queue (see ``worker.py``). It is provided
    as a simple way to trigger the workflow without going through the
    higher-level CLI or notebook examples.
    """
    import asyncio  # noqa: PLC0415

    from temporalio.client import Client  # noqa: PLC0415
    from temporalio.contrib.pydantic import pydantic_data_converter  # noqa: PLC0415

    client = await Client.connect(ADDRESS, data_converter=pydantic_data_converter)

    input_data = PersonalAssistantInput(
        query="Schedule a team standup for tomorrow at 9am and email the team.",
    )
    result = await client.execute_workflow(
        PersonalAssistantWorkflow.run,
        input_data,
        id="multi-agent-personal-assistant",
        task_queue=TASK_QUEUE,
    )

    print("\nFinal assistant response:\n")  # noqa: T201
    print(result.final_response)  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(main())
