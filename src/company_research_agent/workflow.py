from typing import List
from datetime import timedelta

from temporalio import workflow

from src.resources.custom_types.types import (
    AgentInput,
    AgentStepInput,
    AgentStepOutput,
    ToolCall,
)
from src.resources.myprompts.history import PromptHistory
from src.resources.myprompts.models import SystemPrompt, TaskPrompt, BasePrompt

from .config import PROVIDER


### DEFINE PROMPTS ###
SYSTEM_PROMPT = """
You are an expert Competitive Analysis Agent.
Given a company name,
validate it using LLM knowledge,
determine its sector,
identify top 3 competitors,
gather real-time strategy data using tools,
analyze their strategies, and
output a beautifully formatted comparison table with actionable insights.
"""

PLANNING_PROMPT_INITIAL_PLAN = """
Step-by-step plan:
1. Validate that the company name exists using LLM knowledge.
2. Determine the sector using LLM knowledge.
3. Identify top 3 competitors using LLM knowledge.
4. Gather data on strategies using web search, page browsing and social media websites.
5. Analyze strategies and generate a comparison table.
6. Propose actionable insights.
Do not repeat steps unless the output becomes inaccurate or inadmissible.
"""

PLANNING_PROMPT_UPDATE_FACTS_PRE = "Reassess facts with new information:"
PLANNING_PROMPT_UPDATE_FACTS_POST = "Updated facts considered."
PLANNING_PROMPT_UPDATE_PLAN_PRE = "Revise the analysis plan based on new data:"
PLANNING_PROMPT_UPDATE_PLAN_POST = "Analysis plan revised."

MANAGED_AGENT_TASK = """
Your task is to analyze the strategies of the top 3 competitors for {task_description} and
produce a comparison table with actionable insights.
"""

@workflow.defn
class AgentLoopWorkflow:
    def __init__(self):
        self.history: PromptHistory = PromptHistory()
        self.tools_used: List[str] = []
        self.step_counter: int = 0
        self.max_steps: int = 30

    @workflow.run
    async def run(self, input: AgentInput) -> dict:
        """
        Main loop:
        - Build initial prompts from the task
        - Call LLM step activity
        - Optionally invoke a tool
        - Repeat until final answer or max_steps
        """

        # Build initial prompt history using the prompt models
        final_answer_instructions = (
            "\n\nWhen you have completed all necessary tool calls and analysis "
            "and are ready to give the final result, respond with a single "
            "message starting with 'FINAL ANSWER:' followed by the final report. "
        )

        non_repetition_instructions = (
            "You must not call a tool again with the same arguments if it has already succeeded, "
            "unless new information makes that result invalid."
        )
        provider = PROVIDER
        system_prompt = SystemPrompt(
            text=(SYSTEM_PROMPT.strip() + non_repetition_instructions + final_answer_instructions)
        )
        task_text = MANAGED_AGENT_TASK.format(task_description=input.task).strip()
        task_prompt = TaskPrompt(text=task_text)

        self.history.add(system_prompt)
        self.history.add(task_prompt)

        # Seed the model with an initial explicit plan.
        initial_plan_prompt = BasePrompt(
            role="user",
            text=PLANNING_PROMPT_INITIAL_PLAN.strip(),
        )
        self.history.add(initial_plan_prompt)

        # Assemble into provider-specific messages for Gemini
        history_messages = self.history.to_messages(provider=provider)

        next_input = AgentStepInput(task=input.task, history=history_messages)

        last_output: AgentStepOutput | None = None

        for step in range(1, self.max_steps + 1):
            self.step_counter = step

            # ----- Step 1: Ask LLM what to do next -----
            llm_result = await workflow.execute_activity(
                "company_research_llm_step_activity",
                next_input,
                schedule_to_close_timeout=timedelta(seconds=90),
            )

            # Temporal + data converter may deserialize to dict; normalize.
            if isinstance(llm_result, dict):
                llm_result = AgentStepOutput(**llm_result)

            last_output = llm_result

            # Record assistant message if present
            if llm_result.output_text:
                assistant_prompt = BasePrompt(role="assistant", text=llm_result.output_text)
                self.history.add(assistant_prompt)
                history_messages = self.history.to_messages(provider=provider)

            # ----- Step 2: Check if workflow is finished -----
            if llm_result.is_final:
                workflow.logger.info("Agent finished.")
                raw_text = llm_result.output_text or ""

                # Strip the FINAL ANSWER marker to get pure Markdown
                markdown = raw_text
                stripped = raw_text.lstrip()
                lower = stripped.lower()
                if lower.startswith("final answer:"):
                    # Find index in original stripped string, then slice after the marker
                    marker_len = len("final answer:")
                    markdown = stripped[marker_len:].lstrip()

                # Render PDF from the Markdown report
                pdf_b64 = await workflow.execute_activity(
                    "render_report_pdf",
                    markdown,
                    schedule_to_close_timeout=timedelta(seconds=60),
                )

                return {
                    "markdown_report": markdown,
                    "pdf_base64": pdf_b64,
                }

            # ----- Step 3: If tool call requested -----
            if llm_result.tool_call is not None:
                tool_req: ToolCall = llm_result.tool_call
                self.tools_used.append(tool_req.name)

                tool_result: str = await workflow.execute_activity(
                    "tool_activity",
                    tool_req,
                    schedule_to_close_timeout=timedelta(seconds=30),
                )

                # Add tool result to history as a tool message
                tool_prompt = BasePrompt(role="tool", text=tool_result)
                self.history.add(tool_prompt)

                # After each tool call, explicitly update facts and plan.
                facts_update_text = (
                    f"{PLANNING_PROMPT_UPDATE_FACTS_PRE}\n\n"
                    f"Latest tool: {tool_req.name}\n"
                    f"Arguments: {tool_req.arguments}\n"
                    f"Result:\n{tool_result}\n\n"
                    f"{PLANNING_PROMPT_UPDATE_FACTS_POST}"
                )
                plan_update_text = (
                    f"{PLANNING_PROMPT_UPDATE_PLAN_PRE}\n\n"
                    f"Tools used so far: {', '.join(self.tools_used)}\n"
                    f"Completed steps: validation if any validate_company call succeeded; "
                    f"sector identification if identify_sector succeeded; "
                    f"competitor identification if identify_competitors succeeded.\n\n"
                    f"{PLANNING_PROMPT_UPDATE_PLAN_POST}"
                )

                self.history.add(BasePrompt(role="user", text=facts_update_text))
                self.history.add(BasePrompt(role="user", text=plan_update_text))

                history_messages = self.history.to_messages(provider=provider)

                # Send updated history back to the LLM
                next_input = AgentStepInput(task=input.task, history=history_messages)
                continue

            # If not final and no tool call, continue with updated history
            next_input = AgentStepInput(task=input.task, history=history_messages)

        # Max steps reached
        workflow.logger.info("Max steps reached without final answer.")
        if last_output and last_output.output_text:
            # Best-effort: return whatever we have as markdown without PDF.
            return {
                "markdown_report": last_output.output_text,
                "pdf_base64": "",
            }
        return {
            "markdown_report": "",
            "pdf_base64": "",
        }
