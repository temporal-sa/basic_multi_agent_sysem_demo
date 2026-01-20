import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker
from .activities import llm_step_activity, tool_activity
from .config import ADDRESS, TASK_QUEUE
from .workflow import ChatPersonalAssistantWorkflow, PersonalAssistantWorkflow
from src.company_research_agent.workflow import AgentLoopWorkflow

interrupt_event = asyncio.Event()


async def main() -> None:
    """Start a worker that can run the personal assistant workflows."""
    client = await Client.connect(ADDRESS, data_converter=pydantic_data_converter)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[PersonalAssistantWorkflow, ChatPersonalAssistantWorkflow, AgentLoopWorkflow],
        activities=[llm_step_activity, tool_activity],
    ):
        # Keep the worker alive until interrupted (Ctrl+C during demos)
        await interrupt_event.wait()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
