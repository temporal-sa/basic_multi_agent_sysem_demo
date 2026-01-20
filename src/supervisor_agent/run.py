"""CLI tool for chatting with the multi-agent personal assistant workflow.

This script starts a long-lived chat workflow and then enters an
interactive loop where user messages are sent to the workflow via
signals. After each message, the CLI polls a workflow query to retrieve
the most recent assistant reply and prints it to the terminal.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.service import RPCError

from .agent_types import ChatMessage, ChatSessionConfig
from .config import ADDRESS, TASK_QUEUE
from .workflow import ChatPersonalAssistantWorkflow


async def main() -> None:
    """Start an interactive chat session with the assistant."""
    client = await Client.connect(ADDRESS, data_converter=pydantic_data_converter)

    workflow_id = f"multi-agent-personal-assistant-chat-{uuid4()}"

    handle = await client.start_workflow(
        ChatPersonalAssistantWorkflow.run,
        ChatSessionConfig(),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    print(f"\nStarted chat session with workflow id: {workflow_id}\n")  # noqa: T201
    print("Type 'exit' or 'quit' to end the session.\n")  # noqa: T201

    last_seen_turn = -1

    while True:
        try:
            user_text = input("you> ").strip()  # noqa: T201
        except EOFError:
            user_text = "exit"

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            await handle.signal(ChatPersonalAssistantWorkflow.close)
            print("Closing chat session...")  # noqa: T201
            break

        await handle.signal(
            ChatPersonalAssistantWorkflow.submit_user_message,
            ChatMessage(text=user_text),
        )

        # Wait for a new assistant response for this turn.
        while True:
            try:
                response = await handle.query(ChatPersonalAssistantWorkflow.get_latest_response)
            except RPCError:
                # Query timeouts or expirations are transient in this CLI;
                # back off briefly and retry.
                await asyncio.sleep(0.5)
                continue

            if response is not None and response.turn_index > last_seen_turn:
                text = response.text or ""
                prefix = "FINAL SUMMARY: "
                if text.startswith(prefix):
                    text = text[len(prefix) :].lstrip()

                print(f"assistant> {text}")  # noqa: T201
                last_seen_turn = response.turn_index
                break

            await asyncio.sleep(0.3)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
