import asyncio
import uuid
import base64
from datetime import timedelta
from pathlib import Path

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.resources.custom_types.types import AgentInput
from .workflow import AgentLoopWorkflow
from .config import TASK_QUEUE, ADDRESS


async def main(prompt: str = "Temporal") -> dict:
    client = await Client.connect(
        ADDRESS,
        data_converter=pydantic_data_converter,
    )

    handle = await client.start_workflow(
        AgentLoopWorkflow.run,
        AgentInput(task=prompt),
        id=f"durable-test-id-{uuid.uuid4()}",
        task_queue=TASK_QUEUE,
        run_timeout=timedelta(minutes=10),
    )

    try:
        result = await handle.result()

        markdown = result.get("markdown_report", "")
        pdf_b64 = result.get("pdf_base64", "")

        # Write PDF to disk if available
        pdf_path = None
        if pdf_b64:
            pdf_bytes = base64.b64decode(pdf_b64)
            safe_name = "".join(c for c in prompt if c.isalnum() or c in ("-", "_")) or "report"
            pdf_path = Path(f"{safe_name}_report.pdf")
            pdf_path.write_bytes(pdf_bytes)

        print("\n=== Agent Result (Markdown) ===\n")
        print(markdown)

        if pdf_path:
            print(f"\nPDF written to: {pdf_path}")
        else:
            print("\nNo PDF generated (empty pdf_base64).")

        return result
    except Exception as exc:
        print(f"Workflow finished with exception: {exc}")
        return {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Gemini research agent for a given company."
    )
    parser.add_argument(
        "company",
        nargs="?",
        default="Temporal",
        help="Company name to analyze (default: Temporal)",
    )
    args = parser.parse_args()

    asyncio.run(main(prompt=args.company))
