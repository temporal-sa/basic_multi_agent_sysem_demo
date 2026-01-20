import base64
import inspect
import io
import json
import textwrap
import typing
from typing import Any, Dict

from google import genai
from pydantic import BaseModel
from temporalio import activity

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    _REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    LETTER = None
    canvas = None
    _REPORTLAB_AVAILABLE = False

from src.resources.mytools import (
    TOOL_DISPATCH,
    TOOL_SCHEMAS,
    OPENAI_TOOLS,
    load_tool_modules,
)
from src.resources.custom_types.types import AgentStepInput, AgentStepOutput, ToolCall
from .config import PROVIDER, MODEL

# Dynamically load the app-specific tool module.
# We use the fully-qualified package path to avoid ambiguity.
load_tool_modules("src.company_research_agent.company_research_tools")


def _provider_name(provider) -> str:
    if hasattr(provider, "value"):
        return str(getattr(provider, "value")).lower()
    return str(provider).lower()


_PROVIDER_NAME = _provider_name(PROVIDER)

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


@activity.defn(name="company_research_llm_step_activity")
async def company_research_llm_step_activity(step: AgentStepInput) -> AgentStepOutput:
    contents = step.history

    if _PROVIDER_NAME == "gemini":
        config = genai.types.GenerateContentConfig(
            tools=TOOL_SCHEMAS,
        )

        resp = _llm_client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )

        # Be defensive in case the model returns no candidates or empty parts.
        candidates = getattr(resp, "candidates", None) or []
        if not candidates or not getattr(candidates[0], "content", None):
            # Fallback: treat whole response as plain text.
            txt = str(resp)
            return AgentStepOutput(
                is_final=False,
                output_text=txt,
                model_message={},
            )

        msg = candidates[0].content
        parts = getattr(msg, "parts", None) or []

        if not parts:
            txt = str(msg)
            return AgentStepOutput(
                is_final=False,
                output_text=txt,
                model_message={"role": getattr(msg, "role", None)},
            )

        # Parse first part for a potential function call
        part = parts[0]
        func_call = getattr(part, "function_call", None)

        if func_call:
            return AgentStepOutput(
                is_final=False,
                tool_call=ToolCall(
                    name=func_call.name,
                    arguments=dict(func_call.args),
                ),
                model_message={"role": getattr(msg, "role", None)},
            )

        # Otherwise plain text; decide if this is truly final
        txt = getattr(part, "text", None)
        if txt is None:
            txt = str(part)

        normalized = txt.strip().lower()
        # Treat as final only if the model explicitly marks it as such.
        is_final = normalized.startswith("final answer:") or normalized.startswith("final answer")

        return AgentStepOutput(
            is_final=is_final,
            output_text=txt,
            model_message={"role": getattr(msg, "role", None)},
        )

    if _PROVIDER_NAME == "openai":
        # Expect `contents` to already be OpenAI-style messages from PromptHistory.
        response = _llm_client.chat.completions.create(
            model=MODEL,
            messages=contents,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )
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
        normalized = txt.strip().lower()
        is_final = normalized.startswith("final answer:") or normalized.startswith("final answer")

        return AgentStepOutput(
            is_final=is_final,
            output_text=txt,
            model_message={"role": getattr(message, "role", "assistant")},
        )

    raise NotImplementedError(
        f"company_research_llm_step_activity not implemented for provider={PROVIDER!r}"
    )


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


@activity.defn
async def tool_activity(tool_call: ToolCall) -> str:
    tool_fn = TOOL_DISPATCH[tool_call.name]
    result = _invoke_tool(tool_fn, tool_call.arguments)

    # Convert to text for Gemini consumption
    return str(result)


@activity.defn
async def render_report_pdf(markdown_report: str) -> str:
    """
    Render a Markdown report into a simple, nicely formatted PDF.

    Returns:
        Base64-encoded PDF bytes as a UTF-8 string.
    """
    # If ReportLab is not installed, degrade gracefully by returning an
    # empty string. The calling workflow treats an empty string as "no
    # PDF generated" and will still return the Markdown report.
    if not _REPORTLAB_AVAILABLE:
        activity.logger.warning(
            "ReportLab is not installed; skipping PDF rendering and "
            "returning an empty pdf_base64 string."
        )
        return ""

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER

    # Simple layout: title + body text with wrapping
    margin_x = 72  # 1 inch
    margin_top = height - 72
    line_height = 14

    lines = markdown_report.splitlines()

    y = margin_top

    for raw_line in lines:
        line = raw_line.rstrip()

        # Interpret H1 headings as larger bold text
        if line.startswith("# "):
            text = line[2:].strip()
            c.setFont("Helvetica-Bold", 18)
            c.drawString(margin_x, y, text)
            y -= line_height * 2
            continue
        elif line.startswith("## "):
            text = line[3:].strip()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin_x, y, text)
            y -= line_height * 1.5
            continue
        elif line.startswith("### "):
            text = line[4:].strip()
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, y, text)
            y -= line_height * 1.3
            continue

        # Normal paragraph text; wrap to page width
        if not line.strip():
            y -= line_height
            continue

        c.setFont("Helvetica", 11)
        wrapped = textwrap.wrap(line, width=90)
        for wline in wrapped:
            if y <= 72:
                c.showPage()
                y = margin_top
                c.setFont("Helvetica", 11)
            c.drawString(margin_x, y, wline)
            y -= line_height

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()

    # Return as base64 so it is easy to store / transmit
    return base64.b64encode(pdf_bytes).decode("utf-8")
