# Multi‑Agent Personal Assistant Demo – Overview

This directory documents the `multi_agent_demo` example so a teammate can:

- Understand every file in `src/supervisor_agent`.
- Run a successful end‑to‑end demo (including restarting the worker mid‑demo).
- See how the looping workflow is generalized across model providers and tools.

The demo shows how to turn a LangChain‑style multi‑agent assistant into a **Temporal‑backed** system:

- All LLM and tool I/O runs in **activities**.
- The assistant logic runs in **deterministic workflows**.
- A **long‑lived chat workflow** exposes a simple query/signal API for a CLI.

---

## Files and Responsibilities

All paths are under `src/supervisor_agent/`.

### `config.py`

- Defines shared configuration:
  - `ADDRESS`: Temporal server address (defaults to `localhost:7233`).
  - `TASK_QUEUE`: Task queue used by workflows and activities (e.g. `"supervisor-task-queue"`).
  - `PROVIDER`: LLM provider (`LLMProvider.OPENAI`, `LLMProvider.GEMINI`, …).
  - `MODEL`: Model identifier (e.g. `"gpt-5"` or a Gemini model name).
- This file is the **single switch** for:
  - Which Temporal server to talk to.
  - Which LLM backend and model the demo uses.

### `agent_types.py`

- Pydantic models that define the **contract between workflows and activities**:
  - `AgentStepInput`: messages/history passed into the LLM step activity.
  - `LlmResponse`: JSON schema describing what the LLM should return
    (tool calls, final text, etc.).
  - `ToolCall`: name + arguments for a single tool invocation.
  - `AgentStepOutput`: normalized activity result containing:
    - `output_text` (assistant text).
    - `tool_call` (if the model requested a tool).
    - `is_final` (whether the model says the turn is done).
  - `ChatMessage`, `ChatResponse`, `ChatSessionConfig`,
    `PersonalAssistantInput`, `PersonalAssistantResult`, etc.
- These types keep the **loop generic**:
  - Workflows don’t care whether the provider is OpenAI or Gemini.
  - Activities always serialize to the same `AgentStepOutput` shape.

### `tools.py`

- Defines **domain‑specific tools** the assistant can call:
  - Calendar tools: e.g., `schedule_event`, `get_available_time_slots`.
  - Email tools: e.g., `manage_email` / `send_email`.
  - Weather tools: e.g., `get_weather`.
- Tools are regular Python functions decorated with `@tool` from
  `src.resources.mytools.decorators`, which:
  - Registers them in a global `TOOL_DISPATCH` registry.
  - Exposes JSON schemas for OpenAI / Gemini tool calling.
- These functions are **pure Python** and side‑effecting:
  - Sending emails, creating calendar events, or fetching weather.
  - Workflows only ever see a **logical tool name + string result**.

### `activities.py`

- Hosts the **LLM step** and **generic tool execution** activities.

Key pieces:

- Provider adaptation:
  - Chooses client implementation based on `PROVIDER` (`OpenAI`, `genai.Client`, …).
  - Calls provider‑specific APIs but always returns an `AgentStepOutput`.
- `llm_step_activity(step: AgentStepInput) -> AgentStepOutput`:
  - For **Gemini**:
    - Uses `response_json_schema=LlmResponse.model_json_schema()` so the model
      returns JSON conforming to `LlmResponse`.
    - Interprets:
      - `function_call` parts as `ToolCall`.
      - Text parts as assistant output.
    - Marks `is_final` when the text starts with `FINAL SUMMARY:` or `FINAL ANSWER:`.
  - For **OpenAI**:
    - Uses `tools=build_openai_tools()` and `tool_choice="auto"`.
    - Maps `tool_calls` into `ToolCall` objects.
    - Checks the assistant content for `FINAL SUMMARY:` / `FINAL ANSWER:` to set `is_final`.
- `tool_activity(tool_call: ToolCall) -> str`:
  - Looks up the Python function in `TOOL_DISPATCH` by name.
  - Instantiates any Pydantic argument models if needed.
  - Returns the tool result as text.

Together, `agent_types.py` + `activities.py` form a **generalized loop**:

- Any provider that can either emit JSON or call tools can be slotted in.
- Any tool registered via `@tool` can be used by the agents.

### `workflow.py`

Defines two workflows:

1. `PersonalAssistantWorkflow` (one‑shot request → answer).
2. `ChatPersonalAssistantWorkflow` (long‑lived chat session).

#### `PersonalAssistantWorkflow`

- Deterministic loop:
  - Builds an initial `PromptHistory` with:
    - `SUPERVISOR_PROMPT`: system‑level instructions for the multi‑agent supervisor.
    - A task prompt describing what the user might ask.
    - The initial user query (`PersonalAssistantInput.query`).
  - For up to `max_steps`:
    1. Converts history + provider into `AgentStepInput`.
    2. Calls `llm_step_activity` to ask “what next?”.
    3. If `tool_call` is present:
       - Executes `tool_activity` (or a special weather branch).
       - Appends `[Tool <name> output]: ...` to the prompt history.
       - Optionally synthesizes a **weather FINAL SUMMARY** after
         several calls to avoid infinite loops.
       - Continues the loop.
    4. Otherwise, if `output_text` is present:
       - Appends a `ModelPrompt` to history.
       - If `is_final` is `True`, breaks.
  - Returns `PersonalAssistantResult` with:
    - `final_response`: last assistant text (often beginning with `FINAL SUMMARY:`).
    - `tool_calls` and total `steps`.

#### `ChatPersonalAssistantWorkflow`

- Long‑lived workflow that supports interactive chat:
  - Signals:
    - `submit_user_message(ChatMessage)`: enqueue a user message.
    - `close()`: request a graceful shutdown.
  - Query:
    - `get_latest_response() -> ChatResponse | None`: exposes the latest assistant turn.
- Main loop:
  - Starts with the same `SUPERVISOR_PROMPT` and a chat‑specific task prompt.
  - For each turn:
    - Waits until there is a pending user message or `_closed` is set.
    - Appends the user message to `PromptHistory`.
    - Runs a bounded agent loop (similar to `PersonalAssistantWorkflow`) to produce:
      - One final assistant message for this **turn** (possibly using tools).
    - Stores it as `_latest_response = ChatResponse(text=..., turn_index=...)`.
- Important behaviors:
  - Both workflows **never** call tools directly; they only orchestrate activities.
  - The chat workflow is **looping and generalized**:
    - Unaware of provider internals (Gemini vs OpenAI).
    - Unaware of specific tools (calendar, email, weather, company research).
    - It just reacts to `AgentStepOutput`.

### `run.py` (CLI)

- Small terminal UI for `ChatPersonalAssistantWorkflow`:
  - Connects to Temporal using `ADDRESS` and Pydantic data converter.
  - Starts `ChatPersonalAssistantWorkflow.run(ChatSessionConfig())` on `TASK_QUEUE`.
  - Enters a REPL:
    - Reads `you>` input.
    - Sends it via `submit_user_message` signal.
    - Polls `get_latest_response` until a new `turn_index` appears.
    - Prints `assistant> ...`.
- Quality‑of‑life logic:
  - Strips a leading `FINAL SUMMARY:` prefix from responses so the user
    sees only the natural language summary.
  - Catches `RPCError` around the query call and retries on transient
    “query not found / expired” conditions.
  - Treats `exit` / `quit` as a signal to close the workflow and end the session.

### `worker.py`

- Launches a Temporal worker that can run:
  - Workflows:
    - `PersonalAssistantWorkflow`
    - `ChatPersonalAssistantWorkflow`
    - `AgentLoopWorkflow` (from the `company_research_agent` package)
  - Activities:
    - `llm_step_activity`
    - `tool_activity`
- Connects to the same `ADDRESS` and `TASK_QUEUE` as `run.py`.
- Uses an `asyncio.Event` to stay alive until interrupted (Ctrl+C).

### `langchain_version.py`

- Reference implementation of the original LangChain multi‑agent example:
  - Shows how the same **business logic** looked before Temporalization.
  - Useful for comparing:
    - Ad‑hoc Python + LangChain loops vs.
    - Structured Temporal workflows + activities + tools.

---

## How the Looping Workflow Is Generalized

The core pattern is:

1. **History**: Build a provider‑agnostic prompt history in Python (`PromptHistory`).
2. **Step**: Convert history to provider messages and call a single **LLM step activity**.
3. **Branch**:
   - If the step requests a tool (`tool_call`):
     - Execute a generic **tool activity** that dispatches by name.
     - Record the tool output back into history.
     - Repeat from step 2.
   - If the step returns only text with `is_final=True`:
     - Stop and return the final text (and any summaries, reports, etc.).
4. **Provider independence**:
   - `activities.py` is the only file that knows how to talk to OpenAI vs Gemini.
   - Workflows and tools are written once and reused across providers.

This makes the system:

- **Simple and robust**:
  - Temporal guarantees determinism and failure handling at the workflow level.
  - Activities encapsulate all non‑deterministic work.
- **Extensible**:
  - New tools: add a `@tool` function and register it.
  - New agents: define different prompts and/or step types; reuse the same loop.
  - New providers: add another branch in `llm_step_activity` that returns
    `AgentStepOutput` and honors the same JSON schema.

Once this loop is in place, your specialists can focus on:

- Prompt tuning and instructions (`SUPERVISOR_PROMPT`, task prompts).
- Conversation design and communication systems.
- Bandit‑style tool selection (e.g., when to call which tool).
- New domain‑specific agents (like the `company_research_agent`).

---

## Running the Demo End‑to‑End

These instructions assume:

- Temporal server is available on `localhost:7233`.
- Your Python environment has been set up via `uv sync --dev`.
- Any required LLM credentials (OpenAI, Gemini) are configured in the environment.

### 1. Start Temporal Server

In a terminal:

```bash
temporal server start-dev
```

Leave this running.

### 2. Start the Multi‑Agent Worker

In a second terminal, from the repo root:

```bash
uv run -m src.supervisor_agent.worker
```

This starts a worker on `TASK_QUEUE` (from `config.py`) that can run:

- The personal assistant workflows.
- The company research sub‑agent.
- All LLM/tool activities.

### 3. Start the Chat CLI

In a third terminal, from the repo root:

```bash
uv run -m src.supervisor_agent.run
```

You should see:

- A printed workflow ID for the chat session.
- A `you>` prompt for entering messages.

Try:

```text
you> Schedule a 30-minute sync with my team for tomorrow afternoon and email them a brief agenda.
```

You should see one or more `assistant>` messages as the workflow calls tools and produces a final summary.

---

## Mid‑Demo Worker Restart (Intentional)

One goal of this demo is to show that the system is **resilient to worker restarts**, even mid‑conversation.

To observe this:

1. With the CLI running, ask a question that triggers tools:

   ```text
   you> What's the weather in Seattle tomorrow afternoon, and can you send me an email summary?
   ```

2. While the assistant is thinking (or after a few turns), **restart the worker**:

   - Go to the worker terminal and press `Ctrl+C` to stop it.
  - Then restart:

    ```bash
    uv run -m src.supervisor_agent.worker
    ```

3. Continue using the CLI:

   - The chat workflow execution **remains active in Temporal**.
   - When the worker comes back up, it:
     - Replays the workflow history deterministically.
     - Resumes processing new signals and queries.

You may see transient log warnings like:

- `"Query not found when attempting to respond to it"` or
- `RPCError: Timeout expired`

These occur when:

- The server discards expired query tasks while the worker is restarting.

The CLI is tolerant of this and simply retries; your conversation and state are preserved.

---

## Company Research During the Demo

The demo integrates a **separate company research agent**:

- The supervisor agent has access to a `company_research` tool.
- Internally this delegates to the `company_research_agent.AgentLoopWorkflow`,
  which runs a longer‑horizon iterative analysis with its own tools.

To highlight this during a demo:

1. Start a chat session as above.
2. Ask about the company you’re demoing to. For example:

   ```text
   you> We're meeting with Trimble. Can you research them and summarize their strategy versus competitors?
   ```

3. The assistant will:
   - Call the `company_research` tool.
   - The tool starts the `AgentLoopWorkflow` as a child workflow on the
     `company-research-task-queue`.
   - That workflow:
     - Validates the company.
     - Identifies the sector and top competitors.
     - Browses relevant pages.
     - Generates a Markdown report (and PDF) via `generate_report`.
   - The main assistant receives a **markdown excerpt** and synthesizes it into:
     - A concise `FINAL SUMMARY` (like the Trimble example you saw).
     - A follow‑up question such as “Would you like the full report…?”.

Because the company research agent is itself built using the **same loop pattern**
(history → LLM step → tools → final marker), it is easy to:

- Plug it in as a tool to other supervisors.
- Swap providers.
- Extend its tool set (e.g., adding more web/search integrations).

---

## Extending and Specializing the System

Once this infrastructure is in place, your specialists can focus on:

- **Prompt tuning**:
  - Adjust `SUPERVISOR_PROMPT`, task prompts, and system instructions.
  - Add specialized prompts for new domains (finance, healthcare, etc.).
- **Communication systems**:
  - Experiment with different trace and history strategies
    (e.g., summarization, memory slots, or retrieval‑augmented patterns).
- **Bandit problems and tool selection**:
  - Implement scoring and bandit‑style policies inside workflows:
    - When to choose which tool.
    - How to balance exploration vs. exploitation.
- **New agents**:
  - Implement additional sub‑agents (like the company research agent) as
    independent workflows with their own tools and prompts.
  - Expose each as a simple tool to the supervisor agent.

Because the Temporal workflows are deterministic and all external I/O is
inside activities, you can iterate on agent logic safely while relying on:

- Temporal for durability and retries.
- The provider abstraction for model portability.
- The shared tools layer for reusing integrations across agents.
