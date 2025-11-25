"""Agentic MCP helper for Translation Helps.

This module provides a lightweight planner/executor/finalizer flow that lets
the LLM pick which Translation Helps MCP tools/prompts to call, validate the
arguments, execute them, and compose a final response.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, cast

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from pydantic import BaseModel, Field
from translation_helps import TranslationHelpsClient  # type: ignore  # pylint: disable=import-error

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import track_openai_usage
from utils.perf import add_tokens

logger = get_logger(__name__)

ALLOWED_AGENTIC_TOOLS: set[str] = {
    "prompt_translation-helps-for-passage",
    "prompt_get-translation-words-for-passage",
    "prompt_get-translation-academy-for-passage",
    "fetch_translation_notes",
    "fetch_translation_word_links",
    "fetch_translation_word",
    "fetch_translation_academy",
    "fetch_translation_questions",
    "fetch_scripture",
}

PLAN_SYSTEM_PROMPT = """
You select the best MCP tools/prompts to answer the user's request.

Rules:
- Choose 1-3 calls from the allowed list.
- Use the required parameters shown in the manifest.
- Do not guess parameters; prefer the user's wording and references.
- Return ONLY a JSON object matching the schema.
"""

FINALIZER_SYSTEM_PROMPT = """
You are composing a reply using ONLY the provided MCP tool results.
- Be concise and helpful.
- If scripture text appears, keep it verbatim.
- If nothing useful is present, say so briefly.
- Offer 2 short follow-up questions that relate to the data retrieved.
"""

FAILURE_MESSAGE_EN = (
    "Sorry. I couldn't figure out what to do. Please pass this along to my creators."
)


class ToolCall(BaseModel):
    """Single MCP invocation request."""

    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """Structured plan produced by the planner LLM."""

    calls: List[ToolCall] = Field(default_factory=list)
    rationale: Optional[str] = None


@dataclass(slots=True)
class MCPToolSpec:
    """Metadata for validating tool calls."""

    name: str
    required: List[str]
    is_prompt: bool = False


@dataclass(slots=True)
class MCPAgenticDependencies:
    """External helpers needed by the agentic MCP flow."""

    openai_client: OpenAI
    extract_cached_tokens_fn: Callable[..., Any]


async def _load_manifest(client) -> Dict[str, MCPToolSpec]:
    """Load tools/prompts and reduce to the allowed manifest."""
    specs: Dict[str, MCPToolSpec] = {}
    tools = await client.list_tools()
    for tool in tools:
        name = str(tool.get("name", "")).strip()
        if name not in ALLOWED_AGENTIC_TOOLS:
            continue
        schema = tool.get("inputSchema") or {}
        required = schema.get("required") or []
        required_list = [str(item) for item in required if isinstance(item, str)]
        specs[name] = MCPToolSpec(name=name, required=required_list, is_prompt=False)

    prompts = await client.list_prompts()
    for prompt in prompts:
        name = str(prompt.get("name", "")).strip()
        full_name = f"prompt_{name}" if name else ""
        if full_name not in ALLOWED_AGENTIC_TOOLS:
            continue
        args = prompt.get("arguments") or []
        required_args: list[str] = []
        for arg in args:
            if isinstance(arg, Mapping) and arg.get("required"):
                arg_name = arg.get("name")
                if isinstance(arg_name, str):
                    required_args.append(arg_name)
        specs[full_name] = MCPToolSpec(name=full_name, required=required_args, is_prompt=True)
    return specs


def _manifest_summary(specs: Mapping[str, MCPToolSpec]) -> str:
    """Return a compact JSON manifest for the planner prompt."""
    summary = [
        {
            "name": spec.name,
            "required": spec.required,
            "is_prompt": spec.is_prompt,
        }
        for spec in specs.values()
    ]
    return json.dumps(summary, ensure_ascii=False)


def _plan_calls(
    deps: MCPAgenticDependencies,
    user_message: str,
    intent: IntentType,
    manifest_summary: str,
    prior_error: str | None = None,
) -> Plan:
    """Ask the LLM to select MCP calls."""

    messages: list[EasyInputMessageParam] = [
        {
            "role": "user",
            "content": (
                f"User request: {user_message}\n"
                f"Intent: {intent.value}\n"
                f"Allowed tools: {manifest_summary}\n"
                "Return a JSON object with calls and optional rationale."
            ),
        }
    ]
    if prior_error:
        messages.append(
            {
                "role": "user",
                "content": f"The previous plan failed validation: {prior_error}",
            }
        )

    resp = deps.openai_client.responses.parse(
        model="gpt-4o",
        instructions=PLAN_SYSTEM_PROMPT,
        input=cast(Any, messages),
        text_format=Plan,
        temperature=0,
        store=False,
    )
    usage = getattr(resp, "usage", None)
    track_openai_usage(usage, "gpt-4o", deps.extract_cached_tokens_fn, add_tokens)
    parsed = resp.output_parsed
    return parsed if isinstance(parsed, Plan) else Plan(calls=[])


def _validate_plan(plan: Plan, specs: Mapping[str, MCPToolSpec]) -> str | None:
    """Validate the planned calls against the manifest."""
    for call in plan.calls:
        spec = specs.get(call.name)
        if spec is None:
            return f"unknown tool: {call.name}"
        if not isinstance(call.args, dict):
            return f"arguments for {call.name} must be an object"
        missing = [
            req for req in spec.required if req not in call.args or call.args[req] in (None, "")
        ]
        if missing:
            return f"{call.name} is missing required fields: {', '.join(missing)}"
    if not plan.calls:
        return "no calls were provided"
    return None


async def _execute_prompt(client, name: str, params: Dict[str, Any]) -> str:
    """Execute a prompt via the REST endpoint."""
    if not client._http_client:  # pylint: disable=protected-access
        raise RuntimeError("MCP client HTTP session is not initialized.")
    url = client.server_url.replace("/api/mcp", "/api/execute-prompt")
    response = await client._http_client.post(  # pylint: disable=protected-access
        url,
        json={"promptName": name, "parameters": params},
    )
    response.raise_for_status()
    data = response.json()
    return json.dumps(data, ensure_ascii=False)


def _extract_text_from_content(content: Iterable[Any]) -> str:
    """Extract text blobs from MCP tool content."""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if "text" in item:
                chunks.append(str(item.get("text", "")))
            elif item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        elif isinstance(item, str):
            chunks.append(item)
    return "".join(chunks)


async def _execute_calls(
    client,
    plan: Plan,
    specs: Mapping[str, MCPToolSpec],
) -> list[dict[str, Any]]:
    """Execute planned calls and capture text outputs."""
    results: list[dict[str, Any]] = []
    for call in plan.calls:
        spec = specs[call.name]
        try:
            if spec.is_prompt:
                prompt_name = call.name.replace("prompt_", "", 1)
                text = await _execute_prompt(client, prompt_name, call.args)
            else:
                resp = await client.call_tool(call.name, call.args)
                content = resp.get("content", []) if isinstance(resp, dict) else []
                text = _extract_text_from_content(content)
                if not text and resp.get("text"):
                    text = str(resp["text"])
            results.append({"name": call.name, "args": call.args, "content": text})
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[agentic-mcp] %s failed: %s", call.name, exc)
            results.append({"name": call.name, "args": call.args, "content": f"[ERROR] {exc}"})
    return results


def _finalize_response(
    deps: MCPAgenticDependencies,
    user_message: str,
    intent: IntentType,
    tool_results: list[dict[str, Any]],
) -> str:
    """Ask the LLM to compose the final reply from tool outputs."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": FINALIZER_SYSTEM_PROMPT},
        {"role": "user", "content": f"User request: {user_message}\nIntent: {intent.value}"},
        {
            "role": "assistant",
            "content": json.dumps(tool_results, ensure_ascii=False),
        },
    ]
    completion = deps.openai_client.chat.completions.create(
        model="gpt-4o",
        messages=cast(Any, messages),
        temperature=0.4,
    )
    usage = getattr(completion, "usage", None)
    track_openai_usage(usage, "gpt-4o", deps.extract_cached_tokens_fn, add_tokens)
    content = completion.choices[0].message.content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else "" for part in content)
    return content or ""


async def _run_agentic_mcp_async(
    deps: MCPAgenticDependencies,
    user_message: str,
    intent: IntentType,
    max_plan_attempts: int = 2,
) -> str:
    """Async workflow to plan, validate, execute, and finalize MCP calls."""
    server_url = os.getenv("MCP_SERVER_URL")
    client_options = {"serverUrl": server_url} if server_url else {}
    mcp_client = TranslationHelpsClient(client_options)
    await mcp_client.connect()
    try:
        manifest = await _load_manifest(mcp_client)
        if not manifest:
            logger.warning("[agentic-mcp] Manifest is empty; aborting.")
            return FAILURE_MESSAGE_EN
        logger.info("[agentic-mcp] Loaded manifest with %d tools", len(manifest))

        manifest_summary = _manifest_summary(manifest)
        validation_error: str | None = None
        plan: Plan = Plan(calls=[])

        for _ in range(max_plan_attempts):
            plan = _plan_calls(
                deps,
                user_message=user_message,
                intent=intent,
                manifest_summary=manifest_summary,
                prior_error=validation_error,
            )
            validation_error = _validate_plan(plan, manifest)
            if validation_error is None:
                break
        if validation_error is not None:
            logger.info("[agentic-mcp] Plan failed after retries: %s", validation_error)
            return FAILURE_MESSAGE_EN

        tool_results = await _execute_calls(mcp_client, plan, manifest)
        logger.info(
            "[agentic-mcp] Executed %d calls successfully for intent=%s",
            len(tool_results),
            intent.value,
        )
        return _finalize_response(deps, user_message, intent, tool_results)
    finally:
        await mcp_client.close()


def run_agentic_mcp(
    deps: MCPAgenticDependencies,
    user_message: str,
    intent: IntentType,
) -> str:
    """Public entry point for the agentic MCP flow."""
    try:
        return asyncio.run(_run_agentic_mcp_async(deps, user_message=user_message, intent=intent))
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("[agentic-mcp] Unexpected failure: %s", exc, exc_info=True)
        return FAILURE_MESSAGE_EN


__all__ = ["MCPAgenticDependencies", "run_agentic_mcp", "FAILURE_MESSAGE_EN"]
