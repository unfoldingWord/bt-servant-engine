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
from pydantic import BaseModel, ConfigDict, Field
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

MAX_TOOL_CONTENT_CHARS = int(os.getenv("MCP_TOOL_MAX_CHARS", "5000"))
MAX_PLAN_TOTAL_CHARS = int(os.getenv("MCP_PLAN_TOTAL_MAX_CHARS", "15000"))

PLAN_SYSTEM_PROMPT = """
You select the best MCP tools/prompts to answer the user's request.

Rules:
- Choose 1-3 calls from the allowed list.
- Use the required parameters shown in the manifest.
- Only use argument keys from the manifest (reference, language, organization, format,
  includeIntro, includeContext, includeVerseNumbers, term, category, rcLink, moduleId, path).
- Do not guess parameters; prefer the user's wording and references.
- Return ONLY a JSON object matching the schema.
"""

FINALIZER_SYSTEM_PROMPT = """
You are composing a reply using ONLY the provided MCP tool results.
- Be concise and helpful.
- If scripture text appears, keep it verbatim.
- If the intent is get-passage-summary, produce a concise summary (3–5 sentences max).
- If nothing useful is present, say so briefly.
"""

FAILURE_MESSAGE_EN = (
    "Sorry. I couldn't figure out what to do. Please pass this along to my creators."
)


class ToolCallArgs(BaseModel):
    """Allowed arguments for MCP tools/prompts (schema must forbid extras)."""

    model_config = ConfigDict(extra="forbid")

    reference: Optional[str] = None
    language: Optional[str] = None
    organization: Optional[str] = None
    format: Optional[str] = None
    includeIntro: Optional[bool] = None
    includeContext: Optional[bool] = None
    includeVerseNumbers: Optional[bool] = None
    term: Optional[str] = None
    category: Optional[str] = None
    rcLink: Optional[str] = None
    moduleId: Optional[str] = None
    path: Optional[str] = None


class ToolCall(BaseModel):
    """Single MCP invocation request."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: ToolCallArgs = Field(default_factory=ToolCallArgs)


class Plan(BaseModel):
    """Structured plan produced by the planner LLM."""

    model_config = ConfigDict(extra="forbid")

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


def _args_to_payload(args: ToolCallArgs | Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize args into a plain dict, dropping Nones."""
    if isinstance(args, ToolCallArgs):
        return args.model_dump(exclude_none=True)
    if isinstance(args, Mapping):
        return {k: v for k, v in args.items() if v is not None}
    return {}


def _truncate_text(text: Optional[str], limit: int, source: str) -> str:
    """Trim oversized text to control token costs."""
    if not text:
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated to {limit} chars from {source}]"


def _enforce_total_limit(results: list[dict[str, Any]], total_limit: int) -> list[dict[str, Any]]:
    """Ensure the aggregate content sent to the finalizer stays within budget."""
    if total_limit <= 0:
        return results
    total = sum(len(r.get("content") or "") for r in results)
    if total <= total_limit:
        return results

    # Scale down each item proportionally first.
    ratio = total_limit / max(total, 1)
    for item in results:
        content = item.get("content") or ""
        new_len = max(1, int(len(content) * ratio))
        if new_len < len(content):
            item["content"] = (
                content[:new_len] + f"\n[truncated to fit total limit {total_limit} chars]"
            )

    # If still over, trim the largest items until under the cap.
    total = sum(len(r.get("content") or "") for r in results)
    if total <= total_limit:
        return results
    for item in sorted(results, key=lambda r: len(r.get("content") or ""), reverse=True):
        if total <= total_limit:
            break
        content = item.get("content") or ""
        excess = total - total_limit
        keep = max(0, len(content) - excess)
        if keep < len(content):
            item["content"] = (
                content[:keep] + f"\n[truncated to fit total limit {total_limit} chars]"
            )
            total = sum(len(r.get("content") or "") for r in results)
    return results


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
        args_dict = _args_to_payload(call.args)
        missing = [
            req for req in spec.required if req not in args_dict or args_dict[req] in (None, "")
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
        arg_payload = _args_to_payload(call.args)
        try:
            logger.info(
                "[agentic-mcp] Calling %s (prompt=%s) with args=%s",
                call.name,
                spec.is_prompt,
                sorted(arg_payload.keys()),
            )
            if spec.is_prompt:
                prompt_name = call.name.replace("prompt_", "", 1)
                text = await _execute_prompt(client, prompt_name, arg_payload)
            else:
                resp = await client.call_tool(call.name, arg_payload)
                content = resp.get("content", []) if isinstance(resp, dict) else []
                text = _extract_text_from_content(content)
                if not text and resp.get("text"):
                    text = str(resp["text"])
            text = _truncate_text(text, MAX_TOOL_CONTENT_CHARS, call.name)
            results.append({"name": call.name, "args": arg_payload, "content": text})
            logger.info("[agentic-mcp] Completed %s (chars=%d)", call.name, len(text or ""))
            logger.info("[agentic-mcp] Raw content for %s: %s", call.name, text[:500])
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[agentic-mcp] %s failed: %s", call.name, exc)
            results.append({"name": call.name, "args": arg_payload, "content": f"[ERROR] {exc}"})
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
                logger.info(
                    "[agentic-mcp] Plan validated with %d call(s): %s",
                    len(plan.calls),
                    ", ".join(call.name for call in plan.calls),
                )
                break
        if validation_error is not None:
            logger.info("[agentic-mcp] Plan failed after retries: %s", validation_error)
            return FAILURE_MESSAGE_EN

        tool_results = await _execute_calls(mcp_client, plan, manifest)
        tool_results = _enforce_total_limit(tool_results, MAX_PLAN_TOTAL_CHARS)
        logger.info(
            "[agentic-mcp] Executed %d calls successfully for intent=%s",
            len(tool_results),
            intent.value,
        )
        final_response = _finalize_response(deps, user_message, intent, tool_results)
        logger.info(
            "[agentic-mcp] Final response length=%d chars for intent=%s",
            len(final_response or ""),
            intent.value,
        )
        return final_response
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
