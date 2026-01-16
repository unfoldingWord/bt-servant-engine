"""ReAct Agentic MCP helper for Translation Helps.

This module provides a ReAct (Reason + Act) agentic loop that dynamically
discovers available MCP tools and orchestrates multi-step workflows to
answer Bible translation questions.

Key features:
- Dynamic tool discovery from MCP manifest (no hardcoded whitelists)
- ReAct loop: Think → Act → Observe → Repeat (max configurable iterations)
- Discovery on demand: tries direct fetches first, discovers on failure
- Extensive structured logging for verification and debugging
- Language fallback to English when resources unavailable
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, cast

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field as PydanticField
from translation_helps import ClientOptions, TranslationHelpsClient  # type: ignore  # pylint: disable=import-error

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import track_openai_usage
from bt_servant_engine.services.progress_messaging import maybe_send_progress
from bt_servant_engine.services import status_messages
from utils.perf import add_tokens

logger = get_logger(__name__)

FAILURE_MESSAGE_EN = (
    "Sorry. I couldn't figure out what to do. Please pass this along to my creators."
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(slots=True)
class ReActConfig:
    """Configuration for the ReAct agentic loop."""

    max_iterations: int = 5
    think_model: str = "gpt-4o"
    tool_max_chars: int = 5000
    total_max_chars: int = 15000
    fallback_language: str = "en"

    @classmethod
    def from_settings(cls) -> "ReActConfig":
        """Create config from application settings."""
        return cls(
            max_iterations=config.REACT_MAX_ITERATIONS,
            think_model=config.REACT_THINK_MODEL,
            tool_max_chars=config.REACT_TOOL_MAX_CHARS,
            total_max_chars=config.REACT_TOTAL_MAX_CHARS,
            fallback_language=config.REACT_FALLBACK_LANGUAGE,
        )


@dataclass(slots=True)
class MCPAgenticDependencies:
    """External helpers needed by the agentic MCP flow."""

    openai_client: OpenAI
    extract_cached_tokens_fn: Callable[..., Any]
    brain_state: Optional[Dict[str, Any]] = None  # For progress messaging


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(slots=True)
class MCPToolSpec:
    """Metadata for a tool discovered from MCP manifest."""

    name: str
    required: List[str]
    description: str = ""


@dataclass(slots=True)
class Observation:
    """Result from executing a single tool call."""

    tool_name: str
    success: bool
    content: str
    error: Optional[str]
    elapsed_ms: float


@dataclass(slots=True)
class ThinkResult:
    """Result from the Think phase of ReAct."""

    reasoning: str
    action: str  # "call_tools" | "done"
    tool_calls: List[Dict[str, Any]]
    final_response: Optional[str]


@dataclass
class IterationLog:
    """Log of a single ReAct iteration."""

    iteration: int
    think: ThinkResult
    observations: List[Observation]
    elapsed_ms: float


@dataclass
class ReActContext:
    """Context maintained across ReAct iterations."""

    user_message: str
    intent: IntentType
    manifest: Dict[str, MCPToolSpec]
    history: List[IterationLog] = field(default_factory=list)
    accumulated_content: str = ""


# =============================================================================
# Pydantic Models for LLM Parsing
# =============================================================================


class ToolCallArgs(BaseModel):
    """Allowed arguments for MCP tools."""

    model_config = ConfigDict(extra="allow")  # Allow dynamic args from manifest

    # Common fetch tool params
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

    # Discovery tool params
    stage: Optional[str] = None
    subjects: Optional[str] = None
    limit: Optional[int] = None
    topic: Optional[str] = None
    subject: Optional[str] = None
    languages: Optional[List[str]] = None


class ToolCall(BaseModel):
    """Single MCP tool invocation request."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: ToolCallArgs = PydanticField(default_factory=ToolCallArgs)


class ThinkResponse(BaseModel):
    """Structured response from the Think phase LLM."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    action: str  # "call_tools" | "done"
    tool_calls: List[ToolCall] = PydanticField(default_factory=list)
    final_response: Optional[str] = None


# =============================================================================
# System Prompt
# =============================================================================


REACT_SYSTEM_PROMPT = """
You are a Bible translation assistant with access to MCP tools.

# Available Tools
{manifest_json}

# Your Process
1. THINK: Analyze what information you need
2. ACT: Choose tool(s) to call
3. OBSERVE: Review results in next turn
4. Repeat until you have enough information to respond

# Tool Categories
- **Discovery**: list_languages, list_subjects, list_resources_for_language,
  search_translation_word_across_languages - use to find what's available
- **Fetch**: fetch_scripture, fetch_translation_notes, fetch_translation_word_links,
  fetch_translation_word, fetch_translation_academy, fetch_translation_questions

# Decision Rules
- For common languages (en, es, fr): try fetch directly
- For unfamiliar languages or failures: discover first with list_resources_for_language
- If fetch fails with "not found": use discovery tools, then retry with correct params
- The server maps language codes automatically (es -> es-419)
- If resource unavailable in requested language: fall back to English, note this

# Orchestration Examples
When user asks for "translation helps" for a passage, orchestrate:
1. fetch_scripture - get the text
2. fetch_translation_notes - get notes for the passage
3. fetch_translation_word_links - get linked words
4. fetch_translation_word - get definitions for key words (call multiple times if needed)
5. Combine results in your final response

When user asks about a specific term:
1. fetch_translation_word - get the definition
2. If not found, search_translation_word_across_languages to find alternatives

# Response Format (JSON)
{{
  "reasoning": "Brief explanation of your thinking",
  "action": "call_tools" | "done",
  "tool_calls": [{{"name": "...", "args": {{...}}}}],
  "final_response": "Only if action is done - your complete answer to the user"
}}

# Context
User: {user_message}
Intent: {intent}
History: {history_summary}
"""


# =============================================================================
# Helper Functions
# =============================================================================


def _truncate_text(text: Optional[str], limit: int, source: str) -> str:
    """Trim oversized text to control token costs."""
    if not text:
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated to {limit} chars from {source}]"


def _manifest_summary(specs: Dict[str, MCPToolSpec]) -> str:
    """Return a compact JSON manifest for the planner prompt."""
    summary = [
        {
            "name": spec.name,
            "required": spec.required,
            "description": spec.description[:100] if spec.description else "",
        }
        for spec in specs.values()
    ]
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _history_summary(history: List[IterationLog]) -> str:
    """Summarize iteration history for context."""
    if not history:
        return "No previous iterations."

    lines = []
    for log in history:
        tools_called = [obs.tool_name for obs in log.observations]
        successes = sum(1 for obs in log.observations if obs.success)
        failures = len(log.observations) - successes
        lines.append(
            f"Iteration {log.iteration}: Called {tools_called}, "
            f"{successes} succeeded, {failures} failed"
        )
        # Include key content snippets
        for obs in log.observations:
            if obs.success and obs.content:
                snippet = obs.content[:200].replace("\n", " ")
                lines.append(f"  - {obs.tool_name}: {snippet}...")

    return "\n".join(lines)


def _should_discover(observations: List[Observation]) -> bool:
    """Check if we need to discover available resources due to failures."""
    for obs in observations:
        if not obs.success and obs.error:
            error_lower = obs.error.lower()
            discovery_keywords = ["language", "not found", "unavailable", "404", "error"]
            if any(kw in error_lower for kw in discovery_keywords):
                return True
    return False


def _args_to_payload(args: ToolCallArgs | Dict[str, Any] | None) -> Dict[str, Any]:
    """Normalize args into a plain dict, dropping Nones."""
    if isinstance(args, ToolCallArgs):
        return {k: v for k, v in args.model_dump().items() if v is not None}
    if isinstance(args, dict):
        return {k: v for k, v in args.items() if v is not None}
    return {}


async def _send_progress(
    brain_state: Optional[Dict[str, Any]],
    message_key: str,
    force: bool = False,
    **format_args: Any,
) -> None:
    """Send a progress message if brain_state is available.

    Args:
        brain_state: The BrainState dict (or None if not available)
        message_key: The status message key (e.g., REACT_ANALYZING_REQUEST)
        force: If True, bypass throttling
        **format_args: Format arguments for templated messages
    """
    if brain_state is None:
        return

    try:
        message = status_messages.get_progress_message(
            message_key, brain_state, **format_args
        )
        await maybe_send_progress(brain_state, message, force=force)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("[react-mcp] Failed to send progress: %s", exc)


def _get_tool_progress_key(tool_name: str) -> Optional[str]:
    """Map tool name to appropriate progress message key."""
    tool_message_map = {
        "fetch_scripture": status_messages.REACT_FETCHING_SCRIPTURE,
        "fetch_translation_notes": status_messages.REACT_FETCHING_NOTES,
        "fetch_translation_word": status_messages.REACT_FETCHING_WORD_INFO,
        "fetch_translation_word_links": status_messages.REACT_FETCHING_WORD_INFO,
        "fetch_translation_academy": status_messages.REACT_FETCHING_NOTES,
        "fetch_translation_questions": status_messages.REACT_FETCHING_NOTES,
        "list_languages": status_messages.REACT_DISCOVERING_RESOURCES,
        "list_subjects": status_messages.REACT_DISCOVERING_RESOURCES,
        "list_resources_for_language": status_messages.REACT_DISCOVERING_RESOURCES,
        "list_resources_by_language": status_messages.REACT_DISCOVERING_RESOURCES,
        "search_translation_word_across_languages": status_messages.REACT_FETCHING_WORD_INFO,
    }
    return tool_message_map.get(tool_name)


# =============================================================================
# Core ReAct Functions
# =============================================================================


async def _load_full_manifest(client: Any) -> Dict[str, MCPToolSpec]:
    """Load ALL tools from MCP server. No whitelist - agent orchestrates."""
    specs: Dict[str, MCPToolSpec] = {}

    tools = await client.list_tools()
    for tool in tools:
        name = str(tool.get("name", "")).strip()
        if not name:
            continue
        schema = tool.get("inputSchema") or {}
        required = [str(r) for r in (schema.get("required") or [])]
        description = str(tool.get("description", ""))
        specs[name] = MCPToolSpec(name=name, required=required, description=description)

    # NOTE: We intentionally skip list_prompts().
    # MCP "prompts" are orchestration workflows - the ReAct agent handles
    # orchestration itself by calling individual tools.

    logger.info(
        "[react-mcp] Loaded tools from manifest",
        extra={"tool_count": len(specs), "tools": list(specs.keys())},
    )
    return specs


def _think(
    deps: MCPAgenticDependencies,
    context: ReActContext,
    react_config: ReActConfig,
) -> ThinkResult:
    """Ask the LLM to reason about what to do next."""
    prompt = REACT_SYSTEM_PROMPT.format(
        manifest_json=_manifest_summary(context.manifest),
        user_message=context.user_message,
        intent=context.intent.value,
        history_summary=_history_summary(context.history),
    )

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "What should I do next? Respond with JSON."},
    ]

    # Add accumulated content if we have observations
    if context.history:
        last_iter = context.history[-1]
        obs_content = []
        for obs in last_iter.observations:
            if obs.success:
                obs_content.append(f"[{obs.tool_name}]: {obs.content}")
            else:
                obs_content.append(f"[{obs.tool_name} FAILED]: {obs.error}")
        if obs_content:
            messages.insert(
                1,
                {
                    "role": "assistant",
                    "content": "Previous tool results:\n" + "\n\n".join(obs_content),
                },
            )

    try:
        completion = deps.openai_client.chat.completions.create(
            model=react_config.think_model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        usage = getattr(completion, "usage", None)
        track_openai_usage(
            usage, react_config.think_model, deps.extract_cached_tokens_fn, add_tokens
        )

        content = completion.choices[0].message.content or "{}"
        parsed = json.loads(content)

        return ThinkResult(
            reasoning=parsed.get("reasoning", ""),
            action=parsed.get("action", "done"),
            tool_calls=parsed.get("tool_calls", []),
            final_response=parsed.get("final_response"),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("[react-mcp] Think phase failed", extra={"error": str(exc)})
        return ThinkResult(
            reasoning=f"Error during thinking: {exc}",
            action="done",
            tool_calls=[],
            final_response=FAILURE_MESSAGE_EN,
        )


async def _act(
    client: Any,
    think_result: ThinkResult,
    react_config: ReActConfig,
) -> List[Observation]:
    """Execute the tool calls from the Think phase."""
    observations: List[Observation] = []

    for call in think_result.tool_calls:
        tool_name = call.get("name", "")
        args = call.get("args", {})
        arg_payload = _args_to_payload(args)

        start = time.time()
        try:
            logger.info(
                "[react-mcp] Calling tool",
                extra={
                    "tool": tool_name,
                    "args": json.dumps(arg_payload, ensure_ascii=False),
                },
            )

            resp = await client.call_tool(tool_name, arg_payload)

            # Extract text content from response
            if isinstance(resp, dict):
                content_items = resp.get("content", [])
                text_parts = []
                for item in content_items:
                    if isinstance(item, dict):
                        if "text" in item:
                            text_parts.append(str(item["text"]))
                        elif item.get("type") == "text":
                            text_parts.append(str(item.get("text", "")))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text = "".join(text_parts)
                if not text and resp.get("text"):
                    text = str(resp["text"])
            else:
                text = str(resp) if resp else ""

            text = _truncate_text(text, react_config.tool_max_chars, tool_name)
            elapsed = (time.time() - start) * 1000

            logger.info(
                "[react-mcp] Tool completed",
                extra={
                    "tool": tool_name,
                    "success": True,
                    "content_len": len(text),
                    "elapsed_ms": round(elapsed, 2),
                },
            )

            observations.append(
                Observation(
                    tool_name=tool_name,
                    success=True,
                    content=text,
                    error=None,
                    elapsed_ms=elapsed,
                )
            )

        except Exception as exc:  # pylint: disable=broad-except
            elapsed = (time.time() - start) * 1000
            error_msg = str(exc)

            logger.warning(
                "[react-mcp] Tool failed",
                extra={
                    "tool": tool_name,
                    "success": False,
                    "error": error_msg,
                    "elapsed_ms": round(elapsed, 2),
                },
            )

            observations.append(
                Observation(
                    tool_name=tool_name,
                    success=False,
                    content="",
                    error=error_msg,
                    elapsed_ms=elapsed,
                )
            )

    return observations


async def _discover_resources(client: Any, react_config: ReActConfig) -> List[Observation]:
    """Call discovery tools to learn what's available."""
    observations: List[Observation] = []

    # Try list_languages
    start = time.time()
    try:
        result = await client.call_tool("list_languages", {})
        content = json.dumps(result, ensure_ascii=False)
        content = _truncate_text(content, react_config.tool_max_chars, "list_languages")
        elapsed = (time.time() - start) * 1000

        lang_count = 0
        if isinstance(result, dict) and "languages" in result:
            lang_count = len(result["languages"])

        logger.info(
            "[react-mcp] Discovery: languages",
            extra={
                "tool": "list_languages",
                "success": True,
                "language_count": lang_count,
                "elapsed_ms": round(elapsed, 2),
            },
        )

        observations.append(
            Observation(
                tool_name="list_languages",
                success=True,
                content=content,
                error=None,
                elapsed_ms=elapsed,
            )
        )
    except Exception as exc:  # pylint: disable=broad-except
        elapsed = (time.time() - start) * 1000
        logger.warning(
            "[react-mcp] Discovery failed: list_languages",
            extra={"error": str(exc), "elapsed_ms": round(elapsed, 2)},
        )
        observations.append(
            Observation(
                tool_name="list_languages",
                success=False,
                content="",
                error=str(exc),
                elapsed_ms=elapsed,
            )
        )

    return observations


async def _handle_language_fallback(
    deps: MCPAgenticDependencies,
    react_config: ReActConfig,
    requested_lang: str,
    reason: str,
) -> str:
    """Log warning, send progress message, and return fallback language."""
    logger.warning(
        "[react-mcp] Language fallback",
        extra={
            "requested": requested_lang,
            "fallback": react_config.fallback_language,
            "reason": reason,
        },
    )
    # Send progress message about fallback
    await _send_progress(
        deps.brain_state,
        status_messages.REACT_LANGUAGE_FALLBACK,
        requested_language=requested_lang,
    )
    return react_config.fallback_language


# =============================================================================
# Main ReAct Loop
# =============================================================================


async def _run_react_mcp_async(
    deps: MCPAgenticDependencies,
    react_config: ReActConfig,
    user_message: str,
    intent: IntentType,
) -> str:
    """Async ReAct workflow: Think → Act → Observe → Repeat."""
    start_time = time.time()

    logger.info(
        "[react-mcp] Starting ReAct loop",
        extra={
            "intent": intent.value,
            "user_message": user_message[:100],
            "max_iterations": react_config.max_iterations,
        },
    )

    # Send initial progress message
    await _send_progress(
        deps.brain_state,
        status_messages.REACT_CONNECTING_TO_RESOURCES,
        force=True,
    )

    # Connect to MCP server
    server_url = os.getenv("MCP_SERVER_URL")
    if not server_url:
        logger.error("[react-mcp] MCP_SERVER_URL not set - cannot connect to MCP server")
        return FAILURE_MESSAGE_EN

    client_options = cast(ClientOptions, {"serverUrl": server_url})
    mcp_client = TranslationHelpsClient(client_options)

    try:
        await mcp_client.connect()
    except Exception as conn_exc:
        logger.error(
            "[react-mcp] Failed to connect to MCP server",
            extra={"server_url": server_url, "error": str(conn_exc)},
        )
        return FAILURE_MESSAGE_EN

    try:
        # Load full manifest (no whitelist)
        manifest = await _load_full_manifest(mcp_client)
        if not manifest:
            logger.warning("[react-mcp] Manifest is empty; aborting.")
            return FAILURE_MESSAGE_EN

        # Initialize context
        context = ReActContext(
            user_message=user_message,
            intent=intent,
            manifest=manifest,
        )

        # Main ReAct loop
        for iteration in range(1, react_config.max_iterations + 1):
            iter_start = time.time()

            logger.info(
                "[react-mcp] Iteration starting",
                extra={"iteration": iteration},
            )

            # Send progress: analyzing request
            if iteration == 1:
                await _send_progress(
                    deps.brain_state,
                    status_messages.REACT_ANALYZING_REQUEST,
                )

            # THINK: LLM reasons about what to do next
            think_result = _think(deps, context, react_config)

            logger.info(
                "[react-mcp] Think result",
                extra={
                    "iteration": iteration,
                    "action": think_result.action,
                    "reasoning": think_result.reasoning[:200],
                    "tool_count": len(think_result.tool_calls),
                },
            )

            # Check if done
            if think_result.action == "done":
                # Send combining results progress
                await _send_progress(
                    deps.brain_state,
                    status_messages.REACT_COMBINING_RESULTS,
                )
                total_elapsed = (time.time() - start_time) * 1000
                logger.info(
                    "[react-mcp] Loop complete - agent done",
                    extra={
                        "iteration": iteration,
                        "total_elapsed_ms": round(total_elapsed, 2),
                        "intent": intent.value,
                    },
                )
                return think_result.final_response or FAILURE_MESSAGE_EN

            # Send progress based on what tools we're calling
            if think_result.tool_calls:
                first_tool = think_result.tool_calls[0].get("name", "")
                progress_key = _get_tool_progress_key(first_tool)
                if progress_key:
                    await _send_progress(deps.brain_state, progress_key)

            # ACT: Execute tool calls
            observations = await _act(mcp_client, think_result, react_config)

            # Log each observation
            for obs in observations:
                logger.info(
                    "[react-mcp] Observation",
                    extra={
                        "iteration": iteration,
                        "tool": obs.tool_name,
                        "success": obs.success,
                        "content_len": len(obs.content),
                        "elapsed_ms": round(obs.elapsed_ms, 2),
                        "error": obs.error,
                    },
                )

            # Check for failures that need discovery
            if _should_discover(observations):
                await _send_progress(
                    deps.brain_state,
                    status_messages.REACT_DISCOVERING_RESOURCES,
                )
                logger.info(
                    "[react-mcp] Triggering discovery due to failure",
                    extra={"iteration": iteration},
                )
                discovery_obs = await _discover_resources(mcp_client, react_config)
                observations.extend(discovery_obs)

            # Update context history
            iter_elapsed = (time.time() - iter_start) * 1000
            context.history.append(
                IterationLog(
                    iteration=iteration,
                    think=think_result,
                    observations=observations,
                    elapsed_ms=iter_elapsed,
                )
            )

            logger.info(
                "[react-mcp] Iteration complete",
                extra={
                    "iteration": iteration,
                    "elapsed_ms": round(iter_elapsed, 2),
                },
            )

        # Max iterations reached - try to finalize with what we have
        total_elapsed = (time.time() - start_time) * 1000
        logger.warning(
            "[react-mcp] Max iterations reached",
            extra={
                "max_iterations": react_config.max_iterations,
                "total_elapsed_ms": round(total_elapsed, 2),
            },
        )

        # One final think to get a response
        final_think = _think(deps, context, react_config)
        if final_think.final_response:
            return final_think.final_response
        return FAILURE_MESSAGE_EN

    finally:
        await mcp_client.close()
        total_elapsed = (time.time() - start_time) * 1000
        logger.info(
            "[react-mcp] Session closed",
            extra={
                "total_elapsed_ms": round(total_elapsed, 2),
                "iterations": len(context.history) if "context" in dir() else 0,
            },
        )


def run_agentic_mcp(
    deps: MCPAgenticDependencies,
    user_message: str,
    intent: IntentType,
) -> str:
    """Public entry point for the ReAct agentic MCP flow."""
    react_config = ReActConfig.from_settings()

    try:
        return asyncio.run(
            _run_react_mcp_async(
                deps,
                react_config,
                user_message=user_message,
                intent=intent,
            )
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("[react-mcp] Unexpected failure", extra={"error": str(exc)})
        return FAILURE_MESSAGE_EN


__all__ = ["MCPAgenticDependencies", "run_agentic_mcp", "FAILURE_MESSAGE_EN", "ReActConfig"]
