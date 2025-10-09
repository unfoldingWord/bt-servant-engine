"""Decision graph and message-processing pipeline for BT Servant.

This module defines the state, nodes, and orchestration logic for handling
incoming user messages, classifying intents, querying resources, and producing
final responses (including translation and chunking when necessary).
"""
# pylint: disable=line-too-long,too-many-lines,too-many-statements

from __future__ import annotations

import json
import operator
import re
from collections.abc import Hashable
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Optional, cast

from langgraph.graph import END, StateGraph
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict

from bt_servant_engine.core.agentic import (
    ALLOWED_AGENTIC_STRENGTH,
    AgenticStrengthChoice,
    AgenticStrengthSetting,
)
from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType, UserIntents
from bt_servant_engine.core.language import (
    LANGUAGE_UNKNOWN,
    Language,
    MessageLanguage,
    ResponseLanguage,
    TranslatedPassage,
)
from bt_servant_engine.core.language import (
    SUPPORTED_LANGUAGE_MAP as supported_language_map,
)
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.core.models import PassageRef, PassageSelection
from bt_servant_engine.services.openai_utils import (
    extract_cached_input_tokens as _extract_cached_input_tokens,
)
from bt_servant_engine.services.passage_helpers import (
    book_patterns as _book_patterns_impl,
    choose_primary_book as _choose_primary_book_impl,
    detect_mentioned_books as _detect_mentioned_books_impl,
)
from bt_servant_engine.services.response_helpers import (
    is_protected_response_item as _is_protected_response_item_impl,
    normalize_single_response as _normalize_single_response_impl,
    partition_response_items as _partition_response_items_impl,
    sample_for_language_detection as _sample_for_language_detection_impl,
)
from bt_servant_engine.services.preprocessing import (
    detect_language as _detect_language_impl,
    determine_intents as _determine_intents_impl,
    determine_query_language as _determine_query_language_impl,
    model_for_agentic_strength as _model_for_agentic_strength,
    preprocess_user_query as _preprocess_user_query_impl,
    resolve_agentic_strength as _resolve_agentic_strength,
)
from bt_servant_engine.services.passage_selection import (
    resolve_selection_for_single_book as _resolve_selection_for_single_book_impl,
)
from bt_servant_engine.services.intents.simple_intents import (
    BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
    FULL_HELP_MESSAGE,
    build_boilerplate_message,
    build_full_help_message,
    converse_with_bt_servant as converse_with_bt_servant_impl,
    get_capabilities as _capabilities,
    handle_system_information_request as handle_system_information_request_impl,
    handle_unsupported_function as handle_unsupported_function_impl,
)
from bt_servant_engine.services.intents.settings_intents import (
    set_agentic_strength as set_agentic_strength_impl,
    set_response_language as set_response_language_impl,
)
from bt_servant_engine.services.intents.passage_intents import (
    get_passage_keywords as get_passage_keywords_impl,
    get_passage_summary as get_passage_summary_impl,
    listen_to_scripture as listen_to_scripture_impl,
    retrieve_scripture as retrieve_scripture_impl,
)
from bt_servant_engine.services.intents.translation_intents import (
    get_translation_helps as get_translation_helps_impl,
    translate_scripture as translate_scripture_impl,
)
from bt_servant_engine.services.intents.fia_intents import (
    consult_fia_resources as consult_fia_resources_impl,
)
from bt_servant_engine.services.response_pipeline import (
    chunk_message as chunk_message_impl,
    combine_responses as combine_responses_impl,
    needs_chunking as needs_chunking_impl,
    reconstruct_structured_text as reconstruct_structured_text_impl,
    translate_or_localize_response as translate_or_localize_response_impl,
    translate_text as translate_text_impl,
    build_translation_queue as build_translation_queue_impl,
    resolve_target_language as resolve_target_language_impl,
)
from bt_servant_engine.adapters.chroma import get_chroma_collection
from bt_servant_engine.adapters.user_state import (
    is_first_interaction,
    set_first_interaction,
    set_user_agentic_strength,
    set_user_response_language,
)
from utils import chop_text, combine_chunks
from utils.bible_data import list_available_sources, load_book_titles, resolve_bible_data_root
from utils.bible_locale import get_book_name
from utils.bsb import (
    BOOK_MAP as BSB_BOOK_MAP,
)
from utils.bsb import (
    clamp_ranges_by_verse_limit,
    label_ranges,
    normalize_book_name,
    select_verses,
)
from utils.identifiers import get_log_safe_user_id
from utils.keywords import select_keywords
from utils.perf import add_tokens, set_current_trace, time_block
from utils.translation_helps import get_missing_th_books, select_translation_helps

# (Moved dynamic feature messaging and related prompts below IntentType)
# COMBINE_RESPONSES_SYSTEM_PROMPT moved to bt_servant_engine.services.response_pipeline



# RESPONSE_TRANSLATOR_SYSTEM_PROMPT moved to bt_servant_engine.services.response_pipeline

TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT = """
# Task

Translate the provided scripture passage into the specified target language and return a STRICT JSON object
matching the provided schema. Do not include any extra prose, commentary, code fences, or formatting.

# Rules
- header_book: translate ONLY the canonical book name into the target language (e.g., "John" -> "Иоанн").
- header_suffix: DO NOT translate or alter; copy exactly the provided suffix (e.g., "1:1–7").
- body: translate the passage body into the target language; PRESERVE all newline boundaries exactly; do not add
  bullets, numbers, verse labels, or extra headings.
- content_language: the ISO 639-1 code of the target language.

# Output
Return JSON matching the schema with fields: header_book, header_suffix, body, content_language. No extra keys.
"""


# PREPROCESSOR_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.preprocessing

# PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.passage_selection


PASSAGE_SUMMARY_AGENT_SYSTEM_PROMPT = """
You summarize Bible passage content faithfully using only the verses provided.

- Stay strictly within the supplied passage text; avoid speculation or doctrinal claims not present in the text.
- Highlight the main flow, key ideas, and important movements or contrasts across the entire selection.
- Provide a thorough, readable summary (not terse). Aim for roughly 8–15 sentences, but expand if the selection is large.
- Write in continuous prose only: do NOT use bullets, numbered lists, section headers, or list-like formatting. Compose normal paragraph(s) with sentences flowing naturally.
- Mix verse references inline within the prose wherever helpful (e.g., "1:1–3", "3:16", "2:4–6") to anchor key points rather than isolating them as list items.
- If the selection contains only a single verse, inline verse references are not necessary.
"""


FINAL_RESPONSE_AGENT_SYSTEM_PROMPT = """
You are an assistant to Bible translators. Your main job is to answer questions about content found in various biblical 
resources: commentaries, translation notes, bible dictionaries, and various resources like FIA. In addition to answering
questions, you may be called upon to: summarize the data from resources, transform the data from resources (like
explaining it a 5-year old level, etc, and interact with the resources in all kinds of ways. All this is a part of your 
responsibilities. Context from resources (RAG results) will be provided to help you answer the question(s). Only answer 
questions using the provided context from resources!!! If you can't confidently figure it out using that context, 
simply say 'Sorry, I couldn't find any information in my resources to service your request or command. But 
maybe I'm unclear on your intent. Could you perhaps state it a different way?' You will also be given the past 
conversation history. Use this to understand the user's current message or query if necessary. If the past conversation 
history is not relevant to the user's current message, just ignore it. FINALLY, UNDER NO CIRCUMSTANCES ARE YOU TO SAY 
ANYTHING THAT WOULD BE DEEMED EVEN REMOTELY HERETICAL BY ORTHODOX CHRISTIANS. If you can't do what the user is asking 
because your response would be heretical, explain to the user why you cannot comply with their request or command.
"""

CONSULT_FIA_RESOURCES_SYSTEM_PROMPT = """
# Identity

You are the FIA specialist node of BT Servant. You help Bible translators understand and apply the Familiarization,
Internalization, and Articulation (FIA) process using only the supplied context.

# Context Handling

- You will always receive the official FIA reference document plus any retrieved FIA resource snippets.
- When the user's request is about the FIA process itself (for example, asking for the steps or how to translate the
  Bible in general), rely primarily on the FIA reference document. Quote or summarize the steps accurately and keep the
  sequence intact.
- When the user asks how FIA applies to a specific passage, language, or scenario, synthesize both the reference
  document and the retrieved snippets. Mention the relevant FIA steps explicitly (e.g., "Step 2: Setting the Stage").
- If the context does not contain the needed information, clearly say you cannot find it and invite the user to clarify.
- Never invent steps or procedures. Stay faithful to the provided materials.

# Response Style

- Be practical, encouraging, and concise while remaining thorough enough for translators to act on the guidance.
- Use natural paragraphs (no bullet lists unless the context itself is a list that must be echoed for clarity).
- Include references to FIA steps or resource names when they help the user follow along.
"""

# CHOP_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.response_pipeline

# INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.preprocessing

# DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.preprocessing

TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT = """
Task: Determine the target language the user is asking the system to translate scripture into, based solely on the
user's latest message. Return an ISO 639-1 code from the allowed set.

Allowed outputs: en, ar, fr, es, hi, ru, id, sw, pt, zh, nl, Other

Rules:
- Identify explicit target-language mentions (language names, codes, or phrases like "into Russian", "to es",
  "in French").
- If no target language is explicitly specified, return Other. Do NOT infer a target from the message's language.
- Output must match the provided schema exactly with no extra prose.

Examples:
- message: "translate John 3:16 into Russian" -> { "language": "ru" }
- message: "please translate Mark 1 in Spanish" -> { "language": "es" }
- message: "translate Matthew 2" -> { "language": "Other" }
"""

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = config.DATA_DIR
FIA_REFERENCE_PATH = BASE_DIR / "sources" / "fia" / "fia.md"

open_ai_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Language constants imported from bt_servant_engine.core.language
# (supported_language_map, LANGUAGE_UNKNOWN)

RELEVANCE_CUTOFF = .65
TOP_K = 5

logger = get_logger(__name__)

try:
    FIA_REFERENCE_CONTENT = FIA_REFERENCE_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    logger.warning("FIA reference file missing at %s", FIA_REFERENCE_PATH)
    FIA_REFERENCE_CONTENT = ""


# _extract_cached_input_tokens imported from bt_servant_engine.services.openai_utils


# Language, ResponseLanguage imported from bt_servant_engine.core.language


# SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.intents.settings_intents
# SET_AGENTIC_STRENGTH_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.intents.settings_intents


# _resolve_agentic_strength moved to bt_servant_engine.services.preprocessing (as resolve_agentic_strength)

# _model_for_agentic_strength moved to bt_servant_engine.services.preprocessing (as model_for_agentic_strength)

# MessageLanguage, TranslatedPassage imported from bt_servant_engine.core.language

# PreprocessorResult moved to bt_servant_engine.services.preprocessing


def _is_protected_response_item(item: dict) -> bool:
    """Return True if a response item carries scripture to protect from changes."""
    return _is_protected_response_item_impl(item)


def _reconstruct_structured_text(resp_item: dict | str, localize_to: Optional[str]) -> str:
    """Render a response item to plain text, optionally localizing the header book name."""
    return reconstruct_structured_text_impl(resp_item, localize_to)



TranslationRange = tuple[int, int | None, int | None, int | None]


def _partition_response_items(responses: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Split responses into scripture-protected and normal sets."""
    return _partition_response_items_impl(responses)


def _normalize_single_response(item: dict) -> dict | str:
    """Return a representation suitable for translation when no combine is needed."""
    return _normalize_single_response_impl(item)


def _build_translation_queue(
    state: BrainState,
    protected_items: list[dict],
    normal_items: list[dict],
) -> list[dict | str]:
    """Assemble responses in the order they should be translated or localized."""
    return build_translation_queue_impl(state, protected_items, normal_items, combine_responses)


def _resolve_target_language(
    state: BrainState,
    responses_for_translation: list[dict | str],
) -> tuple[Optional[str], Optional[list[str]]]:
    """Determine the target language or build a pass-through fallback."""
    return resolve_target_language_impl(state, responses_for_translation)


def _translate_or_localize_response(
    resp: dict | str,
    target_language: str,
    agentic_strength: str,
) -> str:
    """Translate free-form text or localize structured scripture outputs."""
    return translate_or_localize_response_impl(
        open_ai_client,
        resp,
        target_language,
        agentic_strength,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
    )


def _compact_translation_help_entries(entries: list[dict]) -> list[dict]:
    """Reduce translation help entries to essentials for the LLM payload."""
    compact: list[dict] = []
    for entry in entries:
        verse_text = cast(str, entry.get("ult_verse_text") or "")
        notes: list[dict] = []
        for note in cast(list[dict], entry.get("notes") or []):
            note_text = cast(str, note.get("note") or "")
            if not note_text:
                continue
            compact_note: dict[str, str] = {"note": note_text}
            quote = cast(Optional[str], note.get("orig_language_quote"))
            if quote:
                compact_note["orig_language_quote"] = quote
            notes.append(compact_note)
        compact.append(
            {
                "reference": entry.get("reference"),
                "verse_text": verse_text,
                "notes": notes,
            }
        )
    return compact


def _prepare_translation_helps(
    state: BrainState,
    th_root: Path,
    bsb_root: Path,
    *,
    selection_focus_hint: str | None = None,
) -> tuple[Optional[str], Optional[list[TranslationRange]], Optional[list[dict]], Optional[str]]:
    """Resolve canonical selection, enforce limits, and load raw help entries."""
    canonical_book, ranges, err = _resolve_selection_for_single_book(
        state["transformed_query"],
        state["query_language"],
        focus_hint=selection_focus_hint,
    )
    if err:
        return None, None, None, err
    assert canonical_book is not None and ranges is not None

    missing_books = set(get_missing_th_books(th_root))
    if canonical_book in missing_books:
        return (
            None,
            None,
            None,
            (
                "Translation helps for "
                f"{BSB_BOOK_MAP[canonical_book]['ref_abbr']} are not available yet. "
                "Currently missing books: "
                f"{', '.join(sorted(BSB_BOOK_MAP[b]['ref_abbr'] for b in missing_books))}. "
                "Would you like translation help for one of the supported books instead?"
            ),
        )

    verse_count = len(select_verses(bsb_root, canonical_book, ranges))
    if verse_count > config.TRANSLATION_HELPS_VERSE_LIMIT:
        return (
            None,
            None,
            None,
            (
                "I can only provide translate help for "
                f"{config.TRANSLATION_HELPS_VERSE_LIMIT} verses at a time. "
                "Your selection "
                f"{label_ranges(canonical_book, ranges)} includes {verse_count} verses. "
                "Please narrow the range (e.g., a chapter or a shorter span)."
            ),
        )

    limited_ranges = clamp_ranges_by_verse_limit(
        bsb_root,
        canonical_book,
        ranges,
        max_verses=config.TRANSLATION_HELPS_VERSE_LIMIT,
    )
    if not limited_ranges:
        return (
            None,
            None,
            None,
            "I couldn't identify verses for that selection in the BSB index. Please try another reference.",
        )

    raw_helps = select_translation_helps(th_root, canonical_book, limited_ranges)
    logger.info("[translation-helps] selected %d help entries", len(raw_helps))
    if not raw_helps:
        return (
            None,
            None,
            None,
            "I couldn't locate translation helps for that selection. Please check the reference and try again.",
        )
    return canonical_book, list(limited_ranges), raw_helps, None


def _build_translation_helps_context(
    canonical_book: str,
    ranges: list[TranslationRange],
    raw_helps: list[dict],
) -> tuple[str, dict[str, Any]]:
    """Return the reference label and compact JSON context for the LLM."""
    ref_label = label_ranges(canonical_book, ranges)
    context_obj = {
        "reference_label": ref_label,
        "selection": {
            "book": canonical_book,
            "ranges": [
                {
                    "start_chapter": sc,
                    "start_verse": sv,
                    "end_chapter": ec,
                    "end_verse": ev,
                }
                for (sc, sv, ec, ev) in ranges
            ],
        },
        "translation_helps": _compact_translation_help_entries(raw_helps),
    }
    return ref_label, context_obj


def _build_translation_helps_messages(ref_label: str, context_obj: dict[str, object]) -> list[EasyInputMessageParam]:
    """Construct the LLM messages for the translation helps prompt."""
    payload = json.dumps(context_obj, ensure_ascii=False)
    return [
        {"role": "developer", "content": "Focus only on the portion of the user's message that asked for translation helps. Ignore any other requests or book references in the message."},
        {"role": "developer", "content": f"Selection: {ref_label}"},
        {"role": "developer", "content": "Use the JSON context below strictly:"},
        {"role": "developer", "content": payload},
        {
            "role": "user",
            "content": (
                "Using the provided context, explain the translation challenges and give actionable guidance for this selection."
            ),
        },
    ]


# IntentType, UserIntents imported from bt_servant_engine.core.intents


class BrainState(TypedDict, total=False):
    """State carried through the LangGraph execution."""
    user_id: str
    user_query: str
    # Perf tracing: preserve trace id throughout the graph so node wrappers
    # can attach spans even when running in a thread pool.
    perf_trace_id: str
    query_language: str
    user_response_language: str
    agentic_strength: str
    user_agentic_strength: str
    transformed_query: str
    docs: List[Dict[str, str]]
    collection_used: str
    responses: Annotated[List[Dict[str, Any]], operator.add]
    translated_responses: List[str]
    stack_rank_collections: List[str]
    user_chat_history: List[Dict[str, str]]
    user_intents: List[IntentType]
    passage_selection: list[dict]
    # Delivery hint for bt_servant to send a voice message instead of text
    send_voice_message: bool
    voice_message_text: str


# Centralized capability registry and builders for feature help/boilerplate
# Capability class moved to bt_servant_engine.services.intents.simple_intents
# Imported above as part of simple_intents imports

# _capabilities moved to bt_servant_engine.services.intents.simple_intents (as get_capabilities)
# Imported above as _capabilities

def _capabilities_DEPRECATED() -> List[Capability]:
    """Return the list of user-facing capabilities with examples.

    Centralized here so greetings, help, and unsupported-function prompts stay in sync
    as new intents are added. Update this list to change feature messaging.
    """
    return [
        {
            "intent": IntentType.GET_PASSAGE_SUMMARY,
            "label": "Summarize a passage",
            "description": "Summarize books, chapters, or verse ranges.",
            "examples": [
                "Summarize Titus 1.",
                "Summarize Mark 1:1–8.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.CONSULT_FIA_RESOURCES,
            "label": "FIA process guidance",
            "description": "Use FIA resources to explain the workflow or apply steps to a passage.",
            "examples": [
                "What are the steps of the FIA process?",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.GET_TRANSLATION_HELPS,
            "label": "Translation helps",
            "description": "Point out typical translation challenges.",
            "examples": [
                "Translation challenges for John 1:1?",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.GET_PASSAGE_KEYWORDS,
            "label": "Keywords",
            "description": "List key terms in a passage.",
            "examples": [
                "Important words in Romans 1.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.RETRIEVE_SCRIPTURE,
            "label": "Show scripture text",
            "description": "Display the verse text for a selection.",
            "examples": [
                "Show John 3:16–18.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.LISTEN_TO_SCRIPTURE,
            "label": "Read aloud",
            "description": "Hear the passage as audio.",
            "examples": [
                "Read Romans 8:1–4 aloud.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.TRANSLATE_SCRIPTURE,
            "label": "Translate scripture",
            "description": "Translate a passage into another language.",
            "examples": [
                "Translate John 3:16 into Indonesian.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.SET_AGENTIC_STRENGTH,
            "label": "Adjust agentic strength",
            "description": "Tune how assertive the assistant should be (normal, low, or very low).",
            "examples": [
                "Set my agentic strength to low.",
            ],
            "include_in_boilerplate": False,
            "developer_only": True,
        },
        {
            "intent": IntentType.SET_RESPONSE_LANGUAGE,
            "label": "Set response language",
            "description": "Choose your preferred reply language.",
            "examples": [
                "Set my response language to Spanish.",
            ],
            "include_in_boilerplate": True,
        },
    ]


def build_boilerplate_message() -> str:
    """Build a concise 'what I can do' list with examples."""
    caps = [
        c
        for c in _capabilities()
        if c.get("include_in_boilerplate") and not c.get("developer_only", False)
    ]
    lines: list[str] = ["Here’s what I can do:"]
    for idx, c in enumerate(caps, start=1):
        example = c["examples"][0] if c.get("examples") else ""
        lines.append(f"{idx}) {c['label']} (e.g., '{example}')")
    lines.append("Which would you like me to do?")
    return "\n".join(lines)


def build_full_help_message() -> str:
    """Build a full help message with descriptions and examples for each capability."""
    lines: list[str] = ["Features:"]
    visible_caps = [c for c in _capabilities() if not c.get("developer_only", False)]
    for idx, c in enumerate(visible_caps, start=1):
        lines.append(f"{idx}. {c['label']}: {c['description']}")
        if c.get("examples"):
            for ex in c["examples"]:
                lines.append(f"   - Example: '{ex}'")
    return "\n".join(lines)


BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE = build_boilerplate_message()
FULL_HELP_MESSAGE = build_full_help_message()

FIRST_INTERACTION_MESSAGE = f"""
Hello! I am the BT Servant. This is our first conversation. Let's work together to understand and translate God's word!

{BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}
"""

UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT = f"""
# Identity

You are a part of a RAG bot system that assists Bible translators. You are one node in the decision/intent processing 
lang graph. Specifically, your job is to handle the perform-unsupported-function intent. This means the user is trying 
to perform an unsupported function.

# Instructions

Respond appropriately to the user's request to do something that you currently can't do. Leverage the 
user's message and the conversation history if needed. Make sure to always end your response with some version of  
the boiler plate available features message (see below).

<boiler_plate_available_features_message>
    {BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}
</boiler_plate_available_features_message>
"""

CONVERSE_AGENT_SYSTEM_PROMPT = f"""
# Identity

You are a part of a RAG bot system that assists Bible translators. You are one node in the decision/intent processing 
lang graph. Specifically, your job is to handle the converse-with-bt-servant intent by responding conversationally to 
the user based on the provided context.

# Instructions

If we are here in the decision graph, the converse-with-bt-servant intent has been detected. You will be provided with 
the user's most recent message and conversation history. Your job is to respond conversationally to the user. Unless it 
doesn't make sense to do so, aim to end your response with some version of  the boiler plate available features message 
(see below).

<boiler_plate_available_features_message>
    {BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}
</boiler_plate_available_features_message>
"""

HELP_AGENT_SYSTEM_PROMPT = f"""
# Identity

You are a part of a WhatsApp RAG bot system that assists Bible translators called BT Servant. You sole purpose is to 
provide help information about the BT Servant system. If this node has been hit, it means the system has already 
classified the user's most recent message as a desire to receive help or more information about the system. This is 
typically the result of them saying something like: 'help!' or 'tell me about yourself' or 'how does this work?' Thus, 
make sure to always provide some help, to the best of your abilities. Always provide help to the user.

# Instructions
You will be supplied with the user's most recent message and also past conversation history. Using this context,
provide the user with information detailing how the system works (the features of the BT Servant system). Use the
feature information below. End your response with a single question inviting the user to pick one capability
(for example: 'Which of these would you like me to do?').

<features_full_help_message>
{FULL_HELP_MESSAGE}
</features_full_help_message>

# Using prior history for better responses

Here are some guidelines for using history for better responses:
1. If you detect in conversation history that you've already said hello, there's no need to say it again.
2. If it doesn't make sense to say "hello!" to the user, based on their most recent message, there's no need to say 
'Hello!  I'm here to assist with Bible translation tasks' again.
"""


def start(state: Any) -> dict:
    """Handle first interaction greeting, otherwise no-op."""
    s = cast(BrainState, state)
    user_id = s["user_id"]
    if is_first_interaction(user_id):
        set_first_interaction(user_id, False)
        return {"responses": [
            {"intent": "first-interaction", "response": FIRST_INTERACTION_MESSAGE}]}
    return {}


def determine_intents(state: Any) -> dict:
    """Classify the user's transformed query into one or more intents."""
    s = cast(BrainState, state)
    query = s["transformed_query"]
    user_intents = _determine_intents_impl(open_ai_client, query)
    return {
        "user_intents": user_intents,
    }


def set_response_language(state: Any) -> dict:
    """Detect and persist the user's desired response language."""
    s = cast(BrainState, state)
    return set_response_language_impl(
        open_ai_client,
        s["user_id"],
        s["user_query"],
        s["user_chat_history"],
        supported_language_map,
        set_user_response_language,
    )


def set_agentic_strength(state: Any) -> dict:
    """Detect and persist the user's preferred agentic strength."""
    s = cast(BrainState, state)
    return set_agentic_strength_impl(
        open_ai_client,
        s["user_id"],
        s["user_query"],
        s["user_chat_history"],
        set_user_agentic_strength,
        config.LOG_PSEUDONYM_SECRET,
    )


def combine_responses(chat_history, latest_user_message, responses) -> str:
    """Ask OpenAI to synthesize multiple node responses into one coherent text."""
    return combine_responses_impl(
        open_ai_client,
        chat_history,
        latest_user_message,
        responses,
        _extract_cached_input_tokens,
    )


def translate_responses(state: Any) -> dict:
    """Translate or localize responses into the user's desired language."""
    s = cast(BrainState, state)
    raw_responses = [
        resp for resp in cast(list[dict], s["responses"]) if not resp.get("suppress_text_delivery")
    ]
    if not raw_responses:
        if bool(s.get("send_voice_message")):
            logger.info("[translate] skipping text translation because delivery is voice-only")
            return {"translated_responses": []}
        raise ValueError("no responses to translate. something bad happened. bailing out.")

    protected_items, normal_items = _partition_response_items(raw_responses)
    responses_for_translation = _build_translation_queue(s, protected_items, normal_items)
    if not responses_for_translation:
        if bool(s.get("send_voice_message")):
            logger.info("[translate] no text responses after queue assembly; voice-only delivery")
            return {"translated_responses": []}
        raise ValueError("no responses to translate. something bad happened. bailing out.")

    target_language, passthrough = _resolve_target_language(s, responses_for_translation)
    if passthrough is not None:
        return {"translated_responses": passthrough}
    assert target_language is not None

    agentic_strength = _resolve_agentic_strength(s)
    translated_responses = [
        _translate_or_localize_response(resp, target_language, agentic_strength)
        for resp in responses_for_translation
    ]
    return {"translated_responses": translated_responses}


def translate_text(
    response_text: str,
    target_language: str,
    *,
    agentic_strength: Optional[str] = None,
) -> str:
    """Translate a single text into the target ISO 639-1 language code."""
    return translate_text_impl(
        open_ai_client,
        response_text,
        target_language,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        agentic_strength=agentic_strength,
    )


# detect_language moved to bt_servant_engine.services.preprocessing


def determine_query_language(state: Any) -> dict:
    """Determine the language of the user's original query and set collection order."""
    s = cast(BrainState, state)
    query = s["user_query"]
    agentic_strength = _resolve_agentic_strength(s)
    query_language, stack_rank_collections = _determine_query_language_impl(
        open_ai_client, query, agentic_strength
    )
    return {
        "query_language": query_language,
        "stack_rank_collections": stack_rank_collections
    }


def preprocess_user_query(state: Any) -> dict:
    """Lightly clarify or correct the user's query using conversation history."""
    s = cast(BrainState, state)
    query = s["user_query"]
    chat_history = s["user_chat_history"]
    transformed_query, reason, changed = _preprocess_user_query_impl(open_ai_client, query, chat_history)
    return {
        "transformed_query": transformed_query
    }


def query_vector_db(state: Any) -> dict:
    """Query the vector DB (Chroma) across ranked collections and filter by relevance."""
    # pylint: disable=too-many-locals
    s = cast(BrainState, state)
    query = s["transformed_query"]
    stack_rank_collections = s["stack_rank_collections"]
    filtered_docs = []
    # this loop is the current implementation of the "stacked ranked" algorithm
    for collection_name in stack_rank_collections:
        logger.info("querying stack collection: %s", collection_name)
        db_collection = get_chroma_collection(collection_name)
        if not db_collection:
            logger.warning("collection %s was not found in chroma db.", collection_name)
            continue
        col = cast(Any, db_collection)
        results = col.query(
            query_texts=[query],
            n_results=TOP_K
        )
        docs = results["documents"]
        similarities = results["distances"]
        metadata = results["metadatas"]
        logger.info("\nquery: %s\n", query)
        logger.info("---")
        hits = 0
        for i in range(len(docs[0])):
            cosine_similarity = round(1 - similarities[0][i], 4)
            doc = docs[0][i]
            m = metadata[0][i]
            resource_name = m.get("name", "")
            source = m.get("source", "")
            logger.info("processing %s from %s.", resource_name, source)
            logger.info("Cosine Similarity: %s", cosine_similarity)
            logger.info("Metadata: %s", resource_name)
            logger.info("---")
            if cosine_similarity >= RELEVANCE_CUTOFF:
                hits += 1
                filtered_docs.append({
                    "collection_name": collection_name,
                    "resource_name": resource_name,
                    "source": source,
                    "document_text": doc
                })
        if hits > 0:
            logger.info("found %d hit(s) at stack collection: %s", hits, collection_name)

    return {
        "docs": filtered_docs
    }


# pylint: disable=too-many-locals
def query_open_ai(state: Any) -> dict:
    """Generate the final response text using RAG context and OpenAI."""
    s = cast(BrainState, state)
    docs = s["docs"]
    query = s["transformed_query"]
    chat_history = s["user_chat_history"]
    try:
        if len(docs) == 0:
            no_docs_msg = (f"Sorry, I couldn't find any information in my resources to service your request "
                           f"or command.\n\n{BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}")
            return {"responses": [
                {"intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE, "response": no_docs_msg}]}

        # build context from docs
        # context = "\n\n".join([item["doc"] for item in docs])
        context = json.dumps(docs, indent=2)
        logger.info("context passed to final node:\n\n%s", context)
        rag_context_message = "When answering my next query, use this additional" + \
            f"  context: {context}"
        chat_history_context_message = (f"Use this conversation history to understand the user's "
                                        f"current request only if needed: {json.dumps(chat_history)}")
        messages = cast(List[EasyInputMessageParam], [
            {
                "role": "developer",
                "content": rag_context_message
            },
            {
                "role": "developer",
                "content": "Focus only on the portion of the user's message requesting general Bible translation assistance. Ignore unrelated requests or passages mentioned elsewhere in the message.",
            },
            {
                "role": "developer",
                "content": chat_history_context_message
            },
            {
                "role": "user",
                "content": query
            }
        ])
        agentic_strength = _resolve_agentic_strength(s)
        model_name = _model_for_agentic_strength(agentic_strength, allow_low=False, allow_very_low=True)
        response = open_ai_client.responses.create(
            model=model_name,
            instructions=FINAL_RESPONSE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, messages)
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            it = getattr(usage, "input_tokens", None)
            ot = getattr(usage, "output_tokens", None)
            tt = getattr(usage, "total_tokens", None)
            if tt is None and (it is not None or ot is not None):
                tt = (it or 0) + (ot or 0)
            cit = _extract_cached_input_tokens(usage)
            add_tokens(it, ot, tt, model=model_name, cached_input_tokens=cit)
        bt_servant_response = response.output_text
        logger.info('response from openai: %s', bt_servant_response)
        logger.debug("%d characters returned from openAI", len(bt_servant_response))

        resource_list = ", ".join({
            f"{item.get('resource_name', 'unknown')} from {item.get('source', 'unknown')}"
            for item in docs
        })
        cascade_info = (
            f"bt servant used the following resources to generate its response: {resource_list}."
        )
        logger.info(cascade_info)

        return {"responses": [
            {"intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE, "response": bt_servant_response}]}
    except OpenAIError:
        logger.error("Error during OpenAI request", exc_info=True)
        error_msg = "I encountered some problems while trying to respond. Let Ian know about this one."
        return {"responses": [{"intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE, "response": error_msg}]}


def consult_fia_resources(state: Any) -> dict:
    """Answer FIA-specific questions using FIA collections and reference material."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return consult_fia_resources_impl(
        open_ai_client,
        s["transformed_query"],
        s["user_chat_history"],
        s.get("user_response_language"),
        s.get("query_language"),
        get_chroma_collection,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        agentic_strength,
    )


def chunk_message(state: Any) -> dict:
    """Chunk oversized responses to respect WhatsApp limits, via LLM or fallback."""
    s = cast(BrainState, state)
    responses = s["translated_responses"]
    text_to_chunk = responses[0]
    chunk_max = config.MAX_META_TEXT_LENGTH - 100
    chunks = chunk_message_impl(
        open_ai_client,
        text_to_chunk,
        _extract_cached_input_tokens,
        responses[1:],
        chunk_max,
    )
    return {"translated_responses": chunks}


def needs_chunking(state: BrainState) -> str:
    """Return next node key if chunking is required, otherwise finish."""
    return needs_chunking_impl(state)


def process_intents(state: Any) -> List[Hashable]:  # pylint: disable=too-many-branches
    """Map detected intents to the list of nodes to traverse."""
    s = cast(BrainState, state)
    user_intents = s["user_intents"]
    if not user_intents:
        raise ValueError("no intents found. something went very wrong.")

    nodes_to_traverse: List[Hashable] = []
    if IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE in user_intents:
        nodes_to_traverse.append("query_vector_db_node")
    if IntentType.CONSULT_FIA_RESOURCES in user_intents:
        nodes_to_traverse.append("consult_fia_resources_node")
    if IntentType.GET_PASSAGE_SUMMARY in user_intents:
        nodes_to_traverse.append("handle_get_passage_summary_node")
    if IntentType.GET_PASSAGE_KEYWORDS in user_intents:
        nodes_to_traverse.append("handle_get_passage_keywords_node")
    if IntentType.GET_TRANSLATION_HELPS in user_intents:
        nodes_to_traverse.append("handle_get_translation_helps_node")
    if IntentType.RETRIEVE_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_retrieve_scripture_node")
    if IntentType.LISTEN_TO_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_listen_to_scripture_node")
    if IntentType.SET_RESPONSE_LANGUAGE in user_intents:
        nodes_to_traverse.append("set_response_language_node")
    if IntentType.SET_AGENTIC_STRENGTH in user_intents:
        nodes_to_traverse.append("set_agentic_strength_node")
    if IntentType.PERFORM_UNSUPPORTED_FUNCTION in user_intents:
        nodes_to_traverse.append("handle_unsupported_function_node")
    if IntentType.RETRIEVE_SYSTEM_INFORMATION in user_intents:
        nodes_to_traverse.append("handle_system_information_request_node")
    if IntentType.CONVERSE_WITH_BT_SERVANT in user_intents:
        nodes_to_traverse.append("converse_with_bt_servant_node")
    if IntentType.RETRIEVE_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_retrieve_scripture_node")
    if IntentType.LISTEN_TO_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_listen_to_scripture_node")
    if IntentType.TRANSLATE_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_translate_scripture_node")

    return nodes_to_traverse


def handle_unsupported_function(state: Any) -> dict:
    """Generate a helpful response when the user requests unsupported functionality."""
    s = cast(BrainState, state)
    return handle_unsupported_function_impl(open_ai_client, s["user_query"], s["user_chat_history"])


def handle_system_information_request(state: Any) -> dict:
    """Provide help/about information for the BT Servant system."""
    s = cast(BrainState, state)
    return handle_system_information_request_impl(open_ai_client, s["user_query"], s["user_chat_history"])


def converse_with_bt_servant(state: Any) -> dict:
    """Respond conversationally to the user based on context and history."""
    s = cast(BrainState, state)
    return converse_with_bt_servant_impl(open_ai_client, s["user_query"], s["user_chat_history"])


def _book_patterns() -> list[tuple[str, str]]:
    """Return (canonical, regex) patterns to detect book mentions (ordered)."""
    return _book_patterns_impl(BSB_BOOK_MAP)


def _detect_mentioned_books(text: str) -> list[str]:
    """Detect canonical books mentioned in text, preserving order of appearance."""
    return _detect_mentioned_books_impl(text, BSB_BOOK_MAP)


def _choose_primary_book(text: str, candidates: list[str]) -> str | None:
    """Heuristic to pick a primary book when multiple are mentioned.

    Prefer the first mentioned that appears near chapter/verse digits; else None.
    """
    return _choose_primary_book_impl(text, candidates, BSB_BOOK_MAP)


def _resolve_selection_for_single_book(
    query: str,
    query_lang: str,
    focus_hint: str | None = None,
) -> tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]:
    """Parse and normalize a user query into a single canonical book and ranges."""
    return _resolve_selection_for_single_book_impl(
        client=open_ai_client,
        query=query,
        query_lang=query_lang,
        book_map=BSB_BOOK_MAP,
        detect_mentioned_books_fn=_detect_mentioned_books,
        translate_text_fn=translate_text,
        focus_hint=focus_hint,
    )


def handle_get_passage_summary(state: Any) -> dict:
    """Handle get-passage-summary: extract refs, retrieve verses, summarize."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return get_passage_summary_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        s.get("user_response_language"),
        agentic_strength,
    )


def handle_get_passage_keywords(state: Any) -> dict:
    """Handle get-passage-keywords: extract refs, retrieve keywords, and list them."""
    s = cast(BrainState, state)
    return get_passage_keywords_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
    )


def handle_get_translation_helps(state: Any) -> dict:
    """Generate focused translation helps guidance for a selected passage."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return get_translation_helps_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        _prepare_translation_helps,
        _build_translation_helps_context,
        _build_translation_helps_messages,
        agentic_strength,
    )


def handle_retrieve_scripture(state: Any) -> dict:
    """Handle retrieve-scripture with optional auto-translation."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return retrieve_scripture_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        s.get("user_response_language"),
        agentic_strength,
    )


def handle_listen_to_scripture(state: Any) -> dict:
    """Delegate to retrieve-scripture and request voice delivery."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return listen_to_scripture_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _reconstruct_structured_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        s.get("user_response_language"),
        agentic_strength,
    )

def handle_translate_scripture(state: Any) -> dict:
    """Handle translate-scripture: return verses translated into a target language."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return translate_scripture_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        s.get("user_response_language"),
        agentic_strength,
    )


def create_brain():
    """Assemble and compile the LangGraph for the BT Servant brain."""
    def wrap_node_with_timing(node_fn, node_name: str):  # type: ignore[no-untyped-def]
        def wrapped(state: Any) -> dict:
            trace_id = cast(dict, state).get("perf_trace_id")
            if trace_id:
                set_current_trace(cast(Optional[str], trace_id))
            with time_block(f"brain:{node_name}"):
                return node_fn(state)
        return wrapped
    def _make_state_graph(schema: Any) -> StateGraph[BrainState]:
        # Accept Any to satisfy IDE variance on schema param; schema is BrainState
        return StateGraph(schema)

    builder: StateGraph[BrainState] = _make_state_graph(BrainState)

    builder.add_node("start_node", wrap_node_with_timing(start, "start_node"))
    builder.add_node("determine_query_language_node", wrap_node_with_timing(determine_query_language, "determine_query_language_node"))
    builder.add_node("preprocess_user_query_node", wrap_node_with_timing(preprocess_user_query, "preprocess_user_query_node"))
    builder.add_node("determine_intents_node", wrap_node_with_timing(determine_intents, "determine_intents_node"))
    builder.add_node("set_response_language_node", wrap_node_with_timing(set_response_language, "set_response_language_node"))
    builder.add_node("set_agentic_strength_node", wrap_node_with_timing(set_agentic_strength, "set_agentic_strength_node"))
    builder.add_node("query_vector_db_node", wrap_node_with_timing(query_vector_db, "query_vector_db_node"))
    builder.add_node("query_open_ai_node", wrap_node_with_timing(query_open_ai, "query_open_ai_node"))
    builder.add_node("consult_fia_resources_node", wrap_node_with_timing(consult_fia_resources, "consult_fia_resources_node"))
    builder.add_node("chunk_message_node", wrap_node_with_timing(chunk_message, "chunk_message_node"))
    builder.add_node("handle_unsupported_function_node", wrap_node_with_timing(handle_unsupported_function, "handle_unsupported_function_node"))
    builder.add_node("handle_system_information_request_node", wrap_node_with_timing(handle_system_information_request, "handle_system_information_request_node"))
    builder.add_node("converse_with_bt_servant_node", wrap_node_with_timing(converse_with_bt_servant, "converse_with_bt_servant_node"))
    builder.add_node("handle_get_passage_summary_node", wrap_node_with_timing(handle_get_passage_summary, "handle_get_passage_summary_node"))
    builder.add_node("handle_get_passage_keywords_node", wrap_node_with_timing(handle_get_passage_keywords, "handle_get_passage_keywords_node"))
    builder.add_node("handle_get_translation_helps_node", wrap_node_with_timing(handle_get_translation_helps, "handle_get_translation_helps_node"))
    builder.add_node("handle_retrieve_scripture_node", wrap_node_with_timing(handle_retrieve_scripture, "handle_retrieve_scripture_node"))
    builder.add_node("handle_listen_to_scripture_node", wrap_node_with_timing(handle_listen_to_scripture, "handle_listen_to_scripture_node"))
    builder.add_node("handle_translate_scripture_node", wrap_node_with_timing(handle_translate_scripture, "handle_translate_scripture_node"))
    builder.add_node("translate_responses_node", wrap_node_with_timing(translate_responses, "translate_responses_node"), defer=True)

    builder.set_entry_point("start_node")
    builder.add_edge("start_node", "determine_query_language_node")
    builder.add_edge("determine_query_language_node", "preprocess_user_query_node")
    builder.add_edge("preprocess_user_query_node", "determine_intents_node")
    builder.add_conditional_edges(
        "determine_intents_node",
        process_intents
    )
    builder.add_edge("query_vector_db_node", "query_open_ai_node")
    builder.add_edge("set_response_language_node", "translate_responses_node")
    builder.add_edge("set_agentic_strength_node", "translate_responses_node")
    # After chunking, finish. Do not loop back to translate, which can recreate
    # the long message and trigger an infinite chunk cycle.

    builder.add_edge("handle_unsupported_function_node", "translate_responses_node")
    builder.add_edge("handle_system_information_request_node", "translate_responses_node")
    builder.add_edge("converse_with_bt_servant_node", "translate_responses_node")
    builder.add_edge("handle_get_passage_summary_node", "translate_responses_node")
    builder.add_edge("handle_get_passage_keywords_node", "translate_responses_node")
    builder.add_edge("handle_get_translation_helps_node", "translate_responses_node")
    builder.add_edge("handle_retrieve_scripture_node", "translate_responses_node")
    builder.add_edge("handle_listen_to_scripture_node", "translate_responses_node")
    builder.add_edge("handle_translate_scripture_node", "translate_responses_node")
    builder.add_edge("query_open_ai_node", "translate_responses_node")
    builder.add_edge("consult_fia_resources_node", "translate_responses_node")

    builder.add_conditional_edges(
        "translate_responses_node",
        needs_chunking
    )
    builder.set_finish_point("chunk_message_node")

    return builder.compile()
LANG_DETECTION_SAMPLE_CHARS = 100


def _sample_for_language_detection(text: str) -> str:
    """Return a short prefix ending at a whitespace boundary for detection."""
    return _sample_for_language_detection_impl(text, LANG_DETECTION_SAMPLE_CHARS)
