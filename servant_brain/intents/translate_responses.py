from __future__ import annotations

from typing import Any, Callable, Optional, cast

from logger import get_logger


logger = get_logger(__name__)


def translate_responses(
    state: Any,
    *,
    combine_responses: Callable[[Any, Any, list[dict]], str],
    is_protected_response_item: Callable[[dict], bool],
    reconstruct_structured_text: Callable[[dict | str, Optional[str]], str],
    detect_language: Callable[[str], str],
    translate_text: Callable[[str, str], str],
    supported_language_map: dict[str, str],
) -> dict:
    """Translate or pass-through responses based on user’s language preferences."""
    s = cast(dict[str, Any], state)
    uncombined = list(cast(list[dict], s["responses"]))

    protected_items: list[dict] = [i for i in uncombined if is_protected_response_item(i)]
    normal_items: list[dict] = [i for i in uncombined if not is_protected_response_item(i)]

    responses_for_translation: list[dict | str] = list(protected_items)
    if normal_items:
        responses_for_translation.append(
            combine_responses(s["user_chat_history"], s["user_query"], normal_items)
        )
    if not responses_for_translation:
        raise ValueError("no responses to translate. something bad happened. bailing out.")

    target_language: str
    if s.get("user_response_language"):
        target_language = cast(str, s["user_response_language"])
    else:
        target_language = cast(str, s["query_language"])
        if target_language == "Other":
            logger.warning('target language unknown. bailing out.')
            passthrough_texts: list[str] = [
                reconstruct_structured_text(resp, None)
                for resp in responses_for_translation
            ]
            passthrough_texts.append(
                "You haven't set your desired response language and I wasn't able to determine the language of your "
                "original message in order to match it. You can set your desired response language at any time by "
                "saying: Set my response language to Spanish, or Indonesian, or any of the supported languages: "
                f"{', '.join(supported_language_map.keys())}."
            )
            return {"translated_responses": passthrough_texts}

    translated_responses: list[str] = []
    for resp in responses_for_translation:
        if isinstance(resp, str):
            if detect_language(resp) != target_language:
                logger.info('preparing to translate to %s', target_language)
                translated_responses.append(translate_text(resp, target_language))
            else:
                logger.info('chunk translation not required. using chunk as is.')
                translated_responses.append(resp)
            continue

        body = cast(dict | str, resp.get("response"))
        if isinstance(body, dict) and isinstance(body.get("segments"), list):
            item_lang = cast(Optional[str], body.get("content_language"))
            header_is_translated = bool(body.get("header_is_translated"))
            localize_to = None if header_is_translated else (item_lang or target_language)
            final_text2 = reconstruct_structured_text(resp, localize_to)
            translated_responses.append(final_text2)
            continue

        translated_responses.append(str(body))
    return {
        "translated_responses": translated_responses
    }
