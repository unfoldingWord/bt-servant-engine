"""Helpers to build localized truncation notices for translation helps responses."""

from __future__ import annotations

from typing import Callable, Optional

from bt_servant_engine.core.logging import get_logger

logger = get_logger(__name__)

_SENTINEL_VERSE_LIMIT = "__VERSE_LIMIT__"
_SENTINEL_ORIGINAL_LABEL = "__ORIGINAL_LABEL__"
_SENTINEL_DELIVERED_LABEL = "__DELIVERED_LABEL__"

_BASE_NOTICE_WITH_ORIGINAL = (
    "I can share translation helps for up to {verse_limit} verses at a time. "
    "You asked for {original_label}, so this response covers {delivered_label}. "
    "Ask for the next section whenever you're ready to continue."
)

_BASE_NOTICE_NO_ORIGINAL = (
    "I can share translation helps for up to {verse_limit} verses at a time. "
    "This response covers {delivered_label}. "
    "Ask for the next section whenever you're ready to continue."
)

_CURATED_WITH_ORIGINAL: dict[str, str] = {
    "en": _BASE_NOTICE_WITH_ORIGINAL,
    "es": (
        "Puedo compartir ayudas de traducción para un máximo de {verse_limit} versículos a la vez. "
        "Pediste {original_label}, así que esta respuesta cubre {delivered_label}. "
        "Pide la siguiente sección cuando quieras continuar."
    ),
}

_CURATED_NO_ORIGINAL: dict[str, str] = {
    "en": _BASE_NOTICE_NO_ORIGINAL,
    "es": (
        "Puedo compartir ayudas de traducción para un máximo de {verse_limit} versículos a la vez. "
        "Esta respuesta cubre {delivered_label}. "
        "Pide la siguiente sección cuando quieras continuar."
    ),
}


def _apply_template(
    template: str, *, verse_limit: int, delivered_label: str, original_label: str | None
) -> str:
    data = {
        "verse_limit": verse_limit,
        "delivered_label": delivered_label,
        "original_label": original_label or "",
    }
    return template.format(**data)


def _translate_with_sentinels(
    base_template: str,
    *,
    language: str,
    translate_text_fn: Callable[[str, str], str],
) -> str | None:
    sentinel_template = (
        base_template.replace("{verse_limit}", _SENTINEL_VERSE_LIMIT)
        .replace("{original_label}", _SENTINEL_ORIGINAL_LABEL)
        .replace("{delivered_label}", _SENTINEL_DELIVERED_LABEL)
    )
    try:
        translated = translate_text_fn(sentinel_template, language)
    except Exception:  # pylint: disable=broad-except
        logger.exception(
            "[translation-notice] failed to translate truncation notice template for language=%s",
            language,
        )
        return None

    expected_tokens = [
        token
        for token in (_SENTINEL_VERSE_LIMIT, _SENTINEL_ORIGINAL_LABEL, _SENTINEL_DELIVERED_LABEL)
        if token in sentinel_template
    ]

    if not all(token in translated for token in expected_tokens):
        logger.warning(
            (
                "[translation-notice] translated template missing sentinel(s) for language=%s; "
                "tokens=%s"
            ),
            language,
            expected_tokens,
        )
        return None

    return translated


def build_truncation_notice(
    *,
    language: str,
    verse_limit: int,
    delivered_label: str,
    original_label: Optional[str],
    translate_text_fn: Optional[Callable[[str, str], str]] = None,
) -> str:
    """Return a localized truncation notice matching the target language.

    If a curated template exists for ``language`` it is used directly. Otherwise, the English
    source template is machine translated using sentinel tokens to guarantee that dynamic
    placeholders survive the process. If translation fails, the English template is returned
    as a safe fallback.
    """

    lang = (language or "en").lower()
    has_original = original_label is not None
    curated_map = _CURATED_WITH_ORIGINAL if has_original else _CURATED_NO_ORIGINAL
    base_template = _BASE_NOTICE_WITH_ORIGINAL if has_original else _BASE_NOTICE_NO_ORIGINAL
    if lang in curated_map:
        return _apply_template(
            curated_map[lang],
            verse_limit=verse_limit,
            delivered_label=delivered_label,
            original_label=original_label,
        )

    if translate_text_fn is None:
        logger.warning(
            (
                "[translation-notice] no translator provided for language=%s; "
                "falling back to English"
            ),
            lang,
        )
        return _apply_template(
            curated_map["en"],
            verse_limit=verse_limit,
            delivered_label=delivered_label,
            original_label=original_label,
        )

    translated_template = _translate_with_sentinels(
        base_template,
        language=lang,
        translate_text_fn=translate_text_fn,
    )
    if translated_template is None:
        return _apply_template(
            curated_map["en"],
            verse_limit=verse_limit,
            delivered_label=delivered_label,
            original_label=original_label,
        )

    notice_text = translated_template
    notice_text = notice_text.replace(_SENTINEL_VERSE_LIMIT, str(verse_limit))
    notice_text = notice_text.replace(_SENTINEL_DELIVERED_LABEL, delivered_label)
    if original_label is not None:
        notice_text = notice_text.replace(_SENTINEL_ORIGINAL_LABEL, original_label)
    else:
        notice_text = notice_text.replace(_SENTINEL_ORIGINAL_LABEL, "")
    return notice_text


__all__ = ["build_truncation_notice"]
