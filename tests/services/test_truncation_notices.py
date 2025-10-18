"""Tests for localized truncation notice generation."""

from __future__ import annotations

from bt_servant_engine.services.truncation_notices import build_truncation_notice


def test_curated_spanish_notice_with_original() -> None:
    """Spanish template keeps placeholders in natural positions."""
    notice = build_truncation_notice(
        language="es",
        verse_limit=5,
        delivered_label="Génesis 1:1-5",
        original_label="Génesis 1:1-11",
    )
    assert "5" in notice
    assert "Génesis 1:1-11" in notice
    assert "Génesis 1:1-5" in notice


def test_curated_english_notice_without_original() -> None:
    """English template without original selection renders correctly."""
    notice = build_truncation_notice(
        language="en",
        verse_limit=7,
        delivered_label="Genesis 1:1-7",
        original_label=None,
    )
    assert "7" in notice
    assert "Genesis 1:1-7" in notice
    assert "You asked for" not in notice


def test_fallback_translation_uses_sentinels() -> None:
    """Fallback translation replaces sentinel tokens with supplied values."""

    def fake_translate(_text: str, language: str) -> str:
        assert language == "fr"
        # Simulate a translated string that keeps sentinel positions.
        return (
            "Je peux fournir des aides à la traduction pour jusqu'à __VERSE_LIMIT__ versets. "
            "Vous avez demandé __ORIGINAL_LABEL__, cette réponse couvre __DELIVERED_LABEL__. "
            "Demandez la section suivante quand vous êtes prêt à continuer."
        )

    notice = build_truncation_notice(
        language="fr",
        verse_limit=3,
        delivered_label="Genèse 1:1-3",
        original_label="Genèse 1:1-9",
        translate_text_fn=fake_translate,
    )
    assert "3" in notice
    assert "Genèse 1:1-9" in notice
    assert "Genèse 1:1-3" in notice


def test_fallback_to_english_when_sentinels_missing() -> None:
    """If the translation drops sentinels, default to English template."""

    def bad_translate(_text: str, _language: str) -> str:
        return "traducción inválida"

    notice = build_truncation_notice(
        language="de",
        verse_limit=2,
        delivered_label="Genesis 1:1-2",
        original_label="Genesis 1:1-8",
        translate_text_fn=bad_translate,
    )
    assert "I can share translation helps" in notice
    assert "Genesis 1:1-8" in notice
