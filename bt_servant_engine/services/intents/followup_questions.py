"""Follow-up questions for intent handlers.

This module provides context-aware follow-up questions that are appended to intent
responses to encourage continued engagement. Each intent has localized follow-up
questions that are only added when no multi-intent follow-up has been added.
"""

from __future__ import annotations

from typing import Any

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger

logger = get_logger(__name__)

# Follow-up questions for each intent type, organized by language
INTENT_FOLLOWUP_QUESTIONS = {
    IntentType.RETRIEVE_SCRIPTURE: {
        "en": "Would you like to look up another Bible passage?",
        "es": "¿Le gustaría buscar otro pasaje bíblico?",
        "fr": "Souhaitez-vous rechercher un autre passage biblique?",
        "pt": "Gostaria de procurar outra passagem bíblica?",
        "sw": "Je, ungependa kutafuta kifungu kingine cha Biblia?",
        "ar": "هل تريد البحث عن مقطع آخر من الكتاب المقدس؟",
        "hi": "क्या आप बाइबिल का कोई अन्य अंश देखना चाहेंगे?",
        "zh": "您想查找另一段圣经经文吗？",
        "ru": "Хотите найти другой библейский отрывок?",
        "id": "Apakah Anda ingin mencari bagian Alkitab lainnya?",
        "nl": "Wilt u nog een Bijbelgedeelte opzoeken?",
    },
    IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE: {
        "en": "Do you have another Bible translation question?",
        "es": "¿Tiene otra pregunta sobre traducción bíblica?",
        "fr": "Avez-vous une autre question sur la traduction biblique?",
        "pt": "Você tem outra pergunta sobre tradução bíblica?",
        "sw": "Je, una swali lingine kuhusu tafsiri ya Biblia?",
        "ar": "هل لديك سؤال آخر عن ترجمة الكتاب المقدس؟",
        "hi": "क्या आपके पास बाइबिल अनुवाद के बारे में कोई अन्य प्रश्न है?",
        "zh": "您还有其他圣经翻译问题吗？",
        "ru": "У вас есть другой вопрос о переводе Библии?",
        "id": "Apakah Anda memiliki pertanyaan lain tentang terjemahan Alkitab?",
        "nl": "Heeft u nog een vraag over Bijbelvertaling?",
    },
    IntentType.CONSULT_FIA_RESOURCES: {
        "en": "Would you like to explore another biblical topic?",
        "es": "¿Le gustaría explorar otro tema bíblico?",
        "fr": "Souhaitez-vous explorer un autre sujet biblique?",
        "pt": "Gostaria de explorar outro tópico bíblico?",
        "sw": "Je, ungependa kuchunguza mada nyingine ya Biblia?",
        "ar": "هل تريد استكشاف موضوع كتابي آخر؟",
        "hi": "क्या आप किसी अन्य बाइबिल विषय का अन्वेषण करना चाहेंगे?",
        "zh": "您想探索另一个圣经主题吗？",
        "ru": "Хотите изучить другую библейскую тему?",
        "id": "Apakah Anda ingin menjelajahi topik Alkitab lainnya?",
        "nl": "Wilt u een ander Bijbels onderwerp verkennen?",
    },
    IntentType.GET_TRANSLATION_HELPS: {
        "en": "Do you need help with another translation question?",
        "es": "¿Necesita ayuda con otra pregunta de traducción?",
        "fr": "Avez-vous besoin d'aide avec une autre question de traduction?",
        "pt": "Precisa de ajuda com outra pergunta de tradução?",
        "sw": "Je, unahitaji msaada na swali lingine la tafsiri?",
        "ar": "هل تحتاج إلى مساعدة في سؤال ترجمة آخر؟",
        "hi": "क्या आपको किसी अन्य अनुवाद प्रश्न में सहायता चाहिए?",
        "zh": "您需要帮助解答另一个翻译问题吗？",
        "ru": "Вам нужна помощь с другим вопросом перевода?",
        "id": "Apakah Anda memerlukan bantuan dengan pertanyaan terjemahan lainnya?",
        "nl": "Heeft u hulp nodig bij een andere vertaalvraag?",
    },
    IntentType.SET_RESPONSE_LANGUAGE: {
        "en": "What else can I help you with today?",
        "es": "¿En qué más puedo ayudarle hoy?",
        "fr": "En quoi d'autre puis-je vous aider aujourd'hui?",
        "pt": "No que mais posso ajudá-lo hoje?",
        "sw": "Je, naweza kukusaidia na nini kingine leo?",
        "ar": "ما الذي يمكنني مساعدتك به اليوم أيضًا؟",
        "hi": "आज मैं आपकी और कैसे मदद कर सकता हूं?",
        "zh": "今天我还能帮您什么？",
        "ru": "Чем еще я могу вам помочь сегодня?",
        "id": "Apa lagi yang bisa saya bantu hari ini?",
        "nl": "Waarmee kan ik u vandaag nog meer helpen?",
    },
    IntentType.SET_AGENTIC_STRENGTH: {
        "en": "Is there anything else I can assist you with?",
        "es": "¿Hay algo más en lo que pueda ayudarle?",
        "fr": "Y a-t-il autre chose avec laquelle je peux vous aider?",
        "pt": "Há mais alguma coisa com a qual eu possa ajudá-lo?",
        "sw": "Je, kuna kitu kingine ninachoweza kukusaidia nacho?",
        "ar": "هل هناك أي شيء آخر يمكنني مساعدتك به؟",
        "hi": "क्या कुछ और है जिसमें मैं आपकी सहायता कर सकता हूं?",
        "zh": "还有什么我可以帮助您的吗？",
        "ru": "Есть ли еще что-нибудь, чем я могу вам помочь?",
        "id": "Apakah ada hal lain yang bisa saya bantu?",
        "nl": "Is er nog iets anders waarmee ik u kan helpen?",
    },
    IntentType.RETRIEVE_SYSTEM_INFORMATION: {
        "en": "Is there anything else I can assist you with?",
        "es": "¿Hay algo más en lo que pueda ayudarle?",
        "fr": "Y a-t-il autre chose avec laquelle je peux vous aider?",
        "pt": "Há mais alguma coisa com a qual eu possa ajudá-lo?",
        "sw": "Je, kuna kitu kingine ninachoweza kukusaidia nacho?",
        "ar": "هل هناك أي شيء آخر يمكنني مساعدتك به؟",
        "hi": "क्या कुछ और है जिसमें मैं आपकी सहायता कर सकता हूं?",
        "zh": "还有什么我可以帮助您的吗？",
        "ru": "Есть ли еще что-нибудь, чем я могу вам помочь?",
        "id": "Apakah ada hal lain yang bisa saya bantu?",
        "nl": "Is er nog iets anders waarmee ik u kan helpen?",
    },
    IntentType.GET_PASSAGE_SUMMARY: {
        "en": "Would you like a summary of another passage?",
        "es": "¿Le gustaría un resumen de otro pasaje?",
        "fr": "Souhaitez-vous un résumé d'un autre passage?",
        "pt": "Gostaria de um resumo de outra passagem?",
        "sw": "Je, ungependa muhtasari wa kifungu kingine?",
        "ar": "هل تريد ملخصًا لمقطع آخر؟",
        "hi": "क्या आप किसी अन्य अंश का सारांश चाहेंगे?",
        "zh": "您想要另一段经文的摘要吗？",
        "ru": "Хотите резюме другого отрывка?",
        "id": "Apakah Anda ingin ringkasan dari bagian lain?",
        "nl": "Wilt u een samenvatting van een ander gedeelte?",
    },
    IntentType.GET_PASSAGE_KEYWORDS: {
        "en": "Would you like keywords from another passage?",
        "es": "¿Le gustarían palabras clave de otro pasaje?",
        "fr": "Souhaitez-vous des mots-clés d'un autre passage?",
        "pt": "Gostaria de palavras-chave de outra passagem?",
        "sw": "Je, ungependa maneno muhimu kutoka kwa kifungu kingine?",
        "ar": "هل تريد كلمات رئيسية من مقطع آخر؟",
        "hi": "क्या आप किसी अन्य अंश से मुख्य शब्द चाहेंगे?",
        "zh": "您想要另一段经文的关键词吗？",
        "ru": "Хотите ключевые слова из другого отрывка?",
        "id": "Apakah Anda ingin kata kunci dari bagian lain?",
        "nl": "Wilt u trefwoorden uit een ander gedeelte?",
    },
    IntentType.LISTEN_TO_SCRIPTURE: {
        "en": "Would you like to hear another passage?",
        "es": "¿Le gustaría escuchar otro pasaje?",
        "fr": "Souhaitez-vous entendre un autre passage?",
        "pt": "Gostaria de ouvir outra passagem?",
        "sw": "Je, ungependa kusikia kifungu kingine?",
        "ar": "هل تريد سماع مقطع آخر؟",
        "hi": "क्या आप कोई अन्य अंश सुनना चाहेंगे?",
        "zh": "您想听另一段经文吗？",
        "ru": "Хотите послушать другой отрывок?",
        "id": "Apakah Anda ingin mendengar bagian lain?",
        "nl": "Wilt u nog een gedeelte horen?",
    },
    IntentType.TRANSLATE_SCRIPTURE: {
        "en": "Would you like help translating another passage?",
        "es": "¿Le gustaría ayuda para traducir otro pasaje?",
        "fr": "Souhaitez-vous de l'aide pour traduire un autre passage?",
        "pt": "Gostaria de ajuda para traduzir outra passagem?",
        "sw": "Je, ungependa msaada wa kutafsiri kifungu kingine?",
        "ar": "هل تريد المساعدة في ترجمة مقطع آخر؟",
        "hi": "क्या आप किसी अन्य अंश का अनुवाद करने में सहायता चाहेंगे?",
        "zh": "您需要帮助翻译另一段经文吗？",
        "ru": "Хотите помощь в переводе другого отрывка?",
        "id": "Apakah Anda ingin bantuan menerjemahkan bagian lain?",
        "nl": "Wilt u hulp bij het vertalen van een ander gedeelte?",
    },
}


def get_followup_for_intent(intent_type: IntentType, language: str) -> str:
    """Get the localized follow-up question for a given intent type.

    Args:
        intent_type: The intent type to get a follow-up question for
        language: The user's preferred language (e.g., 'en', 'es')

    Returns:
        A localized follow-up question string, or English fallback if language not found
    """
    # Get the language-specific questions for this intent
    questions = INTENT_FOLLOWUP_QUESTIONS.get(intent_type)
    if not questions:
        # Handle both IntentType enum and string intents (for test compatibility)
        intent_str = intent_type.value if hasattr(intent_type, "value") else str(intent_type)
        logger.warning(
            "[followup] No follow-up question defined for intent=%s, using generic fallback",
            intent_str,
        )
        # Generic fallback for intents without specific follow-ups
        generic_fallback = {
            "en": "Is there anything else I can help you with?",
            "es": "¿Hay algo más en lo que pueda ayudarle?",
            "fr": "Y a-t-il autre chose avec laquelle je peux vous aider?",
            "pt": "Há mais alguma coisa com a qual eu possa ajudá-lo?",
            "sw": "Je, kuna kitu kingine ninachoweza kukusaidia nacho?",
            "ar": "هل هناك أي شيء آخر يمكنني مساعدتك به؟",
            "hi": "क्या कुछ और है जिसमें मैं आपकी सहायता कर सकता हूं?",
            "zh": "还有什么我可以帮助您的吗？",
            "ru": "Есть ли еще что-нибудь, чем я могу вам помочь?",
            "id": "Apakah ada hal lain yang bisa saya bantu?",
            "nl": "Is er nog iets anders waarmee ik u kan helpen?",
        }
        return generic_fallback.get(language, generic_fallback["en"])

    # Get the localized question, fallback to English if language not available
    question = questions.get(language)
    if not question:
        # Handle both IntentType enum and string intents (for test compatibility)
        intent_str = intent_type.value if hasattr(intent_type, "value") else str(intent_type)
        logger.debug(
            "[followup] No translation for intent=%s language=%s, using English",
            intent_str,
            language,
        )
        question = questions.get("en", "Is there anything else I can help you with?")

    return question


def add_followup_if_needed(response: str, state: Any, intent_type: IntentType) -> str:
    """Add a follow-up question to a response if one hasn't already been added.

    This function checks the state's followup_question_added flag. If it's False,
    it appends the appropriate follow-up question and sets the flag to True.
    If it's already True (e.g., multi-intent follow-up was added), it returns
    the response unchanged.

    Args:
        response: The intent handler's response text
        state: The BrainState dictionary
        intent_type: The intent type being handled

    Returns:
        The response with follow-up appended (if appropriate), or unchanged
    """
    # Check if a follow-up has already been added (e.g., by multi-intent logic)
    if state.get("followup_question_added", False):
        # Handle both IntentType enum and string intents (for test compatibility)
        intent_str = intent_type.value if hasattr(intent_type, "value") else str(intent_type)
        logger.debug(
            "[followup] Follow-up already added for intent=%s, skipping",
            intent_str,
        )
        return response

    # Get the user's language preference
    language = state.get("user_response_language", "en")
    if not language:
        language = "en"

    # Get the appropriate follow-up question
    followup = get_followup_for_intent(intent_type, language)

    # Append with double newlines for spacing
    enhanced_response = f"{response}\n\n{followup}"

    # Mark that we've added a follow-up
    state["followup_question_added"] = True

    # Handle both IntentType enum and string intents (for test compatibility)
    intent_str = intent_type.value if hasattr(intent_type, "value") else str(intent_type)
    logger.info(
        "[followup] Added follow-up for intent=%s language=%s",
        intent_str,
        language,
    )

    return enhanced_response


__all__ = [
    "INTENT_FOLLOWUP_QUESTIONS",
    "get_followup_for_intent",
    "add_followup_if_needed",
]
