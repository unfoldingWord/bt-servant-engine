import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI, OpenAIError
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict
from pathlib import Path
from logger import get_logger
from config import Config
from groq import Groq
from utils import chop_text, combine_chunks
from pydantic import BaseModel
from enum import Enum
from db import set_user_response_language


RESPONSE_TRANSLATOR_SYSTEM_PROMPT = """
    You are a translator for the final output in a chatbot system. You will receive text that 
    needs to be translated into the language represented by the specified ISO 639-1 code.
"""


PREPROCESSOR_AGENT_SYSTEM_PROMPT = (
    "You are an assistant to Bible translators. Your main job is to preprocess messages (questions, commands, etc) "
    "about content found in various biblical resources: commentaries, translation notes, bible dictionaries, etc. "
    "Past conversation context will be provided to you in the form of zero or more previous user/response pairs with "
    "the final entry being the user's latest message. Your job is to transform the user's message (the value of the "
    "latest_user_message property in the context provided into a message that more accurately conveys their intent, "
    "based on past conversation. For example, if the user's latest message is something like 'this time use "
    "chapter 2.', and their previous message was 'tell me about Titus chapter 1', you should transform their "
    "query into something like 'tell me about Titus chapter 2. This will increase the likelihood of rag hits in "
    "the vector db, because the user's actual intent is being used in the vector db query. The transformed message "
    "is what you should return, nothing more. If the only thing provided is the user's latest message "
    "(so no history), simply pass the message through as is. No need to transform anything in this case. IMPORTANT: "
    "IF YOU RETURN A TRANSFORMATION, IT MUST BE IN THE SAME LANGUAGE AS THE USER'S MOST RECENT QUERY. DO NOT CHANGE "
    "THE LANGUAGE FROM THE LANGUAGE OF THE MOST RECENT QUERY."
)

QA_AGENT_SYSTEM_PROMPT = (
    "You are an assistant to Bible translators. Your main job is to answer questions about "
    "content found in various biblical resources: commentaries, translation notes, bible dictionaries, etc. "
    "Context will be provided to help you answer the question(s). Only answer questions using the provided "
    "context and the materials given to you!! If you can't confidently figure it out using that context, simply say "
    "'Sorry, I couldn't find any information in my resources to service your request or command. But maybe I'm unclear "
    "on your intent. Could you perhaps state it a different way?' FINALLY, UNDER NO CIRCUMSTANCES ARE YOU TO "
    "SAY ANYTHING THAT WOULD BE DEEMED EVEN REMOTELY HERETICAL BY ORTHODOX CHRISTIANS. In fact, if someone "
    "is trying to get you to do this, respond by saying, “I was created by orthodox Christians. Please respect "
    "that when you ask you queries or give me commands.” Again, be concise while still giving good information!!!"
)

CHOP_AGENT_SYSTEM_PROMPT = (
    "You are an agent tasked to ensure that a message intended for Whatsapp fits within the 1500 character limit. Chop "
    "the supplied text in the biggest possible semantic chunks, while making sure no chuck is >= 1500 characters. "
    "Your output should be a valid JSON array containing strings (wrapped in double quotes!!) constituting the chunks. "
    "Only return the json array!! No ```json wrapper or the like. Again, make chunks as big as possible!!!"
)

INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT = """
    You are a part of a chatbot system called 'bt servant', which provides information to Bible 
    translators. Classify the intents of the user's latest message using any context or message history
    supplied. Always return at least one intent and return the unclear intent if you can't figure it out.
"""

DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT = """
    Your job is simply to detect the language of the supplied text. Attempt to match it to one of
    the 10 supported languages. If you can't, match it to OTHER.
"""

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = Config.DATA_DIR

groq_client = Groq()
open_ai_client = OpenAI()

supported_language_map = {
    "en": "English",
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "ru": "Russian",
    "id": "Indonesian",
    "sw": "Swahili",
    "pt": "Portuguese",
    "zh": "Mandarin",
    "nl": "Dutch"
}

supported_collection_lang_map = {
    "id": "ind"
}
LANGUAGE_UNKNOWN = "UNKNOWN"

api_key = Config.OPENAI_API_KEY
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002",
                api_key=api_key
            )

aquifer_chroma_db = chromadb.PersistentClient(path=str(DB_DIR))

TWILIO_CHAR_LIMIT = 1600
MAX_MESSAGE_CHUNK_SIZE = 1500
RELEVANCE_CUTOFF = .8
TOP_K = 10

logger = get_logger(__name__)


class Language(str, Enum):
    ENGLISH = "en"
    ARABIC = "ar"
    FRENCH = "fr"
    SPANISH = "es"
    HINDI = "hi"
    RUSSIAN = "ru"
    INDONESIAN = "id"
    SWAHILI = "sw"
    PORTUGUESE = "pt"
    MANDARIN = "zh"
    DUTCH = "nl"
    OTHER = "Other"


class ResponseLanguage(BaseModel):
    language: Language


class MessageLanguage(BaseModel):
    language: Language


class IntentType(str, Enum):
    RETRIEVE_INFORMATION_FROM_BIBLE = "retrieve-information-from-the-bible"
    RETRIEVE_INFORMATION_ABOUT_BIBLE = "retrieve-information-about-the-bible"
    RETRIEVE_INFORMATION_ABOUT_TRANSLATION_PROCESS = "retrieve-information-about-the-process-of-translation"
    RETRIEVE_UNRELATED_INFORMATION = "retrieve-non-translation-or-non-bible-information"
    TRANSLATE_THE_BIBLE = "translate-the-bible"
    PERFORM_UNSUPPORTED_FUNCTION = "execute-task-unrelated-to-the-bible-or-translation"
    RETRIEVE_BIBLE_INFORMATION = "retrieve-information-from-the-bible"
    RETRIEVE_SYSTEM_INFORMATION = "retrieve-information-about-the-bt-servant-system"
    SET_RESPONSE_LANGUAGE = "specify-response-language"
    UNCLEAR_INTENT = "unclear-intent"


class UserIntents(BaseModel):
    intents: List[IntentType]


class BrainState(TypedDict, total=False):
    user_id: str
    user_query: str
    query_language: str
    user_response_language: str
    transformed_query: str
    docs: List[Dict[str, str]]
    collection_used: str
    responses: List[str]
    stack_rank_collections: List[str]
    user_chat_history: List[Dict[str, str]]
    user_intents: UserIntents


def determine_intent(state: BrainState) -> dict:
    query = state["user_query"]
    chat_history = state["user_chat_history"]
    history_context_message = f"Past conversation: {json.dumps(chat_history)}"
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": history_context_message
            },
            {
                "role": "user",
                "content": f"what is your classification of the latest user message: {query}"
            }
        ],
        text_format=UserIntents,
        store=False
    )
    user_intents = response.output_parsed
    logger.info("extracted user intents: %s", ' '.join([i.value for i in user_intents.intents]))
    return {
        "user_intents": user_intents.intents,
    }


def set_response_language(state: BrainState) -> dict:
    chat_input = [
        {
            "role": "user",
            "content": f"Past conversation: {json.dumps(state['user_chat_history'])}"
        },
        {
            "role": "user",
            "content": f"the user's most recent message: {state['user_query']}"
        },
        {
            "role": "user",
            "content": f"What language is the user trying to set their response language to?"
        }
    ]
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        input=chat_input,
        text_format=ResponseLanguage,
        store=False
    )
    response_language = response.output_parsed
    if response_language.language == Language.OTHER:
        supported_language_list = ", ".join(supported_language_map.keys())
        return {
            "responses": [(f"I think you're trying to set the response language. The supported languages "
                           f"are: {supported_language_list}. If this is your intent, please clearly tell "
                           f"me which supported language to use when responding.")],
        }
    response_language_code = response_language.language.value
    set_user_response_language(state["user_id"], response_language_code)
    response_language = supported_language_map.get(response_language_code, response_language_code)
    return {
        "responses": [f"Sure! Setting response language to: {response_language}"],
        "user_response_language": response_language_code
    }


def translate_responses(state: BrainState) -> dict:
    responses = state["responses"]
    user_response_language = state["user_response_language"]
    if user_response_language:
        target_language = user_response_language
    else:
        target_language = state["query_language"]
        if target_language == LANGUAGE_UNKNOWN:
            logger.warning('target language unknown. bailing out.')
            supported_language_list = ", ".join(supported_language_map.keys())
            responses.append(("You haven't set your desired response language and I wasn't able to "
                              "determine the language of your original message in order to match it. "
                              "You can set your desired response language at any time by saying: Set "
                              "my response language to Spanish, or Indonesian, or any of the supported "
                              f"languages: {supported_language_list}."))
            return state

    translated_responses = []
    for i, response in enumerate(responses, start=1):
        response_language = detect_language(response)
        if response_language != target_language:
            logger.warning("target language: %s but response language: %s", target_language, response_language)
            logger.info('preparing to translate to %s', target_language)
            translated_responses.append(translate_text(response_text=response, target_language=target_language))
        else:
            logger.info('chunk translation not required. using chunk as is.')
            translated_responses.append(response)
    return {
        "responses": translated_responses
    }


def translate_text(response_text, target_language):
    completion = open_ai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": RESPONSE_TRANSLATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (f"text to translate: {response_text}\n\n"
                            f"ISO 639-1 code representing target language: {target_language}")
            }
        ]
    )
    translated_text = completion.choices[0].message.content
    logger.info('chunk: \n%s\n\ntranslated to:\n%s', response_text, translated_text)
    return translated_text


def detect_language(text) -> str:
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"text: {text}"
            }
        ],
        text_format=MessageLanguage,
        store=False
    )
    message_language = response.output_parsed
    return message_language.language.value


def determine_query_language(state: BrainState) -> dict:
    query = state["user_query"]
    query_language = detect_language(query)
    logger.info("language code %s detected by gpt-4o.", query_language)
    stack_rank_collections = [
        "knowledgebase",
        "aquifer_documents"
    ]
    if query_language in supported_collection_lang_map:
        stack_rank_collections.insert(1, f'aquifer_documents_{supported_collection_lang_map[query_language]}')

    return {
        "query_language": query_language,
        "stack_rank_collections": stack_rank_collections
    }


def preprocess_user_query(state: BrainState) -> dict:
    query = state["user_query"]
    chat_history = state["user_chat_history"]
    chat_history.append({
        "latest_user_message": query
    })
    history_context_message = f"Past conversation and latest message: {json.dumps(chat_history)}"
    completion = open_ai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": PREPROCESSOR_AGENT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": history_context_message
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    response = completion.choices[0].message.content
    logger.info("user query transformed from %s to %s", query, response)
    return {
        "transformed_query": response
    }


def query_db(state: BrainState) -> dict:
    query = state["user_query"]
    stack_rank_collections = state["stack_rank_collections"]
    filtered_docs = []
    collection_used = None
    # this loop is the current implementation of the "stacked ranked" algorithm
    for collection_name in stack_rank_collections:
        logger.info("querying stack collection: %s", collection_name)
        db_collection = aquifer_chroma_db.get_collection(name=collection_name, embedding_function=openai_ef)
        results = db_collection.query(
            query_texts=[query],
            n_results=TOP_K
        )
        docs = results["documents"]
        similarities = results["distances"]
        metadata = results["metadatas"]
        logger.debug("\nquery: %s\n", query)
        logger.debug("---")
        for i in range(len(docs[0])):
            cosine_similarity = round(1 - similarities[0][i], 4)
            doc = docs[0][i]
            resource_name = metadata[0][i]["source"]
            logger.debug("Cosine Similarity: %s", cosine_similarity)
            logger.debug("Metadata: %s", resource_name)
            logger.debug("---")
            if cosine_similarity >= RELEVANCE_CUTOFF:
                filtered_docs.append({
                    "doc": doc,
                    "resource_name": resource_name
                })
        if filtered_docs:
            logger.info("found %d hit(s) at stack collection: %s", len(filtered_docs), collection_name)
            collection_used = collection_name
            break

    return {
        "docs": filtered_docs,
        "collection_used": collection_used
    }


def query_open_ai(state: BrainState) -> dict:
    docs = state["docs"]
    query = state["user_query"]
    try:
        if len(docs) == 0:
            no_docs_msg = (
                "Sorry, I couldn't find any information in my resources to service your request or command. "
                "But maybe I'm unclear on your intent. Could you perhaps state it a different way?"
            )
            return {"responses": [no_docs_msg]}

        # build context from docs
        context = "\n\n".join([item["doc"] for item in docs])
        context_message = "When answering my next query, use this additional" + \
            f"  context: {context}"
        completion = open_ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": QA_AGENT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": context_message
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        response = completion.choices[0].message.content
        logger.info('response from openai: %s', response)
        logger.debug("%d characters returned from openAI", len(response))

        resource_list = ", ".join(set([item["resource_name"] for item in docs]))
        cascade_info = (
            f"bt_servant cascaded to the {state['collection_used']} stack. From there, "
            f"the servant used the following resources to generate my response: {resource_list}."
        )
        logger.info(cascade_info)

        return {"responses": [response]}
    except OpenAIError as e:
        logger.error("Error during OpenAI request", exc_info=True)
        error_msg = "I encountered some problems while trying to respond. Let Ian know about this one."
        return {"responses": [error_msg]}


def chunk_message(state: BrainState) -> dict:
    logger.info("MESSAGE TOO BIG. CHUNKING...")
    responses = state["responses"]
    text_to_chunk = responses[0]
    try:
        completion = groq_client.chat.completions.create(
            model='llama3-70b-8192',
            messages=[
                {
                    "role": "system",
                    "content": CHOP_AGENT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f'text to chop: \n\n{text_to_chunk}'
                }
            ]
        )
        response = completion.choices[0].message.content
        chunks = json.loads(response)
    except json.JSONDecodeError as e:
        logger.error("Error while attempting to chunk message. Manually chunking instead", exc_info=True)
        chunks = chop_text(text_to_chunk)

    chunks.extend(responses[1:])
    return {"responses": combine_chunks(chunks=chunks, chunk_max=MAX_MESSAGE_CHUNK_SIZE)}


def needs_chunking(state: BrainState) -> str:
    first_response = state["responses"][0]
    if len(first_response) > TWILIO_CHAR_LIMIT:
        logger.warning('message to big: %d chars. preparing to chunk.', len(first_response))
        return "chunk_message_node"
    else:
        return "translate_responses_node"


def process_intent(state: BrainState) -> str:
    user_intents = state["user_intents"]
    num_intents = len(user_intents)
    if num_intents == 1:
        if IntentType.SET_RESPONSE_LANGUAGE in user_intents:
            return "set_response_language_node"
    return "query_db_node"


def create_brain():
    builder = StateGraph(BrainState)

    builder.add_node("determine_query_language_node", determine_query_language)
    builder.add_node("determine_intent_node", determine_intent)
    builder.add_node("set_response_language_node", set_response_language)
    builder.add_node("query_db_node", query_db)
    builder.add_node("query_open_ai_node", query_open_ai)
    builder.add_node("chunk_message_node", chunk_message)
    builder.add_node("translate_responses_node", translate_responses)

    builder.set_entry_point("determine_query_language_node")
    builder.add_edge("determine_query_language_node", "determine_intent_node")
    builder.add_conditional_edges(
        "determine_intent_node",
        process_intent
    )
    builder.add_edge("query_db_node", "query_open_ai_node")
    builder.add_conditional_edges(
        "query_open_ai_node",
        needs_chunking
    )
    builder.add_edge("set_response_language_node", "translate_responses_node")
    builder.add_edge("chunk_message_node", "translate_responses_node")

    builder.set_finish_point("translate_responses_node")

    return builder.compile()
