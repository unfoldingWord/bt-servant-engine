import json
from openai import OpenAI, OpenAIError
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict
from pathlib import Path
from logger import get_logger
from config import config
from groq import Groq
from utils import chop_text, combine_chunks
from pydantic import BaseModel
from enum import Enum
from db import set_user_response_language, get_chroma_collection

HELP_AGENT_SYSTEM_PROMPT = """
# Identity

You are a part of a RAG bot system that assists Bible translators. 

# Instructions

You sole purpose is to 
provide help information about the system, using the help documentation context given to you. You 
are only to answer using the documentation given to you. If you can't, simply say: 'I think you're 
trying to ask a question about the BT Servant system. But I am unable to answer your question. Maybe 
restate your question and try again.' You will also be passed past conversation context. Leverage this 
only if needed -- only if it helps you answer the user's question about the system.

"""

HELP_DOCS = """
I am the Bible Translation Servant. I serve Bible translators. My main job is to answer questions 
about content found in various biblical resources: commentaries, translation notes, bible dictionaries, etc. 
I will attempt to answer any question you have, or follow any commands, using the resources at my 
disposal. If I can't find the answer, or follow the command, based on my available resources, then 
I will tell you. 

Currently supported functions:
1. Answer questions about the Bible or the Bible translation process.
2. Execute commands related to the Bible or the Bible translation process.
3. Respond in a supported language. Simply say "respond in Swahili" or "respond in Dutch".

Currently supported languages: English, Arabic, French, Spanish, Hindi, Russian, Indonesian, 
Swahili, Portuguese, Mandarin, and Dutch.
"""

RESPONSE_TRANSLATOR_SYSTEM_PROMPT = """
    You are a translator for the final output in a chatbot system. You will receive text that 
    needs to be translated into the language represented by the specified ISO 639-1 code.
"""


PREPROCESSOR_AGENT_SYSTEM_PROMPT = """
# Identity

You are a preprocessor agent/node in a retrieval augmented generation (RAG) pipeline. 

# Instructions

Use past conversation context, 
if supplied and applicable, to disambiguate or clarify the intent or meaning of the user's current message. Change 
as little as possible. Change nothing unless necessary. If the intent of the user's message is already clear, 
change nothing. Never greatly expand the user's current message. Changes should be small or none. Feel free to fix 
obvious spelling mistakes or errors, but not logic errors like incorrect books of the Bible. Return the clarified 
message and the reasons for clarifying or reasons for not changing anything. Examples below.

# Examples

## Example 1

<past_conversation>
    user_message: Summarize the book of Titus.
    assistant_response: The book of titus is about...
</past_conversation>

<current_message>
    user_message: Now Mark
</current_message>

<assistant_response>
    new_message: Now Summarize the book of Mark.
    reason_for_decision: Based on previous context, the user wants the system to do the same thing, but this time 
                         with Mark.
    message_changed: True
</assistant_response>
    
## Example 2

<past_conversation>
    user_message: What is going on in 1 Peter 3:7?
    assistant_response: Peter is instructing Christian husbands to be loving to their wives.
</past_conversation>

<current_message>
    user_message: Summarize Mark 3:1
</current_message>
    
<assistant_response>
    new_message: Summarize Mark 3:1.
    reason_for_decision: Nothing was changed. The user's current command has nothing to do with past context and
                         is fine as is.
    message_changed: False
</assistant_response>

## Example 3

<past_conversation>
    user_message: Explain John 1:1
    assistant_response: John claims that Jesus, the Word, existed in the beginning with God the Father.
</past_conversation>

<current_message>
    user_message: Explain Johhn 1:3
</current_message>
    
<assistant_response>
    new_message: Explain John 1:3.
    reason_for_decision: The word 'John' was misspelled in the message.
    message_changed: True
</assistant_response>
"""


FINAL_RESPONSE_AGENT_SYSTEM_PROMPT = """
You are an assistant to Bible translators. Your main job is to answer questions about content found in various biblical 
resources: commentaries, translation notes, bible dictionaries, and various resources like FIA. In addition to answering
questions, you may be called upon to: summarize the data from resources, transform the data from resources (like
explaining it a 5-year old level, etc, and interact with the resources in all kinds of ways. All this is a part of your 
responsibilities. Context from resources (RAG results) will be provided to help you answer the question(s). Only answer 
questions using the provided context from resources!!! If you can't confidently figure it out using that context, 
simply say 'Sorry, I couldn't find any information in my resources to service your request or command. Currently, I only 
have resources loaded from Titus and Mark, and a few other resources related to producing faithful translations. But 
maybe I'm unclear on your intent. Could you perhaps state it a different way?' You will also be given the past 
conversation history. Use this to understand the user's current message or query if necessary. If the past conversation 
history is not relevant to the user's current message, just ignore it. FINALLY, UNDER NO CIRCUMSTANCES ARE YOU TO SAY 
ANYTHING THAT WOULD BE DEEMED EVEN REMOTELY HERETICAL BY ORTHODOX CHRISTIANS. If you can't do what the user is asking 
because your response would be heretical, explain to the user why you cannot comply with their reqeust or command.
"""

CHOP_AGENT_SYSTEM_PROMPT = (
    "You are an agent tasked to ensure that a message intended for Whatsapp fits within the 1500 character limit. Chop "
    "the supplied text in the biggest possible semantic chunks, while making sure no chuck is >= 1500 characters. "
    "Your output should be a valid JSON array containing strings (wrapped in double quotes!!) constituting the chunks. "
    "Only return the json array!! No ```json wrapper or the like. Again, make chunks as big as possible!!!"
)

INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT = """
You are a node in a chatbot system called “BT Servant”, which provides intelligent assistance to Bible translators. Your 
job is to classify the **intent(s)** of the user’s latest message. Always return **at least one** intent from the 
approved list. However, if more than one intent is found, make sure to return those as well. If you're unsure, return 
`unclear-intent`. If the user is asking for something outside the scope of the Bible, Bible translation, the Bible 
translation process, or one of the resources stored in the system (ex. Translation Notes, FIA resources, the Bible, 
Translation Words, Greek or Hebrew resources, commentaries, Bible dictionaries, etc.), or something outside system 
capabilities (defined by the various intents), return the `perform-unsupported-function` intent.

You MUST always return at least one intent. You MUST choose one or more intents from the following five intent types:

<intents>
  <intent name="get-bible-translation-assistance">
    The user is asking for help with Bible translation — including understanding meaning; finding source verses; 
    clarifying language issues; consulting translation resources (ex. Translation Notes, FIA, the Bible, etc); receiving
    explanation of resources; interacting with resource content; asking for transformations of resource content 
    (ex. summaries of resource portions, biblical content, etc); or how to handle specific words, phrases, 
    or translation challenges.
  </intent>
  <intent name="set-response-language">
    The user wants to change the language in which the system responds. They might ask for responses in 
    Spanish, French, Arabic, etc.
  </intent>
  <intent name="retrieve-system-information">
    The user wants information about the BT Servant system itself — how it works, where it gets data, uptime, 
    example questions, supported languages, features, or current system configuration (like the documents currently 
    stored in the ChromaDB (vector database).
  </intent>
  <intent name="perform-unsupported-function">
    The user is asking BT Servant to do something outside the scope of Bible translation help, interacting with the 
    resources in the vector database, or system diagnostics. For example, telling jokes, setting timers, 
    summarizing current news, or anything else COMPLETELY UNRELATED to what BT Servant can do.
  </intent>
  <intent name="unclear-intent">
    The user's request is too ambiguous or vague to classify confidently into one of the categories above.
  </intent>
</intents>

Here are a few examples to guide you:

<examples>
  <example>
    <message>"What is the best way to translate the word 'faith' in this passage?"</message>
    <intent>get-bible-translation-assistance</intent>
  </example>
  <example>
    <message>"What is the fourth step of the FIA process?"</message>
    <intent>get-bible-translation-assistance</intent>
  </example>
  <example>
    <message>"Explain the FIA process to me like I'm a three year old."</message>
    <intent>get-bible-translation-assistance</intent>
  </example>
  <example>
    <message>"What is a FIA process in Mark."</message>
    <intent>get-bible-translation-assistance</intent>
  </example>
  <example>
    <message>"Summarize Mark 3."</message>
    <intent>get-bible-translation-assistance</intent>
  </example>
  <example>
    <message>"Summarize Titus 3:4"</message>
    <intent>get-bible-translation-assistance</intent>
  </example>
  <example>
    <message>"Can you reply to me in French from now on?"</message>
    <intent>set-response-language</intent>
  </example>
  <example>
    <message>"Where does BT Servant get its information from?"</message>
    <intent>retrieve-system-information</intent>
  </example>
  <example>
    <message>"Help"</message>
    <intent>retrieve-system-information</intent>
  </example>
  <example>
    <message>"Can you tell me a joke?"</message>
    <intent>perform-unsupported-function</intent>
  </example>
  <example>
    <message>"Hmm, what was I saying again?"</message>
    <intent>unclear-intent</intent>
  </example>
</examples>

You will return a single structured output like this:
```json
{ "intents": ["get-bible-translation-assistance"] }
```
"""

DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT = """
    Your job is simply to detect the language of the supplied text. Attempt to match it to one of
    the 10 supported languages. If you can't, match it to OTHER.
"""

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = config.DATA_DIR

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

RELEVANCE_CUTOFF = .75
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


class PreprocessorResult(BaseModel):
    new_message: str
    reason_for_decision: str
    message_changed: bool


class IntentType(str, Enum):
    GET_BIBLE_TRANSLATION_ASSISTANCE = "get-bible-translation-assistance"
    PERFORM_UNSUPPORTED_FUNCTION = "perform-unsupported-function"
    RETRIEVE_SYSTEM_INFORMATION = "retrieve-system-information"
    SET_RESPONSE_LANGUAGE = "set-response-language"
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
    query = state["transformed_query"]
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT
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
    history_context_message = f"past_conversation: {json.dumps(chat_history)}"
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        instructions=PREPROCESSOR_AGENT_SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": history_context_message
            },
            {
                "role": "user",
                "content": f'current_message: {query}'
            }
        ],
        text_format=PreprocessorResult,
        store=False
    )
    preprocessor_result = response.output_parsed
    new_message = preprocessor_result.new_message
    reason_for_decision = preprocessor_result.reason_for_decision
    message_changed = preprocessor_result.message_changed
    logger.info("new_message: %s\nreason_for_decision: %s\nmessage_changed: %s",
                new_message, reason_for_decision, message_changed)
    return {
        "transformed_query": new_message if message_changed else query
    }


def query_db(state: BrainState) -> dict:
    query = state["transformed_query"]
    stack_rank_collections = state["stack_rank_collections"]
    filtered_docs = []
    collection_used = None
    # this loop is the current implementation of the "stacked ranked" algorithm
    for collection_name in stack_rank_collections:
        logger.info("querying stack collection: %s", collection_name)
        db_collection = get_chroma_collection(collection_name)
        if not db_collection:
            logger.warning("collection %s was not found in chroma db.", collection_name)
            continue
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
    query = state["transformed_query"]
    chat_history = state["user_chat_history"]
    try:
        if len(docs) == 0:
            no_docs_msg = (
                "Sorry, I couldn't find any information in my resources to service your request or command. "
                "Currently, I only have resources loaded from Titus and Mark, and a few other resources related "
                "to producing faithful translations. But maybe I'm unclear on your intent. Could you perhaps "
                "state it a different way?"
            )
            return {"responses": [no_docs_msg]}

        # build context from docs
        context = "\n\n".join([item["doc"] for item in docs])
        logger.debug("context passed to final node:\n\n%s", context)
        rag_context_message = "When answering my next query, use this additional" + \
            f"  context: {context}"
        chat_history_context_message = (f"Use this conversation history to understand the user's "
                                        f"current request only if needed: {json.dumps(chat_history)}")
        response = open_ai_client.responses.parse(
            model="gpt-4o",
            instructions=FINAL_RESPONSE_AGENT_SYSTEM_PROMPT,
            input=[
                {
                    "role": "developer",
                    "content": rag_context_message
                },
                {
                    "role": "developer",
                    "content": chat_history_context_message
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        bt_servant_response = response.output_text
        logger.info('response from openai: %s', bt_servant_response)
        logger.debug("%d characters returned from openAI", len(bt_servant_response))

        resource_list = ", ".join(set([item["resource_name"] for item in docs]))
        cascade_info = (
            f"bt_servant cascaded to the {state['collection_used']} stack. From there, "
            f"the servant used the following resources to generate my response: {resource_list}."
        )
        logger.info(cascade_info)

        return {"responses": [bt_servant_response]}
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
    chunk_max = config.MAX_META_TEXT_LENGTH - 100
    return {"responses": combine_chunks(chunks=chunks, chunk_max=chunk_max)}


def needs_chunking(state: BrainState) -> str:
    first_response = state["responses"][0]
    if len(first_response) > config.MAX_META_TEXT_LENGTH:
        logger.warning('message to big: %d chars. preparing to chunk.', len(first_response))
        return "chunk_message_node"
    else:
        return "translate_responses_node"


def process_intent(state: BrainState) -> str:
    user_intents = state["user_intents"]
    if not user_intents:
        raise ValueError("no intents found. something went very wrong.")

    if IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE in user_intents:
        return "query_db_node"

    # we need to think through and add support for multiple intents
    # for now, if we have multiple non-get bible translation assistance intents
    # simply process the first one only. FIX SOON. -- IJL
    first_intent = user_intents[0]
    if first_intent == IntentType.SET_RESPONSE_LANGUAGE:
        return "set_response_language_node"
    if first_intent == IntentType.UNCLEAR_INTENT:
        return "handle_unclear_intent_node"
    if first_intent == IntentType.PERFORM_UNSUPPORTED_FUNCTION:
        return "handle_unsupported_function_node"
    if first_intent == IntentType.RETRIEVE_SYSTEM_INFORMATION:
        return "handle_system_information_request_node"


def handle_unsupported_function(state: BrainState) -> str:
    unsupported_function_message = ("Sorry! I can't do that yet. Let my creators know that you "
                                    "desire that feature.")
    return {"responses": [unsupported_function_message]}


def handle_unclear_intent(state: BrainState) -> str:
    unclear_intent_message = ("I'm unclear on your intent. Could you perhaps state it a "
                              "different way?")
    return {"responses": [unclear_intent_message]}


def handle_system_information_request(state: BrainState) -> str:
    query = state["user_query"]
    chat_history = state["user_chat_history"]
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        instructions=HELP_AGENT_SYSTEM_PROMPT,
        input=[
            {
                "role": "developer",
                "content": f'Help docs to use: {HELP_DOCS}'
            },
            {
                "role": "developer",
                "content": f'Conversation history to use if needed: {json.dumps(chat_history)}'
            },
            {
                "role": "user",
                "content": query
            }
        ],
        store=False
    )
    help_response_text = response.output_text
    logger.info('help response from openai: %s', help_response_text)
    return {"responses": [help_response_text]}


def create_brain():
    builder = StateGraph(BrainState)

    builder.add_node("determine_query_language_node", determine_query_language)
    builder.add_node("preprocess_user_query_node", preprocess_user_query)
    builder.add_node("determine_intent_node", determine_intent)
    builder.add_node("set_response_language_node", set_response_language)
    builder.add_node("query_db_node", query_db)
    builder.add_node("query_open_ai_node", query_open_ai)
    builder.add_node("chunk_message_node", chunk_message)
    builder.add_node("handle_unsupported_function_node", handle_unsupported_function)
    builder.add_node("handle_unclear_intent_node", handle_unclear_intent)
    builder.add_node("handle_system_information_request_node", handle_system_information_request)
    builder.add_node("translate_responses_node", translate_responses)

    builder.set_entry_point("determine_query_language_node")
    builder.add_edge("determine_query_language_node", "preprocess_user_query_node")
    builder.add_edge("preprocess_user_query_node", "determine_intent_node")
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

    builder.add_edge("handle_unclear_intent_node", "translate_responses_node")
    builder.add_edge("handle_unsupported_function_node", "translate_responses_node")
    builder.add_edge("handle_system_information_request_node", "translate_responses_node")

    builder.set_finish_point("translate_responses_node")

    return builder.compile()
