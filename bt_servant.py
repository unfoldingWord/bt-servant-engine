import asyncio
import time
from collections import defaultdict
from twilio.rest import Client
import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI, BackgroundTasks, Request, Form
from fastapi.responses import Response
from twilio.base.exceptions import TwilioRestException
from tinydb import TinyDB, Query as TinyQuery
from pathlib import Path
from threading import Lock
from brain import create_brain
from logger import get_logger
from config import Config


app = FastAPI()
brain = None

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = Config.DATA_DIR
DB_PATH = DB_DIR / "chat_history.json"
db = TinyDB(str(DB_PATH))
db_lock = Lock()
CHAT_HISTORY_MAX = 5

logger = get_logger(__name__)

user_locks = defaultdict(asyncio.Lock)


def get_user_chat_history(user_id):
    with db_lock:
        q = TinyQuery()
        result = db.get(q.user_id == user_id)
        return result["history"] if result else []


def update_user_chat_history(user_id, query, response):
    with db_lock:
        q = TinyQuery()
        result = db.get(q.user_id == user_id)
        history = result["history"] if result else []
        history.append({
            "user_message": query,
            "assistant_response": response
        })
        history = history[-CHAT_HISTORY_MAX:]
        db.upsert({"user_id": user_id, "history": history}, q.user_id == user_id)


@app.on_event("startup")
def init():
    logger.info("Initializing bt servant engine...")
    logger.info("Loading brain...")
    global brain
    brain = create_brain()
    logger.info("brain loaded.")


@app.get("/")
def read_root():
    return {"message": "Go to /graphql to access the GraphQL API"}


@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    user_id = From.replace("whatsapp:", "")
    query = Body
    logger.info("Received message from %s: %s", user_id, query)

    background_tasks.add_task(process_message_and_respond, user_id, query)

    # Return immediately with 202 to avoid Twilio timeouts
    return Response(status_code=202)


async def process_message_and_respond(user_id: str, query: str):
    async with user_locks[user_id]:
        start_time = time.time()
        try:
            result = brain.invoke({
                "user_query": query,
                "user_chat_history": get_user_chat_history(user_id=user_id)
            })
            responses = result["responses"]
            response_count = len(responses)
            if response_count > 1:
                responses = [f'({i}/{response_count}) {r}' for i, r in enumerate(responses, start=1)]
            for response in responses:
                logger.info("Response from bt_servant: %s", response)
                client = Client()
                sender = "whatsapp:" + Config.TWILIO_PHONE_NUMBER
                recipient = "whatsapp:" + user_id
                logger.info('Preparing to send message from %s to %s.', sender, recipient)

                client.messages.create(
                    body=response,
                    from_=sender,
                    to=recipient
                )
                # the sleep below is to prevent the (1/3)(3/3)(2/3) situation
                # in a prod situation we would want to handle this better
                # using the Twilio delivery webhook - IJL
                await asyncio.sleep(4)
            update_user_chat_history(user_id=user_id, query=query, response="\n".join(responses).rstrip())
        except (TwilioRestException, RuntimeError, ValueError) as e:
            logger.error("Error occurred during background message handling", exc_info=True)
        finally:
            logger.info("Total processing time: %.2f seconds", time.time() - start_time)


@strawberry.type
class Query:
    @strawberry.field
    def query_bt_servant(self, query: str) -> str:
        user_id = '+1231231234'
        logger.info("Received message from %s: %s", user_id, query)
        result = brain.invoke({
            "user_query": query,
            "user_chat_history": get_user_chat_history(user_id=user_id)
        })
        responses = result["responses"]
        logger.info("Responses from bt_servant: %s", responses)
        update_user_chat_history(user_id=user_id, query=query, response="\n".join(responses).rstrip())
        return responses

    @strawberry.field
    def health_check(self) -> str:
        return "aquifer_whatsapp_bot is healthy!"


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
