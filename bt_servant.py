import time
import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI
from fastapi import Request, Form
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
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

TWILIO_CHAR_LIMIT = 1600


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


@app.get("/")
def read_root():
    return {"message": "Go to /graphql to access the GraphQL API"}


@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...)
):
    start_time = time.time()
    twiml = MessagingResponse()
    try:
        query = Body
        # use the phone number as the user id
        user_id = From.replace("whatsapp:", "")
        logger.info("Received message from %s: %s", user_id, query)
        result = brain.invoke({
            "user_query": query,
            "user_chat_history": get_user_chat_history(user_id=user_id)
        })
        response = result["response"]
        logger.info("Response from bt_servant: %s", response)
        update_user_chat_history(user_id=user_id, query=query, response=response)

        response_length = len(response)
        logger.info("Response length %d characters", response_length)
        if response_length > TWILIO_CHAR_LIMIT:
            logger.warning("Response too long (%d chars), truncating", response_length)
            response = response[:1600]

        twiml.message(response)
    except Exception as e:
        twiml.message(str(e))
        logger.error("Error occurred", exc_info=True)

    logger.info("Total processing time: %.2f seconds",time.time() - start_time)
    return Response(content=str(twiml), media_type="application/xml")


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
        response = result["response"]
        logger.info("Response from bt_servant: %s", response)
        update_user_chat_history(user_id=user_id, query=query, response=response)
        return response

    @strawberry.field
    def health_check(self) -> str:
        return "aquifer_whatsapp_bot is healthy!"


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
