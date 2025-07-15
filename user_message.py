from datetime import datetime, timezone
from config import config

valid_message_types = {'text', 'audio'}


class UserMessage:
    def __init__(self, message_id: str, user_id: str, message_type: str, timestamp: str,
                 text: str = "", media_id: str = ""):
        self.message_id = message_id
        self.user_id = user_id
        self.message_type = message_type
        self.timestamp = int(timestamp)
        self.text = text
        self.media_id = media_id

    @classmethod
    def from_data(cls, data: dict):
        user_id = data.get("from")
        message_id = data.get("id")
        message_type = data.get("type")
        timestamp = data.get("timestamp")

        text = ""
        media_id = ""
        if message_type == "text":
            text = data.get("text", {}).get("body", "")
            if not text:
                raise ValueError("text message received with no message body.")
        elif message_type == "audio":
            media_id = data.get("audio").get("id")
            if not media_id:
                raise ValueError("audio message received with no id.")
        return cls(
            user_id=user_id,
            message_id=message_id,
            message_type=message_type,
            timestamp=timestamp,
            text=text,
            media_id=media_id
        )

    def age(self) -> int:
        message_time_utc = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        return (datetime.now(timezone.utc) - message_time_utc).total_seconds()

    def too_old(self) -> bool:
        return self.age() > config.MESSAGE_AGE_CUTOFF_IN_SECONDS

    def is_unauthorized_sender(self) -> bool:
        return config.IN_META_SANDBOX_MODE and self.user_id != config.META_SANDBOX_PHONE_NUMBER

    def is_supported_type(self) -> bool:
        return self.message_type in valid_message_types
