FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENV BT_SERVANT_LOG_LEVEL=info

CMD ["sh", "-c", "uvicorn bt_servant:app --host 0.0.0.0 --port 8080 --log-level=$BT_SERVANT_LOG_LEVEL"]
