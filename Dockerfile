FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENV BT_SERVANT_LOG_LEVEL=INFO

ENTRYPOINT ["uvicorn", "bt_servant:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["--log-level", "${BT_SERVANT_LOG_LEVEL}"]

