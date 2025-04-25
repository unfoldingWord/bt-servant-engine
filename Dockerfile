FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "bt_servant:app", "--host", "0.0.0.0", "--port", "8080"]
