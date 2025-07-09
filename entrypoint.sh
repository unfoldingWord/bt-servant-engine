#!/bin/sh
echo "▶️ BT_SERVANT_LOG_LEVEL=${BT_SERVANT_LOG_LEVEL}"
exec uvicorn bt_servant:app \
  --host 0.0.0.0 \
  --port 8080 \
  --log-level=${BT_SERVANT_LOG_LEVEL:-info} \
  --access-log
