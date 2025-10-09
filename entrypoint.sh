#!/bin/sh
echo "▶️ BT_SERVANT_LOG_LEVEL=${BT_SERVANT_LOG_LEVEL}"
exec uvicorn bt_servant_engine.apps.api.app:create_app --factory \
  --host 0.0.0.0 \
  --port 8080 \
  --log-level=${BT_SERVANT_LOG_LEVEL:-info} \
  --access-log
