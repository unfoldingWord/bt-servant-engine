# 🧠 bt-servant-engine

An AI-powered WhatsApp assistant for Bible translators, using FastAPI, Twilio, OpenAI, and ChromaDB. Easily runs both locally and in production with environment-aware configuration.

---

## 🚀 Environment Setup

This app uses a `.env` file for local development and Fly.io secrets in production.

### ✅ Step-by-Step Setup (Local)

1. **Create a `.env` file**

Copy the example template and edit your own:

```bash
cp .env.example .env
```

2. **Edit your `.env`**

Set your OpenAI API key and any optional overrides:

```env
FLY_IO=0
OPENAI_API_KEY=sk-...
# Optional: override local data path
# DATA_DIR=/your/custom/path
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

From your project directory:

```bash
uvicorn bt_servant:app --reload
```

---

## 🌐 How Environment Detection Works

We use a centralized config system (`config.py`) to determine behavior based on environment:

| Variable     | Used For                    | Description |
|--------------|-----------------------------|-------------|
| `FLY_IO`     | Detect Fly.io environment   | `1` = running on Fly |
| `DATA_DIR`   | Path to ChromaDB + history  | Optional override of the default data folder |
| `OPENAI_API_KEY` | Auth for OpenAI        | Always required |

If `DATA_DIR` is **not provided**, it defaults to:

- ✅ `/data` when `FLY_IO=1` (Fly volume mount)
- ✅ `./data` relative to the codebase when `FLY_IO=0` (local)

---

## 🔐 Production with Fly.io

This project is Fly.io-ready and uses a Dockerfile for deployment.

In production:

- You **do not need a `.env` file**
- Instead, run:

```bash
fly secrets set FLY_IO=1
fly secrets set OPENAI_API_KEY=sk-...
```

You can also set `DATA_DIR` if needed, but by default it uses the mounted volume at `/data`.

---

## 🧪 Testing Your Config

Add this line in any Python file to see what config is being used:

```python
from config import Config
print("Using data directory:", Config.DATA_DIR)
```

---

## 🛠️ Debugging Tips

- If your app is writing to the wrong folder (`/app/data` instead of `/data`), double-check your `DATA_DIR` resolution.
- Use `fly ssh console` to inspect files on your production volume.
- To test changes, run `fly deploy --no-cache` for a clean rebuild.

---

## 📂 `.env.example`

This file should be included in your repo:

```env
# Copy this to .env and customize
FLY_IO=0
OPENAI_API_KEY=your-key-here
# Optional override
# DATA_DIR=./my-data
```

---

Happy translating 🚀📖
