# 🧠 bt-servant-engine

An AI-powered WhatsApp assistant for Bible translators, now powered by **Meta's Cloud API** (no more Twilio!). The assistant uses FastAPI, OpenAI, and ChromaDB to answer Bible translation questions in multiple languages.

---

## 🚀 Environment Setup

This app uses a `.env` file for local development.

### ✅ Step-by-Step Setup (Local)

1. **Create a `.env` file**

```bash
cp env.example .env
```

2. **Edit your `.env`**

Fill in the required values:

```env
OPENAI_API_KEY=sk-...
META_VERIFY_TOKEN=your-verify-token-here
META_WHATSAPP_TOKEN=your-meta-access-token-here
META_PHONE_NUMBER_ID=your-meta-phone-number-id
BASE_URL=https://your.public.domain
```

> ℹ️ All five above variables are required for the Meta Cloud API to work properly.

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
uvicorn bt_servant:app --reload
```

---

## 🌐 Environment Variables Explained

| Variable               | Purpose                                           |
|------------------------|---------------------------------------------------|
| `OPENAI_API_KEY`       | Auth token for OpenAI's GPT models                |
| `META_VERIFY_TOKEN`    | Custom secret used for Meta webhook verification  |
| `META_WHATSAPP_TOKEN`  | Access token used to send messages via Meta API   |
| `META_PHONE_NUMBER_ID` | Phone number ID tied to Meta app/WABA             |
| `BASE_URL`      | Public base URL used to generate audio file links |
| `BT_SERVANT_LOG_LEVEL` | (Optional) Defaults to DEBUG if not present       |

Other acceptable values for log level: CRITICAL, ERROR, WARNING, and INFO

---

## 🛰 Webhook Setup

Set your webhook URL in the [Meta Developer Console](https://developers.facebook.com/) under your WhatsApp App configuration:

- **Callback URL**: `https://<your-domain>/meta-whatsapp`
- **Verify Token**: Must match `META_VERIFY_TOKEN` in your `.env`

---

## 📤 Sending Messages via Meta Cloud API

When a user sends a WhatsApp message to your number, Meta calls your `/meta-whatsapp` endpoint. This app currently echoes back the received message. All message logic is defined in `bt_servant.py`.

---

## 🛠️ Debugging Tips

- If audio URLs don’t resolve, double-check `BASE_URL`
- If Meta webhook verification fails, confirm `META_VERIFY_TOKEN` matches what you entered in the Meta console

---

## 🧪 Testing Locally

You can test message flow locally using tools like [ngrok](https://ngrok.com/) to expose `localhost:8000` to the public internet, then set your webhook in Meta to use the `https://<ngrok-url>/meta-whatsapp` endpoint.

---

Happy translating 🚀📖
