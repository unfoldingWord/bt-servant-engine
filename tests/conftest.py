"""Pytest configuration: ensure env vars and import path are set early.

This runs before any tests, so modules can import without local path hacks.
Also load .env before setting defaults so real keys are used when present.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure repository root is on sys.path for local package imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Load .env first so OPENAI_API_KEY and friends are available for tests
load_dotenv(override=False)

# Ensure required env vars exist for config import in app (fallbacks only)
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("META_VERIFY_TOKEN", "test")
os.environ.setdefault("META_WHATSAPP_TOKEN", "test")
os.environ.setdefault("META_PHONE_NUMBER_ID", "test")
os.environ.setdefault("META_APP_SECRET", "test")
os.environ.setdefault("FACEBOOK_USER_AGENT", "test")
os.environ.setdefault("BASE_URL", "http://example.com")
