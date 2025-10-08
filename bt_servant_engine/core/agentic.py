"""Agentic strength models and constants for the BT Servant application."""

from enum import Enum

from pydantic import BaseModel

# Allowed agentic strength values
ALLOWED_AGENTIC_STRENGTH = {"normal", "low", "very_low"}


class AgenticStrengthChoice(str, Enum):
    """Accepted agentic strength options for controllable responses."""

    NORMAL = "normal"
    LOW = "low"
    VERY_LOW = "very_low"
    UNKNOWN = "unknown"


class AgenticStrengthSetting(BaseModel):
    """Schema for parsing agentic strength adjustments from the user."""

    strength: AgenticStrengthChoice


__all__ = [
    "ALLOWED_AGENTIC_STRENGTH",
    "AgenticStrengthChoice",
    "AgenticStrengthSetting",
]
