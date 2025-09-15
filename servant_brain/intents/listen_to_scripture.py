from __future__ import annotations

from typing import Any, Callable, Dict


def listen_to_scripture(
    state: Any,
    *,
    retrieve: Callable[[Any], Dict],
) -> Dict:
    """Delegate to retrieve-scripture and request voice delivery."""
    out = retrieve(state)
    out["send_voice_message"] = True
    return out

