# Progress Messages Specification

## Executive Summary

This document specifies the implementation of strategic progress messages in BT Servant Engine to provide users with real-time feedback during long-running operations. The implementation prioritizes clean architecture principles while targeting the slowest execution paths, particularly Bible translation assistance queries.

## Problem Statement

Users experience perceived delays between sending WhatsApp messages to the bot and receiving responses. While typing indicators provide visual feedback, users need more substantive progress updates during computationally expensive operations. The most significant delays occur in:

1. **Vector database queries** - Searching across multiple collections with relevance filtering
2. **OpenAI response generation** - Processing large context windows with RAG results
3. **Intent determination** - Complex query classification
4. **Response translation** - Converting responses to user's preferred language
5. **Voice message generation** - Text-to-speech synthesis for audio responses

## Architecture Principles

The implementation must maintain the clean hexagonal architecture established in the recent refactor:

- **Core layer** remains dependency-free
- **Services layer** orchestrates progress messaging without direct adapter dependencies
- **Adapters layer** handles actual message delivery
- **Apps layer** initiates progress tracking

## Solution Design

### 1. Progress Context in BrainState

Extend the existing `BrainState` TypedDict to include progress messaging context:

```python
class BrainState(TypedDict, total=False):
    # ... existing fields ...

    # Progress messaging fields
    progress_enabled: bool  # Whether to send progress messages
    progress_messenger: Optional[Callable[[str], Awaitable[None]]]  # Callback for sending messages
    last_progress_time: float  # Timestamp of last progress message
    progress_throttle_seconds: float  # Minimum seconds between messages
```

### 2. Progress Message Strategy

Progress messages will be sent at strategic points based on empirical performance data:

#### Critical Progress Points

1. **Conversation Start** (`start_node`)
   - Message: "Give me a few moments to think about your message."
   - Triggers for: Every request (ensures an immediate acknowledgment)
   - **ALWAYS send with force=True** to bypass throttling and ship within the first second

2. **Before Vector DB Query** (query_vector_db_node)
   - Message: "I'm searching various Bible resources to help answer your question."
   - Triggers for: GET_BIBLE_TRANSLATION_ASSISTANCE intent
   - Always send (this is the slowest operation)

3. **Before OpenAI Generation** (query_open_ai_node)
   - Message: "I found potentially relevant documents in the following resources: W, X, Y, and Z. I'm pulling everything together into a helpful response for you." (dynamic list built from `_merged_from` metadata of retrieved docs; falls back to the base sentence when no sources are available)
   - Triggers for: GET_BIBLE_TRANSLATION_ASSISTANCE path (after vector DB)
   - **ALWAYS send with force=True** (OpenAI calls are consistently slow)

4. **Before Translation** (translate_responses_node)
   - Message: "I'm translating my response into your preferred language now."
   - Triggers for: Non-English target languages
   - Threshold: If response > 500 characters

5. **Before Voice Synthesis** (for voice messages)
   - Message: "Creating audio message..."
   - Triggers for: Voice message requests
   - Always send (TTS is slow)

6. **Before Passage Summaries** (handle_get_passage_summary_node)
   - Message: "I'm gathering the passage details so I can summarize them for you."
   - Triggers for: GET_PASSAGE_SUMMARY intent
   - **ALWAYS send with force=True** because users only see this path when summary generation begins

7. **Before Passage Keywords** (handle_get_passage_keywords_node)
   - Message: "I'm reviewing the passage to pull out its key words for you."
   - Triggers for: GET_PASSAGE_KEYWORDS intent
   - **ALWAYS send with force=True** to guarantee feedback before keyword extraction LLM calls

8. **Before Translation Helps** (handle_get_translation_helps_node)
   - Message: "I'm compiling translation helps that will support your work on this passage."
   - Triggers for: GET_TRANSLATION_HELPS intent
   - **ALWAYS send with force=True** to acknowledge heavier translation helps generation

9. **Before Scripture Translation** (handle_translate_scripture_node)
   - Message: "I'm translating this passage into the language you asked for."
   - Triggers for: TRANSLATE_SCRIPTURE intent
   - **ALWAYS send with force=True** to signal the start of long-form translation work

### 3. Implementation Approach

#### Phase 1: Core Infrastructure

1. **Progress Service Module** (`bt_servant_engine/services/progress_messaging.py`):
```python
from time import time
from typing import Any, Awaitable, Callable, cast

from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services import status_messages

logger = get_logger(__name__)

ProgressMessenger = Callable[[status_messages.LocalizedProgressMessage], Awaitable[None]]


async def maybe_send_progress(
    state: Any,
    message: status_messages.LocalizedProgressMessage,
    force: bool = False,
    min_interval: float = 3.0,
) -> None:
    """Send a progress message if conditions are met."""
    from bt_servant_engine.services.brain_orchestrator import BrainState

    s = cast(BrainState, state)
    if not s.get("progress_enabled"):
        return

    messenger = s.get("progress_messenger")
    if not messenger:
        return

    current_time = time()
    last_time = s.get("last_progress_time", 0)
    throttle = s.get("progress_throttle_seconds", min_interval)

    if not force and (current_time - last_time) < throttle:
        logger.debug(
            "Throttling progress message: %s", message.get("text")
        )
        return

    try:
        await messenger(message)
        s["last_progress_time"] = current_time
        logger.info("Sent progress message: %s", message.get("text"))
    except Exception:
        logger.warning("Failed to send progress message", exc_info=True)
```

2. **Node Wrapper Enhancement** (`bt_servant_engine/services/brain_orchestrator.py`):
```python
ProgressMessageInput = (
    status_messages.LocalizedProgressMessage
    | str
    | Callable[[Any], Optional[status_messages.LocalizedProgressMessage | str]]
)


def wrap_node_with_progress(
    node_fn,
    node_name: str,
    progress_message: ProgressMessageInput | None = None,
    condition: Optional[Callable[[Any], bool]] = None,
    force: bool = False,
):
    """Wrap a node with timing and optional progress messaging."""

    async def wrapped(state: Any) -> dict:
        if progress_message is not None and (condition is None or condition(state)):
            raw = progress_message(state) if callable(progress_message) else progress_message
            if isinstance(raw, str):
                payload = status_messages.make_progress_message(raw)
            else:
                payload = raw
            if payload:
                await maybe_send_progress(state, payload, force=force)

        # Execute the original node with timing
        trace_id = cast(dict, state).get("perf_trace_id")
        if trace_id:
            set_current_trace(cast(Optional[str], trace_id))
        with time_block(f"brain:{node_name}"):
            return node_fn(state)

    return wrapped
```

#### Phase 2: Webhook Integration

Modify the webhook handler to inject progress messaging capability:

```python
# In bt_servant_engine/apps/api/routes/webhooks.py

async def progress_callback(
    user_id: str,
) -> Callable[[status_messages.LocalizedProgressMessage], Awaitable[None]]:
    """Create a progress message sender for a specific user."""

    async def send_progress(message: status_messages.LocalizedProgressMessage) -> None:
        try:
            emoji = message.get("emoji", config.PROGRESS_MESSAGE_EMOJI)
            text_msg = message.get("text", "")
            if text_msg:
                await send_text_message(user_id=user_id, text=f"{emoji} {text_msg}")
        except Exception:
            logger.warning("Failed to send progress message", exc_info=True)
    return send_progress

# In process_message function:
brain_payload: dict[str, Any] = {
    # ... existing fields ...
    "progress_enabled": True,  # Could be user preference
    "progress_messenger": await progress_callback(user_message.user_id),
    "progress_throttle_seconds": 3.0,  # Configurable
}
```

#### Phase 3: Strategic Node Updates

Update specific nodes to send progress messages:

```python
# In create_brain() function

builder.add_node(
    "query_vector_db_node",
    wrap_node_with_progress(
        brain_nodes.query_vector_db,
        "query_vector_db_node",
        progress_message=lambda s: status_messages.get_progress_message(
            status_messages.SEARCHING_BIBLE_RESOURCES, s
        ),
        condition=lambda s: IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE in s.get("user_intents", [])
    )
)

builder.add_node(
    "query_open_ai_node",
    wrap_node_with_progress(
        brain_nodes.query_open_ai,
        "query_open_ai_node",
        progress_message=build_translation_assistance_progress_message,
        condition=lambda s: IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE in s.get("user_intents", []),
        force=True  # ALWAYS send this message, bypass throttling
    )
)
```

### 4. Configuration

Add configuration options to control progress messaging:

```python
# In bt_servant_engine/core/config.py

PROGRESS_MESSAGES_ENABLED: bool = env.bool("PROGRESS_MESSAGES_ENABLED", default=True)
PROGRESS_MESSAGE_MIN_INTERVAL: float = env.float("PROGRESS_MESSAGE_MIN_INTERVAL", default=3.0)
PROGRESS_MESSAGE_EMOJI: str = env.str("PROGRESS_MESSAGE_EMOJI", default="⏳")
PROGRESS_MESSAGE_EMOJI_OVERRIDES: dict[str, str] = env.json(
    "PROGRESS_MESSAGE_EMOJI_OVERRIDES",  # Defaults handled in code
    default={}
)
```

### 5. User Preferences

Extend user state to allow per-user progress message preferences:

```python
# In bt_servant_engine/adapters/user_state.py

def get_user_progress_preference(user_id: str) -> Optional[bool]:
    """Get user's preference for progress messages."""
    # Implementation to retrieve from storage
    pass

def set_user_progress_preference(user_id: str, enabled: bool) -> None:
    """Set user's preference for progress messages."""
    # Implementation to persist preference
    pass
```

## Performance Considerations

1. **Throttling**: Minimum 3-second interval between messages to avoid spam
2. **Async Execution**: Progress messages sent asynchronously to avoid blocking
3. **Failure Tolerance**: Progress message failures don't interrupt main flow
4. **Conditional Sending**: Only send when operations exceed time thresholds

## Detailed Implementation Examples

### Throttling Mechanism in Detail

The 3-second throttling prevents message spam when multiple nodes execute in rapid succession. Here's how it works:

#### Problem It Solves
Without throttling, a query triggering multiple intents could send several messages within seconds:
- Node 1 completes at 0.5s → Message sent
- Node 2 completes at 1.2s → Would send another message (too spammy!)
- Node 3 completes at 1.8s → Would send yet another message

#### Implementation
```python
async def maybe_send_progress(
    state: Any,
    message: str,
    force: bool = False,
    min_interval: float = 3.0  # Default 3-second throttle
) -> None:
    """Send a progress message if conditions are met."""
    current_time = time()
    last_time = state.get("last_progress_time", 0)
    time_since_last = current_time - last_time

    if not force and time_since_last < min_interval:
        # Skip this message - too soon after the last one
        logger.debug(f"Throttled: '{message}' (only {time_since_last:.1f}s since last)")
        return

    # Enough time has passed, send the message
    await messenger(message)
    state["last_progress_time"] = current_time
    logger.info(f"Sent progress: '{message}' ({time_since_last:.1f}s since last)")
```

#### Real Timeline Example
```
0.0s - User sends: "Help me translate John 3:16 into Spanish"
0.1s - Typing indicator sent ✓
0.5s - Intent determination complete
       → No message (query under 300 chars)
1.0s - Vector DB query starts
       → "Searching Bible translation resources..." ✓ (first message)
3.5s - Vector DB complete, OpenAI query starts
       → "Analyzing resources and preparing response..." ✓ (FORCED - bypasses throttle)
6.0s - Translation to Spanish starts
       → [THROTTLED] (only 2.5s since last message)
7.2s - Final response sent to user

Result: 2 well-timed progress messages at the genuinely slow operations
```

### Conditional Sending Logic in Detail

The system analyzes operation characteristics to intelligently decide when to send progress messages:

#### 1. Vector DB Query Condition
```python
# Always send for Bible translation assistance (consistently slow)
condition=lambda s: IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE in s.get("user_intents", [])
```

#### 2. OpenAI Generation Condition
```python
def should_show_openai_progress(state):
    """Always show for Bible translation assistance path."""
    # If we're in the assistance path (indicated by having done vector DB query),
    # ALWAYS show progress as OpenAI calls are consistently slow
    return IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE in state.get("user_intents", [])

# In the node wrapper, this will be called with force=True to bypass throttling
# since we know this is a critical slow point
```

#### 3. Translation Condition
```python
def should_show_translation_progress(state):
    """Decide based on target language and response size."""
    responses = state.get("responses", [])
    target_lang = state.get("user_response_language")

    # Skip if English (no translation needed)
    if target_lang == "English" or not target_lang:
        return False

    # Check total response length
    total_length = sum(len(r.get("response", "")) for r in responses)

    # Complex scripts need more processing time
    complex_scripts = ["Arabic", "Chinese", "Japanese", "Hebrew", "Hindi"]
    is_complex = target_lang in complex_scripts

    # Show progress for long responses or complex scripts
    return total_length > 500 or is_complex
```

#### 4. Intent Determination Condition
```python
def should_show_intent_progress(state):
    """Simplified: Only show for very long queries."""
    query = state.get("user_query", "")

    # Simple heuristic: only very long queries warrant a progress message
    # Avoiding brittle keyword detection
    return len(query) > 300  # Only for substantial queries
```

### Scenario-Based Examples

#### Simple Query (No Progress Messages)
```
User: "Show me Psalm 23"

Analysis:
- Query length: 17 chars → No intent progress
- Intent: RETRIEVE_SCRIPTURE → No vector DB query
- Response: English → No translation
Result: Fast response, no progress messages needed
```

#### Complex Query (Multiple Progress Messages)
```
User: "Can you help me understand the Hebrew word chesed in Psalm 136?
       What are the translation challenges and how do different versions
       handle it? Also show me parallel passages."

Analysis:
- Query length: 183 chars → No intent progress (under 300 char threshold)
- Intent: GET_BIBLE_TRANSLATION_ASSISTANCE → Show vector DB progress ✓
- OpenAI after vector DB → ALWAYS show with force=True ✓
- Target language: If non-English → Show translation progress ✓

Timeline:
0.0s - Start processing
1.0s - "Searching Bible translation resources..." (vector DB starts)
4.2s - "Analyzing resources and preparing response..." (OpenAI starts, FORCED)
7.5s - Response delivered (or translation if non-English)

Result: 2-3 strategic progress messages at known bottlenecks
```

#### Voice Message Handling
```python
def handle_voice_message_progress(state):
    """Special handling for voice messages."""
    is_audio_input = state.get("message_type") == "audio"
    needs_voice_output = state.get("send_voice_message", False)

    progress_points = []

    if is_audio_input:
        progress_points.append(("transcribing", "Transcribing your message..."))

    # ... normal processing progress points ...

    if needs_voice_output:
        text_length = len(state.get("voice_message_text", ""))
        if text_length > 100:  # Only for substantial responses
            progress_points.append(("synthesizing", "Creating audio message..."))

    return progress_points
```

### Progress Metrics Storage

The progress metrics system has two storage options:

#### Option 1: In-Memory Only (Recommended for MVP)
```python
# Simple in-memory storage that resets on restart
class InMemoryProgressMetrics:
    """
    Metrics live only during the application lifecycle.
    Pros: Simple, no persistence overhead, no cleanup needed
    Cons: Loses learning on restart, cold start after deployment
    """
    def __init__(self):
        self.operation_timings = defaultdict(list)
        self.started_at = datetime.now()

    def reset_on_restart(self):
        # Metrics are automatically reset when process restarts
        pass
```

#### Option 2: Redis-Backed Persistence (Future Enhancement)
```python
# Persistent storage using Redis with TTL
class RedisProgressMetrics:
    """
    Metrics persist across restarts using Redis.
    Pros: Maintains learning across deployments
    Cons: Requires Redis, needs TTL management
    """
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 86400  # 24 hour TTL on metrics

    def record_timing(self, operation: str, duration: float):
        key = f"progress_metrics:{operation}"
        # Use Redis list with automatic expiration
        self.redis.lpush(key, duration)
        self.redis.ltrim(key, 0, 99)  # Keep last 100
        self.redis.expire(key, self.ttl)
```

**Recommendation**: Start with in-memory storage for the MVP. The cold-start problem is minimal since:
- The system quickly learns typical timings (within 10-20 requests)
- We have good static defaults for known slow operations
- The main benefit is avoiding progress message spam, not perfect prediction

### Dynamic Threshold Adjustment

The system can learn from historical performance:

```python
from collections import defaultdict
from typing import Dict, List
import numpy as np

class AdaptiveProgressManager:
    def __init__(self):
        self.operation_timings: Dict[str, List[float]] = defaultdict(list)
        self.max_history = 100  # Keep last 100 timings per operation

    def record_timing(self, operation: str, duration: float):
        """Record actual operation timing for learning."""
        timings = self.operation_timings[operation]
        timings.append(duration)
        # Keep only recent history
        if len(timings) > self.max_history:
            timings.pop(0)

    def should_show_progress(self, operation: str, characteristics: dict) -> bool:
        """Decide based on historical performance and characteristics."""
        history = self.operation_timings.get(operation, [])

        if len(history) < 10:
            # Not enough history - use static rules
            return self._static_rules(operation, characteristics)

        # Calculate percentiles
        p50 = np.percentile(history, 50)  # Median
        p90 = np.percentile(history, 90)  # 90th percentile

        # Dynamic decision based on historical performance
        if p50 > 3.0:  # If median time > 3s, always show
            return True
        elif p90 > 5.0:  # If 10% of operations take > 5s, be cautious
            # Use characteristics to predict if this will be slow
            return self._predict_slow(operation, characteristics, p50, p90)
        else:
            # Generally fast operation - only show for complex cases
            return self._static_rules(operation, characteristics)

    def _predict_slow(self, operation: str, characteristics: dict, p50: float, p90: float) -> bool:
        """Predict if this specific operation will be slow."""
        if operation == "vector_db_query":
            # Number of collections to search is a good predictor
            collection_count = len(characteristics.get("collections", []))
            return collection_count > 3

        elif operation == "openai_generation":
            # Context size is the best predictor
            context_size = characteristics.get("context_chars", 0)
            # If context is 2x the median context that caused p90 timing
            return context_size > 10000

        return False
```

## Testing Strategy

1. **Unit Tests**:
   - Test progress message throttling
   - Test conditional message sending
   - Test failure handling

2. **Integration Tests**:
   - Test end-to-end flow with progress messages
   - Test WhatsApp API integration
   - Test user preference handling

3. **Performance Tests**:
   - Measure impact on response time
   - Verify throttling behavior under load

## Implementation Plan

### Phase 1: Foundation (Week 1)
- [ ] Create progress messaging service module
- [ ] Extend BrainState with progress fields
- [ ] Add configuration options
- [ ] Write unit tests

### Phase 2: Integration (Week 1-2)
- [ ] Modify webhook handler to inject progress callback
- [ ] Update node wrapper to support progress messages
- [ ] Integrate with existing timing infrastructure
- [ ] Write integration tests

### Phase 3: Strategic Deployment (Week 2)
- [ ] Add progress messages to vector DB query node
- [ ] Add progress messages to OpenAI query node
- [ ] Add progress messages to translation node
- [ ] Test with real WhatsApp messages

### Phase 4: User Experience (Week 2-3)
- [ ] Implement user preference storage
- [ ] Add command to toggle progress messages
- [ ] Refine message content and timing
- [ ] Deploy to staging for user testing

## Risk Mitigation

1. **Architecture Violation**: All messaging remains in the adapter layer
2. **Message Spam**: Throttling prevents excessive messages
3. **API Rate Limits**: Progress messages counted in rate limit calculations
4. **User Disruption**: Feature can be disabled per-user or globally

## Success Metrics

1. **User Satisfaction**: Reduced complaints about response delays
2. **Message Delivery**: >95% successful progress message delivery
3. **Performance Impact**: <5% increase in total response time
4. **Architecture Compliance**: Zero import linter violations

## Alternative Approaches Considered

1. **Streaming Responses**: Not supported by WhatsApp API
2. **Fixed Interval Messages**: Less informative than strategic placement
3. **Detailed Technical Messages**: Too complex for end users
4. **Client-side Indicators**: Limited to typing indicator in WhatsApp

## Conclusion

This specification provides a clean, architecturally sound approach to implementing progress messages that:
- Maintains the hexagonal architecture
- Targets the slowest operations
- Provides meaningful user feedback
- Remains configurable and testable
- Minimizes performance impact

The implementation can be completed in phases, allowing for incremental testing and refinement based on user feedback.
