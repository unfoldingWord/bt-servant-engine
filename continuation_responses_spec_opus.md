# Continuation Responses Specification

## Overview
Add context-aware continuation prompts to all final responses to encourage ongoing conversation. The system will append relevant follow-up questions based on the intent type and content discussed.

## Architecture Decision: Centralized Approach (Option 1)

### Why Centralized?
- **Single point of modification**: Only need to modify the `translate_responses` function in `brain_nodes.py`
- **Consistent across all paths**: Since all graph paths converge through `translate_responses_node` before finishing, we catch every response type
- **Easier maintenance**: All continuation logic in one new module (`continuation_prompts.py`) called from one place
- **Cleaner architecture**: Follows the existing pattern where translation/formatting happens centrally

### Alternative Considered (Option 2)
Individual intent handlers approach was considered but rejected because it would require:
- Changing 10+ different intent handler files
- Duplicating logic across multiple places
- Harder to ensure consistency
- More difficult to maintain and update

## Implementation Plan

### Phase 1: Create Continuation Prompt System

#### New Module: `bt_servant_engine/services/continuation_prompts.py`

Core functions:
```python
def generate_continuation_prompt(
    intent_type: IntentType,
    response_content: str,
    state: BrainState
) -> Optional[str]:
    """Generate a context-aware continuation prompt based on intent and content."""

def extract_biblical_entities(text: str) -> Dict[str, List[str]]:
    """Extract biblical figures, books, and topics from response text."""

def get_related_suggestions(entities: Dict[str, List[str]]) -> List[str]:
    """Generate related biblical topics based on extracted entities."""

def format_continuation_prompt(suggestions: List[str]) -> str:
    """Format suggestions into a natural continuation prompt."""
```

### Phase 2: Integrate with Response Pipeline

#### Modify `bt_servant_engine/services/brain_nodes.py`

```python
def translate_responses(state: Any) -> dict:
    """Translate or localize responses into the user's desired language."""
    from bt_servant_engine.services.brain_orchestrator import BrainState
    from bt_servant_engine.services.continuation_prompts import generate_continuation_prompt

    # ... existing translation logic ...

    # After translation, before returning
    for i, response in enumerate(translated_responses):
        # Get the original intent type from raw responses
        intent_type = raw_responses[i].get("intent") if i < len(raw_responses) else None

        # Generate continuation prompt
        continuation = generate_continuation_prompt(
            intent_type=intent_type,
            response_content=response,
            state=s
        )

        # Append continuation if generated
        if continuation:
            translated_responses[i] = f"{response}\n\n{continuation}"

    return {"translated_responses": translated_responses}
```

### Phase 3: Intent-Specific Continuation Strategies

#### GET_BIBLE_TRANSLATION_ASSISTANCE
- Extract biblical figures/topics from response
- Suggest related figures: "Would you like to learn about [related figure] next?"
- Suggest deeper exploration: "Would you like me to explain [specific aspect] in more detail?"
- Example: After discussing David → suggest Saul, Solomon, or Samuel

#### GET_PASSAGE_SUMMARY
- Suggest adjacent passages: "Would you like a summary of the next/previous chapter?"
- Suggest keywords/helps: "Would you like to see key terms from this passage?"
- Cross-reference related passages: "Would you like to explore parallel accounts?"

#### RETRIEVE_SCRIPTURE
- Suggest analysis: "Would you like a summary or translation helps for this passage?"
- Suggest adjacent verses: "Would you like to see the surrounding context?"
- Offer audio option: "Would you like me to read this passage aloud?"

#### CONSULT_FIA_RESOURCES
- Suggest applying to passage: "Would you like to apply these FIA steps to a specific passage?"
- Suggest next FIA step: "Would you like to explore [next step in process]?"
- Offer examples: "Would you like to see examples of this principle in action?"

#### GET_PASSAGE_KEYWORDS
- Suggest definitions: "Would you like detailed explanations of any of these terms?"
- Suggest passage summary: "Would you like a summary of this passage?"
- Link to translation helps: "Would you like translation guidance for these key terms?"

#### GET_TRANSLATION_HELPS
- Suggest scripture text: "Would you like to see the actual passage text?"
- Suggest related helps: "Would you like translation helps for adjacent passages?"
- Offer FIA guidance: "Would you like FIA process guidance for this passage?"

#### TRANSLATE_SCRIPTURE
- Suggest other languages: "Would you like this translated into another language?"
- Suggest explanation: "Would you like translation notes for this passage?"
- Offer comparison: "Would you like to compare with another translation?"

#### LISTEN_TO_SCRIPTURE
- Suggest text view: "Would you like to see the text of this passage?"
- Suggest adjacent audio: "Would you like to hear the next/previous chapter?"
- Offer translation: "Would you like to hear this in another language?"

#### Settings Intents (SET_RESPONSE_LANGUAGE, SET_AGENTIC_STRENGTH)
- Suggest trying feature: "Would you like me to help with a translation task now?"
- Offer examples: "Would you like to see what I can help you with?"

#### PERFORM_UNSUPPORTED_FUNCTION/RETRIEVE_SYSTEM_INFORMATION
- Already includes "Which of these capabilities would you like to explore?"
- Enhance with specific suggestions based on user's attempted action
- Guide toward supported features

### Phase 4: Smart Content Analysis

#### Biblical Entity Recognition
```python
BIBLICAL_FIGURES = {
    "David": ["Saul", "Solomon", "Samuel", "Jonathan", "Bathsheba"],
    "Moses": ["Aaron", "Miriam", "Joshua", "Pharaoh"],
    "Paul": ["Barnabas", "Timothy", "Silas", "Peter"],
    # ... comprehensive mapping
}

BOOK_RELATIONSHIPS = {
    "Romans": ["Galatians", "Ephesians", "1 Corinthians"],
    "Matthew": ["Mark", "Luke", "John"],
    # ... book connections
}
```

#### Context-Aware Suggestion Generation
- Parse response for biblical references (books, chapters, figures)
- Build knowledge graph connections
- Generate contextually relevant suggestions
- Avoid repetitive suggestions using chat history
- Respect user's language preference

### Phase 5: Configuration & Testing

#### Configuration Options
```python
class ContinuationConfig:
    enabled: bool = True
    max_suggestions: int = 3
    prompt_templates: Dict[str, str] = {
        "default": "Would you like to:",
        "explore": "You might also be interested in:",
        "next_step": "What would you like to explore next?",
    }
    variety_threshold: int = 3  # Vary template after N uses
```

#### Testing Strategy

**Unit Tests** (`tests/services/test_continuation_prompts.py`):
- Test prompt generation for each intent type
- Test biblical entity extraction
- Test suggestion relevance
- Test prompt variety logic

**Integration Tests**:
- Test end-to-end flow through translate_responses
- Verify prompts appear in final output
- Test translation of continuation prompts
- Verify chunking compatibility

**Test Cases**:
```python
def test_translation_assistance_continuation():
    """Test continuation prompts for Bible translation assistance."""
    response = "David was the second king of Israel..."
    prompt = generate_continuation_prompt(
        IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE,
        response,
        mock_state
    )
    assert "Saul" in prompt or "Solomon" in prompt

def test_passage_summary_continuation():
    """Test continuation prompts for passage summaries."""
    response = "Romans 8 discusses life in the Spirit..."
    prompt = generate_continuation_prompt(
        IntentType.GET_PASSAGE_SUMMARY,
        response,
        mock_state
    )
    assert "Romans 7" in prompt or "Romans 9" in prompt
```

### Phase 6: Internationalization

#### Language-Aware Prompts
- Continuation prompts must respect `user_response_language`
- Templates should be translatable
- Maintain natural flow in target language

```python
def get_localized_prompt_template(language: str, template_key: str) -> str:
    """Get continuation prompt template in user's language."""
    # Templates can be stored in language files or generated
    pass
```

## Example Implementations

### Bible Translation Assistance
```
User: "Tell me about David in the Bible"
Bot: David was the second king of Israel, known for defeating Goliath and
writing many Psalms. He established Jerusalem as the capital and despite
his failures, was called "a man after God's own heart."

Would you like to learn more about:
- King Saul, David's predecessor?
- Solomon, David's son and successor?
- David's role in writing the Psalms?
```

### Passage Summary
```
User: "Summarize Romans 8"
Bot: Romans 8 is about life in the Spirit, freedom from condemnation, and
God's love. Paul explains how the Spirit helps us in our weakness and that
nothing can separate us from God's love in Christ.

Would you like to:
- See a summary of Romans 7 or 9?
- Explore key terms from Romans 8?
- Get translation helps for this chapter?
```

### FIA Resources
```
User: "What are the FIA steps?"
Bot: The FIA process includes: [detailed steps explanation]

Would you like to:
- Apply these steps to a specific passage?
- See examples of FIA in practice?
- Learn about specific FIA tools?
```

## Benefits
1. **Increased user engagement** through guided exploration
2. **Natural conversation flow** that feels less transactional
3. **Feature discovery** helps users learn capabilities
4. **Context-aware suggestions** prevent irrelevant prompts
5. **Centralized implementation** ensures consistency
6. **Maintainable architecture** with single point of control

## Risks & Mitigations

### Risk: Prompts might feel repetitive
**Mitigation**:
- Vary prompt templates
- Use chat history to avoid repetition
- Rotate through different suggestion types
- Limit frequency (e.g., not every response needs continuation)

### Risk: Prompts might not translate well
**Mitigation**:
- Ensure prompts go through translation pipeline
- Test with multiple languages
- Use simple, translatable phrases
- Avoid idioms and complex constructions

### Risk: Increased message length
**Mitigation**:
- Keep prompts concise (max 2-3 lines)
- Respect chunking limits
- Make continuation optional via config
- Monitor message length before adding

### Risk: Context switching confusion
**Mitigation**:
- Clear visual separation (double newline)
- Consistent prompt format
- Relevant suggestions only
- Test with actual users

## Success Metrics
- User engagement rate (follow-up questions asked)
- Session length increase
- Feature discovery rate
- User satisfaction scores
- Reduced "what can you do?" queries

## Rollout Strategy
1. **Phase 1**: Implement core system with feature flag (disabled)
2. **Phase 2**: Enable for internal testing
3. **Phase 3**: A/B test with subset of users
4. **Phase 4**: Full rollout with monitoring
5. **Phase 5**: Iterate based on usage data

## Future Enhancements
- Machine learning for suggestion relevance
- Personalization based on user history
- Dynamic prompt generation using LLMs
- Integration with user preferences
- Analytics dashboard for prompt effectiveness