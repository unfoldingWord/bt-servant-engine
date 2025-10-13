# Intent Follow-up Questions Implementation Spec

## Overview
Every intent handler should end with a contextual follow-up question, EXCEPT when the multi-intent handler has already added one.

## Current State

### Intent Handlers WITH Follow-ups
- `help` - already has follow-up
- `unsupported` - already has follow-up
- `converse_with_bt_servant` - already has follow-up (keep as-is)

### Intent Handlers NEEDING Follow-ups
- `scripture_lookup`
- `translation_resources`
- `language_selection`
- `status_message`
- Any other intent handlers discovered during implementation

### No Follow-up Needed
- Error handler (big try/except block)
- Converse with BT Servant (including conversation exits)

## Implementation Requirements

### 1. State Management
Add to `BrainState` TypedDict:
```python
followup_question_added: bool = False  # Set to True when any follow-up is added
```

### 2. Multi-Intent Handler Update
Location: Find where "Would you like help with..." questions are added for multiple intents
Action: Add `state['followup_question_added'] = True` after adding the multi-intent follow-up

### 3. Follow-up Questions by Intent Type

Each intent needs a contextual follow-up that will be added ONLY if `followup_question_added == False`:

| Intent | Follow-up Question (English) |
|--------|------------------------------|
| scripture_lookup | "Would you like to look up another Bible passage?" |
| translation_resources | "Do you need help with another translation question?" |
| language_selection | "What else can I help you with today?" |
| status_message | "Is there anything else I can assist you with?" |

### 4. Implementation Pattern

Each intent handler should follow this pattern at the end:

```python
def handle_[intent_name](state: BrainState, ...) -> str:
    # ... existing intent logic ...

    response = # ... the main response ...

    # Add follow-up if not already added
    if not state.get('followup_question_added', False):
        response += f"\n\n{get_followup_for_intent('[intent_name]', state['language'])}"
        state['followup_question_added'] = True

    return response
```

### 5. Localization Support

Create a follow-up questions dictionary structure:
```python
INTENT_FOLLOWUPS = {
    'scripture_lookup': {
        'en': "Would you like to look up another Bible passage?",
        'es': "¿Le gustaría buscar otro pasaje bíblico?",
        # ... other languages
    },
    'translation_resources': {
        'en': "Do you need help with another translation question?",
        'es': "¿Necesita ayuda con otra pregunta de traducción?",
        # ... other languages
    },
    # ... other intents
}
```

## Implementation Steps

### Step 1: Update BrainState (5 min)
- [ ] Add `followup_question_added: bool` field to BrainState TypedDict
- [ ] Initialize to `False` in brain.py where state is created

### Step 2: Locate Multi-Intent Logic (15 min)
- [ ] Find where "Would you like help with..." is added (likely in brain.py)
- [ ] Add `state['followup_question_added'] = True` after adding that question
- [ ] Verify this happens BEFORE individual intent handlers are called

### Step 3: Create Follow-up Helper (30 min)
- [ ] Create `get_followup_for_intent(intent_type: str, language: str) -> str` function
- [ ] Add INTENT_FOLLOWUPS dictionary with all follow-up questions
- [ ] Include localization for at least English and Spanish

### Step 4: Update Intent Handlers (2 hours)
For each intent handler that needs follow-ups:
- [ ] scripture_lookup
  - [ ] Add follow-up check and append logic
  - [ ] Test single intent scenario
  - [ ] Test multi-intent scenario
- [ ] translation_resources
  - [ ] Add follow-up check and append logic
  - [ ] Test single intent scenario
  - [ ] Test multi-intent scenario
- [ ] language_selection
  - [ ] Add follow-up check and append logic
  - [ ] Test single intent scenario
  - [ ] Test multi-intent scenario
- [ ] status_message
  - [ ] Add follow-up check and append logic
  - [ ] Test single intent scenario
  - [ ] Test multi-intent scenario

### Step 5: Verify Existing Handlers (15 min)
- [ ] Confirm `help` intent follow-up still works
- [ ] Confirm `unsupported` intent follow-up still works
- [ ] Confirm `converse_with_bt_servant` unchanged
- [ ] Confirm error handler has NO follow-up

### Step 6: Testing (1 hour)
- [ ] Single intent with follow-up
- [ ] Multiple intents (only multi-intent follow-up appears)
- [ ] Error case (no follow-up)
- [ ] Localization works for follow-ups

## Test Scenarios

### Scenario 1: Single Intent
Input: "John 3:16"
Expected: Scripture text + "Would you like to look up another Bible passage?"

### Scenario 2: Multiple Intents
Input: "John 3:16 and help"
Expected: Scripture text + "Would you like help with [help intent]?" (NO scripture follow-up)

### Scenario 3: Error Case
Input: Trigger error condition
Expected: Error message with NO follow-up question

### Scenario 4: Existing Follow-up Intent
Input: "help"
Expected: Help text with existing help follow-up (unchanged)

## Success Criteria

1. ✅ All non-exempt intent handlers have follow-up questions
2. ✅ No double follow-ups in any scenario
3. ✅ Multi-intent follow-up takes precedence
4. ✅ Localization works for all follow-ups
5. ✅ Tests pass with ≥65% coverage maintained
6. ✅ No architecture violations (lint-imports passes)

## Files to Modify

1. `bt_servant_engine/core/models.py` or wherever BrainState is defined
2. `brain.py` - for multi-intent logic update
3. `bt_servant_engine/services/intents/scripture_lookup.py`
4. `bt_servant_engine/services/intents/translation_resources.py`
5. `bt_servant_engine/services/intents/language_selection.py`
6. `bt_servant_engine/services/intents/status_message.py`
7. New file: `bt_servant_engine/services/intents/followup_questions.py` (helper functions)

## Estimated Time: 4 hours total