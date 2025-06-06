# ADR 001: Use LangGraph for AI Orchestration

## Status

✅ Accepted

## Context

We needed a flexible, transparent, and modular framework to orchestrate the complex multi-step reasoning required by the BT Servant AI system. This includes:

- Intent detection
- Message preprocessing
- Language detection
- Contextual retrieval using vector databases
- Integration with multiple LLMs and APIs
- Fallback and conditional logic
- Final response translation and formatting

Traditional flow control using chained functions or FSMs quickly became brittle and opaque as the complexity of the system grew.

## Decision

We chose **LangGraph** because:

- It provides a declarative and composable approach to stateful AI workflows
- It enables easy integration of conditional logic and branching using simple functions
- It is compatible with our chosen LLM backend (OpenAI/Groq)
- The `StateGraph` abstraction makes the flow of information and decisions easy to visualize and reason about
- It allows us to treat each step in the system (e.g., preprocessing, retrieval, chunking) as an independent, testable unit

## Consequences

- The mental model of a state graph introduces some initial learning curve
- Runtime inspection of execution flow requires thoughtful logging
- Each node must manage side effects carefully (e.g., DB calls, logging, etc.)
- Tightly couples orchestration logic to LangGraph’s APIs, which would need to be swapped out if the framework is ever deprecated

