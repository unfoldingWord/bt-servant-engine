# ADR 002: Use ChromaDB for Vector RAG Indexing

## Status

✅ Accepted

## Context

To rapidly validate the BT Servant concept, we needed to get a functional Retrieval-Augmented Generation (RAG) chatbot in front of users with minimal setup friction. Our immediate goals were:

- Load and index a small corpus of translation resources
- Retrieve semi-relevant chunks of text in response to natural language queries
- Get the full RAG loop working quickly for feedback and iteration

Given these constraints, we prioritized ease of integration and fast local iteration over long-term scalability and retrieval accuracy.

## Decision

We chose **ChromaDB** because:

- It’s lightweight and easy to spin up for local and containerized development
- No external service provisioning is required
- It integrates well with the Python ecosystem and can be queried with basic cosine similarity out of the box
- It allowed us to validate the end-to-end flow of the bot (input → retrieval → LLM → response) within hours

## Consequences

- Cosine similarity alone is a blunt instrument and occasionally returns irrelevant or misleading results
- More advanced hybrid search techniques (e.g., re-ranking, filtering) require additional layers beyond vanilla ChromaDB
- For production use, we may migrate to **Weaviate**, which offers:
  - Hybrid search (vector + keyword)
  - Better metadata filtering
  - Scalable multi-tenant infrastructure

- Alternative approaches such as **Cognee**, which combines vector search with structured **knowledge graph traversal**, may yield even more accurate and interpretable responses in the long term—but come at the cost of setup time and ontology construction

