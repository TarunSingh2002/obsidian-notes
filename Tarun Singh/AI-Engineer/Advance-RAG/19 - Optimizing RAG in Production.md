---
tags:
  - AIEngineer
  - rag
  - production
  - optimization
---

# 19 - Optimizing RAG Pipelines in Production

> Lesson: Optimising RAG Pipeline (capstone part 3). Theory only.
> No fixed recipe: learn the techniques → know when each fits → **evaluate and experiment** before keeping a change.

## 6 areas
1. Pipeline (component) optimization 2. Caching 3. Cost 4. Monitoring 5. Pitfalls 6. Security
- Optimize for: accuracy, context quality, cost, latency, scaling. Prefer **component-wise** over end-to-end (black box).
- Already "free wins": modular code, caching, session isolation, eval before deploy, query routing, **streaming** (hides latency, lower perceived latency).

## Ingestion (one-off, still matters)
| Component | Lever |
|-|-|
| Loader | `lazy_load()` for GB-size docs |
| Splitter | **chunk size** is the key knob: bigger = more info but worse embeddings + more storage / tokens. Fix: parent-document retriever |
| Embeddings | proprietary (better, paid, rate limits) vs open source. Fewer dims = faster + cheaper (MRL) |
| Vector store | small: Chroma / FAISS · medium: Qdrant / Pinecone · large: Milvus / Weaviate. **Quantization** 32 → 8 bit ≈ 4× less storage |

## Retrieval (runs every query → most effort)
- **Query routing**: skip retrieval if the LLM knows → less cost + latency.
- Rewrite / multi-query for ambiguous queries → run sub-query retrievals **in parallel (async)**.

| Technique | Gain | Cost |
|-|-|-|
| Hybrid (dense + BM25) | exact keywords | a bit of latency |
| Lower-dim embeddings | faster search | some detail lost |
| MMR | less redundancy | tiny |
| RAG Fusion (multi-retriever + RRF) | accuracy, coverage | latency |
| Contextual compression | clean context | LLM calls ($$, latency) |
- Ladder: basic retriever < multi-retriever < LLM-based. **Use the lightest one that meets your accuracy.**
- **Re-rank only when context precision is low** (it only re-orders).
- Context: tight and relevant. Key info at the **start / end** (lost in the middle).
- Slowest + costliest steps: **re-ranking** and **generation**.

## Caching
| Cache | Key → value | Use |
|-|-|-|
| **Document embeddings** | hash(chunk) → vector | ✅ common (50 cached + 5 new chunks = 5 API calls, not 55) |
| **Query vectors** | hash(query) → vector | ✅ common (exact match only) |
| Query → response (semantic cache) | similar query → stored answer | FAQ bots only |
| Query → retrieved docs | similar query → chunks | rare (retrieval should be fresh per query) |
- Stores: RAM / file / Redis.
- **Invalidate** when the chunking strategy, embedding model or knowledge source changes. **Cache only what's static.**

## Cost
| Source | Fix |
|-|-|
| **LLM** (biggest) | structured output for helper calls (fewer tokens), smaller models for helpers, fewer pipeline steps, smaller context, cap output (prompt "max 200 words" + `max_tokens`) |
| Re-ranker | skip if precision is fine; router skips whole pipelines |
| Embeddings | `-3-small`, open source, cache, **MRL** shorter vectors |
| Vector store | shorter vectors (biggest effect), fewer chunks, routing |

### MRL (Matryoshka Representation Learning)
- Trained with a separate loss on nested prefixes (128, 256, 512, 1024, 1536) → the **first dims hold the most important info**.
- Truncate 1536 → 256 / 512 with a small quality loss, **no new model call** (OpenAI `text-embedding-3-*` support it: `dimensions=`).
- Docs and queries must use the same model **and** the same size. The index and cache must match.

## Monitoring
- RAG level: **LangSmith** (tracing), Arize Phoenix, DeepEval (online eval).
- App / infra: Prometheus (metrics + alerts), Grafana (dashboards), PagerDuty (incidents).

## Common pitfalls → fixes
| Pitfall | Fix |
|-|-|
| Random chunk size | experiment on **one shared eval dataset**, same metrics, log and compare |
| Ingestion built once | scheduled re-ingestion (add new, remove stale) + embedding cache |
| Semantic search only | hybrid / Graph RAG / self-query / ensemble, depending on the data |
| No "I don't know" | system prompt rule: empty / irrelevant context → "I don't know" |
| Set-and-forget | RAG is a **living system**: scheduled evals, version the pipeline (drift) |
| Embedding model mismatch | same model for ingestion and queries |
| Context overflow | short, dense context |
| No fallback | backup models + a plain "we're having issues" message |

## Security & compliance
- **Access control** (Google auth / OAuth / JWT). **Prompt injection** (docs, prompt, query).
- **Data privacy**: context goes to LLM APIs, vectors to the cloud. GDPR (EU), HIPAA (US health), SOC 2, PII masking ("Employee 1").
- **Audit logs** of inputs / outputs. **Rate limits per user**. **Output sanitization**.
- Model deprecations → upgrade before the date. **Pin library versions.**
- Protect the **system prompt** (it's product IP): guardrails against "show the text before this prompt".
- **Denial-of-wallet** ("summarize every document", "write 10,000 words") → cap output tokens, retries (≤ 2 rewrites, ≤ 3 retrievals, ≤ 10 docs each).

## Rules
- Pick parameters by **experiment**, never by default.
- Same embeddings (model + size) for docs and queries.
- Treat RAG as a living system: re-ingest, trace, re-evaluate.
- **Hard limits** on every unbounded resource (tokens, context, retries, requests per user).
- Design the failure paths: "I don't know", plus a clean fallback when a service is down.
