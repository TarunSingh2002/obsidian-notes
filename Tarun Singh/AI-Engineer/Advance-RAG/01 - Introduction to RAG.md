---
tags:
  - AIEngineer
  - rag
---

# 01 - Introduction to RAG

> Lessons: Why is RAG Important · What is RAG · How RAG Works
> Code: https://github.com/campusx-official/Advanced_Rag_Codes

## Why RAG: 4 problems of a plain LLM
| Problem | Root cause |
|-|-|
| Knowledge cutoff | trained on a frozen dataset, retraining is slow and costly |
| Hallucination | it's a next-word predictor: missing knowledge still gets a fluent (but false) answer |
| No source attribution | sources are dropped during training, nothing to cite |
| No private data | trained on public data only |
- All 4 = a **knowledge** problem → add **external knowledge** at query time.
- Fine-tuning changes *how* the model answers (style / format), not its knowledge → not the fix.
- **In-context learning** (put the knowledge in the prompt) works, but you can't paste 1000 pages:
  - **context window** limit
  - **lost in the middle**: LLMs focus on the start and end of long prompts
- So: **filter** the knowledge, send only what's relevant per query = **RAG**.

## The 3 jobs
| Job | What | Who |
|-|-|-|
| **Retrieval** | fetch the most relevant chunks for this query (semantic, ranked by similarity, keep top-k) | retriever |
| **Augmentation** | query + retrieved chunks → prompt ("context assembly") | pipeline |
| **Generation** | answer from the augmented prompt | LLM |

## Pipeline
```
INDEXING (once, re-run when data changes)
knowledge source → loader → chunking → embedding model → vector store (knowledge base)

QUERY (every request)
query → same embedding model → similarity search → top-k chunks (text + metadata)
      → prompt (context + query + "say I don't know") → LLM → answer
```
- Chunk **before** embedding: a fixed-size vector can't hold the meaning of a 100-page PDF.
- Vector store keeps **vector** (for search) + **text** (for the prompt) + **metadata** (for citations).
- Query must use the **same** embedding model as the documents.

Prompt shape:
```
From the given context, answer the query.
Context: <chunk 1> <chunk 2> <chunk 3> <chunk 4>
Query: <question>
If the relevant context is not available, answer "I don't know."
```

## Where RAG fails (debug in this order)
1. **Retriever failure**: wrong chunks → garbage in, garbage out.
2. **Generator failure**: right chunks, but the LLM misreads the query or the context.
- RAG **reduces** hallucination, doesn't eliminate it.
