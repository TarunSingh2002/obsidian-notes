---
tags:
  - AIEngineer
  - rag
  - reranking
---

# 08 - Re-Ranking

> Lesson: RAG Re-Ranking
> Code: no repo folder (code from the course notes, below)

## Why
- Embeddings are **lossy compression** (1000 chars → 256 numbers), on **both** query and doc → the similarity score is approximate.
- So a noisy doc can rank high, and the truly relevant doc can drop out of top-k.
- Fix it **after** retrieval with a better (slower) scorer.

## Bi-encoder vs cross-encoder
| | Bi-encoder (retrieval) | Cross-encoder (re-ranking) |
|-|-|-|
| Input | query and doc encoded **separately** | `[CLS] query [SEP] doc [SEP]` **together** |
| Precompute docs? | yes (vector store) | no, runs per (query, doc) pair |
| Speed | fast (1 query embedding per request) | slow (1 model run per candidate) |
| Accuracy | approximate | much better (sees the query-doc relationship) |
- Cross-encoder = BERT-like encoder → `[CLS]` vector → linear → sigmoid → relevance score 0-1.

## Two-stage pipeline (prelims → mains)
```
1M chunks ──bi-encoder──> top 20 candidates ──cross-encoder──> re-sorted ──> top 5 to the LLM
```
- Never run a cross-encoder over the whole knowledge base.
- **Retrieve more than you need** (k > top_n). If k == top_n, it can only re-order, not swap in better docs.

## Code (re-ranker = a compressor in ContextualCompressionRetriever)
```Python
from dotenv import load_dotenv
from langchain_cohere import CohereRerank
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

load_dotenv()                                                  # COHERE_API_KEY

retriever = vectorstore.as_retriever(search_kwargs={'k': 5})   # stage 1: candidate pool

compressor = CohereRerank(model='rerank-english-v3.0', top_n=3)               # paid API, best quality
# compressor = FlashrankRerank(model='ms-marco-MiniLM-L-12-v2', top_n=3)      # free, local (~21.6 MB)

rerank_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                  base_retriever=retriever)
results = rerank_retriever.invoke('How do LLMs handle factual errors in their output?')
```

| | Cohere Rerank | FlashRank |
|-|-|-|
| Type | proprietary API (trial: 10 calls/min) | open-source HF model, runs locally |
| Cost | paid for production | free |
| Use | production | small / local apps |

## Demo result
- "How do LLMs handle factual errors?": base top-5 = LLMs, hallucination, fine-tuning, **RL (rank 4)**, regularization → Cohere: LLMs, **RL ↑ to #2**, hallucination. Weak regularization dropped.
- FlashRank gave the same 3 docs (one query had a different order). Quality is similar.

## Rules
- Bi-encoder shortlists, cross-encoder re-orders.
- Fetch a bigger pool (e.g. 20), re-rank down to top 3-5.
