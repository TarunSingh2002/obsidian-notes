---
tags:
  - AIEngineer
  - rag
  - retrievers
---

# 06 - Retrievers (Similarity, Threshold, MMR, BM25, Hybrid)

> Lessons: Similarity & Threshold Retrievers · MMR Retriever · BM25, Hybrid and Ensemble Retriever
> Code: `05_retrievers/notebooks/` (01_similarity_search → 06_custom_ensemble_retriever)

## Basics
- Retriever = text query **in** → list of `Document` (text + metadata) **out**. Never vectors.
- Retrieval ≈ 80% of a basic RAG's quality.
- 🎯 Interview trap: the vector store keeps **only document embeddings**. The query vector is made per request, used once, discarded.
- Why `as_retriever()` and not `similarity_search()`? It has tuning (threshold, MMR) and it's a **Runnable** (plugs into chains).

```Python
retriever = vectorstore.as_retriever(search_type='similarity',          # or similarity_score_threshold, mmr
                                     search_kwargs={'k': 3})
results = retriever.invoke('How do cells generate their energy?')
```
- **k = 2-5**. Too high → weak docs sneak in. Too low → misses.
- `search_kwargs={'k': 3, 'filter': {'topic': 'biology'}}` = metadata filter. It's static while queries are dynamic → use it deliberately.

## Similarity score threshold
```Python
retriever = vectorstore.as_retriever(search_type='similarity_score_threshold',
                                     search_kwargs={'score_threshold': 0.7})
```
- Returns **all** docs above the threshold (0, 1 or 50). No good match → empty list (better than forcing k bad docs).
- ⚠ Chroma quirk: the score it returns behaves like **1 − similarity** (lower = better). Real cosine = `1 - score`. Check with `similarity_search_with_score` before picking a threshold.
- Set the metric per collection: `Chroma(..., collection_configuration={'hnsw': {'space': 'cosine'}})` (`cosine`, `l2`, `ip`).

## MMR (Maximal Marginal Relevance) = relevance + **diversity**
- Problem: top-k can be 3 near-duplicate chunks (one idea said 3 ways) → wasted context.
- Step 1: plain similarity → `fetch_k` candidates (~10× k).
- Step 2: pick the most relevant first, then repeatedly pick the candidate with the best:
```
MMR(D) = λ · sim(D, query)  −  (1 − λ) · max sim(D, already selected)
```
- Example λ = 0.5, D1 picked: D2 (q 0.90, D1 0.88) → 0.01. D3 (q 0.75, D1 0.22) → 0.265 → **D3 wins**.
```Python
retriever = vectorstore.as_retriever(search_type='mmr',
                                     search_kwargs={'k': 3, 'fetch_k': 10, 'lambda_mult': 0.7})
```
| lambda_mult | Result in the demo |
|-|-|
| 1 | = plain similarity (3 duplicates) |
| **0.7** | on-topic + distinct sub-topics ✅ |
| 0.5 | 3rd doc drifted off-topic |
| 0 | random-ish topics |
- Tune λ per use case. Good for research papers / exploratory questions.

## BM25 (keyword, sparse)
- TF-IDF with 2 fixes: **term-frequency saturation** (keyword stuffing stops helping) + **document length normalization**.
- No embedding model. Matches literal words: "killing pathogens" misses "neutralise pathogens".
- 🚫 **Never standalone**. Use it as the keyword half of hybrid search.
```Python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(docs, k=2, bm25_variant='plus')   # 'plus' for short docs
```

## Hybrid search = semantic + BM25, merged by RRF
- `EnsembleRetriever` merges several retrievers' lists with **Reciprocal Rank Fusion** (ranks, not scores: scales differ):
```
RRF(d) = Σ over retrievers  weight · 1 / (rank + 60)        (rank starts at 0)
```
- A doc found by several retrievers gets its scores **added** → floats up.
```Python
from langchain_classic.retrievers import EnsembleRetriever

chroma_retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 4})
ensemble = EnsembleRetriever(retrievers=[chroma_retriever, bm25_retriever],
                             weights=[0.8, 0.2])           # semantic-heavy usually best
results = ensemble.invoke('How do vaccines protect against disease?')
```
- Demo: semantic found all 4 health docs. BM25 pulled in an off-topic REST API doc. Ensemble 0.8/0.2 → 4 good docs on top, the junk last.

### Custom ensemble (`langchain_classic` support ends Dec 2026)
```Python
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


class MyEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]
    rrf_k: int = 60

    def _get_relevant_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        doc_scores = {}                                  # page_content -> [score, doc]
        for i in range(len(self.retrievers)):
            results = self.retrievers[i].invoke(query)
            weight = self.weights[i]
            for rank in range(len(results)):
                doc = results[rank]
                rrf_score = weight * (1.0 / (rank + self.rrf_k))
                key = doc.page_content
                if key in doc_scores:
                    doc_scores[key][0] = doc_scores[key][0] + rrf_score   # found again → add up
                else:
                    doc_scores[key] = [rrf_score, doc]

        ranked = sorted(doc_scores.values(), key=lambda pair: pair[0], reverse=True)
        final_docs = []
        for score, doc in ranked:
            final_docs.append(doc)
        return final_docs
```
- Gave exactly the same output as the built-in one.

## Rules
- Relevance (similarity) ≠ diversity (MMR) ≠ exact keywords (BM25). **Hybrid usually beats either alone.**
- Merge multiple retrievers by **rank (RRF)**, never by raw scores.
