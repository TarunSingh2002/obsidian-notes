---
tags:
  - AIEngineer
  - rag
  - rag-fusion
---

# 09 - RAG Fusion

> Lesson: RAG Fusion
> Code: `06_rag_fusion/` (`rag_fusion.py` class, `rag_pipeline.ipynb`, `rag_pipeline_ensemble.ipynb`)

## Idea
- The **user's query is the bottleneck**: users don't know what's in the knowledge base or which words match.
- Multi-query fixes coverage, but **plain dedupe throws away rank info** (D2 was #1 in two lists, yet becomes just "one entry").
- **RAG Fusion = multi-query + RRF re-ranking** (repeats add to a score instead of being dropped).

```
query → LLM → N sub-queries → retrieve each → RRF fuse → top-k → prompt → LLM
```

## RRF (rewards consistency)
```
RRF(d) = Σ over lists  1 / (60 + rank)        rank starts at 1, missing = 0
```
| Doc | Q1 | Q2 | Q3 | RRF |
|-|-|-|-|-|
| D2 | 1 | 2 | 1 | **0.0489** 🥇 |
| D3 | - | 1 | 3 | 0.0323 |
| D1 | 3 | 3 | - | 0.0317 |
| D4 / D5 | one list at rank 2 | | | 0.0161 |
- Gaps between ranks are tiny (1/61 vs 1/63) → **showing up in more lists beats a better position in one list**. Rank still breaks ties (D3 > D1).
- `k = 60`: large k → consistency dominates. Small k → top rank dominates.

## Code (`rag_fusion.py`, simplified)
```Python
import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.ensemble import EnsembleRetriever

logger = logging.getLogger('rag_fusion_logger')        # logs sub-queries, per-query docs, RRF scores


class SubQuerySchema(BaseModel):
    sub_queries: list[str] = Field(..., description='List of sub-queries to be generated from the main query.')


class RAGFusion:
    def __init__(self, retriever, llm_chain=None, num_subqueries=3, k=5):
        self.retriever = retriever
        self.llm_chain = llm_chain
        self.num_subqueries = num_subqueries
        self.k = k

    @classmethod
    def from_llm(cls, llm, retriever, num_subqueries=3, k=5):
        prompt = ChatPromptTemplate([
            ('system', 'You are a helpful assistant that generates sub-queries from a main query to enhance retrieval.'),
            ('user', "Given the main query: '{main_query}', generate {num_subqueries} sub-queries "
                     'that can be used to retrieve relevant documents.'),
        ])
        llm_chain = prompt | llm.with_structured_output(SubQuerySchema)
        return cls(retriever=retriever, llm_chain=llm_chain, num_subqueries=num_subqueries, k=k)

    @classmethod
    def from_retrievers(cls, base_retrievers, weights=None, k=5):
        # several retrievers on the same query (e.g. similarity + MMR, or BM25 + vector)
        ensemble = EnsembleRetriever(retrievers=base_retrievers, weights=weights)   # does RRF itself
        return cls(retriever=ensemble, k=k)

    def _reciprocal_rank_fusion(self, retrieved_docs):
        doc_scores = {}                                    # page_content -> [score, doc]
        for doc_list in retrieved_docs:
            rank = 1
            for doc in doc_list:
                rrf_score = 1.0 / (rank + 60)
                key = doc.page_content
                if key in doc_scores:
                    doc_scores[key][0] = doc_scores[key][0] + rrf_score
                else:
                    doc_scores[key] = [rrf_score, doc]
                rank = rank + 1
        ranked = sorted(doc_scores.values(), key=lambda pair: pair[0], reverse=True)
        final_docs = []
        for score, doc in ranked:
            final_docs.append(doc)
        return final_docs

    def invoke(self, query):
        if self.llm_chain is None:                         # ensemble mode
            return self.retriever.invoke(query)[:self.k]
        result = self.llm_chain.invoke({'main_query': query, 'num_subqueries': self.num_subqueries})
        logger.info(f'sub-queries: {result.sub_queries}')
        all_docs = []
        for sub_query in result.sub_queries:
            all_docs.append(self.retriever.invoke(sub_query))
        return self._reciprocal_rank_fusion(all_docs)[:self.k]
```

## Full pipeline
```Python
llm = ChatOpenAI(model='gpt-5-mini')
base_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
fusion = RAGFusion.from_llm(llm=llm, retriever=base_retriever, num_subqueries=2, k=3)
# or: RAGFusion.from_retrievers([similarity_retriever, mmr_retriever], weights=[0.5, 0.5], k=3)

question = 'How does NotebookLM retrieve relevant information from uploaded documents?'
docs = fusion.invoke(question)

context_parts = []
for doc in docs:
    context_parts.append(doc.page_content)
context = '\n\n'.join(context_parts)

prompt = ChatPromptTemplate.from_messages([
    ('system', "Answer ONLY from the context. If the answer is not there, say \"I don't know\".\n\nContext:\n{context}"),
    ('human', '{question}'),
])
answer = (prompt | llm).invoke({'context': context, 'question': question})
```
- Only the **retrieval** step changes vs vanilla RAG. Augmentation and generation stay the same.

## Trade-offs
- ✅ robust to bad queries, simple, explainable.
- ❌ an extra LLM call + N retrievals (latency, cost). Pick N deliberately.
- **Log the intermediate steps** (sub-queries, per-query docs, RRF scores) or you can't debug it.
