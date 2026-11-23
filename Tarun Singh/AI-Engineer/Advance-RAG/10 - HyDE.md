---
tags:
  - AIEngineer
  - rag
  - hyde
---

# 10 - HyDE (Hypothetical Document Embeddings)

> Lesson: HyDE
> Code: `07_rag_hyde/` (`hyde.py`, `hyde_implementation.ipynb`)

## Idea
- A short, vague query vs long, detailed docs = **apples vs oranges** → the query vector can land between clusters.
- LLM writes a **hypothetical document** that answers the query → **search with that** (doc vs doc).
```
query → LLM (temp 0) → hypothetical doc → retriever (same embedding model) → REAL docs → prompt(query + real docs) → LLM → answer
```
- The hypothetical doc = **search key, not the answer** (it may hallucinate). Facts come from the real retrieved docs.
- The final generation still answers the **original query**.

| | RAG Fusion | HyDE |
|-|-|-|
| Changes | number of query phrasings | the *kind* of text compared |
| Ranking | RRF over many retrievals | plain similarity on 1 doc |

- Paper (Dec 2022): generated several docs and **averaged their embeddings**. Today one doc is enough. Multi-doc version: generate N → embed each → `np.mean` → `similarity_search_by_vector`.

## Code (`hyde.py`, simplified)
```Python
import logging
from langchain_core.prompts import ChatPromptTemplate

handler = logging.FileHandler('logs.log', encoding='utf-8')   # utf-8: Windows crashed on '≤' without it
logger = logging.getLogger('hyde_logger')
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class CustomHypotheticalDocumentEmbedder:
    def __init__(self, llm_chain, retriever):
        self.llm_chain = llm_chain
        self.retriever = retriever

    @classmethod
    def from_llm(cls, llm, retriever):
        prompt = ChatPromptTemplate([
            ('system',
             'You are an expert document writer. Given a user query, your task is to generate a hypothetical document '
             'that would directly and thoroughly answer the query. The document should be written as if it were a real, '
             'authoritative passage retrieved from a knowledge base — not a response to the user, but a self-contained '
             "piece of text that contains the answer. Write in a factual, informative tone. Do not include phrases like "
             "'Based on your query' or 'Here is a document'. Output only the hypothetical document text, nothing else."),
            ('human', 'query: {query}'),
        ])
        return cls(llm_chain=prompt | llm, retriever=retriever)

    def invoke(self, query):
        hypothetical_doc = self.llm_chain.invoke({'query': query}).content
        logger.info(f'Hypothetical document:\n{hypothetical_doc}')
        return self.retriever.invoke(hypothetical_doc)        # search with the DOC, not the query
```

## Pipeline
```Python
doc_llm = ChatOpenAI(model='gpt-5-mini', temperature=0)      # writes the search key: focused, repeatable
answer_llm = ChatOpenAI(model='gpt-5-mini', temperature=0.5) # final answer: a bit of freedom is fine

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
hyde = CustomHypotheticalDocumentEmbedder.from_llm(llm=doc_llm, retriever=retriever)

question = 'How do transformer models use attention to process sequences?'
docs = hyde.invoke(question)          # → transformers, attention mechanism, LLMs docs

context_parts = []
for doc in docs:
    context_parts.append(doc.page_content)
context = '\n\n'.join(context_parts)
answer = (prompt | answer_llm).invoke({'context': context, 'question': question})   # same "only from context" prompt
```

## Rules
- Compare like with like. Embed the hypothetical doc with the **same model** as the store (the retriever does this for you).
- Prompt it to look like a corpus passage (no "Here is a document..." preamble), temperature 0.
- Never use the hypothetical doc as context.
