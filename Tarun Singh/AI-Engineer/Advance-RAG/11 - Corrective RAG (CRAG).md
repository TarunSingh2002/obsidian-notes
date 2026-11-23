---
tags:
  - AIEngineer
  - rag
  - crag
  - langgraph
---

# 11 - Corrective RAG (CRAG)

> Lesson: Corrective RAG (paper: *Corrective Retrieval Augmented Generation*, 2024)
> Code: no repo folder (code from the course notes; LangGraph + FAISS + Tavily)

## Problem: blind trust
- A vector DB **always returns something**, even when nothing relevant exists.
- Demo (ML books): "What is a transformer?" → a fluent correct answer, but **none of the 4 chunks mention transformers** (MLP, CNN, index pages). The LLM used its own weights. For a private topic (leave policy) → hallucination.
- 🎯 A fluent answer ≠ good retrieval. **Inspect the retrieved docs.**

## CRAG = evaluate retrieval → act on the verdict
| Verdict | Rule (per-doc score 0-1) | Action |
|-|-|-|
| **CORRECT** | any doc > **0.7** | refine good docs → generate (internal knowledge) |
| **INCORRECT** | all docs < **0.3** | rewrite query → **web search** → refine → generate (external) |
| **AMBIGUOUS** | anything else | good docs **+** web docs → refine → generate (both) |
- Only docs scoring **> 0.3** are kept for generation, even when the verdict is CORRECT.
- Paper uses a fine-tuned T5-Large (not released) → here `gpt-4o-mini`, temperature 0. Thresholds 0.7 / 0.3 are the lecturer's choice.

```
START → retrieve → eval_each_doc ─CORRECT──────────────────────────────→ refine → generate → END
                                 └INCORRECT / AMBIGUOUS→ rewrite_query → web_search ↗
```

## Knowledge refinement (decompose → filter → recompose)
- A relevant chunk still carries junk (900-char cuts ignore topics). "What is gradient descent?" → keep S1, S2, drop the CNN sentences.
- Split into sentence **strips** → LLM keep / drop each → join the kept ones.

## Code
```Python
import re
from typing import List, TypedDict
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
retriever = FAISS.from_documents(chunks, OpenAIEmbeddings(model='text-embedding-3-large')) \
                 .as_retriever(search_type='similarity', search_kwargs={'k': 4})
tavily = TavilySearchResults(max_results=5)            # TAVILY_API_KEY
UPPER_TH = 0.7
LOWER_TH = 0.3


class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str


def join_docs(docs):
    parts = []
    for d in docs:
        parts.append(d.page_content)
    return '\n\n'.join(parts).strip()


# ---------- 1. retrieve ----------
def retrieve_node(state):
    return {'docs': retriever.invoke(state['question'])}


# ---------- 2. evaluate each doc ----------
class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a strict retrieval evaluator for RAG.\n'
               'You will be given ONE retrieved chunk and a question.\n'
               'Return a relevance score in [0.0, 1.0].\n'
               '- 1.0: chunk alone is sufficient to answer fully/mostly\n'
               '- 0.0: chunk is irrelevant\n'
               'Be conservative with high scores.\nAlso return a short reason.\nOutput JSON only.'),
    ('human', 'Question: {question}\n\nChunk:\n{chunk}'),
])
doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def eval_each_doc_node(state):
    scores = []
    good = []
    for d in state['docs']:
        out = doc_eval_chain.invoke({'question': state['question'], 'chunk': d.page_content})
        scores.append(out.score)
        if out.score > LOWER_TH:
            good.append(d)

    best = max(scores)
    if best > UPPER_TH:
        return {'good_docs': good, 'verdict': 'CORRECT', 'reason': f'A chunk scored > {UPPER_TH}'}
    if best < LOWER_TH:                                  # all below the lower threshold
        return {'good_docs': [], 'verdict': 'INCORRECT', 'reason': f'All chunks < {LOWER_TH}'}
    return {'good_docs': good, 'verdict': 'AMBIGUOUS', 'reason': 'Mixed relevance signals'}


def route_after_eval(state):
    if state['verdict'] == 'CORRECT':
        return 'refine'
    return 'rewrite_query'                               # INCORRECT and AMBIGUOUS share a path


# ---------- 3. rewrite query (for web search) ----------
class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Rewrite the user question into a web search query composed of keywords.\nRules:\n'
               '- Keep it short (6-14 words).\n'
               '- If the question implies recency (recent/latest/last week/last month), add a constraint like (last 30 days).\n'
               '- Do NOT answer the question.\n- Return JSON with a single key: query'),
    ('human', 'Question: {question}'),
])
rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_query_node(state):
    return {'web_query': rewrite_chain.invoke({'question': state['question']}).query}


# ---------- 4. web search ----------
def web_search_node(state):
    q = state.get('web_query') or state['question']
    web_docs = []
    for r in tavily.invoke({'query': q}) or []:
        title = r.get('title', '')
        url = r.get('url', '')
        content = r.get('content', '') or r.get('snippet', '')
        text = f'TITLE: {title}\nURL: {url}\nCONTENT:\n{content}'
        web_docs.append(Document(page_content=text, metadata={'url': url, 'title': title}))
    return {'web_docs': web_docs}


# ---------- 5. refine (strips) ----------
def decompose_to_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    strips = []
    for s in sentences:
        if len(s.strip()) > 20:                          # drop tiny fragments
            strips.append(s.strip())
    return strips


class KeepOrDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a strict relevance filter.\n'
               'Return keep=true only if the sentence directly helps answer the question.\n'
               'Use ONLY the sentence. Output JSON only.'),
    ('human', 'Question: {question}\n\nSentence:\n{sentence}'),
])
filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


def refine(state):
    if state['verdict'] == 'CORRECT':
        docs_to_use = state['good_docs']                       # internal only
    elif state['verdict'] == 'INCORRECT':
        docs_to_use = state['web_docs']                        # external only
    else:
        docs_to_use = state['good_docs'] + state['web_docs']   # both

    strips = decompose_to_sentences(join_docs(docs_to_use))
    kept = []
    for s in strips:
        if filter_chain.invoke({'question': state['question'], 'sentence': s}).keep:
            kept.append(s)
    return {'strips': strips, 'kept_strips': kept, 'refined_context': '\n'.join(kept).strip()}


# ---------- 6. generate ----------
answer_prompt = ChatPromptTemplate.from_messages([
    ('system', "Answer only from the context. If the context is empty or insufficient, say \"I don't know\"."),
    ('human', 'Question: {question}\n\nContext:\n{context}'),
])


def generate(state):
    out = (answer_prompt | llm).invoke({'question': state['question'], 'context': state['refined_context']})
    return {'answer': out.content}


g = StateGraph(State)
g.add_node('retrieve', retrieve_node)
g.add_node('eval_each_doc', eval_each_doc_node)
g.add_node('rewrite_query', rewrite_query_node)
g.add_node('web_search', web_search_node)
g.add_node('refine', refine)
g.add_node('generate', generate)
g.add_edge(START, 'retrieve')
g.add_edge('retrieve', 'eval_each_doc')
g.add_conditional_edges('eval_each_doc', route_after_eval,
                        {'refine': 'refine', 'rewrite_query': 'rewrite_query'})
g.add_edge('rewrite_query', 'web_search')
g.add_edge('web_search', 'refine')
g.add_edge('refine', 'generate')
g.add_edge('generate', END)
app = g.compile()

result = app.invoke({'question': 'Batch normalization versus layer normalization'})   # → AMBIGUOUS
```

## Test queries
| Query | Verdict |
|-|-|
| Bias-variance tradeoff | CORRECT (in the books) |
| AI news from last month | INCORRECT → web |
| Batch norm vs layer norm | AMBIGUOUS (books have batch norm only) |
- Query rewriting barely helped in practice (Tavily handles vague queries). It's kept because the paper has it, and it's cheap.

## Rules
- Never trust retrieval blindly. Ask "did retrieval work?" before generating.
- Filter at **sentence** level, not chunk level.
- Strong evidence → internal. Failed → external. Mixed → both.
