---
tags:
  - AIEngineer
  - rag
  - self-rag
  - langgraph
---

# 12 - Self-RAG (Self-Reflective RAG)

> Lesson: Self RAG
> Code: no repo folder. The code below is rebuilt from the course notes (LangGraph, OpenAI, no fine-tuned model like the paper). Use case: "Nexa AI Solutions" company docs, chunks 600 / 150.

## 3 problems of fixed retrieve → generate
1. **Indiscriminate retrieval**: "seconds in a minute?" → retrieved junk → "*about* 60 seconds, depending on context". Less confident + wasted compute.
2. **Blind trust**: "What causes diabetes?" → chunk says what diabetes *is* → circular answer. **Similarity ≠ relevance.**
3. **No verification** of the answer.

## The 4 reflection questions → 4 nodes
| Question | Node | On failure |
|-|-|-|
| Is retrieval needed? | `decide_retrieval` | answer directly from LLM knowledge |
| Are the docs relevant? | `is_relevant` (per doc) | none relevant → "No relevant document found" |
| Is the answer grounded? | `is_supported` (fully / partially / no support) | `revise_answer` loop (cap: 10) |
| Is the answer useful? | `is_use` | `rewrite_question` → retrieve again (cap: 3) → "No answer found" |
- Grounded ≠ useful: an answer can be 100% from the docs and still not answer the question ("why does ice float?" → "ice is solid water").
- Partial support example: the docs say "24 paid leaves" + "sick & casual leave exist". The answer "24 leaves, **which include** sick & casual" = an invented link.

```
START → decide_retrieval ─no──→ generate_direct → END
              └yes→ retrieve → is_relevant ─none→ no_relevant_docs → END
                                  └some→ generate_from_context → is_supported
   is_supported ─partial/none & retries<10→ revise_answer → is_supported (loop)
                └fully (or cap hit)→ is_use ─useful→ END
                                            └not useful & tries<3→ rewrite_question → retrieve (loop)
                                            └not useful & cap hit→ no_answer_found → END
```

## Code
```Python
from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
MAX_REVISIONS = 10
MAX_REWRITE_TRIES = 3


class State(TypedDict):
    question: str
    retrieval_query: str          # starts = question, changes on rewrite
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    is_supported: str             # fully supported / partially supported / no support
    evidence: List[str]           # debugging aid
    retries: int
    is_use: str                   # useful / not useful
    reason: str


# ---------- Q1: retrieve at all? ----------
class ShouldRetrieve(BaseModel):
    should_retrieve: bool = Field(description='True if external documents are needed to answer reliably, else False')


decide_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Decide if retrieval is needed. Return JSON matching the schema.\n'
               'should_retrieve=True if answering requires specific facts, citations, or info likely not in the model.\n'
               'should_retrieve=False for general explanations, definitions, or reasoning.\n'
               'If unsure, choose True.'),
    ('human', 'Question: {question}'),
])
decide_chain = decide_prompt | llm.with_structured_output(ShouldRetrieve)


def decide_retrieval(state):
    out = decide_chain.invoke({'question': state['question']})
    return {'need_retrieval': out.should_retrieve}


def route_after_decide(state):
    if state['need_retrieval']:
        return 'retrieve'
    return 'generate_direct'


def generate_direct(state):
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Answer using general knowledge. Don't assume access to external documents. "
                   "If unsure or it needs specific sources, say \"I don't know based on my general knowledge\"."),
        ('human', '{question}'),
    ])
    return {'answer': (prompt | llm).invoke({'question': state['question']}).content}


def retrieve(state):
    return {'docs': retriever.invoke(state['retrieval_query'])}      # uses the (possibly rewritten) query


# ---------- Q2: relevant docs ----------
class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(description='True if the document helps answer the question, else False')


relevance_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You judge document relevance. A document is relevant if it contains information useful for answering the question.'),
    ('human', 'Question: {question}\n\nDocument:\n{document}'),
])
relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceDecision)


def is_relevant(state):
    relevant_docs = []
    for doc in state['docs']:
        decision = relevance_chain.invoke({'question': state['question'], 'document': doc.page_content})
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {'relevant_docs': relevant_docs}


def route_after_relevance(state):
    if len(state['relevant_docs']) > 0:
        return 'generate_from_context'
    return 'no_relevant_docs'


def no_relevant_docs(state):
    return {'answer': 'No relevant document found.'}     # placeholder: could add web search here


def generate_from_context(state):
    parts = []
    for doc in state['relevant_docs']:
        parts.append(doc.page_content)
    context = '\n\n'.join(parts)
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a business RAG assistant. Answer using ONLY the provided context.'),
        ('human', 'Question: {question}\n\nContext:\n{context}'),
    ])
    answer = (prompt | llm).invoke({'question': state['question'], 'context': context}).content
    return {'context': context, 'answer': answer}


# ---------- Q3: grounded? ----------
class SupportCheck(BaseModel):
    is_supported: Literal['fully supported', 'partially supported', 'no support']
    evidence: List[str] = Field(description='Facts from the context that back the answer')


support_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Check if EVERY claim in the answer is present in the context.\n'
               'fully supported = all facts come from the context.\n'
               'partially supported = some facts are not in the context.\n'
               'no support = the key claims are not in the context.\n'
               'Also list the supporting facts you found.'),
    ('human', 'Question: {question}\n\nAnswer: {answer}\n\nContext:\n{context}'),
])
support_chain = support_prompt | llm.with_structured_output(SupportCheck)


def is_supported(state):
    out = support_chain.invoke({'question': state['question'], 'answer': state['answer'], 'context': state['context']})
    return {'is_supported': out.is_supported, 'evidence': out.evidence}


def route_after_support(state):
    if state['is_supported'] == 'fully supported':
        return 'is_use'
    if state['retries'] >= MAX_REVISIONS:
        return 'is_use'                                   # cap hit → keep the answer as-is
    return 'revise_answer'


def revise_answer(state):
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'Rewrite the answer using ONLY direct quotes from the context, as bullet points. '
                   'No explanations, no extra wording.'),
        ('human', 'Question: {question}\n\nCurrent answer: {answer}\n\nContext:\n{context}'),
    ])
    new_answer = (prompt | llm).invoke({'question': state['question'], 'answer': state['answer'],
                                        'context': state['context']}).content
    return {'answer': new_answer, 'retries': state['retries'] + 1}


# ---------- Q4: useful? ----------
class UsefulnessCheck(BaseModel):
    is_use: Literal['useful', 'not useful']
    reason: str


use_chain = ChatPromptTemplate.from_messages([
    ('system', 'Judge whether the answer actually addresses the question.'),
    ('human', 'Question: {question}\n\nAnswer: {answer}'),
]) | llm.with_structured_output(UsefulnessCheck)


def is_use(state):
    out = use_chain.invoke({'question': state['question'], 'answer': state['answer']})
    return {'is_use': out.is_use, 'reason': out.reason}


def route_after_use(state):
    if state['is_use'] == 'useful':
        return END
    if state['rewrite_tries'] >= MAX_REWRITE_TRIES:
        return 'no_answer_found'
    return 'rewrite_question'


class RewrittenQuery(BaseModel):
    retrieval_query: str


rewrite_chain = ChatPromptTemplate.from_messages([
    ('system', 'Rewrite the question into a query optimized for vector retrieval over internal company PDFs. '
               'Keep it short, preserve key entities, add 2-5 high-signal keywords.'),
    ('human', 'Question: {question}\nPrevious query: {retrieval_query}\nAnswer that was not useful: {answer}'),
]) | llm.with_structured_output(RewrittenQuery)


def rewrite_question(state):
    out = rewrite_chain.invoke({'question': state['question'], 'retrieval_query': state['retrieval_query'],
                                'answer': state['answer']})
    return {'retrieval_query': out.retrieval_query, 'rewrite_tries': state['rewrite_tries'] + 1, 'retries': 0}


def no_answer_found(state):
    return {'answer': 'No answer found.'}


g = StateGraph(State)
for name, fn in [('decide_retrieval', decide_retrieval), ('generate_direct', generate_direct),
                 ('retrieve', retrieve), ('is_relevant', is_relevant), ('no_relevant_docs', no_relevant_docs),
                 ('generate_from_context', generate_from_context), ('is_supported', is_supported),
                 ('revise_answer', revise_answer), ('is_use', is_use),
                 ('rewrite_question', rewrite_question), ('no_answer_found', no_answer_found)]:
    g.add_node(name, fn)

g.add_edge(START, 'decide_retrieval')
g.add_conditional_edges('decide_retrieval', route_after_decide, ['retrieve', 'generate_direct'])
g.add_edge('generate_direct', END)
g.add_edge('retrieve', 'is_relevant')
g.add_conditional_edges('is_relevant', route_after_relevance, ['generate_from_context', 'no_relevant_docs'])
g.add_edge('no_relevant_docs', END)
g.add_edge('generate_from_context', 'is_supported')
g.add_conditional_edges('is_supported', route_after_support, ['is_use', 'revise_answer'])
g.add_edge('revise_answer', 'is_supported')
g.add_conditional_edges('is_use', route_after_use, [END, 'rewrite_question', 'no_answer_found'])
g.add_edge('rewrite_question', 'retrieve')
g.add_edge('no_answer_found', END)
app = g.compile()

q = 'Describe Nexa AI company culture'
result = app.invoke({'question': q, 'retrieval_query': q, 'retries': 0, 'rewrite_tries': 0})
```

## Demo results
| Question | What happened |
|-|-|
| Who is the CEO of Nexa AI? | retrieve → 4 docs, **1 relevant** → fully supported → useful |
| What is machine learning? | no retrieval → direct answer |
| How many employees? | fully supported (85+) |
| Describe company culture | partially supported → **1 revision** → fully supported |
| Free trial? How many days? | **no support**: the LLM said "14 days" from its weights (caught!) |
| Refund policy? | no support → not useful → "No answer found" |
- Limitation of the notebook: after the revision cap, `is_use` doesn't recheck grounding.

## Rules
- Retrieve only when needed (if unsure → retrieve).
- Filter docs for relevance **before** generating.
- Verify **groundedness** and **usefulness** separately.
- **Cap every loop** + a fallback ("No answer found").
- Build the graph one node at a time and test each step.
