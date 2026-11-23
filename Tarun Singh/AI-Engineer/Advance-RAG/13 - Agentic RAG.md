---
tags:
  - AIEngineer
  - rag
  - agentic-rag
  - langgraph
---

# 13 - Agentic RAG

> Lessons: Agentic RAG Concept · Agentic RAG Implementation using LangGraph
> Code: `08_agentic_rag/notebooks/` → `00_base_rag` → `01_conditional_retrieval` → `02/03_tool_use_retrieval` → `04_retrieval_evaluation_and_rewriting` → `05_query_decomposition` → `agentic_rag` (final)
> Data: 15-page "Will EVs dampen the oil price shock?" PDF, chunks 900 / 150 → 51 chunks, Chroma in-memory, gpt-5-mini.

## Idea
| | Fixes | Still missing |
|-|-|-|
| CRAG | verifies retrieval (evaluator + web fallback) | fixed flow |
| Self-RAG | decides + verifies (4 questions) | questions always in the **same order** |
| **Agentic RAG** | an **agent** (LLM + memory + tools) decides at every step with a **ReAct loop** (reason → act → observe) | |
- Retrieval = a **tool**. The agent picks: **If** (retrieve?), **Where** (vector store / web), **How** (k, filters, MMR params), **When** (order, retry, delay).

### Must-haves
1. Retrieval decision. 2. Query decomposition. 3. Query rewriting. 4. Source selection (tool calling).
5. Retrieval parameters. 6. Retry logic. 7. Relevancy check on docs. 8. Response verification (grounded + useful).
- Good-to-have: short-term memory (reuse docs already fetched), **dynamic generation prompt**, graceful fallback ("I don't know" > a wrong answer).

## Final graph
```
START → route_question ─no retrieval→ build_prompt → generate → END
            └retrieval→ check_decomposition ─complex→ decompose_query ┐
                                             └simple──────────────────→ agent ⇄ tools → check_retrieval_limit
   check_retrieval_limit ─under limit→ agent        └ agent has no more tool calls ┐
                         └limit hit──→ collect_tool_output ←────────────────────────┘
   collect_tool_output → evaluate_docs ─relevant→ build_prompt
                                       └none→ rewrite_query ─tries<3→ agent
                                                            └tries=3→ build_prompt (FALLBACK)
```

## Code (key pieces)
```Python
import os
import operator
from typing import Annotated, Literal
from pydantic import BaseModel
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

llm = ChatOpenAI(model='gpt-5-mini')


class AgenticRAGState(MessagesState):          # messages = chat history + tool calls
    query: str
    retrieved_docs: Annotated[list[Document], operator.add]   # reducer: APPEND across tool calls
    relevant_docs: list[Document]
    context: str
    needs_retrieval: bool
    needs_decomposition: bool
    retrieval_count: int
    max_retrieval_steps: int
    is_relevant: bool
    rewritten_query: str
    rewrite_count: int
    constructed_prompt: str
    generation: str
```

### 1. If: route_question
```Python
class RouteDecision(BaseModel):
    needs_retrieval: bool


def route_question(state):
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'Classify whether the following question requires retrieving information from a '
                   'specialized document, or can be answered from your own general knowledge.'),
        ('human', '{query}'),
    ])
    decision = (prompt | llm.with_structured_output(RouteDecision)).invoke({'query': state['query']})
    return {'needs_retrieval': decision.needs_retrieval}


def route_after_classification(state) -> Literal['check_decomposition', 'build_prompt']:
    if state['needs_retrieval']:
        return 'check_decomposition'
    return 'build_prompt'
```

### 2. Plan: decomposition
```Python
class DecompositionDecision(BaseModel):
    needs_decomposition: bool
    sub_queries: list[str]                     # empty when not needed


def check_decomposition(state):
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Determine if the user's question asks for multiple distinct pieces of information that each "
                   'require separate retrieval. If yes, needs_decomposition=True, else False.'),
        ('human', '{query}'),
    ])
    result = (prompt | llm.with_structured_output(DecompositionDecision)).invoke({'query': state['query']})
    return {'needs_decomposition': result.needs_decomposition}


def decompose_query(state):
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Break the user's question into focused, self-contained sub-queries — one per retrieval step. "
                   'Each sub-query should target a single distinct piece of information.'),
        ('human', '{query}'),
    ])
    result = (prompt | llm.with_structured_output(DecompositionDecision)).invoke({'query': state['query']})
    lines = []
    for i in range(len(result.sub_queries)):
        lines.append(f'Step {i + 1}: {result.sub_queries[i]}')
    steps = '\n'.join(lines)
    return {'query': f'You must call a retrieval tool for each of the following {len(lines)} steps '
                     f'before answering. Do not skip any step.\n\n{steps}'}
```

### 3. Where + How: tools (agent fills the args)
- A tool needs: `@tool`, **type hints**, a clear **docstring** (the agent picks tools by reading it).
- `content_and_artifact` → `content` (text the LLM sees) + `artifact` (Document list we keep in state).
```Python
@tool(response_format='content_and_artifact')
def vector_store_search(query: str, k: int = 3):
    """Search the vector store for relevant document passages.
    Adjust k (default 3) to retrieve more or fewer passages."""
    docs = vectorstore.as_retriever(search_kwargs={'k': k}).invoke(query)
    parts = []
    for d in docs:
        parts.append(d.page_content)
    return '\n\n## Vector Store Results\n\n' + '\n\n'.join(parts), docs


@tool(response_format='content_and_artifact')
def web_search(query: str, max_results: int = 3):
    """Search the web for current or real-time information.
    Adjust max_results (default 3) to control how many results are returned."""
    response = TavilyClient(api_key=os.environ['TAVILY_API_KEY']).search(query, max_results=max_results)
    docs = []
    parts = []
    for r in response['results']:
        docs.append(Document(page_content=r['content'], metadata={'source': r['url'], 'title': r.get('title', '')}))
        parts.append(r['content'])
    return '\n\n## Web Search Results\n\n' + '\n\n'.join(parts), docs


tools = [vector_store_search, web_search]
agent_llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)                     # runs tool calls → ToolMessages

AGENT_SYSTEM_PROMPT = (
    'You are a retrieval agent with access to two tools:\n\n'
    "1. vector_store_search — use for questions answerable from the internal document ('Will EVs Dampen the "
    "Oil Price Shock?' ...). You may increase k for broader coverage.\n\n"
    '2. web_search — use for current or real-time information not covered by the document. '
    'Always rephrase the query into a concise, keyword-optimized search string first.\n\n'
    'You may call one tool, both tools, or no tool depending on what the query requires.\n\n'
    'When the query contains numbered steps, each step is a separate retrieval sub-task. '
    'Make sure every step receives at least one retrieval call.')


def agent(state):
    active_query = state.get('rewritten_query') or state['query']
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), HumanMessage(content=active_query)]
    for m in state['messages']:                  # keep earlier tool calls / results visible
        messages.append(m)
    return {'messages': [agent_llm_with_tools.invoke(messages)]}


def route_after_agent(state) -> Literal['tools', 'collect_tool_output']:
    if state['messages'][-1].tool_calls:
        return 'tools'
    return 'collect_tool_output'
```
- Demo: domain query → vector store with **k = 8** (agent chose it). Weather → web only. "2030 projection vs today's demand" → **both in parallel**. "Kinetic vs potential energy" → no retrieval.

### 4. Loop cap (retry logic)
```Python
def check_retrieval_limit(state):
    return {'retrieval_count': state.get('retrieval_count', 0) + 1}     # 1 round = all parallel calls of 1 AI message


def route_after_limit_check(state) -> Literal['agent', 'collect_tool_output']:
    if state['retrieval_count'] >= state['max_retrieval_steps']:
        return 'collect_tool_output'
    return 'agent'


def collect_tool_output(state):
    docs = []
    for m in state['messages']:
        if isinstance(m, ToolMessage):
            docs.extend(m.artifact)
    return {'retrieved_docs': docs}
```

### 5. Verify docs + rewrite
```Python
class RelevanceEvaluation(BaseModel):
    is_relevant: bool


def evaluate_docs(state):
    docs = state.get('retrieved_docs') or []
    if not docs:
        return {'is_relevant': False, 'relevant_docs': [], 'context': ''}
    active_query = state.get('rewritten_query') or state['query']
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a relevance evaluator. Return is_relevant=True only if the passage directly addresses '
                   'the query with specific facts or analysis. False if off-topic, too vague, or unhelpful.'),
        ('human', 'Query: {query}\n\nDocument passage:\n{doc}'),
    ])
    chain = prompt | llm.with_structured_output(RelevanceEvaluation)
    relevant_docs = []
    parts = []
    for d in docs:
        if chain.invoke({'query': active_query, 'doc': d.page_content}).is_relevant:
            relevant_docs.append(d)
            parts.append(d.page_content)
    return {'is_relevant': len(relevant_docs) > 0, 'relevant_docs': relevant_docs, 'context': '\n\n'.join(parts)}


def rewrite_query(state):
    prompt = ChatPromptTemplate.from_messages([
        ('system', "A retrieval system searched for the user's query but none of the retrieved documents were "
                   'relevant. Rewrite the query to be more specific, use different terminology, or focus on a '
                   'narrower aspect. Return only the rewritten query string.'),
        ('human', 'Original query: {query}\n\nRewritten query:'),
    ])
    response = (prompt | llm).invoke({'query': state['query']})          # always from the ORIGINAL query (no drift)
    return {'rewritten_query': response.content.strip(), 'rewrite_count': state.get('rewrite_count', 0) + 1}


def route_after_evaluation(state) -> Literal['build_prompt', 'rewrite_query']:
    if state['is_relevant']:
        return 'build_prompt'
    return 'rewrite_query'


def route_after_rewrite(state) -> Literal['agent', 'build_prompt']:
    if state.get('rewrite_count', 0) >= 3:
        return 'build_prompt'
    return 'agent'
```
- Demo: an ambiguous query retrieved **54 docs → 1 relevant**. "Nuclear power?" (not in the report) → rewritten twice → found on the web.

### 6. Dynamic generation prompt + fallback
```Python
FALLBACK_MSG = ('I was unable to find relevant information to answer your query after multiple retrieval '
                'attempts. The knowledge source does not appear to contain content that addresses this question.')


def build_prompt(state):
    if state.get('rewrite_count', 0) >= 3 and not state.get('is_relevant', False):
        return {'constructed_prompt': 'FALLBACK'}
    sections = ['You are a knowledgeable assistant.']
    if state.get('needs_decomposition'):
        sections.append('The question was complex and split into sub-queries for retrieval. '
                        'Synthesize a single coherent answer covering all parts.')
    if state.get('rewritten_query'):
        sections.append('The original query was reformulated during retrieval to improve result quality.')

    context = state.get('context') or ''
    if context:
        web_docs = []
        pdf_docs = []
        for d in state.get('relevant_docs') or []:
            if d.metadata.get('source', '').startswith('http'):
                web_docs.append(d)
            else:
                pdf_docs.append(d)
        if web_docs and pdf_docs:
            sections.append('Answer using the following context from the internal document and web search:')
        elif web_docs:
            sections.append('Answer using the following context from web search:')
        else:
            sections.append('Answer using the following context from the internal document:')
        sections.append(f'Context:\n{context}')
        if web_docs:
            urls = []
            for d in web_docs:
                urls.append('- ' + d.metadata['source'])
            sections.append('Sources:\n' + '\n'.join(urls))
    else:
        sections.append('No relevant documents were found. Answer from your general knowledge if possible.')
    return {'constructed_prompt': '\n\n'.join(sections)}


def generate(state):
    if state.get('constructed_prompt') == 'FALLBACK':
        return {'generation': FALLBACK_MSG}                     # no LLM call → no hallucination
    prompt = ChatPromptTemplate.from_messages([('system', '{constructed_prompt}'), ('human', '{query}')])
    response = (prompt | llm).invoke({'constructed_prompt': state['constructed_prompt'], 'query': state['query']})
    return {'generation': response.content}
```

### Wiring
```Python
def route_after_decomposition_check(state) -> Literal['decompose_query', 'agent']:
    if state['needs_decomposition']:
        return 'decompose_query'
    return 'agent'


g = StateGraph(AgenticRAGState)
for name, fn in [('route_question', route_question), ('check_decomposition', check_decomposition),
                 ('decompose_query', decompose_query), ('agent', agent), ('tools', tool_node),
                 ('check_retrieval_limit', check_retrieval_limit), ('collect_tool_output', collect_tool_output),
                 ('evaluate_docs', evaluate_docs), ('rewrite_query', rewrite_query),
                 ('build_prompt', build_prompt), ('generate', generate)]:
    g.add_node(name, fn)

g.add_edge(START, 'route_question')
g.add_conditional_edges('route_question', route_after_classification)
g.add_conditional_edges('check_decomposition', route_after_decomposition_check)
g.add_edge('decompose_query', 'agent')
g.add_conditional_edges('agent', route_after_agent)
g.add_edge('tools', 'check_retrieval_limit')
g.add_conditional_edges('check_retrieval_limit', route_after_limit_check)
g.add_edge('collect_tool_output', 'evaluate_docs')
g.add_conditional_edges('evaluate_docs', route_after_evaluation)
g.add_conditional_edges('rewrite_query', route_after_rewrite)
g.add_edge('build_prompt', 'generate')
g.add_edge('generate', END)
graph = g.compile()

result = graph.invoke({'query': 'Impact of EVs on the oil industry and the current Brent crude price in USD',
                       'messages': [], 'retrieved_docs': [],           # reducer fields need initial values
                       'retrieval_count': 0, 'max_retrieval_steps': 3, 'rewrite_count': 0})
print(result['generation'])
```

## Known gaps in the course code (fix before reusing)
- It doesn't run the answer-level checks (grounded / useful) → add Self-RAG nodes.
- `retrieval_count` isn't reset after a rewrite, and the 3rd rewrite is never retrieved.
- Tool results from earlier attempts can leak into `collect_tool_output` after a rewrite.
- Agentic runs are **non-deterministic**: the same query can take different paths.

## Rules
- Don't retrieve by reflex.
- Retrievers = tools with good docstrings; let the agent pick the source and `k`.
- **Bound every loop** (tool rounds + rewrites).
- Filter docs before generating. Rewrite from the original query.
- Honest fallback > hallucination. Tell the generator how the context was built.
