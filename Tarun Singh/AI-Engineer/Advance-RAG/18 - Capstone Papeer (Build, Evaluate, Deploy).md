---
tags:
  - AIEngineer
  - rag
  - agentic-rag
  - project
  - deployment
---

# 18 - Capstone: Papeer (Agentic RAG for Research Papers)

> Lessons: Introduction and Plan of Action · Evaluation & Deployment
> Code: https://github.com/Himanshu-1703/rag-papeer-project
> Stack: LangGraph + SQLite checkpointer · GPT-5.4 mini · text-embedding-3-small (cached) · **Qdrant Cloud** · Tavily · Streamlit · DeepEval · uv · Docker → Docker Hub → AWS EC2

## What it does
| Feature | How |
|-|-|
| Agentic RAG chat | router → agent + tools (Qdrant / Tavily) → relevancy check → answer |
| Sources | local PDF / TXT / MD, web URLs, **arXiv** (by ID or title) |
| Memory + sessions | `SqliteSaver` checkpointer, `thread_id` = session id |
| **Per-session collections** | each session gets its own Qdrant collection (no leakage) |
| Auto session rename | small LLM makes a 3-5 word title from the 1st message |
| `/btw` side question | answered (LLM or web), **never saved** to history |
| Claim verification | Tavily (general + arxiv.org) → superseded? + up to 3 newer papers |
| Cached embeddings | chunk hash → vector in a file store (re-upload = no re-embedding) |
- vs the course's agentic RAG: **removed** decomposition + dynamic prompt (latency), **added** memory + cache.

## Graph
```
router ─direct_answer─────────────────────────────────────────────→ generate_answer → END
       ─verify_claim──→ verify_claim ─────────────────────────────→ generate_answer
       ─retrieve──→ agent_node ⇄ retrieval (ToolNode)
                    agent_node ─attempts ≥ 3──────────────────────→ generate_answer
                               └no tool call→ relevancy_check ─relevant─→ generate_answer
                                                              └not relevant, rewrite left→ query_rewrite → agent_node
                                                              └no rewrites left→ generate_answer ("couldn't find...")
```

| File | Job |
|-|-|
| `app.py` | Streamlit UI |
| `backend/rag_graph.py` | the agentic graph |
| `backend/vector_store.py` | Qdrant, embedding cache, search |
| `backend/models.py` | Pydantic schemas (RouterDecision, RelevancyDecision, ClaimVerificationResult) |
| `backend/paper_loader.py` | files / URLs / arXiv loading |
| `backend/btw_handler.py` | `/btw` |
| `evaluate.py`, `goldens.json` | DeepEval |

## vector_store.py: cache + collection per session
```Python
import os
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

EMBEDDING_DIM = 1536
base_embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
embeddings = CacheBackedEmbeddings.from_bytes_store(base_embeddings,
                                                    LocalFileStore('./embedding_cache/'),   # Redis at scale
                                                    namespace=base_embeddings.model,        # 1 cache per model
                                                    query_embedding_cache=True,
                                                    key_encoder='blake2b')                  # hash of chunk text = key
qdrant_client = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=120)


def get_collection_name(session_id):
    return 'papeer_' + session_id.replace('-', '_')


def get_vectorstore(session_id):
    name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(name):
        qdrant_client.create_collection(collection_name=name,
                                        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
    return QdrantVectorStore(client=qdrant_client, collection_name=name, embedding=embeddings)


def add_paper(docs, session_id):
    get_vectorstore(session_id).add_documents(docs)


def search(query, session_id, k=4):
    return get_vectorstore(session_id).similarity_search(query, k=k)
```
- `list_papers` scrolls the collection (`qdrant_client.scroll(..., limit=100, offset=offset)`) and collects unique `metadata['title']` → every chunk must be stamped with a `title`.
- Qdrant free: Docker locally, or Cloud free tier (~4 GB, 1 GB RAM). 34-page report → 92 points.

## paper_loader.py
- PDF `PyMuPDFLoader`, text `TextLoader`, web `WebBaseLoader`; chunks **1000 / 200**; markdown via `RecursiveCharacterTextSplitter.from_language('markdown', ...)`.
- arXiv: regex `(\d{4}\.\d{4,5}(?:v\d+)?)` → strip `vN` → download `https://arxiv.org/pdf/{id}` → temp file → load → delete. No ID → search by title. **Prefer IDs** (titles can match the wrong paper).

## rag_graph.py key ideas
```Python
from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command


class RetrieverInput(BaseModel):
    query: str = Field(description='Semantic query to search research paper chunks')
    k: int = Field(default=4, ge=1, le=10, description='Number of chunks to retrieve')


@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(query: str, k: int,
                              session_id: Annotated[str, InjectedState('session_id')],      # from state, not the LLM
                              current_docs: Annotated[list, InjectedState('retrieved_docs')],
                              tool_call_id: Annotated[str, InjectedToolCallId]):
    """Search the uploaded research paper vector store for relevant passages."""
    docs = search(query=query, session_id=session_id, k=k)
    if not docs:
        return [ToolMessage(content='No relevant documents found in the vector store.', tool_call_id=tool_call_id)]
    return [ToolMessage(content=f'Retrieved {len(docs)} chunk(s) from the vector store.', tool_call_id=tool_call_id),
            Command(update={'retrieved_docs': (current_docs or []) + docs})]   # bulk data → state, summary → agent


RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
MAX_RETRIEVAL_ATTEMPTS = 3


def agent_node(state):
    attempts = state.get('retrieval_attempts', 0)
    if attempts >= MAX_RETRIEVAL_ATTEMPTS:
        lm = llm                       # plain LLM: CAN'T emit tool calls (an unanswered call would corrupt history)
    else:
        lm = retrieval_llm
    response = lm.invoke([{'role': 'system', 'content': RETRIEVE_SYSTEM}] + state['messages'])
    updates = {'messages': [response]}
    if getattr(response, 'tool_calls', None):
        updates['retrieval_attempts'] = attempts + 1
    return updates


def agent_routing(state):
    if tools_condition(state) == 'tools':           # always answer pending tool calls first
        return 'retrieval'
    if state.get('retrieval_attempts', 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return 'generate_answer'
    return 'relevancy_check'


def relevancy_check_node(state):
    docs = state.get('retrieved_docs') or []
    snippets = []
    for doc in docs[:3]:
        snippets.append(doc.page_content[:300])     # ONE cheap call on 3 × 300 chars (not per chunk)
    if not snippets:
        return {'is_relevant': False}
    prompt = f"Question: {state['query']}\n\nRetrieved chunks:\n" + '\n\n---\n\n'.join(snippets) + \
             '\n\nAre these chunks relevant to answering the question?'
    decision = relevancy_llm.invoke([{'role': 'system', 'content': RELEVANCY_CHECK_SYSTEM},   # lenient prompt
                                     {'role': 'user', 'content': prompt}])
    return {'is_relevant': decision.is_relevant}


def after_relevancy_routing(state):
    if state.get('is_relevant', False):
        return 'generate_answer'
    if state.get('rewrite_count', 0) < 1:           # rewrite cap
        return 'query_rewrite'
    return 'generate_answer'
```
- `query_rewrite` returns: new `HumanMessage`, `query`, `retrieved_docs=[]`, `retrieval_attempts=0`, `rewrite_count + 1`, `is_relevant=None`.
- Router rules: content of uploaded papers / live info → `retrieve`; "is this still true?" → `verify_claim`; stable general knowledge → `direct_answer`; unsure → `retrieve`.
- Always return the **`tool_call_id`** in the ToolMessage.
- `build_graph()`: `SqliteSaver(sqlite3.connect('checkpoints.db', check_same_thread=False))` → `graph.compile(checkpointer=...)`.

## app.py (Streamlit)
```Python
@st.cache_resource              # Streamlit reruns the whole script on every click → build the graph once
def get_graph():
    return build_graph()

config = {'configurable': {'thread_id': active_sid}}           # checkpointer restores history by thread
for chunk, metadata in graph.stream(input_state, config, stream_mode='messages'):
    if metadata.get('langgraph_node') == 'generate_answer' and chunk.content:
        response_text = response_text + chunk.content          # stream ONLY the final answer node
        placeholder.markdown(response_text + '|')
```
- `sessions.json` maps session id → name / created_at / is_named. A fresh input state every turn (messages come from the checkpointer).

## Evaluation with DeepEval
- Same 5 metrics as RAGAS + a **Synthesizer** that auto-generates goldens: load → sample chunks → LLM quality filter (≥ 0.5) → group similar chunks → contexts → (input, expected_output).
```Python
from deepeval import evaluate
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric,
                              AnswerRelevancyMetric, FaithfulnessMetric)

goldens = Synthesizer().generate_goldens_from_docs(
    document_paths=[PDF_PATH], include_expected_output=True, max_goldens_per_context=2,
    context_construction_config=ContextConstructionConfig(max_contexts_per_document=5))   # 5 × 2 = 10 goldens
# save to goldens.json, reuse next time (slow + costs money)

M = 'gpt-5.4-mini'
metrics = [ContextualPrecisionMetric(threshold=0.7, model=M), ContextualRecallMetric(threshold=0.7, model=M),
           ContextualRelevancyMetric(threshold=0.7, model=M), AnswerRelevancyMetric(threshold=0.7, model=M),
           FaithfulnessMetric(threshold=0.7, model=M)]

test_cases = []
for pair in pairs:
    session_id = f'evaluation_session_{uuid4()}'               # fresh session per case: no memory leaks
    add_paper(docs, session_id)
    answer, retrieval_context = run_rag_query(graph, pair['input'] + ' as per the report in knowledge base', session_id)
    test_cases.append(LLMTestCase(input=pair['input'], actual_output=answer,
                                  expected_output=pair['expected_output'], retrieval_context=retrieval_context))

results = evaluate(test_cases, metrics,
                   async_config=AsyncConfig(max_concurrent=3, throttle_value=5))   # avoid OpenAI rate limits
```
| Metric | Avg | Passed |
|-|-|-|
| Contextual precision | 0.98 | 10/10 |
| Contextual recall | 1.00 | 10/10 |
| **Contextual relevancy** | **0.58** | 5/10 |
| Answer relevancy | 0.99 | 10/10 |
| Faithfulness | 0.98 | 10/10 |
- Low contextual relevancy = the price of the **single cheap relevancy call** (multiple tool calls pile up noisy chunks). Per-chunk filtering would fix it but adds latency.
- 5-20 test cases while learning, 50-100+ for a real app. Save results to JSON.

## Docker → Docker Hub → EC2
```dockerfile
FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt      # deps first (layer cache)
COPY backend/ backend/
COPY documents/ documents/
COPY app.py evaluate.py main.py goldens.json sessions.json ./   # most-changed files last
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```
```bash
docker build -t <user>/papeer:latest .
docker run --env-file .env -p 8501:8501 <user>/papeer:latest      # .env is NOT baked into the image
docker push <user>/papeer:latest
# EC2: Ubuntu, t3.micro, key pair, SG allows SSH/HTTP/HTTPS + port 8501
# install docker → add user to docker group → docker pull → docker run -e ... -p 8501:8501 → http://<public-ip>:8501
```
- Demo opened "All TCP from anywhere". For real: open only 8501, SSH from your IP.

## Rules
- Cap every loop, and know what each cap does (stopping tool calls ≠ judging evidence).
- Isolate state per session and per eval test case.
- Bulk data in state, a summary to the agent, always return `tool_call_id`.
- Save expensive artifacts (goldens, results). Keep secrets out of images.
