---
tags:
  - AIEngineer
  - llm-eval
  - rag
  - deepeval
---

# 13 - How to Test RAG Retrievers (Hands-on with DeepEval)

> Video: https://www.youtube.com/watch?v=9Dkz3ckRj8c
> Code: https://github.com/campusx-official/rag-eval-deepeval
> Component-level eval, part 1: **build the retriever → evaluate it → improve it**. (The generator is next lecture.)

## Project setup
```
rag-eval/
├── data/       # lecture transcripts (.vtt subtitle files) = knowledge base
├── src/        # app code: retriever.py, reranker.py, generator.py, rag_pipeline.py, app.py
├── evals/      # eval pipelines: eval_retriever.py, ...
├── goldens/    # golden datasets (.json)
└── .env        # OPENAI_API_KEY
```
```bash
uv init --python 3.11
uv add langchain-core langchain-openai langchain-chroma langchain-text-splitters openai deepeval pytest python-dotenv sentence-transformers
```

## 1. Build the retriever (`src/retriever.py`)
- Load the `.vtt` files → **drop timestamp lines** (they hurt semantic meaning) → one `Document` per session with `metadata={'session': N}` (for citations later).
- Chunk → embed (OpenAI) → **Chroma** (saved to `chroma_store/`, built once) → `as_retriever(k=5)`.
- When transcripts or chunking settings change → **delete `chroma_store/`** and rebuild.
```Python
import os
import re
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()
DATA_DIR = 'data'
DB_DIR = 'chroma_store'


def load_transcripts():
    docs = []
    for path in glob.glob(f'{DATA_DIR}/*.vtt'):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == 'WEBVTT' or '-->' in line:   # skip blanks + timestamps
                continue
            lines.append(line)
        text = ' '.join(lines)
        session = re.search(r'Session[ _]*(\d+)', path).group(1)
        docs.append(Document(page_content=text, metadata={'session': session}))
    return docs


def load_store():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    if os.path.exists(DB_DIR):                       # already built -> just open it
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    docs = load_transcripts()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={'k': 5})


if __name__ == '__main__':
    retriever = build_retriever()
    results = retriever.invoke('what is regression testing?')
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...")
```

---

## 2. How a retriever fails → 2 metrics
| Failure | Metric |
|-|-|
| **Misses** the right chunks | **Recall** = of all correct chunks, how many were retrieved? |
| Brings the right chunks **plus noise** | **Precision** = of the retrieved chunks, how many are useful? |
- Correct = {1, 15, 13}. Retrieved = {1, 15, 27, 28, 29} → recall 2/3, precision 2/5.
- **Trade-off via k**: bigger k → recall ↑ but precision ↓ (more noise).
- Both are **reference-based** (they need a golden dataset with the truth).

### ❌ Approach 1: golden = question → correct **chunk IDs** (Recall@k)
- Q: "What is regression testing?" → gold chunks {72, 89, 100}; per-question recall / precision → average.
- **Why it fails here:**
  1. Building it is miserable: read ~800 chunks for each of 50 questions.
  2. **Change chunk size/overlap → chunks and IDs change → the golden dataset becomes invalid** → rebuild it every time you tune.
- OK only when the chunking never changes (cleanly separated docs that you never re-chunk).

### ✅ Approach 2: golden = question → **ideal answer** (+ LLM-as-a-judge)
- The ideal answer is written **from your own transcripts** (what was taught), not from Google.
- **Survives re-chunking**: the ideal answer doesn't depend on chunk IDs.

**Contextual Recall (DeepEval):**
1. A judge LLM splits the **ideal answer into claims** (atomic statements).
2. For each claim: is it supported by **any** retrieved chunk?
3. recall = supported claims / total claims (3 claims, all found → 1.0; 2 claims, 1 found → 0.5).
- Write ideal answers as **atomic claims** + use a good judge model so claim splitting is reliable.

**Contextual Precision (DeepEval):**
- For each retrieved chunk the judge decides relevant / not. Judge prompt idea:
  > "Here is the question. Here is the expected answer. Here is one retrieved chunk. Does it contain information that helps produce the expected answer? Answer yes / no with a reason."
- ⭐ **Rank-aware**: relevant chunks should be **at the top**.
  - Case A: ✅✅❌❌❌ vs Case B: ❌❌❌✅✅ → plain precision is equal (2/5), but A is better.
  - Precision at each position: A = 1/1, 2/2, 2/3, 2/4, 2/5 → **high average**. B = 0, 0, 0, 1/4, 2/5 → **low average**.

---

## 3. Build the golden dataset (4 ways)
| Way | Pros / cons |
|-|-|
| **1. Hand-authored** (by someone who knows the data, e.g. the teacher) | best quality, not scalable / costly |
| **2. LLM-assisted drafting + human review** (used here: transcripts → Claude, one question at a time, 15 Qs, each reviewed) | fast and cheap; must remove things the LLM added from outside knowledge |
| **3. DeepEval `Synthesizer`** | automated, but the questions were unnatural ("Assess methodologies to validate LLM robustness against adversarial exploits...") → not used |
| **4. Production logs** (👍 interactions) | real questions; only after deployment |

Golden format (`goldens/retriever_goldens.json`):
```json
[
  {
    "id": "g001",
    "query": "What is online eval and how is it different from offline eval?",
    "ideal_answer": "Online eval is evaluating your system on live production traffic after deployment ... offline eval checks the app works correctly, online eval tells you it is running normally ...",
    "source": "Session 4"
  },
  {
    "id": "g002",
    "query": "faithfulness vs groundedness kya hota hai?",
    "ideal_answer": "...",
    "source": "..."
  }
]
```
- Student-like wording (even Hinglish) = realistic.

### Synthesizer (for reference)
```Python
from deepeval.synthesizer import Synthesizer

contexts = []
for chunk in sample_chunks:          # e.g. 15 random chunks
    contexts.append([chunk])         # each context = one chunk

synthesizer = Synthesizer(model='gpt-4.1-mini')
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,    # gives the ideal answer
    max_goldens_per_context=1,
)
for g in goldens:
    print(g.input, '->', g.expected_output)   # REVIEW every one by hand
```

---

## 4. DeepEval basics (3 building blocks)
| Block | What |
|-|-|
| **`LLMTestCase`** | **one row** of your golden dataset (`input`, `actual_output`, `expected_output`, `retrieval_context`) |
| **Metric** | what to measure (e.g. `AnswerRelevancyMetric`, `ContextualRecallMetric`). Options: `model` (judge LLM), `threshold` (pass / fail cut-off), `include_reason=True` |
| **`evaluate()`** | runs every metric on every test case (batched, parallel) → report with scores, pass / fail, reasons |

```Python
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv()

case_1 = LLMTestCase(input='What is the capital of France?',
                     actual_output='The capital of France is Paris.')                         # should pass
case_2 = LLMTestCase(input='What is the capital of France?',
                     actual_output='France is a beautiful country famous for its food and wine.')  # should fail

metric = AnswerRelevancyMetric(threshold=0.7, model='gpt-4.1', include_reason=True)
evaluate(test_cases=[case_1, case_2], metrics=[metric])
```

## 5. Evaluate the retriever (`evals/eval_retriever.py`)
```Python
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from src.retriever import build_retriever

load_dotenv()
GOLDEN_PATH = 'goldens/retriever_goldens.json'
JUDGE_MODEL = 'gpt-4.1-mini'
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

retriever = build_retriever()

test_cases = []
for g in goldens:
    retrieved = retriever.invoke(g['query'])
    retrieval_context = []
    for doc in retrieved:
        retrieval_context.append(doc.page_content)

    test_cases.append(LLMTestCase(
        input=g['query'],
        expected_output=g['ideal_answer'],
        retrieval_context=retrieval_context,
        actual_output='(generator not evaluated in this run)',   # placeholder
    ))

metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={                      # logged with the run -> compare configs later
        'embedding_model': 'text-embedding-3-large',
        'chunk_size': 1000,
        'chunk_overlap': 150,
        'top_k': 5,
        'judge_model': JUDGE_MODEL,
    },
)
```
- Run it as a module from the project root (add empty `__init__.py` to `src/` and `evals/`, or you get `No module named src`):
```bash
python -m evals.eval_retriever
```
- Then **read the reasons** of the failed cases.

## 6. Improve it (iterate on the same goldens)
| Change | Recall | Precision | Failed (of 15) |
|-|-|-|-|
| Baseline: chunk 750 / overlap 100, `text-embedding-3-small`, k=5 | 0.80 | 0.80 | 5 |
| Chunk **1000 / 150** (697 chunks) | **0.97** | 0.83 | 3 |
| + **Reranker** | 0.92 | **0.85** | **2** |
| + `text-embedding-3-large` | **0.99** | 0.85 | 3 |
| k = 3 | ↓ | 0.84 | no gain |
- Final: **recall 95%+, precision ~85%** → good enough to build the generator.
- With only 15 goldens, scores wobble between runs (variance). More goldens = more stable.
- Next ideas: a better reranker, more chunking experiments (e.g. 1500 / 200).

### Reranker (`src/reranker.py`): over-retrieve, then rerank with a cross-encoder
- The bi-encoder fetches `fetch_k=10` fast → a **cross-encoder** scores each (query, chunk) pair together → keep the best `top_k=5`.
- Rerankers push relevant chunks to the top → **rank-aware precision goes up**.
```Python
from sentence_transformers import CrossEncoder
from src.retriever import load_store

CROSS_ENCODER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'   # small, CPU-friendly (~80MB)


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5):
        self.store = load_store()
        self.reranker = CrossEncoder(CROSS_ENCODER)
        self.fetch_k = fetch_k
        self.top_k = top_k

    def invoke(self, query):
        # 1. over-retrieve with the vector store
        candidates = self.store.similarity_search(query, k=self.fetch_k)

        # 2. score every (query, chunk) pair
        pairs = []
        for doc in candidates:
            pairs.append((query, doc.page_content))
        scores = self.reranker.predict(pairs)

        # 3. sort by score (highest first) and keep top_k
        scored = []
        for i in range(len(candidates)):
            scored.append((scores[i], i))
        scored.sort(reverse=True)

        best = []
        for score, i in scored[:self.top_k]:
            best.append(candidates[i])
        return best
```
- In the eval, swap `build_retriever()` for `RerankingRetriever()` (same `.invoke()` interface).

## Remember
- Retriever metrics: **contextual recall** (claims of the ideal answer found in the retrieved chunks) + **contextual precision** (relevant chunks, **rank-aware**).
- Golden = **question + ideal answer** (not chunk IDs), so it survives re-chunking.
- DeepEval = `LLMTestCase` + metrics + `evaluate()`.
- Tune: chunk size / overlap, embedding model, **reranker**, k.
