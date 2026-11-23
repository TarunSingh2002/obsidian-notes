---
tags:
  - AIEngineer
  - llm-eval
  - rag
  - deepeval
---

# 14 - Evaluating RAG: The Generator + the Full Pipeline (RAG Triad)

> Video: https://www.youtube.com/watch?v=PATGn2XhmCY
> Code: https://github.com/campusx-official/rag-eval-deepeval
> Component level part 2 (**generator**) + pipeline level (**RAG triad**).

## 1. Build the generator (`src/generator.py`)
- Input: question + context (list of chunks) → output: answer. A `prompt | llm | StrOutputParser()` chain.
- `gpt-4o-mini`, **temperature 0**.
- First prompt (simple): use only the context, no outside knowledge; if not enough info, say exactly *"I don't have enough information in the course material to answer that."*; clear and concise.
```Python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a helpful teaching assistant for a course on LLM evaluations.
Answer the student's question using ONLY the information in the context below.
- Use only information present in the context. Do not add outside knowledge.
- If the context does not contain enough information, say exactly:
  "I don't have enough information in the course material to answer that."
- Keep the answer clear and concise.

<COURSE_CONTEXT>
{context}
</COURSE_CONTEXT>

<STUDENT_QUESTION>
{question}
</STUDENT_QUESTION>

Answer:
""")

chain = prompt | llm | StrOutputParser()


def generate(query, context):
    context_text = '\n\n'.join(context)        # list of chunks -> one string
    return chain.invoke({'question': query, 'context': context_text})


if __name__ == '__main__':
    ctx = ['Online eval means evaluating your system on live production traffic after '
           'deployment. It works without an answer key, unlike offline eval.']
    print(generate('what is online eval?', ctx))
```

## 2. How a generator fails → 2 metrics
| Failure | Example | Metric |
|-|-|-|
| **Unfaithful** (hallucinates beyond the context) | Q: "Does the AI Engineering program have **live classes**?" Context: recorded lessons, assignments, projects, weekly doubt sessions. Answer: "Yes, **2 live classes every week**..." ❌ (invented, like Air Canada) | **Faithfulness** |
| **Irrelevant** (faithful, but doesn't answer) | Answer: "The program includes assignments, projects, recorded lessons and doubt sessions." (faithful, but never says yes/no about live classes) | **Answer relevancy** |
- Good answer: "The context does not confirm live classes; it only mentions recorded lessons, ..." → faithful **and** relevant.
- ⚠️ **Faithful ≠ correct**: with wrong context, a faithful generator gives a wrong answer (correctness is an application-level metric, next lecture).

### Faithfulness (needs the context)
1. Golden dataset: `query` + **`ideal_context`** (the right chunks for that question).
2. Give the generator **question + ideal context** (**isolation**: NOT the retriever's output).
3. Judge LLM splits the **generated answer into claims**.
4. Each claim: supported by the context? → faithfulness = supported / total (e.g. 2/3 = 0.67). Average over questions.
- Opposite direction of contextual recall: recall = ideal answer's claims vs retrieved context; faithfulness = generated answer's claims vs given context.

### Answer relevancy (reference-free)
1. Judge LLM splits the generated answer into claims.
2. Each claim: does it help answer the **question**? → relevant / total.
- "What does benchmark saturation mean?" → claims: (1) scores cluster high ✅ (2) can't tell models apart ✅ (3) "contamination is test data leaking into training" ❌ off-topic → **0.67**.
- The goldens only supply questions, not a reference → **reference-free**.
- One judge does both claim splitting and checking (separate API calls). A judge can be wrong, but it's consistently wrong across runs → comparisons still work; use a strong judge (few samples, so affordable).

### Golden dataset for faithfulness
- Export every chunk from Chroma (`export_chroma_chunks.py` → `chunks_dump.json`, ~862 chunks) → give it to Claude → generate **one (question, ideal_context) pair at a time** → **human review** (the teacher knows what was taught) → 15 rows in `goldens/faithfulness_dataset.json`.
```json
[
  {
    "id": "f001",
    "query": "what is vibe testing and why isn't it enough for deploying an llm app?",
    "ideal_context": ["...chunk text 1...", "...chunk text 2..."]
  }
]
```
```Python
# export_chroma_chunks.py (idea)
import json
from src.retriever import load_store

store = load_store()
data = store._collection.get(include=['documents', 'metadatas'])   # documents are not returned by default

dump = []
for i in range(len(data['ids'])):
    dump.append({'id': data['ids'][i], 'text': data['documents'][i], 'meta': data['metadatas'][i]})

with open('chunks_dump.json', 'w') as f:
    json.dump(dump, f, indent=2, ensure_ascii=False)
```

### `evals/eval_generator.py`
```Python
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from src.generator import generate

load_dotenv()
JUDGE_MODEL = 'gpt-4o-mini'
THRESHOLD = 0.7

with open('goldens/faithfulness_dataset.json') as f:
    goldens = json.load(f)

test_cases = []
for g in goldens:
    context = g['ideal_context']                 # GOLDEN context (isolation)
    answer = generate(g['query'], context)       # run the generator
    test_cases.append(LLMTestCase(
        input=g['query'],
        actual_output=answer,                    # judged answer
        retrieval_context=context,               # faithfulness checks against this
    ))

metrics = [
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]
evaluate(test_cases=test_cases, metrics=metrics)     # test cases run in parallel
```
```bash
python -m evals.eval_generator
```

### Results and improvement
| Version | Faithfulness | Answer relevancy |
|-|-|-|
| Simple prompt | 0.91 | **0.73** |
| Refined prompt (3-4 iterations) | **0.96** | **0.92** (14/15 pass) |
- Faithfulness is easy to score high (modern LLMs follow "use only the context" well). Relevancy is harder.
- Generator levers: **(1) a better model**, **(2) the system prompt**. Process: run → read the failed cases' reasons → give them to an LLM → refine the prompt → re-run.
- Rules added: don't strengthen / overstate claims; context is an informal transcript, so synthesize and rephrase; don't need the question's exact wording to appear; answer **every part** of the question; teacher-like prose; don't pad.
- ⚠️ Don't over-tune the prompt to the goldens (**overfitting**). Check it on real retriever context (next section).
- **Confident AI** (DeepEval's company): `deepeval view` → log in + API key → runs, failures, config per run = experiment tracking (like MLflow).

---

## 3. Build the pipeline (`src/rag_pipeline.py`)
```Python
from src.reranker import RerankingRetriever
from src.generator import generate


class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)   # load once

    def invoke(self, query):
        docs = self.retriever.invoke(query)          # 1. retrieve + rerank
        context = []
        for doc in docs:
            context.append(doc.page_content)         # 2. Documents -> list of strings
        answer = generate(query, context)            # 3. generate
        return {'query': query, 'context': context, 'answer': answer}   # all 3 triad legs


if __name__ == '__main__':
    rag = RagPipeline()
    result = rag.invoke('What is drift and why does it matter after deployment?')
    print(result['answer'])
```
```bash
python -m src.rag_pipeline
```

## 4. The RAG Triad (pipeline level)
| Pair | Metric | Checks |
|-|-|-|
| Answer ↔ Context | **Faithfulness** | answer grounded in the **retrieved** context |
| Answer ↔ Question | **Answer relevancy** | answer addresses the question |
| Context ↔ Question | **Contextual relevancy** | retrieved context is relevant to the question |
- Same faithfulness / relevancy as before, **but now the context comes from the retriever**, not the goldens → different scores.
- **Contextual relevancy** (reference-free): the judge splits **each retrieved chunk into statements** → each statement relevant to the question? → relevant / total (e.g. 10/15).

### `evals/eval_rag_pipeline.py`
```Python
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from src.rag_pipeline import RagPipeline

load_dotenv()
JUDGE_MODEL = 'gpt-4o-mini'
THRESHOLD = 0.7

with open('goldens/faithfulness_dataset.json') as f:    # reuse only the queries
    goldens = json.load(f)

rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g['query'])                      # retrieve -> rerank -> generate
    test_cases.append(LLMTestCase(
        input=g['query'],
        actual_output=result['answer'],                  # from the generator
        retrieval_context=result['context'],             # from the RETRIEVER now
    ))

metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]
evaluate(test_cases=test_cases, metrics=metrics)
```

### Results and the "curious case"
| Metric | Score |
|-|-|
| Faithfulness | **0.93** |
| Answer relevancy | **0.86** (the prompt generalizes; not overfit) |
| Contextual relevancy | **0.42** ❌ |
| (Retriever alone) Recall / Precision | 0.99 / 0.89 |
- Why is the retriever "good" (recall, precision) and "bad" (contextual relevancy) at once?
  - **Precision** asks: is the **chunk** useful? A chunk with 1 useful sentence out of 5 counts as useful.
  - **Contextual relevancy** asks: how much **text inside** each chunk is relevant → measures **noise inside chunks**.
- Fix: **smaller chunks / less overlap** (watch for trade-offs with the other metrics). With faithfulness, relevancy, recall and precision all good, low contextual relevancy can be acceptable.

## DeepEval's 5 core RAG metrics
| Metric | Level | Reference? |
|-|-|-|
| Contextual recall | retriever | reference-based (ideal answer) |
| Contextual precision | retriever | reference-based (ideal answer) |
| Faithfulness | generator / pipeline | needs context (no ideal answer) |
| Answer relevancy | generator / pipeline | reference-free |
| Contextual relevancy | pipeline | reference-free |
- Correctness, completeness, style → custom metrics with **G-Eval** (next lecture).

## Remember
- Generator in isolation = golden **ideal context**. Pipeline = **retriever's** context.
- Faithful ≠ correct. Relevant ≠ correct.
- Generator levers: model + system prompt (iterate on the failure reasons, avoid overfitting).
- Low contextual relevancy with high precision = noisy chunks → smaller chunks.
