---
tags:
  - AIEngineer
  - rag
  - evaluation
  - ragas
---

# 16 - RAGAS (RAG Assessment)

> Lesson: RAG Assessment
> Code: `10_rag_evaluation/` (`evaluate.py`, `rag_pipeline.py`, `main.py`, `notebooks/` one per metric)
> Related: [[13 - Testing RAG Retrievers (DeepEval)]] · [[14 - Testing the Generator & RAG Triad]] (DeepEval versions)

## Idea
- Evaluate only 2 things: the **retriever** and the **generator LLM** → 2 failure types.
- Metrics = a **compass**: change one knob (k 3 → 5) = one **experiment** → compare scores.
- **LLM-as-a-judge** (semantic) > keyword matching ("capital of India is New Delhi" = "New Delhi is the capital...").
- Pick **~4-5 metrics** you can act on, not all ~10.

| Term | Meaning |
|-|-|
| `user_input` | the query |
| `retrieved_contexts` | list of chunk strings from the retriever |
| `response` | the LLM answer |
| `reference` | human-written ideal answer (answer key) |

## The 5 metrics
| Metric | Component | Needs | Calc | Better |
|-|-|-|-|-|
| **Context Recall** | retriever | query, contexts, reference | reference claims found in contexts / all reference claims | ↑ |
| **Context Precision** | retriever (ranking) | query, contexts, reference | Σ(precision@k × v_k) / #relevant | ↑ |
| **Faithfulness** | generator | query, response, contexts | response claims supported by contexts / all response claims | ↑ |
| **Response Relevancy** | generator | query, response | LLM makes N (=3) questions from the response → avg cosine with the real query | ↑ |
| **Noise Sensitivity** | generator robustness | query, response, reference, contexts | response claims that came from **noise** / all response claims | **↓** |

### Quick examples
- **Recall**: reference Eiffel = Paris, 1889, 330 m, Gustave Eiffel. Contexts have Paris + 1889 → **2/4 = 0.5**. Low → small chunks, low k, weak embeddings.
- **Precision**: relevant at ranks 1, 3, 5 → (1 + 0.67 + 0.6) / 3 = **0.756**. Same 3 at ranks 1, 2, 3 → **1.0**. Low → re-ranking.
- **Faithfulness**: Einstein answer with 5 claims, 3 in context (no "IQ 160") → **0.6**. Low → hallucination, fix the prompt ("only from context").
- **Response relevancy**: "capital of France?" answered with wine & population → generated questions drift → low. Doesn't check facts. (Same trick as HyDE.)
- **Noise sensitivity**: 4 claims, 2 pulled from unrelated chunks (Louvre visitors, Arc height) → **0.5** (bad). 0 = fully robust.

## Single-metric code
```Python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (ContextRecall, ContextPrecision, Faithfulness,
                                       AnswerRelevancy, NoiseSensitivity)

client = AsyncOpenAI()                                    # async client (RAGAS runs async)
llm = llm_factory('gpt-5-mini', client=client)
embeddings = embedding_factory('openai', model='text-embedding-3-small', client=client)

scorer = ContextRecall(llm=llm)
result = await scorer.ascore(
    user_input='What are the symptoms and causes of Type 2 diabetes?',
    retrieved_contexts=['Type 2 diabetes symptoms include frequent urination and excessive thirst.',
                        'People with Type 2 diabetes often experience fatigue and blurred vision.'],
    reference='Type 2 diabetes is caused by insulin resistance, often linked to obesity and a sedentary '
              'lifestyle. Its symptoms include frequent urination, excessive thirst, fatigue, blurred vision, '
              'and slow-healing sores.')
print(result.value)       # 0.57 → the causes were missed

# others, same pattern:
# ContextPrecision(llm=llm).ascore(user_input, reference, retrieved_contexts)
# Faithfulness(llm=llm).ascore(user_input, response, retrieved_contexts)
# AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(user_input, response)          # strictness default 3
# NoiseSensitivity(llm=llm).ascore(user_input, response, reference, retrieved_contexts)
```

## Evaluate a real pipeline (`evaluate.py`)
```Python
from openai import AsyncOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings.base import embedding_factory

QA_PAIRS = [   # (question, human-written reference)
    ('What does sustainable development mean?',
     "Development that meets today's needs without preventing future generations from meeting their own needs"),
    ('What are the three pillars of sustainable development?',
     'Economic growth, social inclusion, and environmental protection'),
    ('What is the target year for the Sustainable Development Goals?', '2030'),
]


def run_evaluation(chain, retriever):
    client = AsyncOpenAI()
    embeddings = embedding_factory('openai', model='text-embedding-3-small', client=client)

    dataset = []
    for question, ground_truth in QA_PAIRS:
        contexts = []
        for doc in retriever.invoke(question):
            contexts.append(doc.page_content)
        response = chain.invoke({'context': '\n\n'.join(contexts), 'question': question}).content
        dataset.append({'user_input': question,
                        'retrieved_contexts': contexts,
                        'response': response,
                        'reference': ground_truth})

    results = evaluate(dataset=EvaluationDataset.from_list(dataset),
                       embeddings=embeddings)                     # ⚠ needed for response relevancy
    results.to_pandas().to_csv('evaluation_results.csv', index=False)
```
```Python
# main.py
from dotenv import load_dotenv
load_dotenv()
from rag_pipeline import build_rag_chain       # PDF → 800/150 chunks → Chroma → k=4 → gpt (temp 0, "only from context")
from evaluate import run_evaluation

chain, retriever = build_rag_chain()
run_evaluation(chain, retriever)
```
- Response relevancy came back blank in the demo → cause: no embedding model passed to `evaluate()`.

## Rules
- Bad answer? Check **recall / precision** (retriever) before blaming the LLM (**faithfulness / relevancy / noise**).
- Semantic (LLM) judging over keyword matching.
- Few, well-chosen metrics. Change one knob per experiment.
