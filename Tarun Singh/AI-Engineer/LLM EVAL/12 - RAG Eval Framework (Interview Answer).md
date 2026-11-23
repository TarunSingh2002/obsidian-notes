---
tags:
  - AIEngineer
  - llm-eval
  - rag
---

# 12 - How to Answer "How Do You Evaluate Your RAG App?"

> Video: https://www.youtube.com/watch?v=4zn-gSckVTQ
> Model evals are done (benchmarks + custom model evals). Now: **application evals**, covering **RAG** (next ~4 lectures) and **agents**.

## Why RAG and agents?
- App types: simple chatbot, **RAG chatbot**, **agents**, multimodal apps (image generators), **fixed-schema output** apps (Zomato email classifier).
- Most professional chatbots have RAG; agents are everywhere → these two are covered. Simpler apps follow from them.

## The project: CampusX Doubt Solver
- A simple RAG chatbot over the **LLM Evals course lecture transcripts** → students ask doubts; it answers + **cites which session**.
- Kept simple on purpose: the focus is on **evaluating** it well, not building a fancy RAG.

## ⭐ The RAG evaluation framework (the interview answer)
```
EVAL SUITE (3 levels)
├── Component level  (build + test each part in isolation)
│     1. build Retriever  → 2. eval: Recall, Precision
│     3. build Generator  → 4. eval (fed hand-made question + context, NOT the retriever):
│                              Faithfulness, Answer relevance, Citation accuracy
├── Pipeline level
│     5. connect Retriever + Generator → 6. eval: RAG TRIAD
└── Application level
      7. Quality: Correctness, Completeness, Style (CampusX teaching tone)
      8. Safety: Toxicity, PII leakage, Jailbreak resistance
      9. Ops: Latency, Cost per query, Tokens
        ↓
REGRESSION TESTING (run the whole suite on every new version)
        ↓
DEPLOY → ONLINE EVALS (tracing, computed metrics, drift, alerts)
        ↓
SELF-IMPROVING LOOP (bad production cases → golden datasets)
```
- Build and test **together** (like software: unit tests → integration tests → system tests).

### Component metrics
| Component | Metric | Meaning |
|-|-|-|
| Retriever | **Recall** | of all correct docs, how many were fetched? |
| Retriever | **Precision** | of the fetched docs, how many were useful? |
| Generator | **Faithfulness** | answer only from the given context (no hallucination) |
| Generator | **Answer relevance** | answer addresses the question |
| Generator | **Citation accuracy** | it cites the right source (which lecture transcript) |

### The RAG Triad (pipeline level)
3 things exist: **Question**, **Context** (retrieved), **Answer** → one metric for each pair:
```
              Question
     Context /        \ Answer
   relevance/          \relevance
           /            \
     Context ─────────── Answer
             Faithfulness
```
| Pair | Metric |
|-|-|
| Question ↔ Context | **Context relevance**: is the retrieved context relevant to the question? |
| Context ↔ Answer | **Faithfulness**: is the answer grounded in the context? |
| Question ↔ Answer | **Answer relevance**: does the answer address the question? |

### Application-level metrics
- **Correctness** (right answer), **Completeness** (all parts of a multi-part question answered), **Style** (matches CampusX teachers' explanation style).
- **Safety**: toxicity, PII, jailbreaks. **Ops**: latency, cost / query, tokens.

## Tool: DeepEval
- Built-in metrics: RAG (answer relevancy, faithfulness, contextual precision / recall / relevancy), safety (toxicity, PII leakage...), agents, multi-turn, multimodal.
- **pytest-like syntax** (feels like normal software testing).
- Why not RAGAS? RAGAS was already covered in the Advanced RAG course, and DeepEval is **broader** (agents, chatbots, images...), with growing adoption.

## Project layout
```
project/
├── src/              # the RAG app: retriever.py, generator.py, rag_pipeline.py, (FastAPI / Streamlit)
├── evals/            # EVAL SUITE
│   ├── eval_retriever.py
│   ├── eval_generator.py
│   ├── eval_rag_pipeline.py
│   ├── eval_application.py
│   ├── eval_safety.py
│   └── eval_ops.py
└── run_evals.py      # runs every eval → one report
```

## Regression testing
> **Run the whole eval suite on the app** → is **v2 objectively not worse than v1**?

| Level | How |
|-|-|
| 1. Basic | run `run_evals.py`, record the first results as the **baseline**, compare each new run manually |
| 2. Experiment tracking | log **config** (chunk size, overlap, temperature, embedding model) + **metrics** per run to **MLflow** (or Confident AI, Weights & Biases) → dashboard comparing runs |
| 3. CI gating | **GitHub Actions**: every push → run the eval suite → compare with the baseline. Worse than baseline − threshold (e.g. 0.03) → **block the deploy**. Better → deploy + **update the baseline** |
- Learn the **concept**, not the tool (no standard tool for LLMs yet, unlike MLflow for ML).

## After deployment: online evals
- **Capture** (tracing / observability: LangSmith, Langfuse, Confident AI): latency, cost, tokens, 👍/👎.
- **Compute**: faithfulness, answer relevance, correctness on live traffic.
- **Drift**: e.g. faithfulness dropping over the last 8 hours → **alert** → fix.
- **Self-improving loop**: misbehaving chats → added to the **golden datasets** → better offline evals next version.

## The interview answer (say it as a framework)
1. "I build an **eval suite** at 3 levels."
2. **Component**: retriever (recall, precision), generator (faithfulness, answer relevance, citation accuracy).
3. **Pipeline**: the **RAG triad** (context relevance, faithfulness, answer relevance).
4. **Application**: correctness, completeness, style + **safety** (toxicity, PII, jailbreak) + **ops** (latency, cost, tokens).
5. **Regression testing** on every change (baseline → experiment tracking → CI gate).
6. **Deploy → online evals** (tracing, computed metrics, drift alerts).
7. Failures → back into the golden datasets.
> Don't just list 3-4 metric names. A **framework** answer shows real depth.

## Next lectures
1. Component level (build + eval retriever and generator; first DeepEval use)
2. Pipeline level (RAG triad)
3. Application level (quality, safety, ops)
4. Regression testing + online evals
