---
tags:
  - AIEngineer
  - llm-eval
---

# 04 - Why an AI App Needs Multiple Eval Pipelines

> Video: https://www.youtube.com/watch?v=DcZ-XCk-O_M
> One LLM app → **many eval pipelines** (99.99% of the time). There are 2 reasons: **multiple failure points** and **multiple risk categories**.

## Reason 1: Multiple failure points
### Example: a RAG chatbot
```
query → Retriever (vector DB) → top-k docs ─┐
query ──────────────────────────────────────┴→ Generator (LLM) → answer
```
Obvious failure points:
- **Retriever** fetches the wrong docs → wrong answer.
- **Generator** ignores the docs and hallucinates.

→ One eval pipeline for each:
| Pipeline       | Checks                                                                                  |
| -------------- | --------------------------------------------------------------------------------------- |
| Retriever eval | given a query → are the **relevant docs** fetched?                                      |
| Generator eval | given the context → is the answer **faithful / grounded** (only facts from the context, nothing added)? E.g. context says "ML course = 3 weeks" → answer "3 weeks", NOT "3 weeks, also buy our 4-week Python course" |

### Both components pass, but the app still fails ⚠️
- Query: "Duration of the ML course?", with `k = 5`.
- Retrieved: D1-D4 = random docs (one says "Python course = 6 weeks"), **D5** = "ML course = 8 weeks".
  - Retriever: ✅ the right doc is within its 5 tries.
- The generator's prompt says to prioritize the top docs → it mixes D1-D4 → "ML course = **6 weeks**" ❌.
  - Generator: ✅ it followed its instructions and didn't invent anything (it got bad ordering).
- **Each component works on its own, yet the pipeline broke.**
- → You need a **workflow-level eval** (retriever + generator together). It flags this.
  - Fix: add a **reranker** → moves D5 to the top → the pipeline works.

### Even a correct workflow can fail at the app level
- Everything is correct, but each answer takes **10 seconds** → not deployable.
- → **Application-level eval** (latency threshold, cost per query, time to first token).

### 3 levels of failure points
| Level           | Examples                                                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| **Component**   | system prompt, retriever, reranker, query rewriter, embedding model, vector DB, output parser, tool selector, memory, guardrails |
| **Workflow**    | RAG flow (retriever + generator), agent workflow, multi-turn chatbot flow                             |
| **Application** | whole-app latency, token cost per query, time to first token                                          |

## Reason 2: Multiple risk categories
Every failure point has **several aspects** that can go wrong (e.g. the retriever fetches the right docs but takes 5-10s). They're grouped into 3 risk categories:

| Category                | Question                                                   |
| ----------------------- | ---------------------------------------------------------- |
| **Application quality** | Does the app do its job well (correct, relevant, complete)? |
| **Safety**              | Is the output harmful (toxic, dangerous, biased, data leaks, jailbreaks)? |
| **Operations**          | Does it run **fast, cheap, reliably** when deployed?        |

### Risk categories cheat table
| Area | Risks to evaluate |
|-|-|
| **General LLM app** (e.g. summarizer) | correctness / accuracy, relevance, completeness (every question answered), instruction following (format, length) |
| **RAG** | context relevance, retriever recall, **groundedness / faithfulness**, citation accuracy |
| **Agents** | tool selection, parameter correctness, task completion, error recovery |
| **Multi-turn chatbot** | context retention (remembers the old chat), clarification behavior (asks when the query is ambiguous) |
| **Safety** | toxicity, harmful content (self-harm, weapons, illegal), bias (different answers per user profile), PII leakage (someone's card / contact details), prompt injection & jailbreak resistance |
| **Operations** | latency, cost per request, token efficiency, error / failure rate, latency under load |

→ Same app: one pipeline for latency, one for safety, one for correctness...

## Remember
- Component evals passing ≠ workflow working ≠ app ready.
- Evaluate at **component + workflow + application** levels.
- For each, think about **quality + safety + operations**.
