---
tags:
  - AIEngineer
  - llm-eval
---

# 02 - What are LLM Evals? Model Evals vs Application Evals

> Video: https://www.youtube.com/watch?v=cNF_MO82Qew

## Definition
> **LLM Evals** = **systematic, repeatable tests** used to judge an **LLM** or an **LLM-powered system** against **clear criteria**.

| Property        | Meaning                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------- |
| **Systematic**  | not vibe testing. Build a proper **dataset** that covers edge cases (e.g. take 100 real user chats from the CampusX chatbot) |
| **Repeatable**  | change the prompt / model / retriever / chunking → run the **same tests** again → compare v1 vs v2 |
| **Clear criteria** | decide what "good" means. CampusX chatbot: correct, simple explanation, from our own course content, safe (no abuse/scary tone) |

## Eval ≠ metric
- Coming from ML, you may think eval = a metric (accuracy, precision, recall). **Wrong.**
- **LLM Eval = the complete testing setup:**
  - **What** you test (e.g. the retriever of a RAG app)
  - **Criteria** (e.g. retriever accuracy)
  - **Dataset** you test on
  - **When** (offline before deploy / online in production)
  - **Tool** (e.g. RAGAS for RAG)
- The goal is not "a score". It's **answering practical questions**:
  - Can this model be used for this task?
  - Is this system **good enough to ship**?
  - Did **prompt v2** improve over v1?
  - Is the RAG answer **grounded** in the retrieved context?
  - Is the agent completing the task correctly?
  - Is the chatbot **safe** for real users?
  - Is latency under control?

## 2 types of LLM evals
> ⚠️ "Model evals" / "Application evals" are **Nitish's own terms** to simplify. The industry just says "LLM evals" for both.

| | **Model Evals** | **Application Evals** |
|-|-|-|
| Evaluates | the **LLM itself** | the **whole LLM-based app** (and its parts) |
| Question | "**Can the model** do this?" | "**Does my product** work?" |
| Done by | frontier labs (OpenAI, Anthropic, Google...) | **you, the AI engineer** |
| How | **benchmarks**, leaderboards | your own datasets + criteria |
| Your job | just **read and understand** benchmarks → pick the right LLM | **actually do it** → main topic of this playlist |

### Model evals: 8 capabilities tested today
| Capability                | Example benchmark                 |
| ------------------------- | --------------------------------- |
| Reasoning                 | MMLU (with knowledge)             |
| Knowledge (world / GK till cutoff) | MMLU (science, history, law, medicine...) |
| Maths                     | GSM8K (grade school maths)        |
| Coding                    | SWE-bench, HumanEval              |
| Instruction following     | IFEval                            |
| Long context handling     | Needle in a Haystack              |
| Multimodal understanding  | MMMU                              |
| Tool use                  | -                                 |
- New model released → tested on benchmarks → "tops benchmark X" is published.
- For you: knowing benchmarks → better decision on **which LLM** (OpenAI / Anthropic / open source) for your project.

### Application evals: why they're needed
- In a real LLM app, the **LLM is just one component**. Also:
  - UI, system prompt, tools / APIs, orchestration code (LangGraph branching, parallel...), guardrails, output parsers, memory / context
  - RAG: retriever, embedding model, vector DB, reranker
  - After deploy: monitoring, feedback loop
- **Smartphone analogy:** a great chip (Snapdragon benchmark score) ≠ a great phone. Camera, OS, battery, screen, sound must also work, and each gets tested.
  - LLM = the chip (frontier labs test it). The system around it = **your** responsibility.
- Done at **2 levels:**
  - **Component level**: is the retriever / embedding model / reranker working?
  - **System level**: final response quality, latency, cost per token
- CampusX chatbot questions: answered correctly? course material used? faithful? easy for beginners? hallucination? fast? safe?

## Remember
- LLM eval = systematic + repeatable + clear criteria.
- Eval = the **whole testing setup**, not just a metric.
- Model evals → benchmarks (just learn to read them). Application evals → **your real job** (99% of "LLM evaluation" videos/blogs mean this).
