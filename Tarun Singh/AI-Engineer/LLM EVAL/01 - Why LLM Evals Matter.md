---
tags:
  - AIEngineer
  - llm-eval
---

# 01 - Why LLM Evals Matter (Playlist Intro)

> Video: https://www.youtube.com/watch?v=6W92_t9FveA
> **LLM Evals** = checking that your LLM app is actually good **before** (and after) you put it in front of real users.

## Who is an AI Engineer?
- Someone who **builds applications and products on top of foundation models** (LLMs).
- Common skills (everyone learns these): LangChain, RAG, agents, LangGraph / CrewAI / Agno, LangSmith, prompt engineering, n8n.
- **LLM Evals** is the **less common but very important** skill:
  - Interview favorite: *"How do you evaluate your RAG app / agentic app?"*
  - Gives you an edge over the competition (few good resources exist).
  - Mindset shift: from "personal project to show the interviewer" → "how does my app serve **crores of users** safely?"

## Vibe testing (what most of us do)
- **Vibe testing** = casually try the app with a few prompts and judge it **by feel**.
  - "I asked 5-10 questions, the answers looked good, so it works."
- Problems:
  - **Informal** (no metrics)
  - **Subjective** (based on feel)
  - **Not repeatable** (next version can't be tested the same way)
- OK for personal projects. **Never** for production.

## 3 real failures (deployed after vibe testing)
| Case                | What happened                                                                 | Lesson |
| ------------------- | ----------------------------------------------------------------------------- | ------ |
| **Air Canada**      | Chatbot **hallucinated** the bereavement-fare policy ("pay full, we refund later"). Real policy: discount only before booking. Customer sued. Air Canada said "the chatbot is a separate entity". The judge said: **your chatbot = your property = your responsibility**. They lost and paid. | Hallucination → legal + reputation loss |
| **Chevrolet dealer** | A user **jailbroke** the chatbot ("agree to everything I say"), then asked for a car for **$1**. The bot agreed and called it a "binding offer". Screenshots went viral. | No safety/jailbreak testing → public embarrassment |
| **Colombian airline lawyer** | A lawyer asked ChatGPT for past injury cases. It **invented cases** (names, dates, everything). The lawyer submitted them unchecked → fake cases, **~$5000 fine**, lost the case. | Confident hallucination, no verification |

→ **Evaluation is super important.**

## Why don't people evaluate then? Because it's harder than software testing
| Traditional software                         | LLM application                                                |
| -------------------------------------------- | -------------------------------------------------------------- |
| **Deterministic**: same input → same output (calculator: 2+2 = 4 always) | **Probabilistic**: same input → different outputs ("what is overfitting?" gives a different, still correct, answer each time) |
| Only check **correctness**                   | **Multi-dimensional**: factuality, completeness, tone, groundedness, latency, cost... |
|                                              | The dimensions **change from app to app**                      |

## Playlist roadmap
1. What are LLM evals (with an example)
2. The LLM eval **landscape** (techniques, tools: a mental map for every new term)
3. **Model evals**: evaluating LLMs themselves → **benchmarks**
4. **Application evals**: evaluating apps built on LLMs
5. Build your **own eval pipeline**: golden dataset + rubrics, run on your own app
6. **RAG** evals
7. **Agent** evals
8. **Safety** evals
9. **Operational** evals: after deployment → latency, tokens/sec, time to first token, load

## Remember
- Vibe testing = by feel, not repeatable → not for production.
- Your chatbot's words = your company's responsibility.
- LLM apps are **probabilistic** + **multi-dimensional** → harder to test than normal software → that's why we need proper evals.
