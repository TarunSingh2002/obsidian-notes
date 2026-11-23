---
tags:
  - AIEngineer
  - llm-eval
---

# 06 - Offline Evals vs Online Evals

> Video: https://www.youtube.com/watch?v=SahaDGzN-Bk
> **Offline eval** = evaluate **before deployment** (on a golden dataset).
> **Online eval** = evaluate **after deployment**, on **live production traffic**, with **no answer key**.

## Offline evals
- Everything from lectures 3-5 (Zomato router, retriever recall, UPSC grader) = **offline**. Nothing new, just a name.
- 3 big benefits:

| Benefit | What |
|-|-|
| **1. Pre-release testing** | never ship untested (Air Canada...). Can be automated as a **release gate** in **CI/CD**: push code → GitHub Actions runs the eval script → score ≥ threshold (e.g. 95%) → auto-deploy, else notify / roll back |
| **2. Version comparison** | Claude vs OpenAI model, prompt A vs B, reranker A vs B, vector DB, architecture... run the **same eval** (same golden dataset = level field) on both → pick the winner |
| **3. Regression testing** | improving one thing can break another. E.g. a "be very kind and polite" prompt fixed cold refund replies but made the bot round the price ("~19,000" instead of 19,500). The golden dataset has all question types (refund, pricing, curriculum) → check each category's score stays the same after a change (refund was 90% → must not drop to 80%) |

## Why offline isn't enough: 3 production risks
1. **Unanticipated inputs**: the golden dataset had 200-500 expected questions, but real users send anything: Hinglish, ambiguous half-questions, angry rants, adversarial prompt injections, edge cases.
2. **Emergent / systematic failures** (only visible at scale):
   - new course launch → thousands of concurrent users → **latency spikes**
   - **subtle bias** (e.g. worse answers for non-technical users) only visible across thousands of chats
3. **Drift**: prices, curriculum, policies change over a year → docs change → the old golden dataset becomes **obsolete** → offline scores still look good, but users give negative feedback.
- Offline needs a golden dataset + correct answers. In production there are **neither**.

## Offline vs Online
| | Offline | Online |
|-|-|-|
| When | **before** deployment | **after** deployment, continuously |
| Data | fixed golden dataset | **live production traffic** |
| Answer key | mostly yes | **no** → estimate on the go |
| Inputs | what you **anticipated** | **anything** |
| Catches | **regressions** | **drift**, surprises, emergent bugs |
| Best for | **gating** (CI), version comparison | drift detection, real-world performance |
| Cost / speed | fast, cheap, repeatable | can be costly at scale → **sampling** |

- **Not rivals, complementary. Always use both.**
- ⭐ **Offline checks the app works *correctly*. Online checks the app is running *normally*.**

### "Correct" vs "normal" (UPSC grader)
- Offline: LLM marks ≈ human marks (MAE) → **correctness**.
- Online: no human marks for new answers → can't measure correctness.
  - Instead: plot the **weekly distribution of scores** (baseline). Weeks 1-4 look the same; week 5 suddenly has many 800+ scores → **something changed** → investigate (could be a bug, could be smarter students).
- No answer key? Options:
  - Metrics that **don't need one**: e.g. **faithfulness** (context + answer → LLM judge).
  - **Proxies**: thumbs up/down rate, score distributions vs baseline.

---

## The online eval pipeline
### Step 1: Logging (everything first)
> Capture a **structured, replayable record of every conversation turn**.

| What to log | Fields |
|-|-|
| IDs | conversation_id, turn_id, user_id, session_id, timestamp |
| Content | user question, retrieved context (RAG), model output |
| Operational | latency (ms), prompt tokens, completion tokens, cost, errors / status code |
| User signals | 👍/👎, escalation ("talk to a human", email support), rephrasing the same question |

Engineering properties:
- **Non-blocking**: logging must not add latency to the chat.
- **Durable & queryable**: a data warehouse / observability tool (**LangSmith**) → fetch it anytime.
- **Late signal attachment**: signals that come later (an escalation email next day) get linked by `conversation_id`.
- **PII handling**: **mask** phone, card, Aadhaar, DOB **before** storing.

### Step 2: Know your signals
| **Captured** (just record) | **Computed** (needs an evaluator) |
|-|-|
| 👍/👎, latency, cost per conversation, token usage, error rate | faithfulness, answer relevance, correctness, hallucination, toxicity, bias & fairness |

### Flow for captured signals
```
Log → Dashboard (aggregated over time: last 1h / 24h / 7d / 6m) → Alerting
```
- Look at **aggregates**, not single chats (one slow chat ≠ a problem; the average rising for an hour = a problem).
- E.g. course launch → latency graph jumps → add EC2 instances / load balancer.
- **Alerting**: nobody watches graphs 24/7 → threshold crossed → Slack / email / PagerDuty / webhook.
- LangSmith → **Monitoring** (trace latency, error rate, LLM call count, cost; time window selector) + **Alerts** (metric, threshold, last 5 min, Slack / PagerDuty / custom webhook).

### Flow for computed signals (e.g. hallucination)
```
Log → Sampling → Evaluator (LLM-as-a-judge, reference-free) → Dashboard → Alerting
```
- No answer key → **reference-free** → **LLM-as-a-judge** with a rubric (sees question + retrieved context + output).
- Running a judge on all 50,000 chats/day = cost doubles → **sample** (e.g. 1,000).
- **Stratified sampling > random:** split chats into categories and sample more from the **risky** ones: 👎, abrupt endings, escalations, repeated rephrasing, money topics (refund / fees / admission). Skip 👍 chats mostly.
- Many evaluators run in parallel (one pipeline per signal).

### LangSmith evaluators
- **Evaluators → templates** (all LLM-as-a-judge): Security (PII leakage, prompt injection, code injection), Safety (toxicity, bias & fairness), Quality (hallucination, correctness, conciseness, ...), conversation, agent trajectory, image, voice.
- Setup: name → application → judge model (OpenAI / Claude + API key, temperature) → prompt / rubric → output format.
- Run on:
  - **Tracing project** → **online** evaluator
  - **Dataset** → **offline** evaluator
- **Datasets & Experiments** → create datasets, run offline experiments (from code).
- In a bad trace → **"Add to dataset"** / **annotation queue** → production failures become offline test cases.

## The self-improving loop
```
Offline eval → deploy → online eval finds failures → add them to the golden dataset → offline eval → deploy → ...
```

## Q&A
- **Is faithfulness 0.87 good?** Compare against a **baseline** (usually from offline evals). Baseline 0.85 → fine. It drops to 0.75 in 24h → alert / fix.
- **Offline score went 92 → 99, is it good?** Only if **other metrics didn't regress**.

## Remember
- Offline = correct (before deploy). Online = normal (after deploy).
- Online pipeline: **log → (sample → evaluate) → dashboard → alert**.
- Captured signals go straight to the dashboard. Computed ones need a sampled LLM judge.
- Feed production failures back into the golden dataset.
- Tools ahead: LangSmith, DeepEval, RAGAS.
