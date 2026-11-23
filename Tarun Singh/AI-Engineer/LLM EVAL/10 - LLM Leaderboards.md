---
tags:
  - AIEngineer
  - llm-eval
---

# 10 - How to Use LLM Leaderboards

> Video: https://www.youtube.com/watch?v=SoZPmKb5uGc
> **LLM leaderboard** = a **public ranking / comparison table** showing how different LLMs perform on a **common set of evaluations**.
> Benchmark = the exam. Leaderboard = where the results are published.

## Why leaderboards exist
1. **Compare models across labs** with a common reference.
2. **Trust**: usually **third-party** (they have less at stake than OpenAI / Anthropic claiming their own scores).
3. **Model selection without resources**: you can't evaluate hundreds of models yourself → take the top ones.
4. **Spot saturation**: top 10 clustered at 92-94% → the benchmark is saturating.
5. **Discover new models**: scroll past the top 3-4 (Google / OpenAI / Anthropic) to rank 10-20 → cheaper models that may fit your use case.

## Who uses them
| Who | For |
|-|-|
| **AI engineers** | **shortlisting** candidate models (100 → 5), then run your own evals |
| **Frontier labs** | where they stand; **when to release** (if the new model doesn't beat the rival, don't release it; test in **stealth mode** under hidden names, e.g. Google's "Nano Banana" first appeared anonymously on an image leaderboard) |
| **Researchers** | which benchmarks are saturated → new research directions |
| **Policy makers / safety institutes** | monitor if a model jumps far ahead → step in |
| **Open-source community** | **discovery & publicity** (a small lab's model topping one benchmark gets famous, e.g. Chinese labs) |

## 4 types of leaderboards
| Type | What | Example | Usefulness |
|-|-|-|-|
| **1. Benchmark-specific** | ranking on **one** benchmark | HLE's own leaderboard (Gemini 3 Pro 38%, calibration error) | **least useful**: narrow view |
| **2. Multi-benchmark** | combines many benchmarks + often **cost / latency / speed / context window** | **LiveBench** (contamination-free, 23 tasks, 7 categories), **Artificial Analysis** (intelligence, speed, cost per task, HLE, GPQA Diamond, coding agents, speech, image...) | **most useful**: "which model has the best overall mix of capability, cost and performance?" |
| **3. Human-preference** | users vote between 2 anonymous answers (A / B / tie / both bad) → ranking from votes | **LMArena** (battle mode; text, code, image, video arenas) | popular for marketing. Bias: humans prefer long, confident, well-formatted, entertaining answers |
| **4. Application-specific** | one domain / task (may combine several benchmarks) | **Berkeley Function Calling Leaderboard** (tool calling), coding, agents, SQL, medical | use it when building for that domain |

## Why you can't blindly trust leaderboards
1. **Benchmark performance ≠ real-app performance** (like Kaggle vs real data science): clean data vs messy reality (ambiguous requests, missing info, company data, tool failures, edge cases).
2. **Contamination** → inflated scores.
3. **Over-optimizing for the leaderboard**. **Goodhart's law:** *"When a measure becomes a target, it ceases to be a good measure."* (e.g. tuning a model to please LMArena voters with formatting / flattery; a car engineered only for mileage).
4. **Hidden composition** in multi-benchmark boards: which benchmarks are in, how scores are normalized and **weighted**.
5. **Small differences don't matter**: 84.3 vs 84.1 = effectively the same model (like JEE rank 1 vs 25).
6. **Human-preference bias** (style over substance).
7. **Stale / incomplete / self-reported scores**: newest models missing, old versions, discontinued models, companies uploading their own numbers.

## How an AI engineer should read leaderboards (5 steps)
1. **Write your requirements first:** app type, latency needs, cost budget, context needs, deployment constraints (public API vs **on-prem** → open source). Avoids "rank 1 bias".
2. **Go to the right leaderboard** for your task:
   - agent → agent / tool-calling boards
   - chatbot → LMArena
   - RAG → **MTEB** (embedding model rankings)
   - budget → Artificial Analysis / Vellum (speed + price)
3. **Read it correctly** (read the fine print / FAQ):
   - what is scored and how? who evaluated?
   - inference budget, reasoning on or off?
   - dataset age, updated? private test set?
   - saturated?
   - confidence intervals (small dataset → close models are equal)
   - weighting in composite scores
4. **Shortlist the top 3-5** for your criteria.
5. **Run your own custom evals** on them → pick the winner (next lecture).

> ⭐ **Leaderboards are a *filtering* tool, not a *decision* tool.**

## Remember
- 4 types: single-benchmark, multi-benchmark (most useful), human-preference (LMArena), application-specific (e.g. BFCL, MTEB).
- Goodhart's law. Tiny score gaps are noise.
- Requirements → right board → read the details → shortlist 3-5 → custom evals → decide.
