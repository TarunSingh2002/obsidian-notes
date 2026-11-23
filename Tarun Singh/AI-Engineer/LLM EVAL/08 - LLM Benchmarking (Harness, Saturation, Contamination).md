---
tags:
  - AIEngineer
  - llm-eval
---

# 08 - What is LLM Benchmarking? Eval Harness, Saturation, Contamination

> Video: https://www.youtube.com/watch?v=qIiU3lyjrhM
> Code (Colab): https://colab.research.google.com/drive/1Jg-ujeyTfYn9yNqKSt7uvN0gJOtySiIg
> **Benchmark** = a **standardized test** used to measure a **specific model capability**.

## 4 parts of every benchmark (example: GSM8K)
**GSM8K** = **G**rade **S**chool **M**aths, ~**8.5K** questions (class 6-8 level word problems). Old (2021), now saturated. On Hugging Face / Kaggle.

| Part | What | GSM8K |
|-|-|-|
| **1. Dataset + task** | questions **+ answer key** (like a golden dataset) + what the model must do | "Natalia sold clips to 48 friends in April, and half as many in May. How many altogether?" → 48 + 24 = **72** |
| **2. Run configuration** | the fixed settings, **same for every model** | see below |
| **3. Scoring** | **extract** the answer from the output, then **compare** | extract `72` from "The answer is 72" (regex / structured output) → exact match. Open-ended answers → LLM-as-a-judge |
| **4. Aggregation** | combine per-question scores | mean of 0/1 → 920/1000 = 92%. MMLU (57 subjects) → per-subject scores, **weighted mean** (subjects have different question counts) |

### Run configuration details
- **Prompt construction**
  - **Zero-shot** (just the question) vs **few-shot** (show solved examples first). GSM8K = **8-shot**.
  - **Chain-of-thought** (solve step by step) vs direct answer. GSM8K allows CoT.
- **Decoding / sampling**: **temperature ≈ 0** (no randomness), **max tokens** (too low → CoT gets cut; too high → unfair advantage; same for all models).
- **Scoring strategy**
  - **pass@1**: ask once, right or wrong (strict)
  - **pass@k**: ask k times, correct if **any** attempt is right (lenient)
  - **maj@k**: ask k times, take the **most common** answer (mode)
  - → When someone says "82% on X", ask **which one**.
- **Environment / tools**: tools allowed? (web search / code interpreter would make maths trivial). GSM8K: no tools. SWE-bench: tools needed (fetch GitHub issues).
- All of this is written in the benchmark's **research paper** (most benchmarks started as papers).

## How a lab evaluates a model on a benchmark (a loop)
```Python
results = []
for item in dataset:                                  # every question = one row
    prompt = build_prompt(item, fewshot=8, chat_template=True)
    raw_output = model.generate(prompt, temperature=0, max_tokens=512, stop=['Question:'])
    prediction = extract_answer(raw_output)          # "The answer is 72" -> "72"
    score = 1 if prediction == item['answer'] else 0
    results.append(score)

final_score = sum(results) / len(results)            # aggregation
```
- Looks simple, but at 8,000+ API calls you need: answer extraction, exact paper scoring, **batching**, **retries** on failures, **rate-limit** handling...
- **Eval harness** = the plumbing code that runs model evaluation. Use a library instead of writing it:
  - **lm-evaluation-harness** (EleutherAI): the industry standard
  - Inspect, HELM
  - DeepEval also has benchmarks (more code, and mostly saturated ones; DeepEval is more for **application** evals)
- Analogy: **benchmark = exam paper**, **harness = the exam administration**.

### Code: run GSM8K with lm-evaluation-harness
```Python
!pip install -q "lm_eval[api]"

import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key: ")
```
```bash
!lm-eval run \
    --model openai-chat-completions \
    --model_args "model=gpt-5.6-luna,num_concurrent=5,max_retries=5" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --apply_chat_template \
    --limit 20 \
    --output_path ./gsm8k_results \
    --log_samples
```
| Flag | Meaning |
|-|-|
| `--model openai-chat-completions` + `model=...` | which model (Hugging Face models also work) |
| `num_concurrent=5` | parallel requests |
| `max_retries=5` | retry failed API calls |
| `--tasks gsm8k_cot` | the benchmark (with chain-of-thought) |
| `--num_fewshot 8` | 8-shot prompting |
| `--apply_chat_template` | use the chat format |
| `--limit 20` | only 20 questions (**testing only**: a full run costs ~₹2300, 20 questions ~₹3-4). Remove it for real metrics |
| `--output_path`, `--log_samples` | save results + a per-question JSON (doc_id, question, answer, score) |
- Result: ~90% (18/20). No loop code needed.

## Who runs benchmarks?
| Who | Why | How much to trust |
|-|-|-|
| **Frontier labs** (OpenAI, Anthropic, Google DeepMind) | track training checkpoints (pre-training direction), **release gating**, **marketing** | **a ceiling**, not reality: favorable in-house conditions + **cherry-picking** (show the wins, hide the losses). Like a car's claimed mileage of 25 → 10 in real driving |
| **Third-party evaluators** (LMArena, independent companies) | evaluation **is** their product; same conditions for all models | **most reliable**. Also show cost & latency. Like IMDb ratings for models |
| **AI engineering teams** (you) | run public benchmarks yourself (lm-eval-harness) in **your** conditions → also measure latency & cost | yours |

## Why benchmarks can mislead (take them with a pinch of salt)
1. **Contamination**: public benchmark questions + answers are on the internet → scraped into pre-training data → the model may have **memorized** them. Fix: **private** benchmarks, **dynamic** benchmarks (refreshed regularly).
2. **Saturation**: a new benchmark → models score 25-36% → models improve → everyone clusters at 92-97% → it **can't differentiate** models anymore → retired, replaced. Saturated: **GSM8K, MMLU, SWE-bench**. Lifecycle: hard → improving → saturated → retired.
3. **Configuration gaming**: favorable settings for your own model (e.g. give it a Python interpreter on GSM8K, higher max tokens / reasoning effort) and defaults for rivals → 5-10% swings. Unknown temperature, max tokens, latency, cost.
4. **Aggregation tricks**: report only the MMLU **average**, hiding weak subjects (great physics, bad economics) → your economics chatbot fails.

## Remember
- Benchmark = dataset + run config + scoring + aggregation (all in the paper).
- Same config for every model, or the comparison is meaningless.
- pass@1 vs pass@k vs maj@k matter.
- Use **lm-evaluation-harness** instead of writing the loop.
- Trust third-party > your own runs > lab claims. Watch for contamination, saturation, config gaming, aggregation hiding.
