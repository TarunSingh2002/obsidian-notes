---
tags:
  - AIEngineer
  - llm-eval
  - rag
  - ops
---

# 17 - RAG Operational Evals: Latency, Cost, Reliability

> Video: https://www.youtube.com/watch?v=kuTgQM9zhq0
> Code: https://github.com/campusx-official/rag-eval-deepeval (`evals/eval_ops.py`, `eval_latency.py`, `eval_cost.py`, `eval_reliability.py`)
> The last part of the eval suite: can the app run **fast, cheap and reliably** at scale?

## What operational evals are
- **Software + telemetry measurements**: **no golden dataset, no LLM judge** → free to run (apart from the app's own LLM calls).
- 3 main ones: **latency** (how long users wait), **cost** (money per query), **reliability** (how often it fails).
- (**Throughput** needs load / stress testing with dedicated tools → out of scope.)

### Why run them offline too (not only after deploy)?
| | Correctness | Faithfulness | Answer rel. | Avg latency | P95 latency | Cost / query | Timeout | Success |
|-|-|-|-|-|-|-|-|-|
| v1 (k=5) | 91% | 94% | 93% | 2.3 s | 4.8 s | ₹0.72 | 2% | 99.8% |
| v2 (+reranker, k=10, bigger LLM) | 95% | 96% | 95% | **4.1 s** | **6.2 s** | **₹1.08** | 1% | 99.9% |
- Quality improved, but latency nearly **doubled** and cost went up 50%. Without ops evals you'd find out only from angry users and the bill.
- Laptop numbers aren't absolute truth, but the **difference between versions (the direction)** is meaningful (same setup each time).
> **"Don't wait until production to discover your RAG pipeline is too slow or too expensive."**

---

## 1. Latency
> **Latency** = the time the system takes to respond to a request (question sent → full answer shown).

Basic idea: start a timer → `rag.invoke(question)` → stop → `end - start`. 6 things to get right:

1. **Distributions, not averages**: report **P50 (median), P95, P99**. P95 = 95% of requests finish within this time. The **tail** shows how bad the worst user experience is. Dashboards show it over windows (last 1h / 24h / 7d).
2. **Break it down**: end-to-end **and** per component (embedding, retrieval, rerank, generation) → find the slow part. E.g. 4.5 s = 1.5 s retrieval + 3 s generation.
3. **TTFT (time to first token)**: with **streaming**, users see text early (easier to read, feels faster). Measure TTFT separately (it includes retrieval).
4. **Cold starts**: the first calls are slow (connection setup, model loading e.g. downloading the reranker, vector DB connection, network handshake, cache warm-up, serverless boot) → **discard warm-up runs**.
5. **Report token counts / context size** with latency (longer answers and bigger context = slower).
6. **Latency ≠ throughput**: latency = one request's wait (how long a diner waits for food). **Throughput** = requests handled per unit time (meals the kitchen serves per hour). Above the throughput limit, requests **queue** → latency rises. A laptop test = single user.

### Simple latency eval
```Python
import math
import time
from src.rag_pipeline import RagPipeline
from src.generator import generate_stream

QUESTIONS = [
    'What is the difference between reference-based and reference-free evals?',
    'Explain what faithfulness measures in a RAG pipeline.',
    'How does the G-Eval metric assign a score?',
    'What is MMLU and why is contamination a problem?',
]
REPEATS = 5
WARMUP_RUNS = 2
SLO_P95_MS = 3000          # full answer p95 under 3 s
SLO_TTFT_P95_MS = 1200     # first token p95 under 1.2 s


def percentile(values, p):
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


rag = RagPipeline()

# warm-up: run and throw away (cold start)
for i in range(WARMUP_RUNS):
    rag.invoke(QUESTIONS[i % len(QUESTIONS)])

total_ms = []
retrieval_ms = []
ttft_ms = []
for question in QUESTIONS:
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        docs = rag.retriever.invoke(question)
        context = []
        for doc in docs:
            context.append(doc.page_content)
        t1 = time.perf_counter()

        first_token_time = None
        for piece in generate_stream(question, context):      # streaming generator
            if first_token_time is None:
                first_token_time = time.perf_counter()
        t2 = time.perf_counter()

        retrieval_ms.append((t1 - t0) * 1000)
        ttft_ms.append((first_token_time - t0) * 1000)
        total_ms.append((t2 - t0) * 1000)

p95 = percentile(total_ms, 95)
print('end-to-end p50/p95/p99:', percentile(total_ms, 50), p95, percentile(total_ms, 99))
print('retrieval p95:', percentile(retrieval_ms, 95))
print('TTFT p95:', percentile(ttft_ms, 95))
if p95 <= SLO_P95_MS:
    print('SLO full answer: PASS')
else:
    print('SLO full answer: FAIL')
```
- `generate_stream()` = the same chain with `.stream()` instead of `.invoke()`, yielding non-empty chunks.
- Latency levers: smaller / faster model, streaming, fewer / smaller chunks, faster reranker or none, caching, better infra.

---

## 2. Cost
> **Cost = tokens × price**: derived, not measured. Near-deterministic (temperature 0 → stable token counts) → an honest **offline** metric: "can we afford N users/day?"

```
cost/query = uncached_input × input_price + cached_input × cached_price + output × output_price   (per 1M tokens)
```
- gpt-4o-mini (example prices per 1M): input $0.15, **cached input $0.075**, output $0.60 (output ≈ 4× input → long answers dominate).
- Read real usage from the LLM message: call `prompt | llm` (**without** `StrOutputParser`, which drops usage) → `msg.usage_metadata`: `input_tokens`, `output_tokens`, `input_token_details['cache_read']`.
- Project: × queries/day (e.g. 2000) × 30 × USD→INR → monthly. Budget line, e.g. **≤ $0.0015 / query**.
```Python
from src.generator import prompt, llm

measured_chain = prompt | llm           # keep the AIMessage (has usage)

PRICE_INPUT = 0.15
PRICE_CACHED = 0.075
PRICE_OUTPUT = 0.60

docs = rag.retriever.invoke(question)
context_parts = []
for doc in docs:
    context_parts.append(doc.page_content)
context_text = '\n\n'.join(context_parts)

msg = measured_chain.invoke({'question': question, 'context': context_text})
usage = msg.usage_metadata
input_tokens = usage.get('input_tokens', 0)
output_tokens = usage.get('output_tokens', 0)
details = usage.get('input_token_details') or {}
cached_tokens = details.get('cache_read', 0) or 0

uncached = input_tokens - cached_tokens
if uncached < 0:
    uncached = 0
cost = (uncached * PRICE_INPUT + cached_tokens * PRICE_CACHED + output_tokens * PRICE_OUTPUT) / 1_000_000
print('cost per query (USD):', cost)
```
- In the demo the provider auto-cached aggressively (most of ~1750 input tokens cached) because the **same question was repeated**; in production caching will be lower.
- Biggest lever = **the model** (e.g. open source on your own infra). Other tweaks give ~±5%.

---

## 3. Reliability
> Can it serve requests **without errors, timeouts, crashes or broken stages**? 8 of 10 answered → 80% reliable.

| Metric | Meaning |
|-|-|
| **Success rate** / **error rate** | complements (1 − each other) |
| **Timeout rate** | % of requests exceeding the allowed time (≠ error) |
| **Retry rate** | % of requests needing ≥ 1 retry |
- Good practices:
  - **Categorize failures**: LLM API, retriever, reranker, timeout, **rate limit**, parser / formatting, internal exception (try/except around each stage).
  - **Test under load separately** (concurrency breaks things).
  - Use **enough samples** (25 requests may all pass; 1000 shows rare errors).
  - Use **representative requests** (simple, long context, complex, long answers, edge cases) → segment the results.
- Exponential-backoff retry wrapper:
```Python
import time

MAX_RETRIES = 2

success = 0
errors = 0
retried = 0
total = 0
for question in QUESTIONS:
    for _ in range(5):
        total = total + 1
        attempt = 0
        done = False
        while attempt <= MAX_RETRIES and not done:
            try:
                rag.invoke(question)
                done = True
            except Exception:
                attempt = attempt + 1
                time.sleep(2 ** attempt)           # exponential backoff: 2s, 4s
        if attempt > 0:
            retried = retried + 1
        if done:
            success = success + 1
        else:
            errors = errors + 1

print('success rate:', 100 * success / total, '%')
print('error rate:', 100 * errors / total, '%')
print('retry rate:', 100 * retried / total, '%')
```
- Result on a laptop with OpenAI: **100% success, 0% errors, 0 retries** (ideal setup). Real numbers come from production (many users, flakier APIs, e.g. Ollama cloud models, rate limits).

---

## The full RAG eval suite is now complete
```
Component:   retriever (contextual recall, precision) · generator (faithfulness, answer relevancy)
Pipeline:    RAG triad (+ contextual relevancy)
Application: quality (correctness, completeness, style via G-Eval)
             safety  (toxicity, PII / prompt / content leakage, scope)
             ops     (latency p50/p95/p99 + TTFT, cost/query, reliability)
```
Next: **regression testing** (run it all on every change), then **online evals**, then agent evals.

## Remember
- Ops evals = measurements: no goldens, no judge.
- Latency: percentiles (P95 / P99), per stage, TTFT, discard warm-up, report tokens, latency ≠ throughput.
- Cost = tokens × price (watch cached tokens); project to a monthly budget.
- Reliability: success / error / timeout / retry rates; categorize failures; test under load.
- Offline ops numbers matter for the **direction of change** between versions.
