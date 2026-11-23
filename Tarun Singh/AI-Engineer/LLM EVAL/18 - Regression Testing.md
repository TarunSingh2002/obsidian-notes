---
tags:
  - AIEngineer
  - llm-eval
  - rag
---

# 18 - RAG Regression Testing: Preventing Silent AI Failures

> Video: https://www.youtube.com/watch?v=Aov5QczE3Vo
> Code: https://github.com/campusx-official/rag-eval-deepeval (`evals/run_suite.py`, `compare.py`, `metric_registry.py`, `harness.py`, `eval_safety.py`, `eval_ops.py`)

## What is regression?
> **Regression** = returning to a previous, worse state.
> **Regression testing** = after **any** change, re-run the **whole eval suite** and check that **nothing got worse**.
- Example: recall is low → raise k + add a reranker → recall 85% → 90% ✅... but precision / contextual relevancy dropped and latency rose, and you never measured them → shipped → silent failure.
- An old software idea (e.g. Flask → FastAPI v2: is it still as good?). Same here.
- Taught **conceptually** with hand-written code; companies use tools (MLflow, Confident AI...). The concepts stay the same.

## The process
```
1. Build a script that runs ALL evals (quality + safety + ops)
2. First run → save results as  baselines/baseline.json
3. Make a change (k, chunk size, model, system prompt, reranker...)
4. Re-run → save as             baselines/candidate.json
5. Compare metric by metric → verdict
```
```
run_suite.py ──(1st run)──> baseline.json ─┐
run_suite.py ──(after change)──> candidate.json ─┴──> compare.py (+ metric_registry.py) ──> deltas + verdict
```

## 2 complications
1. **Direction differs**: higher = better (faithfulness, recall, success rate), lower = better (latency, cost, toxicity) → store each metric's **direction**.
2. **Noise**: LLM-judge metrics vary run to run **with no change at all** (85.6 → 84.8) → that isn't a regression.
   - Run the **same setup 5-10 times** → compute each metric's **standard deviation** → **noise threshold ≈ 2 × std**.
   - e.g. recall std 0.5 → threshold 1. Baseline 90 → candidate 89.5 = **noise**. 87.5 = **regression**.
   - In practice, add statistics (hypothesis testing, p-values, significance).
- Tiny golden sets (5-15 rows) → big swings (PII 60% ↔ 80%). Bigger goldens → more stable.

## Metric registry (`metric_registry.py`): a rule per metric
| Field | Meaning |
|-|-|
| **direction** | `higher` or `lower` is better |
| **kind** | **gate** (any drop beyond tolerance **blocks** the release, e.g. **safety**), **guardrail** (soft target, a drop → human **REVIEW**, e.g. quality, latency, cost), **info** (tracked, never affects the verdict) |
| **tolerance** | absolute `tol` and/or relative `rel_tol`. Regression only if `worsening > max(tol, rel_tol × abs(baseline))` |
- Measured noise floor (identical pipeline, 2 runs): scores moved up to ~0.033, p95 latency ~20% →
  - safety gates: tol **0.02**, quality guardrails: tol **0.05**, latency guardrail: **25%** relative, cost: **15%**.
  - TTFT is too noisy (~80%) to gate on → info only.
- Gate judge metrics on the **average score**, not the pass rate (the pass rate swings when scores sit near the threshold).

## Code pieces
- **Refactor**: each eval file gets a `run(...)` function that **returns** a summary (still runnable alone via `if __name__ == '__main__':`).
- Merge the 3 ops evals → `eval_ops.py`; merge the 3 safety evals → `eval_safety.py`.
- `harness.py`: shared `load_goldens()` + `summarize_by_metric()` (per-metric pass rate / avg / min / max, so one metric's drop can't hide behind others).
- **`run_suite.py`**: **build the pipeline once and inject it into every eval** (so the snapshot measures exactly one pipeline). Flattens every result into dotted ids (`retriever.contextual_recall.avg_score`, `safety.scope.avg_score`, `ops.latency.e2e_p95_ms`) + metadata (time, git SHA, **prompt hash**, label) → JSON.
```bash
python -m evals.run_suite --baseline --label "k=5 baseline"     # -> baselines/baseline.json
# ... change something ...
python -m evals.run_suite --label "chunk 750 + reranker"        # -> baselines/candidate.json
python -m evals.compare --all                                   # verdict + table
```

### The core compare logic (simplified)
```Python
import json


def load(path):
    with open(path) as f:
        return json.load(f)


def classify(baseline_value, candidate_value, rule):
    if rule['kind'] == 'info':
        return 'info'
    delta = candidate_value - baseline_value
    if rule['direction'] == 'higher':
        improved = delta > 0
    else:
        improved = delta < 0
    if improved:
        return 'improved'                      # improvements are never penalized
    worse = abs(delta)
    tolerance = max(rule['tol'], rule['rel_tol'] * abs(baseline_value))
    if worse <= tolerance:
        return 'flat'                          # inside the noise band
    if rule['kind'] == 'gate':
        return 'blocked'
    return 'regressed'


baseline = load('baselines/baseline.json')['metrics']
candidate = load('baselines/candidate.json')['metrics']

statuses = []
for metric_id in baseline:
    if metric_id not in candidate:
        continue
    rule = rule_for(metric_id)                 # from metric_registry.py
    status = classify(baseline[metric_id], candidate[metric_id], rule)
    statuses.append(status)
    print(metric_id, baseline[metric_id], '->', candidate[metric_id], status)

if 'blocked' in statuses:
    verdict = 'FAIL'       # a gate regressed -> blocked
elif 'regressed' in statuses:
    verdict = 'REVIEW'     # a guardrail regressed -> a human decides
else:
    verdict = 'PASS'       # safe to promote
print('VERDICT:', verdict)
```
- Exit codes make it a **CI gate**: PASS = 0, FAIL = 1, REVIEW = 2 (e.g. GitHub Actions blocks the deploy).

## Demo result
- Most metrics **flat** (within noise), ~4-5 **improved**.
- **Regressed**: **contextual relevancy** 0.45 → **0.38** (the very metric the change aimed at!), **PII leakage** dropped ~20% → a safety drop = big red flag.
- `promote.py` (a simple decision layer): metrics in tiers → **BLOCK** (a critical / safety metric fell), **REVIEW** (only a non-critical one fell; humans weigh it against the gains), **PROMOTE**. Here → **BLOCK**.
- A single weighted "overall quality" number is tempting but not recommended (different metrics = different aspects; a merge hides problems).

## Remember
- Every change (k, chunks, model, **prompt**) → re-run **everything** → compare with the baseline.
- Know each metric's **direction** and **noise threshold** (≈ 2σ from repeated runs).
- Gates (safety) block. Guardrails (quality / ops) → review. Info → just tracked.
- Baseline → candidate → compare → PASS / REVIEW / FAIL → CI/CD gate.
