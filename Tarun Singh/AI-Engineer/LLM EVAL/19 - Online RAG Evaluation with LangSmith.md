---
tags:
  - AIEngineer
  - llm-eval
  - rag
  - langsmith
---

# 19 - Online RAG Evaluation: Monitoring Production with LangSmith

> Video: https://www.youtube.com/watch?v=wON2L2ZvKpE
> Code: https://github.com/campusx-official/rag-eval-deepeval (`evals/eval_online.py`, `src/rag_pipeline.py`, `src/reranker.py`)
> After deployment, keep evaluating **live traffic** to catch breakage. (Concepts: [[06 - Offline vs Online Evals]].)

## Which offline metrics can run online?
- Online = **no golden dataset / no reference answers** (you don't know future questions) → **reference-based metrics can't run**.

| Metric | Online? | Why |
|-|-|-|
| Contextual recall, contextual precision | ❌ | need ideal answers |
| Correctness, completeness | ❌ | need ideal answers |
| Faithfulness, answer relevancy, contextual relevancy (RAG triad) | ✅ | reference-free |
| Style | ✅ (skipped) | barely drifts between offline and online |
| Scope, leakage, toxicity | ✅ | reference-free (demo: **toxicity**) |
| Latency, cost | ✅ | just **logged** |
| Reliability | ✅ (skipped) | needs extra tools (Grafana + Prometheus) |
- The demo's online suite (6 metrics): **faithfulness, answer relevancy, contextual relevancy, toxicity, latency, cost**.

## Step 1: Connect the app to LangSmith (tracing)
1. smith.langchain.com → **Tracing** → new application/project (e.g. `cx doubt solver`) → **generate an API key**.
2. `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=cx doubt solver
```
3. `uv add langsmith`. Run the pipeline → a **trace** appears with input, output, context, **latency** and **cost**, logged automatically. Tracing is **async**, so it doesn't add latency.
- **Latency + cost are now "done"**: just by tracing.

### Problem: retrieval wasn't in the trace
- The auto trace showed only prompt → ChatOpenAI → output parser. **The retriever / reranker was missing** (so latency was understated: 4.14 s).
- Fix: add **`@traceable`** explicitly (children nest under the parent, no duplicates):
```Python
from langsmith import traceable


class RagPipeline:
    @traceable(run_type='chain', name='RagPipeline')
    def invoke(self, query):
        ...


class RerankingRetriever:
    @traceable(run_type='retriever', name='RerankingRetriever')
    def invoke(self, query):
        ...
```
- Now the trace is RagPipeline → Retriever → prompt → LLM → parser, and the latency is correct (4.79 s).
- Ran ~8 different questions to create traffic for the charts.

## Step 2: Dashboard + alerts
- **Monitoring**: LangSmith has a default dashboard (trace latency, error rate, LLM calls, cost...) or you build a **custom dashboard**.
- Add charts per metric, e.g. latency P50 / P95, cost; later **feedback** charts (toxicity, answer relevancy, faithfulness, contextual relevancy) with **avg / min / percentile** aggregation.
- **Alerts** on a chart: metric + threshold + time window (e.g. "faithfulness avg over the last hour < 0.7") → notify via Slack / PagerDuty / webhook.
- Toxicity was set up as a LangSmith online evaluator, so its score appears as **feedback** on each trace.

## Step 3: Run DeepEval (offline metrics) on live traces
```
cron job (every N min) ──> eval_online.py
     │  LangSmith SDK: fetch recent root traces (sample X%)
     │  DeepEval: faithfulness / answer relevancy / contextual relevancy
     └─> client.create_feedback(run_id, key, score, comment) ──> shows on the trace + dashboard
```
- Offline eval files read goldens; the online version reads **traces** instead → a separate file.
- In production use a **cron job**. In the demo: an infinite loop that sleeps 60 s.

```Python
import time
from dotenv import load_dotenv
from langsmith import Client
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric

load_dotenv()

PROJECT = 'cx doubt solver'
JUDGE_MODEL = 'gpt-4o-mini'
THRESHOLD = 0.7
SAMPLE_RATE = 1          # 1 = every trace. In production ~0.3 (3 metrics x several judge calls = expensive)
POLL_SECONDS = 60

client = Client()


def is_sampled(run):
    # deterministic: the same run is always sampled the same way
    return hash(str(run.id)) % 100 < SAMPLE_RATE * 100


def existing_keys(run):
    keys = set()
    for fb in client.list_feedback(run_ids=[run.id]):
        keys.add(fb.key)
    return keys


def score_recent_traces():
    runs = client.list_runs(project_name=PROJECT, is_root=True, run_type='chain')  # RagPipeline roots only
    for run in runs:
        if not is_sampled(run):
            continue

        outputs = run.outputs or {}
        answer = outputs.get('answer')
        context = outputs.get('context')            # post-rerank chunks (same as offline)
        query = (run.inputs or {}).get('query', '')
        if not answer or not context:
            continue

        already = existing_keys(run)

        jobs = [
            ('faithfulness',
             FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             LLMTestCase(input=query, actual_output=answer, retrieval_context=context)),
            ('answer_relevancy',
             AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             LLMTestCase(input=query, actual_output=answer)),
            ('contextual_relevancy',
             ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             LLMTestCase(input=query, actual_output=answer, retrieval_context=context)),
        ]

        for key, metric, test_case in jobs:
            if key in already:
                continue                                # never re-judge the same key
            try:
                metric.measure(test_case)               # single test case, no evaluate()
                client.create_feedback(run_id=run.id, key=key,
                                       score=metric.score, comment=metric.reason)
            except Exception as e:
                print(f'[{key}] failed on run {run.id}: {e}')   # one failing metric doesn't stop the others


if __name__ == '__main__':
    while True:                    # demo only. In production: cron calls score_recent_traces()
        score_recent_traces()
        time.sleep(POLL_SECONDS)
```
```bash
python -m evals.eval_online      # keep it running in its own terminal
```
- Demo trace ("Why do we need golden datasets?"): answer relevancy **0.92**, faithfulness 0.75 / 1.0, contextual relevancy **~0.23-0.29** (low, matching the offline finding), toxicity **0**. Scores show up after a short delay.
- Then add dashboard charts for each feedback key (avg) + alerts.

## Step 4: Close the loop (production failures → goldens)
- A trace with bad scores → **"Add to dataset"** → choose / create a LangSmith **dataset** → an expert writes the **reference answer** → submit.
- Download it and merge into the offline goldens, or better: **keep all golden datasets in LangSmith** from the start and load them via the SDK for offline evals too:
  1. Add failing traces with one click.
  2. Datasets are **versioned** (see how they grew; roll back to an old version).
  3. One source for **offline + online**.

## Wrap-up
- RAG evals (8 sessions) complete: framework → retriever → generator + triad → G-Eval quality → safety → ops → regression testing → online evals.
- Next in the playlist: **agent evals**.

## Remember
- Online = reference-free metrics only (triad, safety, style) + logged ops (latency, cost).
- Tracing first. Use `@traceable` when something (like the retriever) isn't captured.
- Score sampled live traces with DeepEval on a schedule → `create_feedback` → dashboard → alerts.
- Bad traces → datasets (with reference answers) → better offline evals. Keep goldens in LangSmith (versioned).
