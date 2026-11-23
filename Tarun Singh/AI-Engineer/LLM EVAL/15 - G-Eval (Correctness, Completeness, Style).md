---
tags:
  - AIEngineer
  - llm-eval
  - rag
  - deepeval
---

# 15 - G-Eval: A More Deterministic LLM-as-a-Judge (Correctness, Completeness, Style)

> Video: https://www.youtube.com/watch?v=nlyxlKD5cvU
> Code: https://github.com/campusx-official/rag-eval-deepeval (`evals/eval_application.py`)
> Application level → **quality** (this lecture), then safety and ops (next).

## Count-based vs judgment-based metrics
- All 5 metrics so far (recall, precision, faithfulness, answer relevancy, contextual relevancy) are **count-based**: LLM splits into claims → count supported ones → ratio.
- Some metrics **can't be counted**, they need **judgment** of the whole answer:
  - **Style** ("does it sound like CampusX: why → what → how?"): exists at the answer level, not in single sentences.
  - **Correctness**: an **analogy** checked as a lone claim looks "unrelated" and gets wrongly penalized.
  - Also completeness, helpfulness, safety.
- → Use an LLM judge to **score the whole answer**.

## Plain LLM-as-a-judge (and why it's unreliable)
```
You are evaluating whether an AI's answer is correct.
Question: {question}
Expected answer: {expected}
Actual answer: {actual}
Compare the actual answer against the expected answer and decide how factually
correct it is. Give a score from 0 to 10 (10 = fully correct, 0 = completely wrong).
```
- Problem = **high variance**: the same inputs give 6 one run, 8 the next.
  1. A **vague criterion** (one line) → the judge interprets it differently on every call.
  2. **A single integer output**: if P(7) = 0.40 and P(8) = 0.51 → "8"; next run flips → "7". Scores jump.

## ⭐ G-Eval (2023 paper): 2 fixes
1. **Criteria → evaluation steps (via chain-of-thought)**: you give a metric **name** + a **high-level criterion**; the judge (best with GPT-4) expands it into explicit **evaluation steps** = a "rule book / constitution" → less room to improvise → more deterministic. Plus an optional **scoring rubric**.
2. **Probability-weighted score**: don't take the printed token; read the **log-probabilities** of the top score tokens and take a **weighted average**.

```
Top-5 tokens: '8': 0.70, '7': 0.20, '9': 0.05, 'the': 0.01, ':': 0.01
Drop non-numbers → 8: 0.70, 7: 0.20, 9: 0.05 (sum 0.95)
Normalize        → 8: 0.737, 7: 0.211, 9: 0.053
Weighted score   = 8×0.737 + 7×0.211 + 9×0.053 = 7.84   (plain decoding would say "8")
Scale to 0-1     → 0.784 → compare with the threshold (e.g. 0.7) → pass
```
- The weighted score moves smoothly (7.84 → 7.9), so no 6 ↔ 8 jumps.
- G-Eval is still LLM-as-a-judge, just with these 2 innovations.

### G-Eval judge prompt (example shape)
```
You are an evaluator scoring the correctness of an AI-generated answer.
Judge the actual output against the expected output using the evaluation steps below.

Evaluation steps:
1. Compare only the factual claims in the actual output against the expected output.
2. A claim is wrong only if it contradicts the expected output or is factually false.
3. A factually accurate answer scores high even if it is shorter / covers fewer points.
4. Do not deduct for brevity or omitted points; only wrong statements count.
5. Additional correct information must never lower the score.

Rubric: 9-10 all correct | 5-8 one minor inaccuracy | 0-4 clear factual error

Question: ...  Expected output: ...  Actual output: ...
```

## DeepEval `GEval`
| Param | Meaning |
|-|-|
| `name` | metric name (e.g. "Correctness") |
| `criteria` | high-level criterion → DeepEval generates the steps (CoT) |
| `evaluation_steps` | **your own steps** (skips generation → no step variance) |
| `rubric` | `Rubric(score_range=(a, b), expected_outcome=...)` list → you control the scoring too |
| `evaluation_params` | which test-case fields the judge sees (`INPUT`, `ACTUAL_OUTPUT`, `EXPECTED_OUTPUT`) |
| `model`, `threshold` | judge model, pass/fail cut-off (0-1) |
| `strict_mode` | **False** = probability-weighted score (keep it). True = raw integer |
- **Criteria vs your own steps**: start with `criteria` while exploring. Once you understand the failure patterns, write your own `evaluation_steps` + `rubric` = most deterministic.

## Correctness ≠ faithfulness
| | Faithful | Not faithful |
|-|-|-|
| **Correct** | ✅ ideal | model ignored the context but its training knowledge was right |
| **Incorrect** | taught wrong in class, the generator repeated it | hallucinated and wrong |
- Faithfulness = grounded in **your** context. Correctness = true **in the world** (ideal answers written by an expert).

## Golden dataset (`goldens/correctness_goldens.json`, 15 rows)
```json
{
  "id": "C001",
  "question": "why can't we test an LLM app the same way we test normal software?",
  "ideal_answer": "Normal software is deterministic ... LLM applications are probabilistic ... evaluation judges qualities of the output, usually against a golden dataset and often with an LLM as a judge.",
  "source_session": "S1"
}
```

## `evals/eval_application.py`
```Python
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from src.rag_pipeline import RagPipeline

load_dotenv()
JUDGE_MODEL = 'gpt-4o-mini'
THRESHOLD = 0.7

with open('goldens/correctness_goldens.json') as f:
    goldens = json.load(f)

rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g['question'])
    test_cases.append(LLMTestCase(
        input=g['question'],
        actual_output=result['answer'],
        expected_output=g['ideal_answer'],
    ))

# CORRECTNESS: reference-based, judges TRUTH (not coverage or length)
correctness = GEval(
    name='Correctness',
    evaluation_steps=[
        'Compare only the factual claims in the actual output against the expected output.',
        'A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.',
        'A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.',
        'Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.',
        'Additional correct information must NEVER lower the score.',
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome='All stated claims are factually correct and consistent. Brevity is fine.'),
        Rubric(score_range=(5, 8), expected_outcome='Mostly correct but one minor inaccuracy.'),
        Rubric(score_range=(0, 4), expected_outcome='Contains a clear factual error or contradicts the expected output.'),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD, model=JUDGE_MODEL, strict_mode=False,
)

# COMPLETENESS: reference-based, judges COVERAGE (not correctness)
completeness = GEval(
    name='Completeness',
    evaluation_steps=[
        'Identify the key points contained in the expected output.',
        'Check how many of those key points are addressed in the actual output.',
        'Penalize the actual output for each key point it omits or only partially covers.',
        'Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly.',
        'Do NOT penalize the actual output for adding extra information.',
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome='Addresses essentially all key points.'),
        Rubric(score_range=(5, 8), expected_outcome='Covers the main key points but misses one or more.'),
        Rubric(score_range=(0, 4), expected_outcome='Misses several key points.'),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD, model=JUDGE_MODEL, strict_mode=False,
)

# STYLE: reference-free (no EXPECTED_OUTPUT), judges TONE only
style = GEval(
    name='Style',
    evaluation_steps=[
        'Judge only the teaching style and tone, not correctness or completeness.',
        'Reward an intuitive, explanatory tone: plain language, the idea explained before jargon, technical terms briefly unpacked.',
        'Reward a direct, conversational register in prose, as a CampusX lecture would explain it out loud.',
        'An analogy or example is a BONUS for abstract concepts, but a clear direct answer must NOT be penalized for lacking one.',
        'Penalize stiff, bureaucratic, bare-list or jargon-heavy answers.',
        'Do NOT reward or penalize based on correctness, completeness, or length.',
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome='Clearly a CampusX teaching voice: intuitive, conversational, explains before formalizing.'),
        Rubric(score_range=(7, 8), expected_outcome='Clear, conversational prose. Fine without an analogy.'),
        Rubric(score_range=(4, 6), expected_outcome='Understandable but flat, formal, or list-heavy.'),
        Rubric(score_range=(0, 3), expected_outcome='Dry, stiff, bare-list, jargon-heavy, robotic.'),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=THRESHOLD, model=JUDGE_MODEL, strict_mode=False,
)

evaluate(test_cases=test_cases, metrics=[correctness, completeness, style])
```
```bash
python -m evals.eval_application
```

## Iterations (read the failed reasons → fix the judge or the generator)
| Step | Result |
|-|-|
| Correctness v1 (steps only) | **0.66** (8/15). The judge penalized **shorter but correct** answers (the expert's ideal answers are very thorough) |
| Correctness v2 (stated: brevity / omissions are fine + **rubric**) | **0.84** (14/15). Re-run: 0.83, the **same** case fails → deterministic ✅ |
| + Completeness | **0.68** (5/15 pass). The generator prompt said "concise" |
| Generator prompt: *"Answer thoroughly: identify every distinct part of the question and cover each one ..."* | **0.75** (14/15) |
| + Style (generator had no style guidance) | **0.54** |
| Generator prompt: *"Write in flowing, conversational prose like a teacher ... explain the intuition first ..."* + rubric fix ("analogy is a **bonus**", the judge had over-demanded analogies) | **0.74** (9/15; a threshold of 0.6 is reasonable) |
- Levers: **judge** (steps / rubric, when it's unfair) vs **app** (generator prompt). **Prompt engineering matters.**
- Pushing one metric too far can hurt others (e.g. style ↑ → faithfulness ↓).
- More G-Eval ideas: helpfulness, coherence, tonality, safety (next lecture).

## Remember
- Count-based metrics → claims + ratio. Judgment metrics → **G-Eval**.
- G-Eval = criteria → **evaluation steps (CoT)** + **probability-weighted score** → low variance.
- Your own `evaluation_steps` + `rubric` + `strict_mode=False` = most stable.
- Correctness (true in the world) ≠ faithfulness (true to the context) ≠ completeness (coverage) ≠ style (tone).
