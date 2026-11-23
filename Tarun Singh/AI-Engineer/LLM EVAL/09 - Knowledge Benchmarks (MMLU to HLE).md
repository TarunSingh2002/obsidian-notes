---
tags:
  - AIEngineer
  - llm-eval
---

# 09 - Knowledge Benchmarks: from MMLU to Humanity's Last Exam

> Video: https://www.youtube.com/watch?v=QSOB9lNrNj4
> Reference site: https://benchwiki.vercel.app/ (BenchWiki: a "Wikipedia for LLM benchmarks" with status, history, scoring and issues for each benchmark)
> **Knowledge capability** = how much world knowledge (**parametric knowledge** in the weights) the model kept from training. The most fundamental expectation of an LLM; reasoning and coding came later as emergent abilities.

## The evolution story (learn this map)
```
                         MMLU (2020)  "mother of all benchmarks" → saturated ~2024
        ┌────────────────┬──────────┴──────────┬──────────────────┐
  Reliability       Human exams           Depth            Repair MMLU
 TruthfulQA (2021)  AGIEval (2023)      GPQA (2023)      MMLU-Pro (2024)
       │                                     └────────┬─────────┘
  SimpleQA (2024)                       HLE: Humanity's Last Exam (2025)
                                            (breadth × depth)
```
- Asking random questions ≠ systematic → **MMLU** (breadth).
- Big models trained on the internet also learn **misconceptions** → **TruthfulQA** (reliability) → replaced by **SimpleQA**.
- "Why invent tests? Use real human exams" → **AGIEval**.
- MMLU saturated → test **depth** → **GPQA**. Or **fix** MMLU → **MMLU-Pro**.
- Those saturate too → **HLE** = breadth + depth.

## Summary table
| Benchmark | Year | Tests | Data | Format | Status |
|-|-|-|-|-|-|
| **MMLU** | Sep 2020 | **breadth** of knowledge | ~14K Qs, **57 subjects** (humanities, social sci, STEM, other), from GRE / USMLE / AP / textbooks | MCQ, 4 options | **Saturated** |
| **TruthfulQA** | Sep 2021 | **truthfulness** vs common misconceptions | 817 adversarial Qs, 38 categories (correct + incorrect answers) | MCQ (MC1 / MC2) or generation | **Saturated** |
| **AGIEval** | Apr 2023 | performance on **real human exams** | 20 exam sections, 8K+ Qs (SAT, LSAT, LogiQA, AQuA, Gaokao, China civil service...), **bilingual EN + ZH** | 18 MCQ + 2 short-answer | **Saturated** |
| **GPQA** | Nov 2023 | **depth** (PhD-level science) | Physics, Chemistry, Biology. main 448, extended 546, **Diamond 198** (hardest) | MCQ, 4 options | **Near saturation** |
| **MMLU-Pro** | 2024 | MMLU fixed + harder + reasoning | ~12K Qs, **14** disciplines | MCQ, **10 options** | **Near saturation** |
| **SimpleQA** | 2024 (OpenAI) | short **factuality + calibration** | 4,326 short fact-seeking Qs (each one **GPT-4 got wrong**) | **no options**, write the answer | **Active** |
| **HLE** | Jan 2025 | **breadth × depth** | 2,500 public Qs, **100+ subjects**, ~1000 experts, 500 institutes, 50 countries + a **private held-out set** | 80% short answer, 20% MCQ; **10% multimodal** (images) | **Active, state of the art** |

---

## 1. MMLU (Massive Multitask Language Understanding)
- Used to market every model 2021-2024.
- History: GPT-3 = 43.9% vs experts ~90% → scaling-law era (Gopher, Chinchilla, PaLM) → **GPT-4 = 86%** (2023) → frontier models cluster at 86-92% (2024) → retired in 2025.
- Nobody passes ~92% because ~**6.5% of the questions are wrong** (bad / missing answer keys; found by MMLU-Redux).
- **Scoring:** 5-shot, temperature 0, pass@1, no CoT, no tools.
  - Answer extraction: **generate** the letter, or compare the **log-probabilities** of A/B/C/D and pick the highest. The two can differ by 1-3 points.
  - Overall accuracy + per-subject accuracy.
- ⚠️ Very **prompt-format sensitive** (labs gamed the prompt wording).
- Doesn't measure: reasoning depth, calibration, open-ended answers, multilingual (English only, Western curriculum).
- Issues: label errors, heavy contamination (public since 2020), prompt gaming.

## 2. TruthfulQA
- Key finding: **bigger models were often *less* truthful** (more internet → more misconceptions absorbed). "Capability ≠ alignment."
  - E.g. "Cracking knuckles causes arthritis" → a myth. "What happens if you eat watermelon seeds?"
- GPT-3 = 58% vs humans 94%.
- Then alignment techniques (RLHF, instruction tuning, DPO) improved it → saturated by 2024. Followed by **SimpleQA** and **MASK** (safety section).
- **Scoring:**
  - Generation → an LLM judge maps the output to an option.
  - **MC1** = highest log-prob option.
  - **MC2** (default) = total probability on **all true answers** (some questions have several correct ones), averaged.
- Zero-shot, but **6 fixed unrelated QA examples** are prepended to every question. No CoT, temperature 0.
- Doesn't measure: factual recall, **honesty under pressure** (→ MASK), non-Western misconceptions, multilingual.
- Issues: contamination at the **alignment** stage (in fine-tuning data); disputed gold labels; the LLM judge changed over time (GPT judge deprecated) → scores aren't comparable across years.

## 3. AGIEval
- Real exams → a real **measured human baseline**: average human **67%**, top humans **91%**.
- At launch: GPT-4 58%, ChatGPT 43%, text-davinci 37%. Saturated by 2025.
- Zero-shot, CoT allowed.
- ⚠️ "LLM beats humans on IIT-JEE / SAT" headlines ≠ human-level intelligence. It only tests exam knowledge, **not** long-horizon tasks, multi-step reasoning or tool use.

## 4. GPQA (Google-Proof Q&A)
- PhD-level questions, each **validated by 2 domain experts**. A non-expert **with Google and 30 minutes** still can't solve them.
- People report the **Diamond** score.
- History: GPT-4 39% → GPT-4o 56% (Diamond) → o1 78% (OpenAI's hired PhDs scored 69.7%, "beats PhDs" marketing) → Grok 4 ~87% (2025).
- Zero-shot, CoT, temperature 0, pass@1, no tools.
- Doesn't measure: general knowledge (science only), open-ended problem solving, **reasoning-trace correctness** (a lucky guess still gets the marks).
- Issues: **only 198 questions** → wide confidence interval; the PhD baseline is disputed (paper: 81.3% vs OpenAI: 69.7%); contamination.

## 5. MMLU-Pro
- "MMLU rebuilt to fix its flaws":
  - **4 → 10 options** (elimination gets much harder)
  - trivia / noisy questions removed → **reasoning-heavy** ones added (e.g. "2 kg block slides down a 30° frictionless incline, speed after 4 m?")
  - **57 → 14** balanced disciplines
- Reasoning models score ~**20 points higher** than non-reasoning ones → it really needs thinking.
- **MMLU-Redux** = not a benchmark; a paper showing ~6-8% of MMLU questions are wrong.
- 5-shot CoT, temperature 0, pass@1.
- Doesn't measure: open-ended generation, reasoning-trace correctness, **calibration** (does the model know what it doesn't know?).
- Issues: **no human baseline**, near saturation, favors reasoning models, sourced from public STEM problems (HC Verma-like) → contamination.

## 6. SimpleQA
- **No options** → much harder (a model with 88% on MMLU got ~40% here).
- An LLM judge grades each answer as **Correct / Incorrect / Not attempted**.
  - e.g. "Who received the IEEE Frank Rosenblatt Award in 2010?" → right / hallucinated / "I'm not certain".
- Philosophy: get as many right as possible, **don't attempt** what you're unsure of → measures **factuality + calibration** (humility).
- Metrics:
  - **Correct** (headline): correct / all questions
  - **Correct given attempted**: accuracy on attempted questions only
  - **F-score**: harmonic mean of the two
- History: GPT-4o 38%, o1-preview 42%, GPT-4.5 62.5% (Feb 2025). Still **active**.
- Doesn't measure: long-form factuality, RAG hallucination, everyday recall (questions are rare, adversarially picked).
- Issues: LLM-judge drift, **answer-key staleness** (e.g. "current #1 rugby player" changes), **selection bias against GPT-4** (built from GPT-4's failures).

## 7. HLE (Humanity's Last Exam)
- **Breadth (MMLU) × Depth (GPQA)**: expert-written questions from classics to rocket engineering, each **filtered to stump frontier models**.
- The name is a thesis: if models saturate this, closed-ended QA has nothing left to measure → move on to **open-ended / agentic** evaluation.
- **Private held-out test set** kept by the creators → fights contamination.
- Also measures **calibration**: the model states its confidence (e.g. 80%) → RMS calibration error between confidence and correctness.
- History: models were in single digits → Grok 4 ~24%, GPT-5 ~25%, Gemini 3 Pro ~38%. **The current go-to** for knowledge + reasoning + maths.
- No tools. Models without vision effectively face only 90% of the set.
- Doesn't measure: open-ended / agentic problem solving, everyday usefulness, multilingual (English only).
- Issues: disputed answers (3000 → 2500 after removals), LLM-judge grading noise, **failure-filter selection bias** (built from 2024 frontier failures).

## Remember
- MMLU = breadth (the start). TruthfulQA → SimpleQA = truthfulness & calibration. AGIEval = human exams. GPQA = depth. MMLU-Pro = MMLU fixed. **HLE = breadth × depth (today's top)**.
- Lifecycle: launch (models fail) → models improve / contamination → cluster near the ceiling → saturated → retired.
- MCQ is easier than writing the answer. Calibration ("know what you don't know") is measured by SimpleQA and HLE.
- Always check: how many questions, which subset (e.g. GPQA Diamond), the judge model, contamination, the human baseline.
