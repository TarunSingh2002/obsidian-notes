---
tags:
  - AIEngineer
  - llm-eval
---

# 07 - Model Evals & the 8 Core LLM Capabilities

> Video: https://www.youtube.com/watch?v=FPS0rIAQwzo
> **Model eval** = a systematic process of measuring a model's **capabilities, behavior, reliability and operational characteristics** under **controlled conditions**.

## Why should an AI engineer care?
"If you can't measure it, you can't improve it." For an AI engineer:
1. **Pick a model for your app**: in a team meeting you can't say "OpenAI or Claude, both are fine". Give **concrete numbers** on the capabilities your app needs.
2. **Track whether new models really improve**: "Opus 4.8 is deployed, should we switch to the new one?" → justify it with numbers.
3. **Safety**: how much does it hallucinate, can it be jailbroken?
4. **Proprietary API vs self-hosted open source**: Claude (powerful, pricier) vs DeepSeek from Hugging Face on your own servers (cheaper, maybe weaker) → compare.
> Without model evals you're **blind**.

## How a model eval works (4 steps)
1. **Pick the capability** to test (reasoning? coding? safety? instruction following?). There's no single "IQ" score for LLMs → **one eval per capability**.
2. **Bring a test** for it.
3. **Run the model under a fixed protocol** (same prompts, same conditions for all models → repeatable, fair).
4. **Score & interpret** (publish / compare).

## 2 types of tests
| | **Benchmarks** | **Custom evals** |
|-|-|-|
| What | standardized, shared tests (MMLU, SWE-bench) | your **own eval set** from your **actual task** |
| Good for | comparing models on **common ground** (everyone runs the same test) | measuring what **you specifically** care about |

### Why custom evals? Zomato email router example
| | Model A (big, top of leaderboard) | Model B (small, mid-table) |
|-|-|-|
| Price | $15 / 1M tokens | $0.50 / 1M tokens |
| Classification accuracy (your 200-500 labelled emails) | 94% | **91%** |
| Urgency accuracy | 88% | **87%** |
| Cost per 1000 emails | $6 | **< $0.21** |
| Latency | 4.1 s | **0.9 s** |
- Benchmarks alone → Model A wins everything.
- Custom eval → **Model B** = far better value (almost the same accuracy, much cheaper and faster).
- Benchmarks → this lecture + next. Custom model evals → a later lecture.

---

## The 8 core capabilities (most benchmarks target one of these)
| # | Capability | What is measured | Why it matters / real use |
|-|-|-|-|
| 1 | **Knowledge & Reasoning** | factual recall across subjects (MMLU = 57 subjects), **multi-step logical reasoning** (connect facts in the right order → conclusion; e.g. "summarize evolution from the Big Bang → explain today's society") | = **how intelligent** a model is → every lab wants to top it. Research bots, technical docs Q&A, professional assistants (law, teaching) |
| 2 | **Coding & Software Engineering** | working function-level code + tests, **bug fixing in existing codebases**, multi-file long-horizon tasks (refactors), **terminal / CLI** tasks (install packages, configure servers), API / function calling | the biggest **economic** value (Cursor's valuation) → AI coding agents |
| 3 | **Mathematics** | accurate symbolic + numeric reasoning: grade school → competition (olympiad) → undergrad → research-level | scientific computing, financial modelling, engineering simulations, data analysis |
| 4 | **Long Context** | using info from very long inputs (100Ks of tokens): find a small fact in a huge context, track an entity, summarize long docs, keep a large codebase in mind | models claim 128K / 1M windows, but quality **degrades** as context grows → this tells which really works |
| 5 | **Vision & Multimodal** | understanding images, video ("what can I cook from this fridge?") | we live in a multimodal world |
| 6 | **Agentic & Tool Use** | web browsing, structured tool calling, API interaction, using desktops / computers | needed for reliable agentic apps |
| 7 | **Safety & Alignment** | no harmful content, resistance to adversarial attacks, **truthful vs sycophantic** (doesn't flatter bad ideas), cybersecurity skills (crypto, reverse engineering, forensics) | government pressure + reputation risk for labs |
| 8 | **Instruction Following** (underrated) | does exactly what was asked: bullet list, < 200 words, friendly tone; asks **clarifying questions** when instructions are ambiguous | directly drives user satisfaction (ignoring instructions → users leave) |

## Remember
- Model evals help an AI engineer **choose / switch / trust** models.
- 4 steps: capability → test → run under a fixed protocol → score & interpret.
- Benchmarks = standard comparison. **Custom evals = what matters for *your* task** (often a cheaper model wins).
- 8 capabilities: knowledge & reasoning, coding, maths, long context, multimodal, agentic / tools, safety, instruction following.
