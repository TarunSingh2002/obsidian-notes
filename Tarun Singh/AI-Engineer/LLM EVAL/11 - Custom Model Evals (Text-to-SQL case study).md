---
tags:
  - AIEngineer
  - llm-eval
---

# 11 - Selecting the Right LLM: Running Custom Model Evals (Hands-on)

> Video: https://www.youtube.com/watch?v=RG5A-W3eMHI
> Code: https://github.com/campusx-official/llm-sql-eval
> Goal: pick the **best LLM for *your* app** = requirements → leaderboard shortlist → **custom eval on your own data**.

## Case study: "Ask Cricinfo" (Text-to-SQL)
- ESPNcricinfo users ask stats questions during live matches ("Kohli's average vs Pakistan?"). Analysts used to write SQL by hand → doesn't scale (India vs Pakistan = 500 questions in 5 min).
- New feature: user question (English) + **DB schema** → LLM → **SQL** → run on the DB → show the result.
- Not RAG: no retrieved docs, only a fixed schema in the prompt.
- Job: choose the LLM that will be the brain.

```
Stage 1: Write requirements
Stage 2: Leaderboard → shortlist 5-10 candidates
Stage 3: Custom eval on your data → pick the winner
```

---

## Stage 1: Requirements
| Requirement | Decision | Why |
|-|-|-|
| **Task** | text → SQL | |
| **Cost ceiling** | ~₹3 lakh / month (5 lakh used when shortlisting) | a retention feature (users stay longer → more ads) |
| **Latency** | **2-3 seconds** | live match curiosity; no slow reasoning models |
| **Context window** | doesn't matter | single-turn, independent questions, no chat history |
| **Deployment** | public APIs are fine | no private data; managed APIs are more reliable than self-hosting |
| **Correctness** | **critical** | fans screenshot wrong stats → public embarrassment |

### Cost calculation (per month)
- Prompt ≈ **400 input tokens** (system prompt + schema + question), output ≈ **100 tokens** (SQL). ~**50,000 questions / day** (a gut estimate, spiky on big matches).
```
cost/query  = (input_tokens × input_price + output_tokens × output_price) / 1,000,000
cost/month  = cost/query × queries_per_day × 30 × 95 (USD → INR)
```
- E.g. Claude "Fable 5": $10 input, $50 output per 1M tokens:
  - (400 × 10 + 100 × 50) / 1M = **$0.009 / query**
  - × 50,000 = $450/day → × 30 = $13,500 → × 95 = **₹12.8 lakh / month** → 4× over budget ❌ → look lower (e.g. a Sonnet-class model).
- **Blended price** on leaderboards (e.g. "4:1 input:output") = (4 × input + 1 × output) / 5. Fable: (40 + 50) / 5 = $18. Our prompt is also 400:100 = **4:1**, so the blended price fits directly: 500 tokens × $18 / 1M per query.

### Prompt caching (cost saver)
- The system prompt + schema are **the same for every query**; only the question changes → **cache** that prefix on the provider's servers.
- Anthropic-style pricing: **cache write** costs more than normal input (e.g. $12.50 vs $10), **cache hits** are much cheaper (e.g. $1).
  - **5-minute cache**: refreshed if another query arrives within 5 min (fine at 50K queries/day).
  - **1-hour cache**: for low-traffic apps.
- Could cut ~₹12 lakh → ~₹6-7 lakh. Works because of the **KV cache** (attention K/V vectors). Less useful for RAG (the context changes every query).
- Domain-specific column names? Add column descriptions to the system prompt (caching keeps that cheap).

---

## Stage 2: Shortlist from a leaderboard
- Text-to-SQL leaderboards exist (**BIRD-SQL**, **Spider**, **LiveSQLBench**) but were stale, used fine-tuned / agent systems, or had unclear data → not trusted.
- Fallback: **coding leaderboards** (SQL generation ≈ coding; use coding ability as a proxy). Used **llm-stats.com** "Best AI for coding" (146 models, aggregated coding benchmarks + price + speed + context).
- Shortlisting method:
  1. Compute monthly cost for every model → **drop those over budget**.
  2. **Min-max normalize** the coding rating and the throughput (chars/sec) to 0-1.
  3. `score = 0.9 × coding + 0.1 × speed` (speed matters less: the output is a ~100-token SQL query).
  4. Sort → top 10. (Meta's Muse Spark dropped: US only.)
- Picked 5: **GPT-5.6 Terra**, **Kimi K3**, **Grok 4.5**, **Claude Sonnet 5**, **MiniMax M3** (cheap Chinese open-weights option).

---

## Stage 3: Custom eval
### Flow
```
golden dataset (question + gold SQL)
   for each model:
      for each question:
         question + schema → LLM → generated SQL
         run generated SQL on DB  → table A
         run gold SQL on DB       → table B
         compare A vs B (by VALUES) → correct / wrong
      accuracy = correct / total
```
- ⚠️ Don't compare SQL **text** (many queries give the same result). Compare the **result tables** = **execution accuracy**.
- Golden dataset: realistic mix (easy / medium / hard; joins, subqueries, window functions), written & validated by analysts. 50-500 rows; more = more cost (5 models × 500 Qs = 2,500 calls). The demo used **20 very hard** ones (LLM-generated, then validated by running them).

### Project files (12 steps)
| File | Does |
|-|-|
| `data/matches.csv`, `data/deliveries.csv` | Kaggle IPL data (ball-by-ball) |
| `db.py` | keeps 2021-2024 only → SQLite `ipl_2021_2024.db` (tables `matches`, `deliveries`) + indexes |
| `schema_extractor.py` | writes `schema.sql` (tables, columns, types) for the prompt |
| `golden_dataset_generator.py` | 20 questions + gold SQL; runs them to check there are no SQL errors |
| `make_golden_dataset.py` | runs the gold SQL, stores `gold_result_json`, `n_rows`, `order_sensitive` → CSV |
| `first_test.py` | smoke test: 1 question → 1 model → print the SQL |
| `model_openrouter_slug.py` | the 5 models as OpenRouter slugs |
| `evaluator.py` | compares the two result tables |
| `main.py` | orchestrates everything → `eval_results.csv` |
- **OpenRouter**: one API for all providers (OpenAI, Anthropic, xAI, Moonshot, MiniMax). LangChain integration `ChatOpenRouter`. Free credits; the whole run cost ~$5.

### Code (simplified)
```Python
# pip install langchain-openrouter python-dotenv pandas
# .env -> OPENROUTER_API_KEY=sk-or-...
import re
import json
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

MODELS = [
    ("GPT-5.6 Terra",   "openai/gpt-5.6-terra-pro"),
    ("Kimi K3",         "moonshotai/kimi-k3"),
    ("Grok 4.5",        "x-ai/grok-4.5"),
    ("Claude Sonnet 5", "anthropic/claude-sonnet-5"),
    ("MiniMax M3",      "minimax/minimax-m3"),
]

SYSTEM_MSG = (
    "You are a text-to-SQL generator. Given a database schema and a question, "
    "return a single SQL query that answers it. Use SQLite syntax. "
    "Return only the SQL query."
)

with open('schema.sql', encoding='utf-8') as f:
    schema = f.read().strip()

golden = pd.read_csv('golden_dataset.csv')
conn = sqlite3.connect('ipl_2021_2024.db')


def clean_sql(raw):
    # remove ```sql fences and any text before SELECT / WITH
    text = raw.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if start:
        text = text[start.start():]
    return text.strip().rstrip(';')


def run_sql(sql):
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        return None          # SQL error


scoreboard = {}
for name, slug in MODELS:
    llm = ChatOpenRouter(model=slug, temperature=0, max_tokens=800)
    correct = 0

    for _, row in golden.iterrows():
        user_msg = f"Schema:\n{schema}\n\nQuestion: {row['question']}\n\nSQL:"
        response = llm.invoke([SystemMessage(content=SYSTEM_MSG), HumanMessage(content=user_msg)])
        sql = clean_sql(response.content)

        generated_df = run_sql(sql)
        gold = json.loads(row['gold_result_json'])
        gold_df = pd.DataFrame(gold['rows'], columns=gold['columns'])
        order_sensitive = str(row['order_sensitive']).upper() == 'TRUE'

        if generated_df is None:
            print(row['id'], 'SQL ERROR')
        elif tables_match(gold_df, generated_df, order_sensitive):
            correct = correct + 1
            print(row['id'], 'OK')
        else:
            print(row['id'], 'MISMATCH')

    scoreboard[name] = correct / len(golden)

for name in scoreboard:
    print(name, scoreboard[name])
```

### The evaluator: comparing two result tables (simplified idea)
```Python
def canon(value):
    # 24395 == 24395.0 ; small float noise ignored ; strings stripped
    try:
        number = float(value)
        if number.is_integer():
            return 'num:' + str(int(number))
        return 'num:' + str(round(number, 4))
    except (TypeError, ValueError):
        return 'str:' + str(value).strip()


def table_to_rows(df):
    rows = []
    for row in df.values.tolist():
        clean_row = []
        for value in row:
            clean_row.append(canon(value))
        rows.append(tuple(clean_row))
    return rows


def tables_match(gold_df, gen_df, order_sensitive):
    # 1. row count must match
    if len(gold_df) != len(gen_df):
        return False
    gold_rows = table_to_rows(gold_df)
    gen_rows = table_to_rows(gen_df)
    # 2. row order ignored unless the question needs ORDER BY
    if not order_sensitive:
        gold_rows = sorted(gold_rows)
        gen_rows = sorted(gen_rows)
    # 3. compare VALUES only (column names ignored)
    return gold_rows == gen_rows
```
- The repo version is also lenient by **exactly one column**: "Who scored the most?" → a name only, or name + count; both are accepted if the smaller row's values are a **subset** of the bigger row's values.
- Reasons logged: `match`, `mismatch`, `sql_error`, `gen_error`.

## Results (20 hard questions)
| Model | Accuracy | Notes |
|-|-|-|
| **Grok 4.5** | **90%** | fast, ~₹2.5 lakh/month → best value |
| **Claude Sonnet 5** | **85%** | fastest in practice, ~₹2.84 lakh, reliable API |
| GPT-5.6 Terra | 80% | most expensive |
| MiniMax M3 | 65% | several SQL syntax errors |
| Kimi K3 | 55% | hyped (2.7T params) but slow (reasoning) + 6 SQL errors |
- Lesson: **leaderboard hype ≠ performance on your task** (Kimi K3).
- Final choice between Grok and Sonnet: close scores and cost → Nitish leans **Sonnet** (more stable, reliable vendor). Or the team votes.
- Improvements: a bigger dataset (with 20 Qs, 1 question = 5%), **multiple runs** for statistical confidence, also log **latency**.

## Remember
- Requirements (task, cost, latency, context, deployment, correctness) **before** looking at leaderboards.
- Monthly cost = tokens × price × queries/day × 30 × FX. Use prompt caching for a repeated prefix.
- Shortlist: filter by budget → normalize → weighted score.
- Text-to-SQL eval = **execution accuracy** (compare result values, not SQL text).
- The model eval part of the course ends here → next: **application evals** (RAG, agents).
