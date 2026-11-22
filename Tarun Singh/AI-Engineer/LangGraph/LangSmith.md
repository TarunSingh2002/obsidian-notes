---
tags:
  - AIEngineer
---

# LangSmith (Observability for LangGraph)

> **LangSmith** = records every run of your graph (input, output, tokens, latency, every node and LLM call) so you can **see what happened** later.

## 1. Setup
1. Make an account on **smith.langchain.com** → Settings → API Keys → **Create API Key**.
2. Put these in your `.env` file:
```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_api_key
LANGSMITH_PROJECT=chatbot-project
```
3. Load them in your code:
```Python
from dotenv import load_dotenv
load_dotenv()
```
- **That's it. No other code change.** LangSmith traces the graph automatically.
- `LANGSMITH_PROJECT` → the project is created in LangSmith on the first run.

## 2. How it's organized
```
Project (chatbot-project)
 └─ Trace  = one invoke = one chat turn (your message + AI reply)
     └─ chat_node -> ChatGroq call (input, output, tokens, latency, time to first token)
```

## 3. Problem: all chats mixed together
- Every turn becomes a trace in the **same flat list**, even turns from **different conversations** (`thread_id`s).
- **Fix:** LangSmith **Threads**. Pass the `thread_id` in `metadata` of the config, and LangSmith groups the traces by conversation.

```Python
config = {
    'configurable': {'thread_id': thread_id},   # for LangGraph (checkpointer), same as before
    'metadata': {'thread_id': thread_id},       # NEW: for LangSmith -> groups traces into a thread
    'run_name': 'chat_turn',                    # optional: trace name 'chat_turn' instead of 'LangGraph'
}

result = chatbot.invoke({'question': 'what is india capital?'}, config=config)
```
- The metadata key can be `thread_id`, `session_id` or `conversation_id` (any one works).
- Now in LangSmith → project → **Threads** tab: one thread per conversation, and inside it all its turns in order, shown like a chat.

## 4. `@traceable`: trace your own Python functions
- LangChain/LangGraph things (graph, nodes, LLM calls) are traced **automatically**.
- A **normal Python function** (cleaning text, calling some API, your own logic) is **not**. Put `@traceable` on it and it shows up in LangSmith too (input, output, time, errors).

```Python
from langsmith import traceable

@traceable(name='clean_question') #  run_type='tool', tags=['v1'], metadata={'user': 'tarun'}, can also be added here
def clean_question(text: str):
    return text.strip().lower()

def chat_with_ai(state: ChatState):
    question = clean_question(state['question'])   # shows as its own step inside the trace
    response = model.invoke(question)
    return {'answer': response.content}
```
- Called **inside a node** → it shows as a **child step** under that node in the same trace.
- Called **on its own** (outside a graph) → it becomes its **own trace**.
- Useful options: `@traceable(name='...', run_type='tool', tags=['v1'], metadata={'user': 'tarun'})`
  - `run_type`: `'chain'` (default), `'tool'`, `'llm'`, `'retriever'`, ... → just changes the label/icon in LangSmith.
- Still needs the same `.env` setup (`LANGSMITH_TRACING=true` etc.).

## 5. User feedback (👍 / 👎) via code
Idea: every trace (run) has a **`run_id`**. Feedback is attached to a run by that id.
1. **You** make the `run_id` yourself (`uuid4()`) and pass it in the config → the trace gets that exact id.
2. Show the answer, ask the user 👍/👎.
3. Send it with `Client().create_feedback(run_id, ...)`.

```Python
import uuid
from langsmith import Client

client = Client()   # uses LANGSMITH_API_KEY from .env

run_id = uuid.uuid4()   # our own id for THIS trace

config = {
    'run_id': run_id,                    # the trace in LangSmith gets this id
}

result = chatbot.invoke({'question': 'what is india capital?'}, config=config)
print(result['answer'])

# ask the user
user_input = input('Was this answer good? (y/n): ')

if user_input == 'y':
    score = 1
else:
    score = 0

comment = input('Any comment? (press enter to skip): ')

# send feedback to LangSmith -> attached to the trace with this run_id
client.create_feedback(
    run_id=run_id,
    key='user_rating',   # name of the feedback (you choose it)
    score=score,         # 1 = good, 0 = bad (can be any number, e.g. 1-5 stars)
    comment=comment,
)
```
- In LangSmith: open the trace → **Feedback** tab shows `user_rating`. The project's runs list gets a `user_rating` column, and Monitoring shows the average score over time.
- The `key` is just a name. You can send several kinds, e.g. `'user_rating'`, `'correct'`, `'helpful'`.
- Feedback can be sent **any time later** (even minutes after the answer), as long as you kept the `run_id`. In a real app you save the `run_id` with the message, and when the user clicks 👍/👎 you call `create_feedback`.

### What can a `run_id` be?
- It **must be a UUID** → `uuid.uuid4()`. A normal string like `'thread-1'` → **error**.
- It **must be NEW for every trace** (every `invoke` = every chat turn). Never reuse one.
- **Never use the checkpoint `thread_id` as the `run_id`:**
  - `thread_id` **repeats**: every turn of the same chat uses the same `thread_id`.
  - So turn 1 and turn 2 would get the same `run_id` → the 2nd run **clashes** with the 1st one already in LangSmith, and you can't tell which turn a 👍/👎 was for.
```
thread_id = one conversation  -> SAME for every turn of that chat   (checkpointer / LangSmith Threads)
run_id    = one turn (trace)  -> NEW uuid4() before every invoke   (LangSmith feedback)
```

## 6. Other things LangSmith can do (not covered here)
- **Evaluation:** make a **dataset** of inputs (+ expected outputs), run your app on it, and score the answers (with code checks or an **LLM-as-judge**). Compare versions (new prompt / new model) to see if things got better or worse **before** you ship.
- **Monitoring:** dashboards over your traces in production: number of runs, **errors**, **latency**, **token usage / cost**, user feedback, over time.
- **Prompt experimentation:** in the **Playground**, try different prompts / models / settings (temperature etc.) side by side and compare the outputs. Save good prompts in the **Prompt Hub** with versions, and pull them into your code by name, so you can change a prompt without changing code. Run a prompt version against a dataset (Evaluation) to see which one is actually better.
- **Alerting:** set rules on those numbers (e.g. error rate goes up, latency too high, cost spikes) and get **notified** (email / Slack / webhook) when they happen.

## Cheat sheet
| What                      | How                                            |
| ------------------------- | ---------------------------------------------- |
| Turn on tracing           | `.env` with the 4 `LANGSMITH_*` vars + `load_dotenv()` |
| Group turns by chat       | `config['metadata'] = {'thread_id': thread_id}` |
| Rename the trace          | `config['run_name'] = 'chat_turn'`             |
| Set the trace id yourself | `config['run_id'] = uuid.uuid4()`              |
| Send user feedback        | `Client().create_feedback(run_id=run_id, key='user_rating', score=1)` |
| Trace a normal function   | `from langsmith import traceable` → `@traceable(name='...')` on the function |
