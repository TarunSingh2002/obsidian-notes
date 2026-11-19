---
tags:
  - AIEngineer
---

# Persistence in LangGraph

> **Persistence** = the ability to **save and restore the state** of a workflow **over time**.

Persistence is a **foundational** topic. Short-term memory, fault tolerance, human-in-the-loop (HITL) and time travel are all built on top of it.

## The problem persistence solves
- When you call `workflow.invoke(...)`, the state changes node by node and you get a final state at the end.
- **Default behaviour:** once the execution ends, the state is **erased** (it lived only in RAM). You can't get it back later.
- **With persistence:** the state is **saved somewhere (a database)**, so you can look at it or keep using it in the future.

## Key idea: it saves the intermediate values, not just the final one
Example: the state has `name`. START gives `'a'`, node 1 changes it to `'b'`, node 2 changes it to `'c'`.

| Point in the graph | Saved value of `name` |
| ------------------ | --------------------- |
| before start       | (empty)               |
| at start           | `a`                   |
| after node 1       | `b`                   |
| after node 2 (end) | `c`                   |

Without persistence you only get `c`. With persistence you get **every** step. That's why the definition says "**over time**".

This is what makes these features possible:
- **Fault tolerance**: if the workflow crashes at node 2 (server down, API down...), you re-run it and it **resumes from where it crashed**, not from the start.
- **Resume chat**: like ChatGPT's old conversations. The past messages were saved, so you can load them and continue.

Where is it saved? In **some kind of database** (RAM for demos; Postgres/Redis/SQLite for real apps).

## Term 1: Checkpointer
- Persistence in LangGraph is implemented with a **checkpointer**.
- The checkpointer splits the graph's execution into **checkpoints** and saves the full state at each one.
- **Every super-step = one checkpoint.**
  - Super-step = one "round" of execution. Nodes that run **in parallel** count as **one** super-step together and a normal node also as one super-step.

Example (state `numbers: Annotated[list[int], operator.add]`, i.e. with a reducer):
```
START ─> node1 ─┬─> node2 ─┐
                ├─> node3 ─┼─> END
                └─> node4 ─┘
```

| Checkpoint | What happened              | Saved `numbers`   |
| ---------- | -------------------------- | ----------------- |
| 1          | input given at START       | `[1]`             |
| 2          | node1 added 2              | `[1, 2]`          |
| 3          | node2, 3, 4 ran in parallel | `[1, 2, 3, 4, 5]` |
| 4          | reached END                | `[1, 2, 3, 4, 5]` |

So the database holds **4 state snapshots** for this one run.

## Term 2: Threads (`thread_id`)
- Problem: you run the same workflow many times (input `1`, then input `6`, ...). All their checkpoints go into the **same database**. How do you load only **one run's** values?
- Answer: **thread ID**. Every time you execute the workflow, you give it a `thread_id`. All checkpoints of that run are saved **against that thread ID**.
- Later: "give me everything for `thread_id = 2`" → you get only that run's state.
- Chatbot analogy:
  - New conversation → new `thread_id`.
  - All messages of that conversation are stored under that `thread_id`.
  - "Resume that old chat" → fetch its `thread_id` → load its messages → continue.

---

## Code: persistence in practice

Workflow: give a **topic** → `generate_joke` → `generate_explanation` → END.

> The lecture calls an LLM (`ChatOpenAI`) inside both nodes. Here a small fake generator stands in for the LLM so the code runs without an API key. The persistence part is exactly the same. To use a real LLM, replace the body with `llm.invoke(prompt).content`.

### 1. Build the graph with a checkpointer
```Python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver   # the checkpointer, save values in ram
from typing import TypedDict
import random


# State
class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


# Nodes (random choice = behaves like an LLM: a new answer every time)
def generate_joke(state: JokeState):
    templates = [
        "Why did the {t} go to the doctor? It was feeling a little saucy!",
        "Why did the {t} go to the party? It wanted to be the top-ing guest!",
        "Why did the {t} bring a ladder? To rise to the occasion!",
    ]
    joke = random.choice(templates).format(t=state['topic'])
    return {'joke': joke}


def generate_explanation(state: JokeState):
    explanation = f"Explanation: this is a pun-based joke -> {state['joke']}"
    return {'explanation': explanation}


# Graph
graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

# The ONLY extra work for persistence: create a checkpointer and pass it to compile()
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)
```

`InMemorySaver` stores the checkpoints in **RAM**. Close the program and they're gone. It's only for **learning and demos**.
In production, use a database checkpointer: `SqliteSaver` / `PostgresSaver` (and Redis-based ones). Same concept, different storage.

### 2. Invoke with a `thread_id`
With a checkpointer, you **must** pass a `thread_id` in the config:
```Python
config1 = {"configurable": {"thread_id": "1"}}

result = workflow.invoke({'topic': 'pizza'}, config=config1)
print(result)
# {'topic': 'pizza', 'joke': 'Why did the pizza ...', 'explanation': 'Explanation: ...'}
```

### 3. `get_state()`: the final (latest) state of a thread
```Python
snapshot = workflow.get_state(config1)

print(snapshot.values)   # the state dict  -> topic, joke, explanation
print(snapshot.next)     # which node runs next -> () = nothing left, the run finished
print(snapshot.config['configurable']['checkpoint_id'])   # the ID of this checkpoint
```
- It returns a `StateSnapshot` object: `values`, `next`, `config` (has the `checkpoint_id`), `metadata` (has the `step`), `created_at`, `parent_config` (the previous checkpoint).
- With a real database checkpointer, you could close the program, come back 3 days later and still get this.

### 4. `get_state_history()`: ALL checkpoints (intermediate + final)
```Python
for snap in workflow.get_state_history(config1):
    print(snap.metadata['step'], snap.values, '| next ->', snap.next)

# 2 {'topic': 'pizza', 'joke': '...', 'explanation': '...'} | next -> ()
# 1 {'topic': 'pizza', 'joke': '...'}                       | next -> ('generate_explanation',)
# 0 {'topic': 'pizza'}                                      | next -> ('generate_joke',)
# -1 {}                                                     | next -> ('__start__',)
```
- The list is **newest first**.
- **4 checkpoints** for 2 nodes:
  1. before START: empty state, next = `__start__`
  2. before `generate_joke`: only `topic`
  3. before `generate_explanation`: `topic` + `joke`
  4. before END: everything, next = nothing

### 5. Different thread = completely separate history
```Python
config2 = {"configurable": {"thread_id": "2"}}
workflow.invoke({'topic': 'pasta'}, config=config2)

print(workflow.get_state(config1).values['topic'])   # pizza  (thread 1 untouched)
print(workflow.get_state(config2).values['topic'])   # pasta
```
Both threads live in the same checkpointer, but each `thread_id` has its own state and history.

---

## Benefits of persistence (4)
1. **Short-term memory** (chatbots)
2. **Fault tolerance**
3. **Human in the loop (HITL)**
4. **Time travel**

### 1. Short-term memory
- To **resume a past conversation**, the old messages must be stored somewhere. In LangGraph, **persistence is the way** to do that.
- Same `thread_id` → the state from the last call is loaded, and the new input is **added to it** (through the reducer).

Tiny non-LLM chatbot to show it:
```Python
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=key,
    max_tokens=900,  # free tier: 1000 output tokens/min; default asks for 2048
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    humen_message : str

def get_human_message(state: ChatState):
    human_input = input('Put END to end')
    return { 'humen_message': human_input }

def is_chat_ends(state: ChatState) -> Literal[ 'save_user_text_in_history', '__end__']:
    if state['humen_message']=='END':
        return END
    else:
        return 'save_user_text_in_history'

def save_user_text_in_history(state: ChatState):
    return { 'messages': [HumanMessage(state['humen_message'])] }

def talk_to_bot(state: ChatState):
    ai_message=model.invoke(state['messages'])
    print(ai_message)
    print(type(ai_message))
    print("="*50)
    return { 'messages':[ai_message] }

graph = StateGraph(ChatState)

graph.add_node('get_human_message', get_human_message)
graph.add_node('save_user_text_in_history', save_user_text_in_history)
graph.add_node('talk_to_bot', talk_to_bot)

graph.add_edge(START, 'get_human_message')
graph.add_conditional_edges('get_human_message', is_chat_ends)
graph.add_edge('save_user_text_in_history', 'talk_to_bot')
graph.add_edge('talk_to_bot', 'get_human_message')

chatbot = graph.compile(checkpointer=InMemorySaver())

# run 1 
config1 = {"configurable": {"thread_id": "id1"}}

#invoke 
result = chatbot.invoke({'messages': [SystemMessage('You are a chatbot')]}, config=config1)
# day 2 invoke 
result = chatbot.invoke({'messages': [SystemMessage('You are a chatbot')]}, config=config1)

# Final value of alst state
snapshot = chatbot.get_state(config1)
print(snapshot.values)

# seeing all state
for snap in chatbot.get_state_history(config1):
    print(snap.metadata['step'], snap.values, '| next ->', snap.next)
```
The second call on `chat-A` still had `'hi'` in it. That's short-term memory, and it only works because of the checkpointer + `thread_id`.

### 2. Fault tolerance
- If the workflow **crashes** at some node, you resume **exactly from that point**. You don't re-run everything from the start.
- Possible because the state was saved at every step, so LangGraph knows the **last good state** and **which node is next**.
- **How to resume:** call `invoke(None, config)` with the **same `thread_id`**.
  - `None` as input = "don't start fresh, continue from wherever this thread stopped".

> Below, step 2 **raises an error on the first run** instead, so the same idea works in any normal Python script.

```Python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict


class CrashState(TypedDict):
    input: str
    step1: str
    step2: str
    step3: str


attempts = {'count': 0}   # used to make step 2 fail only the first time


def step_1(state: CrashState):
    print("Step 1 executed")
    return {'step1': 'done'}


def step_2(state: CrashState):
    attempts['count'] += 1
    if attempts['count'] == 1:
        print("Step 2 crashed!")
        raise RuntimeError("server went down")     # simulated crash
    print("Step 2 executed")
    return {'step2': 'done'}


def step_3(state: CrashState):
    print("Step 3 executed")
    return {'step3': 'done'}


builder = StateGraph(CrashState)
builder.add_node('step_1', step_1)
builder.add_node('step_2', step_2)
builder.add_node('step_3', step_3)

builder.add_edge(START, 'step_1')
builder.add_edge('step_1', 'step_2')
builder.add_edge('step_2', 'step_3')
builder.add_edge('step_3', END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "thread-1"}}

# Run 1 -> crashes inside step 2
try:
    graph.invoke({'input': 'start'}, config=config)
except RuntimeError as e:
    print("Crash:", e)

snap = graph.get_state(config)
print(snap.values)   # {'input': 'start', 'step1': 'done'}   -> step 1 was saved
print(snap.next)     # ('step_2',)                           -> LangGraph knows where it stopped

# Run 2 -> resume: input = None, SAME thread_id
final = graph.invoke(None, config=config)
print(final)

# Output:
# Step 1 executed
# Step 2 crashed!
# Crash: server went down
# {'input': 'start', 'step1': 'done'}
# ('step_2',)
# Step 2 executed          <- NO "Step 1 executed" again: it resumed at step 2
# Step 3 executed
# {'input': 'start', 'step1': 'done', 'step2': 'done', 'step3': 'done'}
```
Very useful for **long-running workflows**: you don't want to redo 20 minutes of work because step 7 failed.

### 3. Human in the loop (HITL)
- Example: topic → LLM writes a **LinkedIn post** → **ask the human "should I post it?"** → post it through the LinkedIn API.
- The tricky part: the human might answer in 1 second, 1 hour or 2 days. You can't keep the workflow alive in memory for 2 days.
- So LangGraph **interrupts** (pauses) the execution at that point and waits. When the human replies, it **resumes from exactly that point**.
- How does it know where to resume? **Persistence**: the state was checkpointed at the pause.
- It's like fault tolerance, but **on purpose**. Fault tolerance handles external failures; HITL is a pause you planned.
- (The code comes in a separate HITL lecture. It also needs a checkpointer.)

### 4. Time travel
- **Go back** to any old checkpoint and **re-run** the graph from there.
- Main use: **debugging**. Go back to the point where things went wrong and run again from there.
- Every checkpoint has its own **`checkpoint_id`**. That's how you point to one checkpoint.

**Step 0: the joke graph we use** (same as above, with a checkpointer) + one normal run
```Python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
import random


class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


def generate_joke(state: JokeState):
    templates = [
        "Why did the {t} go to the doctor? It was feeling a little saucy!",
        "Why did the {t} go to the party? It wanted to be the top-ing guest!",
        "Why did the {t} bring a ladder? To rise to the occasion!",
    ]
    joke = random.choice(templates).format(t=state['topic'])
    return {'joke': joke}


def generate_explanation(state: JokeState):
    explanation = f"Explanation: this is a pun-based joke -> {state['joke']}"
    return {'explanation': explanation}


graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

workflow = graph.compile(checkpointer=InMemorySaver())

# normal run -> this creates the checkpoints we will go back to
config1 = {"configurable": {"thread_id": "1"}}
workflow.invoke({'topic': 'pizza'}, config=config1)
```

**Step 1: find the old checkpoint**
We want the checkpoint where only `topic` is set (no joke yet), i.e. the one whose next node is `generate_joke`.
```Python
history = list(workflow.get_state_history(config1))

old = None
for snap in history:
    if snap.next == ('generate_joke',):
        old = snap

print(old.values)   # {'topic': 'pizza'}
print(old.config)   # {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f1e...'}}
```
- `old.config` = the "address" of that checkpoint (`thread_id` + `checkpoint_id`). We use it in the next steps.

**Step 2: re-run from that checkpoint**
```Python
result = workflow.invoke(None, old.config)
print(result['joke'])   # a NEW pizza joke
```
- `None` = no new input, just continue from that checkpoint.
- What happens: LangGraph loads `{'topic': 'pizza'}` from that checkpoint and runs `generate_joke` → `generate_explanation` again.
- The old run is **not deleted**. The re-run is saved as a **new branch** next to it:
```
(empty) ─> topic=pizza ─┬─> joke 1 ─> explanation 1       (original run)
                        └─> joke 2 ─> explanation 2       (re-run)
```

**Step 3: change the state, then re-run** with `update_state()`
Example: change `topic` from `pizza` to `samosa` at that checkpoint.
```Python
new_config = workflow.update_state(old.config, {'topic': 'samosa'})

result = workflow.invoke(None, new_config)
print(result['topic'], '->', result['joke'])   # samosa -> a joke about samosa
```
- What happens: `update_state` does **not** edit the old checkpoint. It makes a **new** checkpoint with `topic = samosa` and returns its config (`new_config`).
- So you must re-run with **`new_config`**. If you re-run with `old.config`, you get a pizza joke again.

- Time travel is mostly a **debugging tool**. You won't need it much unless you build complex workflows.

---

## Cheat sheet

| What you want               | Code                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| Turn on persistence         | `workflow = graph.compile(checkpointer=InMemorySaver())`                                         |
| Run on a thread             | `workflow.invoke(state, {"configurable": {"thread_id": "1"}})`                                   |
| Latest state of a thread    | `workflow.get_state(config)` → `.values`, `.next`                                                |
| All checkpoints of a thread | `list(workflow.get_state_history(config))` (newest first)                                        |
| Resume after a crash        | `workflow.invoke(None, config)` (same `thread_id`)                                               |
| State at an old checkpoint  | `workflow.get_state({"configurable": {"thread_id": "1", "checkpoint_id": cid}})`                 |
| Time travel (replay)        | `workflow.invoke(None, {"configurable": {"thread_id": "1", "checkpoint_id": cid}})`              |
| Edit state, then replay     | `new_cfg = workflow.update_state(snap.config, {...})` → `workflow.invoke(None, new_cfg)`         |

**Remember:**
- Checkpointer = how persistence is built. **1 super-step = 1 checkpoint.** Intermediate values are saved too, not just the final one.
- `thread_id` = separates different runs/conversations in the same database.
- `invoke(None, ...)` = "continue / replay", not "start fresh".
- `InMemorySaver` = RAM only, for demos. Use a database checkpointer (SQLite/Postgres/Redis) in real projects.
