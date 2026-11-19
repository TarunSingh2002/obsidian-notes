---
tags:
  - AIEngineer
---

# Streaming in LangGraph

> **Streaming** = the model **sends tokens as soon as they are generated**, instead of waiting for the whole answer to be ready and sending it all at once.

## The problem
- You ask the chatbot: "write a 500 word blog on cricket".
- **Without streaming:** the screen stays **blank for 5-10 seconds**, then the whole 500 words appear **at once**.
- **With streaming:** the answer appears **token by token**, like someone is typing it (the "typewriter effect" you see in ChatGPT).

```
Without streaming:  [ wait ........ ] -> full answer at once
With streaming:     The -> The match -> The match was -> ...
```

## Why streaming is important
1. **Faster response (feels fast):** the first words show up almost immediately. Without it, a non-technical user sees a blank screen, thinks the app froze, and leaves.
2. **Feels human-like:** like talking to a person. It feels alive and keeps the user engaged.
3. **Needed for voice apps (multimodal UI):** an Alexa-type device without streaming would stay silent for 10 seconds, then start speaking. Very awkward.
4. **Easier to read long output (like code):** reading code line by line as it's written is easier than getting a big block all at once.
5. **You can stop it midway:** if you don't like the answer, stop it. The remaining tokens are never generated → **fewer tokens = less money** (LLM providers charge per token).
6. **Progress updates for agents:** not only for LLM text. An agent booking a movie ticket can show "opened BookMyShow → selected movie → selected seat → paying..." instead of 1 minute of silence. (Used later with agents/tools.)

---

## How to implement it
**The only change:** call `.stream()` instead of `.invoke()`.

| `invoke()`                    | `stream()`                                       |
| ----------------------------- | ------------------------------------------------ |
| waits, returns the final state | returns a **generator**: gives pieces one by one |

### Quick reminder: what is a generator?
A generator is a function that uses `yield` instead of `return`. It gives **one value at a time**, and you read the values with a `for` loop.
```Python
def count_to_3():
    yield 1
    yield 2
    yield 3

for number in count_to_3():
    print(number)   # 1, then 2, then 3
```
`.stream()` works the same way: you loop over it and get the LLM's answer piece by piece.

### 1. The chatbot graph (no change needed here)
A normal chatbot graph with one node. Nothing in the graph changes for streaming.
```Python
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=key)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = llm.invoke(state['messages'])
    return {'messages': [response]}


graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=InMemorySaver())
```
- Note: inside the node we still use `llm.invoke(...)`. You don't need to change it. When you call `chatbot.stream(...)` with `stream_mode='messages'`, LangGraph catches the LLM's tokens as they are generated and streams them to you.

### 2. Before: `invoke()` (no streaming)
```Python
config = {'configurable': {'thread_id': 'thread-1'}}

result = chatbot.invoke(
    {'messages': [HumanMessage(content='What is the recipe to make pasta?')]},
    config=config,
)
print(result['messages'][-1].content)   # full answer, all at once, after waiting
```

### 3. After: `stream()` (token by token)
`stream()` takes **3 things**:
1. the **input state** (same as invoke)
2. the **config** (`thread_id`, because we use a checkpointer)
3. **`stream_mode`** → `'messages'` for token-by-token LLM output

```Python
config = {'configurable': {'thread_id': 'thread-1'}}

stream = chatbot.stream(
    {'messages': [HumanMessage(content='What is the recipe to make pasta?')]},
    config=config,
    stream_mode='messages',
)
print(type(stream))   # <class 'generator'>

for message_chunk, metadata in stream:
    if message_chunk.content:
        print(message_chunk.content, end='', flush=True)
```
What happens:
- Every item from the stream is a pair: **`(message_chunk, metadata)`**.
  - `message_chunk` = a small piece of the AI message (a few characters/a word). The text is in `message_chunk.content`.
  - `metadata` = extra info, e.g. `metadata['langgraph_node']` tells which node it came from (`'chat_node'`).
- `if message_chunk.content:` skips empty chunks (some chunks carry no text).
- `end=''` → print the pieces next to each other, not on new lines.
- `flush=True` → show each piece on screen immediately (don't wait for Python's print buffer).

### 4. Keep the full answer too
While streaming, you usually also want the **complete answer** at the end (to save it, show it again later, etc.). Just join the pieces:
```Python
full_answer = ''

for message_chunk, metadata in chatbot.stream(
    {'messages': [HumanMessage(content='Write a 500 word blog on cricket')]},
    config=config,
    stream_mode='messages',
):
    if message_chunk.content:
        print(message_chunk.content, end='', flush=True)
        full_answer = full_answer + message_chunk.content

print()
print(len(full_answer))   # the whole blog is now in one string
```
- Because of the checkpointer, the full AI message is also saved in the thread's state, same as with `invoke()`. Streaming only changes **how you receive** the answer, not what gets saved.

---

## Stream modes
LangGraph has several stream modes. For now, you only need `messages`.

| `stream_mode` | What you get                                         | When to use                          |
| ------------- | ---------------------------------------------------- | ------------------------------------ |
| `'messages'`  | LLM tokens, one by one → `(message_chunk, metadata)` | **chatbots: typewriter effect**      |
| `'values'`    | the **full state** after every step                  | see the whole state as it changes    |
| `'updates'`   | only **what each node changed**, after every step    | progress updates (agents, tools)     |
| `'custom'`    | your own data, sent from inside a node               | custom progress messages             |

The other modes will be used later with agents and tools.

---

## Cheat sheet
| What you want              | Code                                                                       |
| -------------------------- | -------------------------------------------------------------------------- |
| Stream instead of invoke   | `chatbot.stream(input_state, config=config, stream_mode='messages')`       |
| Read the stream            | `for message_chunk, metadata in stream:`                                   |
| Print token by token       | `print(message_chunk.content, end='', flush=True)`                         |
| Skip empty chunks          | `if message_chunk.content:`                                                |
| Which node sent the chunk  | `metadata['langgraph_node']`                                               |

**Remember:**
- Streaming = get the answer **token by token** instead of all at once.
- The graph doesn't change. Only `.invoke()` → `.stream(..., stream_mode='messages')`.
- `.stream()` returns a **generator** → read it with a `for` loop.
- Each item = `(message_chunk, metadata)`. The text is `message_chunk.content`.
