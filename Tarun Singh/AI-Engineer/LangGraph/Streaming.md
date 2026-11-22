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

### 1. The graph (nothing changes here for streaming)
A simple graph: give a `question` → one node asks the LLM → saves the `answer`.
```Python
from typing import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=key,
    max_tokens=900,  # free tier: 1000 output tokens/min; default asks for 2048
)

class ChatState(TypedDict):
    question: str
    answer: str

def chat_with_ai(state: ChatState):
    response = model.invoke(state['question'])
    return {'answer': response.content}   # .content -> save only the text, not the whole AIMessage object

graph = StateGraph(ChatState)
graph.add_node('chat_with_ai', chat_with_ai)

graph.add_edge(START, 'chat_with_ai')
graph.add_edge('chat_with_ai', END)

chatbot = graph.compile(checkpointer=InMemorySaver())
```
- **Inside the node we still call `model.invoke(...)`, not `model.stream(...)`, and streaming still works.** When you run the graph with `chatbot.stream(..., stream_mode='messages')`, LangGraph quietly listens to the LLM while it runs and catches every token. You don't change the node at all.

### 2. Before: `invoke()` (no streaming)
```Python
config1 = {"configurable": {"thread_id": "id1"}}

result = chatbot.invoke(
    {'question': 'hello who is president of india, and how they are and what they do'},
    config=config1,
)
print(result)   # {'question': '...', 'answer': '...'}  -> full state, all at once, after waiting
```

### 3. After: `stream()` (token by token)
`stream()` takes **3 things**:
1. the **input state** (same as invoke)
2. the **config** (`thread_id`, because we use a checkpointer)
3. **`stream_mode`** → `'messages'` for token-by-token LLM output

```Python
config1 = {"configurable": {"thread_id": "id1"}}

stream = chatbot.stream(
    {'question': 'hello who is president of india, and how they are and what they do'},
    config=config1,
    stream_mode='messages',
)
print(type(stream))   # <class 'generator'>

for chunk, metadata in stream:
    print(chunk.content, end='', flush=True)
```
- Every item from the stream is a pair: **`(chunk, metadata)`**.
  - `chunk` = a small piece of the LLM's answer (a word or a few characters). The text is in `chunk.content`.
  - `metadata` = extra info, e.g. `metadata['langgraph_node']` tells which node it came from (`'chat_with_ai'`).
- `end=''` → print the pieces next to each other, not on new lines.
- `flush=True` → show each piece on screen immediately (don't wait for Python's print buffer).

### `stream()` is NOT like `invoke()`: nothing runs until you loop
```Python
stream = chatbot.stream({...}, config=config1, stream_mode='messages')
```
- After this line, `stream` is **not** your result dict. It's a **generator**, and **nothing has run yet**: no LLM call, no tokens.
- The graph only starts when you **loop over it**. Each loop round gives you **one small piece** of the answer.
- If you never loop, it just sits there forever and does nothing.
- After the loop ends, the generator is **used up**. Looping over it again gives nothing.

### How it works inside
```
your for loop starts
   -> LangGraph starts the graph -> chat_with_ai node -> model.invoke(question)
   -> Groq server generates tokens and SENDS each one over the internet as soon as it's made
   -> LangGraph catches each token and puts it in a queue
   -> your loop takes the next token from the queue -> prints it -> takes the next one ...
   -> LLM done -> node returns {'answer': ...} -> checkpoint saved -> loop ends
```
- Once the loop has started, the LLM does **not** wait for your loop token by token. The Groq server generates at its own speed. Tokens pile up in the queue, and your loop just **reads** them one by one.
- **`break` the loop = stop the LLM.** If you `break` out early, LangGraph stops the graph and the remaining tokens are never generated. That's the "stop midway, save tokens = save money" benefit.
```Python
stream = chatbot.stream(
    {'question': 'write a 500 word blog on cricket'},
    config=config1,
    stream_mode='messages',
)

count = 0
for chunk, metadata in stream:
    print(chunk.content, end='', flush=True)
    count = count + 1
    if count == 20:
        break   # stop after 20 chunks -> the LLM stops too
```

### Why do I only get the answer, not the question?
- `stream_mode='messages'` means **"give me only what the LLM generates"**.
- The `question` (and any other key in the state that the LLM didn't write) is **not LLM output**, so it **never comes through this stream**.
- This is not a bug, that's just what this mode does.
- To get the **full state** (question + answer) after streaming, read it from the **checkpoint**:
```Python
config1 = {"configurable": {"thread_id": "id1"}}

stream = chatbot.stream(
    {'question': 'hello who is president of india, and how they are and what they do'},
    config=config1,
    stream_mode='messages',
)

for chunk, metadata in stream:
    print(chunk.content, end='', flush=True)

print()

final = chatbot.get_state(config1).values   # same dict that invoke() would have returned
print(final['question'])
print(final['answer'])
```

### Streaming works with or without a checkpointer
- Streaming does **not** need a checkpointer. It works both ways.
- **With a checkpointer:** the state is saved when the **node finishes** (not after every token). After the loop, `get_state(config)` gives you the full state, same as `invoke()`.
- **Without a checkpointer** (`graph.compile()` with no `checkpointer`, and no `config` needed): streaming works exactly the same, but nothing is saved, so there's no `get_state()`. If you want the full answer at the end, join the pieces yourself:
```Python
chatbot_no_memory = graph.compile()

stream = chatbot_no_memory.stream(
    {'question': 'indias capital??'},
    stream_mode='messages',
)

full_answer = ''
for chunk, metadata in stream:
    print(chunk.content, end='', flush=True)
    full_answer = full_answer + chunk.content

print()
print(full_answer)   # the whole answer in one string
```

---

## Stream modes
LangGraph has several stream modes. For now, you only need `messages`.

| `stream_mode` | What you get                                         | When to use                          |
| ------------- | ---------------------------------------------------- | ------------------------------------ |
| `'messages'`  | LLM tokens, one by one → `(chunk, metadata)`         | **chatbots: typewriter effect**      |
| `'values'`    | the **full state** after every step                  | see the whole state as it changes    |
| `'updates'`   | only **what each node changed**, after every step    | progress updates (agents, tools)     |
| `'custom'`    | your own data, sent from inside a node               | custom progress messages             |

The other modes will be used later with agents and tools.

---

## Cheat sheet
| What you want              | Code                                                                       |
| -------------------------- | -------------------------------------------------------------------------- |
| Stream instead of invoke   | `chatbot.stream(input_state, config=config, stream_mode='messages')`       |
| Read the stream            | `for chunk, metadata in stream:`                                           |
| Print token by token       | `print(chunk.content, end='', flush=True)`                                 |
| Which node sent the chunk  | `metadata['langgraph_node']`                                               |
| Full state after streaming | `chatbot.get_state(config).values` (needs a checkpointer)                  |
| Stop the LLM midway        | `break` out of the loop                                                    |

**Remember:**
- Streaming = get the answer **token by token** instead of all at once.
- The graph doesn't change. Only `.invoke()` → `.stream(..., stream_mode='messages')`.
- `.stream()` returns a **generator** → read it with a `for` loop.
- Each item = `(chunk, metadata)`. The text is `chunk.content`.
- Nothing runs until you loop. `break` the loop = the LLM stops too.
- `model.invoke()` inside the node still streams. No change in the node.
- `'messages'` gives only LLM output → the question/other state keys don't come → get them with `get_state()`.
- Works with or without a checkpointer.
