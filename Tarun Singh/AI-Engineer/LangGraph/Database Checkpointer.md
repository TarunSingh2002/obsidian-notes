---
tags:
  - AIEngineer
---

# Database Checkpointer (MongoDB)

> Swap `InMemorySaver` (RAM) for a **database checkpointer** (`MongoDBSaver`), so chats are **saved permanently** and survive closing the program.

## The problem
- `InMemorySaver` keeps all checkpoints (all messages) in **RAM**.
- Close the program or refresh the page → **everything is gone**.
- **Fix:** a checkpointer that saves into a **database**. Close the app, come back 4 days later, continue the chat from exactly where you left it.

| Checkpointer    | Saves in         | Use it for                                  |
| --------------- | ---------------- | ------------------------------------------- |
| `InMemorySaver` | RAM              | learning / demos (lost when program ends)   |
| `MongoDBSaver`  | MongoDB database | **real apps** (this note)                   |
| (`PostgresSaver`) | PostgreSQL     | same idea, if your app uses Postgres        |

Only **where** the checkpoints are stored changes. Graph, `invoke`, `thread_id`, `get_state` all stay the same.

---

## 1. InMemorySaver (for comparison)
```Python
from langgraph.checkpoint.memory import InMemorySaver

chatbot = graph.compile(checkpointer=InMemorySaver())   # RAM -> gone when the program stops
```

## 2. MongoDBSaver
### Install
```
pip install langgraph-checkpoint-mongodb
```
- Also installs `pymongo`. You need a MongoDB: local (`mongodb://localhost:27017`) or a MongoDB Atlas cluster (`mongodb+srv://...`).
- Keep the connection string (it has your password) in `.env`, not in the code.

### Code
```Python
from typing import TypedDict, Annotated
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver

model = ChatGroq(model="qwen/qwen3.8-27b", api_key=key, max_tokens=900)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]   # new messages get ADDED to the old ones

def chat_node(state: ChatState):
    response = model.invoke(state['messages'])
    return {'messages': [response]}

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# THE ONLY CHANGE vs InMemorySaver: connect to MongoDB, give the client to MongoDBSaver
client = MongoClient(mongodb_url)
checkpointer = MongoDBSaver(client, db_name='chatbot_app')

chatbot = graph.compile(checkpointer=checkpointer)
```
- `MongoClient(mongodb_url)` → connection to your MongoDB server.
- `MongoDBSaver(client, db_name='chatbot_app')` → saves every checkpoint into the database `chatbot_app`. MongoDB creates the database on the first save.
- It creates **2 collections** inside it: **`checkpoints`** and **`checkpoint_writes`** (names can be changed with `checkpoint_collection_name=` / `writes_collection_name=`).

### Proof it's permanent
```Python
config = {'configurable': {'thread_id': 'thread-1'}}

chatbot.invoke({'messages': [HumanMessage(content='hi my name is tarun')]}, config=config)
```
Stop the program. Start it again:
```Python
config = {'configurable': {'thread_id': 'thread-1'}}

result = chatbot.invoke({'messages': [HumanMessage(content='what is my name?')]}, config=config)

for message in result['messages']:
    print(message.content)
# hi my name is tarun          <- loaded from MongoDB
# Hello Tarun! ...             <- loaded from MongoDB
# what is my name?
# Your name is Tarun.
```

---

## 3. How it works: who owns these 2 collections?
- `checkpoints` and `checkpoint_writes` are **LangGraph's internal storage**. The checkpointer writes them on its own on every step. **You never insert / edit / delete documents in them by hand.**
  - The state is stored as a **binary blob** (`msgpack`), not readable JSON. That's why Compass shows `Binary.createFromBase64(...)`.
  - Checkpoints are **linked to each other** by `parent_checkpoint_id`. Editing one by hand can break the chain.
- You read / change them **only through LangGraph's API**:

| You want to                      | Use                                                         |
| -------------------------------- | ----------------------------------------------------------- |
| Latest state of a chat           | `chatbot.get_state(config)`                                 |
| All checkpoints of a chat        | `chatbot.get_state_history(config)`                         |
| Change the state                 | `chatbot.update_state(config, {...})` (makes a **new** checkpoint, never edits an old one) |
| Delete a whole chat              | `checkpointer.delete_thread(thread_id)` (deletes from both collections) |
| Raw checkpoints (low level)      | `checkpointer.list(config)`                                 |

- **Your own app data** (users, chat titles, folders...) goes in **your own collections** in the same database. See section 5.

### `checkpoints` collection: one document = one saved state
| Field                  | Meaning                                                                       |
| ---------------------- | ----------------------------------------------------------------------------- |
| `thread_id`            | which chat (the `thread_id` you passed in the config)                         |
| `checkpoint_ns`        | namespace, `""` for the main graph (only used by subgraphs)                   |
| `checkpoint_id`        | id of this checkpoint. **Time-ordered** (newer id = later), so sorting by it = sorting by time |
| `parent_checkpoint_id` | the checkpoint just before this one (`null` for the first one) → makes a chain |
| `checkpoint`           | the **full state** (all channels: `messages`, ...) as a binary blob            |
| `type`                 | how the blob is encoded: `"msgpack"`                                          |
| `metadata.source`      | how it was made: `input` (the input arrived), `loop` (a step ran), `update` (`update_state`), `fork` (time travel) |
| `metadata.step`        | step number: `-1`, `0`, `1`, ...                                              |
| `metadata.parents`     | parent graphs (empty unless subgraphs)                                        |

### `checkpoint_writes` collection: what each node returned
| Field           | Meaning                                                                      |
| --------------- | ---------------------------------------------------------------------------- |
| `thread_id`, `checkpoint_ns`, `checkpoint_id` | which checkpoint these writes belong to (they happened **after** it) |
| `task_id`       | id of the node run that produced the write                                   |
| `task_path`     | which node: e.g. `~__pregel_pull, get_human_message`                         |
| `channel`       | which state key was written (`messages`, `humen_message`), or `branch:to:<node>` = "go to this node next" |
| `idx`           | order of the writes inside the same task (0, 1, ...)                         |
| `value`, `type` | the written value (binary) and its encoding                                  |

- Why a separate writes collection? **Fault tolerance.** As soon as a node finishes, its output is saved here, **before** the next checkpoint is made. If something crashes in between (or 1 of 3 parallel nodes fails), the finished nodes' outputs are not lost and they don't run again.
- At the next checkpoint, these writes are **applied** to the state, and the new full state is saved as a new document in `checkpoints`.

---



## 6. Good to know
- **Storage grows.** Every step saves a new checkpoint with the **full state** (all messages so far), and nothing is deleted automatically. Long chats → many big documents.
  - Delete finished/unused chats with `checkpointer.delete_thread(thread_id)`.
  - Or auto-delete old checkpoints: `MongoDBSaver(client, db_name='chatbot_app', ttl=60*60*24*30)` → MongoDB removes checkpoints 30 days after they were saved (TTL index).
- **Indexes:** `MongoDBSaver` creates its own index on (`thread_id`, `checkpoint_ns`, `checkpoint_id`), so loading a chat is fast. Indexes on **your** collections are your job (see Setup).
- **Very large scale:** MongoDB can split (shard) collections across servers. A common choice: `threads` by `user_id`, `checkpoints` by `thread_id`, so one chat's data stays together.
- **Learning only:** `checkpointer.list(None)` gives every checkpoint in the database, and `checkpoint.config['configurable']['thread_id']` tells its chat. Fine for exploring, not for a real sidebar.

---

## Cheat sheet
| What you want                 | Code                                                            |
| ----------------------------- | --------------------------------------------------------------- |
| Install                       | `pip install langgraph-checkpoint-mongodb`                      |
| Checkpointer                  | `MongoDBSaver(MongoClient(mongodb_url), db_name='chatbot_app')` |
| Compile                       | `chatbot = graph.compile(checkpointer=checkpointer)`            |
| New chat id                   | `str(uuid.uuid4())`                                             |
| Continue a chat               | `chatbot.invoke({'messages': [...]}, {'configurable': {'thread_id': tid}})` |
| Read a chat's messages        | `chatbot.get_state(config).values['messages']`                  |
| Delete a chat (LangGraph data) | `checkpointer.delete_thread(tid)`                              |
| Auto-delete old checkpoints   | `MongoDBSaver(..., ttl=seconds)`                                |
| User's chats / folders        | your own `threads` / `folders` collections                      |

**Remember:**
- `InMemorySaver` = RAM = lost. `MongoDBSaver` = saved permanently in `checkpoints` + `checkpoint_writes`.
- Those 2 collections belong to **LangGraph**: never edit them by hand, use `get_state` / `update_state` / `delete_thread`.
- Checkpoints: **1 for the input + 1 after every step**, chained by `parent_checkpoint_id`. Writes = each node's output, saved right after the node finishes.
- LangGraph knows only `thread_id`. **Users, titles, folders = your own collections**, linked by `threads._id == thread_id`.
- `thread_id` = `uuid4()`. Always check the chat belongs to the user before using its `thread_id`.
- Resume = same `thread_id` in `invoke`. Nothing else.
