---
tags:
  - python
  - uuid
---

# UUID (Universally Unique Identifier)

> A **UUID** is a **128-bit id** that any computer can create **on its own** (no database, no server, no coordination), and it will practically **never be the same** as any other UUID ever made.

```
3ed030aa-53f9-4559-ba48-ccd1b381d3f7
└──8───┘ └4─┘ └4─┘ └4─┘ └────12────┘   = 32 hex digits (+4 dashes = 36 characters)
              ↑
              the first digit here = the version (4 → uuid4)
```
- 32 hex digits × 4 bits = **128 bits**.

## Why do we need it?
- Normal ids are counters: `1, 2, 3...` or `thread-1, thread-2`. To make the next one, someone must **know the last one** → every server has to ask one central database → slow, and a problem with many servers.
- With UUIDs, **every server / phone / laptop just makes its own id**, and they still don't clash.
- Also: counter ids are **guessable** (`thread-5` → try `thread-6`). UUIDs are not.

## How can it never repeat? (uuid4)
`uuid4` = **122 random bits** (the other 6 bits are fixed: version + variant).
- The randomness comes from the operating system's secure random generator (`os.urandom`), on each machine separately.
- Number of possible uuid4s: **2^122 ≈ 5.3 × 10^36** (5.3 undecillion).
- It's not that a repeat is *impossible*. It's that the chance is so small it never happens in practice:

| How many UUIDs made in total                   | Chance that ANY 2 of them are the same |
| ---------------------------------------------- | -------------------------------------- |
| 100 million users × 1,000 ids each (10^11)     | ~0.000000000000001 (10^-15)            |
| 1 trillion (10^12)                             | ~0.0000000000001 (10^-13)              |
| 2.7 × 10^18 (1 billion per second for **86 years**) | 50%                               |

- So 100 instances, the same user on mobile + laptop at the same time, 100 million users at once: **no problem**. Each device rolls its own 122 random bits; they don't need to know about each other.
- Two `uuid4()` calls right after each other on the same machine → **different** ids (each call is a fresh random draw).
- Safety net anyway: in the database, the id is usually the `_id` / primary key, which is **unique**. In the (practically impossible) case of a clash, the insert fails instead of mixing data.

## UUID versions
| Version   | Made from                                | Use                                                   |
| --------- | ---------------------------------------- | ----------------------------------------------------- |
| `uuid1`   | current time + the machine's MAC address | old; leaks the machine and time                       |
| **`uuid4`** | **random**                             | **the default choice** for ids (chat id, run id, ...) |
| `uuid3` / `uuid5` | a namespace + a name (hash)      | **same input → always the same UUID** (deterministic) |
| `uuid7`   | current time + random                    | like uuid4 but **sorted by creation time** (good for database indexes). In Python 3.14+ |

## In Python
```Python
import uuid

new_id = uuid.uuid4()
print(new_id)             # 3ed030aa-53f9-4559-ba48-ccd1b381d3f7   (a UUID object)
print(type(new_id))       # <class 'uuid.UUID'>
print(new_id.version)     # 4

text_id = str(new_id)     # '3ed030aa-53f9-4559-ba48-ccd1b381d3f7'  -> store this in a DB / send in JSON
print(new_id.hex)         # '3ed030aa53f94559ba48ccd1b381d3f7'      -> without dashes

back = uuid.UUID(text_id) # string -> UUID object again (raises ValueError if the string is not a valid UUID)
```

Check if a string is a valid UUID:
```Python
def is_valid_uuid(text):
    try:
        uuid.UUID(text)
        return True
    except ValueError:
        return False

print(is_valid_uuid('3ed030aa-53f9-4559-ba48-ccd1b381d3f7'))   # True
print(is_valid_uuid('thread-1'))                               # False
```

Same input → same UUID (`uuid5`):
```Python
a = uuid.uuid5(uuid.NAMESPACE_DNS, 'tarun.com')
b = uuid.uuid5(uuid.NAMESPACE_DNS, 'tarun.com')
print(a == b)   # True  -> useful when the same thing must always get the same id
```

## Example: chatbot ids (LangGraph + LangSmith)
Two different UUIDs, for two different things:
```Python
import uuid

# when the user clicks "New chat" -> ONCE per chat
thread_id = str(uuid.uuid4())          # checkpoints / chat history (LangGraph)

# every time the user sends a message -> NEW every turn
run_id = uuid.uuid4()                  # this one trace in LangSmith (for feedback)

config = {
    'configurable': {'thread_id': thread_id},
    'run_id': run_id,
}
```
| id          | Made when            | Same or new?                        | Used for                              |
| ----------- | -------------------- | ----------------------------------- | ------------------------------------- |
| `thread_id` | a new chat starts    | **same** for every message of that chat | LangGraph checkpoints (memory/resume) |
| `run_id`    | every message (turn) | **new** every time                  | LangSmith trace → `create_feedback`   |

- Both come from `uuid4()`, made one after another → they are **different** (never the same).
- They must stay different: `thread_id` repeats across turns, `run_id` must be unique per turn.

## Remember
- UUID = 128-bit id, 36 characters, made **locally** without asking anyone.
- `uuid4()` = random → the default. Chance of a repeat is so small it's treated as zero.
- `str(uuid.uuid4())` to store/send it. `uuid.UUID(text)` to read/validate it.
- `uuid5(namespace, name)` → same input, same id. `uuid7` → time-sorted (Python 3.14+).
- Still keep the id column **unique** in the database as a safety net.
