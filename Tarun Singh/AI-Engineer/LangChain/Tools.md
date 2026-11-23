---
tags:
  - AIEngineer
  - langchain
  - tools
---

# Tools in LangChain

> A **tool** = a normal **Python function**, packaged so that an **LLM can understand it and call it** when needed.

## Why tools? (LLM = brain, no hands and legs)
An LLM can do only 2 things:
1. **Think** (reasoning): understand the question and work out how to answer it.
2. **Speak** (language generation): write the answer word by word.

That's it. It **can't do anything** in the real world:
- fetch **live data** (weather, news, stock price), because it only knows things up to its knowledge cutoff date
- do **maths reliably** (it learned language, not maths)
- call an **API**, send a tweet / email, **book** a train ticket
- **run code**, talk to a **database**

```
LLM   = a body with a brain (think + speak) but NO hands and legs
Tools = the hands and legs -> now it can DO tasks
```
- Example: "Best way from Delhi to Mumbai?" → the LLM answers (flight / train / bus). "Book the train for me" → it **can't**. Give it a "book ticket" tool → now it can.

### Tools and agents
- **AI agent** = an LLM-powered system that can **think, decide, and take actions** with external tools / APIs to reach a goal.
- Thinking and deciding come from the **LLM**. Taking actions comes from the **tools**.
```
Agent = LLM (reasoning) + Tools (actions)
```
- So to build agents (in LangChain or LangGraph), you must know tools well.
- The roadmap: **Tools** (this note) → **Tool calling** (connecting tools to the LLM) → **Agents**.

## 2 kinds of tools
| Kind        | What                                                       | Example                            |
| ----------- | ---------------------------------------------------------- | ---------------------------------- |
| **Built-in** | LangChain already made it: just import and use, no logic to write | DuckDuckGo search, Wikipedia, Shell |
| **Custom**  | **you** write it for your own use case                     | multiply, your company's booking API |

---

## 1. Built-in tools
### Popular Tools

| Tool                   | Description                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| `DuckDuckGoSearchRun`  | Performs web searches using the DuckDuckGo search engine.                                            |
| `WikipediaQueryRun`    | Searches Wikipedia and provides summaries of relevant articles.                                      |
| `PythonREPLTool`       | Executes Python code, such as calculating factorials and performing precise mathematical operations. |
| `ShellTool`            | Executes shell commands and terminal operations on the machine.                                      |
| `RequestsGetTool`      | Sends HTTP GET requests to retrieve data from web APIs and websites.                                 |
| `GmailSendMessage`     | Sends emails through a connected Gmail account.                                                      |
| `SlackSendMessage`     | Sends messages through a connected Slack workspace.                                                  |
| `SQLDatabaseQueryTool` | Executes SQL queries to retrieve and manipulate data in a database.                                  |

- Full list (search, code, productivity like GitHub/Jira/Office, web browsing, databases, finance...): LangChain docs → **Integrations → Tools**. Every tool page has a description + working code.

### DuckDuckGo search
```Python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('ipl news')
print(results)
```
- `pip install langchain-community ddgs` (older versions use the `duckduckgo-search` package).
- **Tools are Runnables**: they have `.invoke()`, just like models, prompts and chains.
- The query goes to the DuckDuckGo search engine, and the search results come back. Useful because the LLM doesn't know today's news.

### Shell tool
```Python
from langchain_community.tools import ShellTool

shell_tool = ShellTool()

results = shell_tool.invoke('whoami')
print(results)   # root   (on Colab)

results = shell_tool.invoke('ls')
print(results)   # sample_data
```
- Also needs `pip install langchain-experimental`.
- ⚠️ **Risky**: it runs real commands on your machine, so it could delete files. Be careful, especially in production.

---

## 2. Custom tools
**When do you need one?** When no built-in tool exists for your case, mostly when you build an agent for **your own app**:
- call **your own API** (e.g. a MakeMyTrip-like app where the agent books flights/hotels through your API)
- wrap your own **business logic**
- let the LLM work with **your own database / product / app**

### 3 Ways to Create a Custom Tool in LangChain

| Method                              | How It Works                                                                                                   | When to Use                                                                                       |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `**@tool**` **Decorator**           | Add the `@tool` decorator to a Python function to convert it into a custom tool.                               | **Simplest approach; suitable for most use cases (around 80–90%).**                               |
| `**StructuredTool**` **+ Pydantic** | Use `StructuredTool.from_function(...)` with a Pydantic input schema to define and validate structured inputs. | When you need strict input validation and production-ready tools.                                 |
| `**BaseTool**` **Class**            | Create a custom class that inherits from `BaseTool` and implement its required methods.                        | When you need full control over tool behavior, including custom logic and asynchronous execution. |

### Way 1: `@tool` decorator (3 steps)
```Python
from langchain_core.tools import tool

# Step 1: a normal function + a docstring
# Step 2: type hints (a: int, b: int -> int)
# Step 3: put @tool on top
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

result = multiply.invoke({'a': 3, 'b': 5})   # inputs are passed as a dict
print(result)   # 15
```
1. **Docstring**: not compulsory, but **highly recommended**. The LLM reads it to understand **what the tool does**.
2. **Type hints**: also not compulsory, but recommended. The LLM learns **what type of input to give** and what it gets back.
3. **`@tool`**: this is the magic. It turns the function into a **tool** that an LLM can talk to.

### Every tool has `name`, `description`, `args`
```Python
print(multiply.name)          # multiply              <- the function name
print(multiply.description)   # Multiply two numbers  <- the docstring
print(multiply.args)          # {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}  <- from the type hints
```
- Built-in tools have them too:
```Python
print(search_tool.name)          # duckduckgo_search
print(search_tool.description)   # A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
print(search_tool.args)          # {'query': {...}}
```

### What the LLM actually sees: the schema
The LLM **never sees your function's code**. When you connect a tool to an LLM, only the tool's **schema** (JSON) is sent:
```Python
print(multiply.args_schema.model_json_schema())
```
```json
{
  "title": "multiply",
  "description": "Multiply two numbers",
  "type": "object",
  "properties": {
    "a": {"title": "A", "type": "integer"},
    "b": {"title": "B", "type": "integer"}
  },
  "required": ["a", "b"]
}
```
- Name + description + inputs (names, types, required). From this the LLM decides **when** to use the tool and **what input** to give.
- That's why the docstring and type hints matter.

### Way 2: `StructuredTool` + Pydantic (stricter)
- A **StructuredTool** = a tool whose input follows a **structured schema**, usually a **Pydantic model**.
- With `@tool`, type hints are a loose way to describe the inputs. A Pydantic model **enforces** them strictly and lets you describe each field.
```Python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# the input schema
class MultiplyInput(BaseModel):
    a: int = Field(description='The first number to multiply')
    b: int = Field(description='The second number to multiply')

# a normal function (no @tool, type hints optional now)
def multiply_func(a: int, b: int) -> int:
    return a * b

# make the tool: function + name + description + input schema, all given here
multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name='multiply',
    description='Multiply two numbers',
    args_schema=MultiplyInput,
)

result = multiply_tool.invoke({'a': 3, 'b': 3})
print(result)              # 9
print(multiply_tool.args)  # now each field also has its 'description'
```
- In the lecture: `Field(required=True, description=...)`. In Pydantic v2, `required=True` is deprecated. A field with **no default is already required**, so just write `Field(description=...)`.
- It's used the same way as Way 1: `.invoke`, `.name`, `.description`, `.args`.

### Way 3: `BaseTool` class (most control)
- **`BaseTool`** = the **abstract base class of ALL tools** in LangChain. It defines how every tool must behave.
- `@tool` tools, `StructuredTool`s and built-in tools are all **built on top of `BaseTool`**.
- So you can also inherit from `BaseTool` yourself:
```Python
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description='The first number to multiply')
    b: int = Field(description='The second number to multiply')

class MultiplyTool(BaseTool):
    name: str = 'multiply'
    description: str = 'Multiply two numbers'
    args_schema: Type[BaseModel] = MultiplyInput    # Type[BaseModel] -> "a Pydantic class"

    def _run(self, a: int, b: int) -> int:          # the name MUST be _run
        return a * b

multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a': 3, 'b': 3})
print(result)   # 9
```
- `name`, `description`, `args_schema` are class attributes. The tool's logic goes in **`_run`** (exactly this name).

#### Sync + async in ONE tool (`_run` + `_arun`)
```Python
import asyncio
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description='The first number to multiply')
    b: int = Field(description='The second number to multiply')

class MultiplyTool(BaseTool):
    name: str = 'multiply'
    description: str = 'Multiply two numbers'
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:            # SYNC version  -> used by .invoke()
        return a * b

    async def _arun(self, a: int, b: int) -> int:     # ASYNC version -> used by .ainvoke()  (name MUST be _arun)
        await asyncio.sleep(1)                        # pretend: async API / DB call
        return a * b

multiply_tool = MultiplyTool()

# 1) hit the SYNC version
print(multiply_tool.invoke({'a': 3, 'b': 3}))         # 9  -> runs _run


# 2) hit the ASYNC version
async def main():
    result = await multiply_tool.ainvoke({'a': 3, 'b': 3})      # runs _arun
    print(result)                                               # 9

    # real benefit: many calls at the SAME time (total ~1 s, not 3 s)
    results = await asyncio.gather(
        multiply_tool.ainvoke({'a': 2, 'b': 3}),
        multiply_tool.ainvoke({'a': 4, 'b': 5}),
        multiply_tool.ainvoke({'a': 6, 'b': 7}),
    )
    print(results)                                              # [6, 20, 42]

asyncio.run(main())        # in a .py script
# await main()             # in Jupyter (an event loop is already running there)
```

| You call                  | Runs    | When                                      |
| ------------------------- | ------- | ----------------------------------------- |
| `tool.invoke(...)`        | `_run`  | Normal scripts, simple chains             |
| `await tool.ainvoke(...)` | `_arun` | FastAPI, async LangGraph, parallel calls  |


#### ⭐ Why `BaseTool` over `@tool` / `StructuredTool`
- Async is **not** only a `BaseTool` thing:
  - `@tool` on an `async def` works, but then you have the async version only.
  - `StructuredTool.from_function(func=multiply_func, coroutine=amultiply_func, ...)` gives both, as 2 loose functions.

|                                                                  | `@tool`          | `StructuredTool`     | **`BaseTool` class**                                 |
| ---------------------------------------------------------------- | ---------------- | -------------------- | ---------------------------------------------------- |
| Input validation                                                 | type hints       | Pydantic ✅           | Pydantic ✅                                           |
| Sync + async                                                     | one or the other | 2 separate functions | ✅ **both in one class** (`_run` + `_arun`)           |
| Own state / config (API key, client, base URL) | ❌ (globals)      | ❌ (globals)          | ✅ **class fields**: `MyTool(api_key=...)` |
| Reuse / extend                                                   | copy-paste       | copy-paste           | ✅ inheritance (one base class, many variants)        |
| Custom error handling, callbacks, `return_direct`, ...           | limited          | via args             | ✅ full control                                       |
| Code needed                                                      | least            | medium               | most                                                 |
- In one line: **`BaseTool` = a tool as a proper object**. It keeps its own settings (fields), sync and async logic and error handling together. That's what you want for production tools such as API, DB or search clients.

#### Example 1: simple API call (weather)
- The API key is a **field** on the tool. It's NOT in `args_schema`, so the LLM never sees it. The LLM only fills `city`.
```Python
import requests
import httpx
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CityInput(BaseModel):
    city: str = Field(description='City name')

class WeatherTool(BaseTool):
    name: str = 'check_weather'
    description: str = 'Get the current weather of a city'
    args_schema: Type[BaseModel] = CityInput
    api_key: str                                   # set once when creating the tool

    def _run(self, city: str) -> str:
        r = requests.get('https://api.weather.com/current', params={'city': city, 'key': self.api_key})
        if r.status_code != 200:
            return f'Could not get weather for {city}'          # simple custom error message
        return r.text

    async def _arun(self, city: str) -> str:
        async with httpx.AsyncClient() as client:
            r = await client.get('https://api.weather.com/current', params={'city': city, 'key': self.api_key})
        if r.status_code != 200:
            return f'Could not get weather for {city}'
        return r.text


weather = WeatherTool(api_key='my-secret-key')
weather.invoke({'city': 'Delhi'})              # sync  → _run
# await weather.ainvoke({'city': 'Delhi'})     # async → _arun
```

#### Example 2: save a user name to the DB
- The collection is already made in `database.py`, just import it.
```Python
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from database import users_collection, async_users_collection   # sync + async Mongo collections

class UserInput(BaseModel):
    name: str = Field(description='Name of the user to save')

class SaveUserTool(BaseTool):
    name: str = 'save_user'
    description: str = 'Save a user name in the database'
    args_schema: Type[BaseModel] = UserInput

    def _run(self, name: str) -> str:
        users_collection.insert_one({'name': name})
        return f'Saved user: {name}'

    async def _arun(self, name: str) -> str:
        await async_users_collection.insert_one({'name': name})
        return f'Saved user: {name}'


save_user = SaveUserTool()
save_user.invoke({'name': 'Tarun'})            # sync  → _run
# await save_user.ainvoke({'name': 'Tarun'})   # async → _arun
```
- Inside `_arun` use **async libraries** (`httpx`, an async Mongo client), not `requests` / normal `MongoClient`. Otherwise it blocks and async gives no benefit.

**Which way to use?** Start with `@tool`. Use `StructuredTool` for strict inputs. Use `BaseTool` when a tool needs config or state, both sync and async, or reuse through inheritance (production).

---

## 3. Toolkits
- **Toolkit** = a collection of **related tools** that serve a common purpose, packed together for **convenience and reusability**.
- Example: a Google Drive toolkit = upload file + search files + read file.
- LangChain has **built-in toolkits**, and you can make your own.
- Main benefit: **reusability**. Make it once and use it in many apps.

```Python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

class MathToolkit:
    def get_tools(self):
        return [add, multiply]

toolkit = MathToolkit()
tools = toolkit.get_tools()

for one_tool in tools:
    print(one_tool.name, '=>', one_tool.description)
# add => Add two numbers
# multiply => Multiply two numbers
```
- The toolkit is just a class with a **`get_tools()`** method that returns the list of tools.

---

## Cheat sheet
| What you want              | Code                                                              |
| -------------------------- | ----------------------------------------------------------------- |
| Web search tool            | `DuckDuckGoSearchRun()` (`langchain_community.tools`)             |
| Shell tool (careful!)      | `ShellTool()` (+ `pip install langchain-experimental`)            |
| Run any tool               | `tool.invoke(...)` (string for built-ins, dict for custom tools)  |
| Simple custom tool         | `@tool` + docstring + type hints                                  |
| Strict inputs              | `StructuredTool.from_function(func=..., name=..., description=..., args_schema=PydanticModel)` |
| Full control / async       | `class X(BaseTool)` with `name`, `description`, `args_schema`, `_run` (and `_arun`) |
| Call sync / async          | `tool.invoke({...})` → `_run` · `await tool.ainvoke({...})` → `_arun` |
| See tool info              | `.name`, `.description`, `.args`                                  |
| What the LLM sees          | `tool.args_schema.model_json_schema()`                            |
| Group related tools        | a toolkit class with `get_tools()` returning a list               |

**Remember:**
- LLM = think + speak. Tools = do. Agent = LLM + tools.
- A tool = a Python function packaged so the LLM can understand and call it.
- Tools are **Runnables** → `.invoke()`.
- Built-in = import and use. Custom = `@tool` (most of the time), `StructuredTool`, or `BaseTool`.
- **Docstring + type hints** matter: the LLM only sees the **schema** (name, description, args), never your code.
- All tools inherit from `BaseTool`.
- This note only makes tools. **Connecting them to the LLM = tool calling** (next note).
