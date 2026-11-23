---
tags:
  - AIEngineer
---

# Tools in LangGraph (Tool Calling)

> Give the chatbot **tools** (calculator, internet search, stock price...) so it can **do things**, not just talk. A normal question gets a normal reply. A task makes the LLM **call a tool**.

---

## 1 new things: `ToolNode` and `tools_condition`
Both come ready-made from LangGraph:
```Python
from langgraph.prebuilt import ToolNode, tools_condition
```

| Thing             | What it is                     | Job                                                                 |
| ----------------- | ------------------------------ | ------------------------------------------------------------------- |
| `ToolNode(tools)` | a **prebuilt node**            | holds all your tools. Reads the LLM's tool call (tool name + input), **runs that tool**, and returns the result as a `ToolMessage` |
| `tools_condition` | a **prebuilt conditional edge** function | looks at the LLM's last message: has **tool calls** → go to `'tools'`, no tool calls → go to `END` |

- Normally you write every node function yourself. `ToolNode` is a node LangGraph **already wrote** for you: the "tool executor".
- `tools_condition` is the "should I chat or call a tool?" decision. You don't write the if/else yourself.

---

## 1. Make the tools
Two kinds of tools:
- **Prebuilt:** LangChain already made it → e.g. `DuckDuckGoSearchRun` (internet search with the DuckDuckGo search engine).
- **Custom:** you write a normal Python function and put **`@tool`** on it. Or use other ways to make custom tools.

```Python
import requests
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Tool 1: prebuilt -> internet search
search_tool = DuckDuckGoSearchRun(region='us-en')

# Tool 2: custom -> calculator
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    if operation == 'add':
        result = first_num + second_num
    elif operation == 'sub':
        result = first_num - second_num
    elif operation == 'mul':
        result = first_num * second_num
    elif operation == 'div':
        if second_num == 0:
            return {'error': 'Division by zero is not allowed'}
        result = first_num / second_num
    else:
        return {'error': f'Unsupported operation: {operation}'}

    return {'first_num': first_num, 'second_num': second_num, 'operation': operation, 'result': result}

# Tool 3: custom -> stock price (Alpha Vantage API, make your own free key)
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using the Alpha Vantage API.
    """
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_key}'
    response = requests.get(url)
    return response.json()
```
- `pip install langchain-community ddgs` for the search tool (older versions used the `duckduckgo-search` package).
- **The docstring is a must.** The LLM **reads the docstring** (and the argument names/types) to decide **which tool to use** and **what input to give**. No good docstring → wrong tool / wrong inputs.

## 2. Give the tools to the LLM
```Python
from langchain_groq import ChatGroq

model = ChatGroq(model="qwen/qwen3.8-27b", api_key=key, max_tokens=900)

tools = [search_tool, calculator, get_stock_price]

llm_with_tools = model.bind_tools(tools)   # the LLM now KNOWS these 3 tools exist
```
- `bind_tools` does **not** run any tool. It only tells the LLM "these tools exist, you may ask for them".
- When a tool is needed, the LLM's reply has **no answer text**. It has `tool_calls` instead: "please run `get_stock_price` with `{'symbol': 'AAPL'}`". Running it is `ToolNode`'s job.

## 3. State + chat node (same as the chatbot)
```Python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state['messages'])   # llm_with_tools, NOT model
    return {'messages': [response]}
```
- Only change vs the normal chatbot: call **`llm_with_tools`** instead of `model`.

---

## 4. The graph: version 1 (has a problem)
```
START -> chat_node --(tools_condition)--> tools -> END
                   \--------------------> END
```

```Python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)          # the name MUST be 'tools' -> tools_condition sends to 'tools'

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)   # -> 'tools' or END
graph.add_edge('tools', END)

chatbot = graph.compile()
```
- `"hello"` → `Hi there! How can I assist you?` ✅ (no tool, went to END)
- `"what is 2 x 3?"` → `{"first_num": 2, "second_num": 3, "operation": "mul", "result": 6}` → the **raw tool output** is shown to the user.

### 2 problems with version 1
1. **Ugly output.** The tool's raw output (JSON) goes straight to the user. The stock tool gives a huge JSON. The user wants: "The current stock price of Apple is $X."
2. **No multi-step.** "What is Apple's stock price, and how much do 50 shares cost?" needs **2 tools one after another**: first the stock price, **then** the calculator using that price. But after one tool run, the graph goes to `END`. The LLM never sees the tool's result, so it can't decide the next step (it gave weird output like `50 x 0 = 0`).

**Fix:** send the tool's output **back to the LLM**, not to `END`.

## 5. The graph: version 2 (correct): a loop between the LLM and the tools
```
         ┌──────────────────────┐
         v                      │
START -> chat_node --(tool calls?)--> tools
                   \
                    \--(no tool calls)--> END
```
**The only change:** `tools -> chat_node` instead of `tools -> END`.
```Python
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')        # <- tool result goes BACK to the LLM

chatbot = graph.compile()
```
- Now the LLM sees the tool's result and either **writes a nice answer** (→ `END`), or **calls another tool** (→ `tools` again). It loops until no more tools are needed.
- `"what is 2 x 3?"` → `The result of 2 × 3 is 6.` ✅
- Multi-step works too.
### Run it
```Python
result = chatbot.invoke({'messages': [HumanMessage(content='What is the stock price of Apple? How much will 50 shares cost?')]})

print(result)   # the final answer
```

### How it works inside (Apple stock + 50 shares)
Everything is stored in `state['messages']`, so the LLM always sees the **full history**:
```
1. HumanMessage   "Apple stock price? cost of 50 shares?"
   chat_node -> LLM decides: need the stock price first
2. AIMessage      tool_calls = get_stock_price(symbol='AAPL')      -> tools_condition: 'tools'
   tools (ToolNode) runs get_stock_price
3. ToolMessage    {... "price": "230.10" ...}                       -> back to chat_node
   chat_node -> LLM sees the price, now needs the maths
4. AIMessage      tool_calls = calculator(50, 230.10, 'mul')         -> tools_condition: 'tools'
   tools runs calculator
5. ToolMessage    {"result": 11505.0}                                -> back to chat_node
   chat_node -> LLM has everything
6. AIMessage      "Apple is $230.10. 50 shares cost $11,505."       -> no tool_calls -> END
```
- 3 message types: `HumanMessage` (user), `AIMessage` (LLM, either text or `tool_calls`), `ToolMessage` (a tool's result).
- In **LangSmith** you can see exactly this: chat_node → tools_condition says `tools` → the tool → chat_node → tools_condition says `__end__`.

---

## 6. With a checkpointer + streaming
Nothing special. Same as before:
```Python
from langgraph.checkpoint.memory import InMemorySaver

chatbot = graph.compile(checkpointer=InMemorySaver())   # or MongoDBSaver(...)
config = {'configurable': {'thread_id': 'thread-1'}}
```


**Problem when streaming:** `stream_mode='messages'` streams **every** message, so the tool's raw output (`ToolMessage`) also gets printed in between.
**Fix:** print only **`AIMessage`** chunks (the LLM's text), skip `ToolMessage`s:
```Python
from langchain_core.messages import AIMessage

stream = chatbot.stream(
    {'messages': [HumanMessage(content='What is the share price of Tesla?')]},
    config=config,
    stream_mode='messages',
)

for chunk, metadata in stream:
    if isinstance(chunk, AIMessage):
        print(chunk.content, end='', flush=True)
```
- `AIMessage` = from the LLM → show it. `ToolMessage` = from a tool → don't show it.
- (Streamed LLM chunks are `AIMessageChunk`, which is a subclass of `AIMessage`, so `isinstance` catches them.)
- Show which tool is running ("Using calculator..."): the chunk is a `ToolMessage` and `chunk.name` is the tool's name.
```Python
for chunk, metadata in stream:
    if isinstance(chunk, ToolMessage):
        print(f'\n[using tool: {chunk.name}]')
    elif isinstance(chunk, AIMessage):
        print(chunk.content, end='', flush=True)
```
(import `ToolMessage` from `langchain_core.messages` too)

---

## Cheat sheet
| What you want                 | Code                                                            |
| ----------------------------- | --------------------------------------------------------------- |
| Imports                       | `from langgraph.prebuilt import ToolNode, tools_condition`      |
| Prebuilt search tool          | `DuckDuckGoSearchRun()` (`langchain_community.tools`)           |
| Custom tool                   | `@tool` on a function **with a docstring**                      |
| LLM knows the tools           | `llm_with_tools = model.bind_tools(tools)`                      |
| Node that runs tools          | `graph.add_node('tools', ToolNode(tools))`                      |
| Chat or tool?                 | `graph.add_conditional_edges('chat_node', tools_condition)`     |
| Loop back to the LLM          | `graph.add_edge('tools', 'chat_node')`                          |
| Stream only the LLM text      | `if isinstance(chunk, AIMessage):`                              |

**Remember:**
- `bind_tools` = the LLM **knows** the tools. `ToolNode` = **runs** them. `tools_condition` = **decides** chat (`END`) or tool (`'tools'`).
- Name the tool node **`'tools'`**: that's where `tools_condition` goes.
- Always `tools -> chat_node` (the loop). `tools -> END` = raw output + no multi-step.
- The docstring of a custom tool tells the LLM when and how to use it.
- Messages: `HumanMessage` → `AIMessage` (`tool_calls`) → `ToolMessage` → ... → final `AIMessage` (text).
- When streaming, print only `AIMessage` chunks.
