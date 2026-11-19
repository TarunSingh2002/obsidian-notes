---
tags:
  - AIEngineer
---

## Sequential Workflows

```Python
from langgraph.graph import StateGraph, START, END  # START and END are dummy nodes: they mark where the graph starts and ends
from typing import TypedDict


# 1. Define the state first
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str


# 2. Define the nodes
# Each node takes the state as input and returns the state (or the part of it that changed)
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state['weight_kg']
    height = state['height_m']
    state['bmi'] = round(weight / (height * height), 2)
    return state


def label_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']

    if bmi < 18.5:
        state['category'] = 'Underweight'
    elif bmi < 25:
        state['category'] = 'Normal'
    elif bmi < 30:
        state['category'] = 'Overweight'
    else:
        state['category'] = 'Obese'

    return state


# 3. Define the graph
graph = StateGraph(BMIState)

# Add nodes
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('label_bmi', label_bmi)

# Add edges
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi', END)

# 4. Compile the graph
workflow = graph.compile()

# 5. Execute the graph
initial_state = {'weight_kg': 100, 'height_m': 1.80}

output_state = workflow.invoke(initial_state)
print(output_state)
# {'weight_kg': 100, 'height_m': 1.8, 'bmi': 30.86, 'category': 'Obese'}
```

## Parallel Workflows

##### Note 1 - In parallel nodes we can't return the complete state
- In parallel node functions we cannot return the complete state
- (unlike the sequential chain, where we can)
- Why? Because LangGraph will think every key-value pair in the state is being edited by all the parallel nodes at the same time
- which raises an error (`InvalidUpdateError`)
- So we do a **partial update**: return only a dict with the key-value pairs that node changed

##### Note 2 - Here we also use a reducer function
- When we calculate multiple values in parallel and need to store them in one list, we need a reducer function
- For example: 5 parallel nodes each return a single value, and we want all 5 values in one list → we need a reducer (`Annotated[list, operator.add]`)

Example: cricket batsman stats. Three nodes run in parallel, then one node builds the summary.

```
            ┌─> strike_rate ───┐
START ──────┼─> boundary_pct ──┼──> summary ──> END
            └─> balls_per_4 ───┘
```

```Python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator


# 1. State
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int

    sr: float
    bpb: float              # balls per boundary
    boundary_percent: float
    summary: str

    # Reducer: every parallel node appends to this list instead of overwriting it
    logs: Annotated[list[str], operator.add]


# 2. Nodes -> each parallel node returns ONLY the keys it changed (partial update)
def calculate_sr(state: BatsmanState):
    sr = round((state['runs'] / state['balls']) * 100, 2)
    return {'sr': sr, 'logs': ['strike rate done']}


def calculate_bpb(state: BatsmanState):
    bpb = round(state['balls'] / (state['fours'] + state['sixes']), 2)
    return {'bpb': bpb, 'logs': ['balls per boundary done']}


def calculate_boundary_percent(state: BatsmanState):
    boundary_runs = state['fours'] * 4 + state['sixes'] * 6
    boundary_percent = round((boundary_runs / state['runs']) * 100, 2)
    return {'boundary_percent': boundary_percent, 'logs': ['boundary % done']}


def summary(state: BatsmanState):
    text = (
        f"Strike Rate - {state['sr']}\n"
        f"Balls per boundary - {state['bpb']}\n"
        f"Boundary percent - {state['boundary_percent']}"
    )
    return {'summary': text}


# 3. Graph
graph = StateGraph(BatsmanState)

graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)

# Fan-out: START -> 3 nodes run in parallel
graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

# Fan-in: summary waits for all 3 to finish
graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

graph.add_edge('summary', END)

# 4. Compile
workflow = graph.compile()

# 5. Execute
initial_state = {'runs': 100, 'balls': 50, 'fours': 6, 'sixes': 4, 'logs': []}

output_state = workflow.invoke(initial_state)
print(output_state['summary'])
print(output_state['logs'])
# Strike Rate - 200.0
# Balls per boundary - 5.0
# Boundary percent - 48.0
# ['boundary % done', 'balls per boundary done', 'strike rate done']
# (all 3 logs are kept thanks to the reducer; the ORDER is not guaranteed, because the nodes run in parallel)
```

## Conditional Workflows

- The graph picks **which node to go to next** based on the state (like `if/else`)
- We use `add_conditional_edges(from_node, router_function)`
- The router function reads the state and returns the **name of the next node**
- Use `Literal[...]` as the return type so it is clear (and LangGraph can draw) which nodes are possible

Example: solve a quadratic equation `ax² + bx + c = 0`. The discriminant decides which branch runs.

```
                                                  ┌─> real_roots ───────┐
START ─> show_equation ─> calculate_discriminant ─┼─> repeated_roots ───┼─> END
                                                  └─> no_real_roots ────┘
```

```Python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


# 1. State
class QuadState(TypedDict):
    a: int
    b: int
    c: int

    equation: str
    discriminant: float
    result: str


# 2. Nodes
def show_equation(state: QuadState):
    equation = f"{state['a']}x² + {state['b']}x + {state['c']} = 0"
    return {'equation': equation}


def calculate_discriminant(state: QuadState):
    discriminant = state['b'] ** 2 - 4 * state['a'] * state['c']
    return {'discriminant': discriminant}


def real_roots(state: QuadState):
    d = state['discriminant']
    root1 = (-state['b'] + d ** 0.5) / (2 * state['a'])
    root2 = (-state['b'] - d ** 0.5) / (2 * state['a'])
    return {'result': f"The roots are {root1} and {root2}"}


def repeated_roots(state: QuadState):
    root = -state['b'] / (2 * state['a'])
    return {'result': f"Only one repeated root: {root}"}


def no_real_roots(state: QuadState):
    return {'result': "No real roots"}


# 3. Router function -> returns the NAME of the next node
def check_condition(state: QuadState) -> Literal['real_roots', 'repeated_roots', 'no_real_roots']:
    if state['discriminant'] > 0:
        return 'real_roots'
    elif state['discriminant'] == 0:
        return 'repeated_roots'
    else:
        return 'no_real_roots'


# 4. Graph
graph = StateGraph(QuadState)

graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')

# Conditional edge: after calculate_discriminant, run check_condition to pick the branch
graph.add_conditional_edges('calculate_discriminant', check_condition)

graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

# 5. Compile and execute
workflow = graph.compile()

for a, b, c in [(1, -5, 6), (1, 2, 1), (1, 1, 5)]:
    output_state = workflow.invoke({'a': a, 'b': b, 'c': c})
    print(output_state['equation'], '->', output_state['result'])
# 1x² + -5x + 6 = 0 -> The roots are 3.0 and 2.0
# 1x² + 2x + 1 = 0 -> Only one repeated root: -1.0
# 1x² + 1x + 5 = 0 -> No real roots
```

## Iterative Workflows

- A **loop**: a conditional edge that can point **back** to an earlier node
- The loop keeps running until a stop condition is true → then go to `END`
- Always keep a **max iteration** counter in the state so the loop can't run forever
- LangGraph also has a safety limit: `recursion_limit` = max number of **super-steps** (one round of node runs; in this simple loop, 1 node run = 1 step) → raises `GraphRecursionError` if crossed
  - Old LangGraph (0.x): default was **25**. New LangGraph 1.x: default is **10007** (or set the env var `LANGGRAPH_DEFAULT_RECURSION_LIMIT`)
  - Set it yourself: `workflow.invoke(state, {'recursion_limit': 50})`
  - When the limit is hit, LangGraph raises a `GraphRecursionError` exception and the program stops at that `invoke` line, like any uncaught Python error. `invoke` never returns, so `output_state` is never set and the `print` lines after it don’t run

Example: number guessing with binary search. The graph keeps guessing until it finds the secret number (or hits the max tries).

```
START ─> guess ─> check ──(correct or max tries)──> END
           ^        │
           └────────┘  (too high / too low → guess again)
```

```Python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
import operator


# 1. State
class GuessState(TypedDict):
    secret: int
    low: int
    high: int

    guess: int
    feedback: str            # 'too low' / 'too high' / 'correct'
    iteration: int
    max_iteration: int

    history: Annotated[list[int], operator.add]   # reducer: keeps every guess


# 2. Nodes
def make_guess(state: GuessState):
    guess = (state['low'] + state['high']) // 2
    return {'guess': guess, 'history': [guess]}


def check_guess(state: GuessState):
    guess = state['guess']
    iteration = state['iteration'] + 1

    if guess == state['secret']:
        return {'feedback': 'correct', 'iteration': iteration}
    elif guess < state['secret']:
        # secret is bigger -> move the lower bound up
        return {'feedback': 'too low', 'low': guess + 1, 'iteration': iteration}
    else:
        # secret is smaller -> move the upper bound down
        return {'feedback': 'too high', 'high': guess - 1, 'iteration': iteration}


# 3. Router -> decides: loop back or stop
def should_continue(state: GuessState) -> Literal['make_guess', '__end__']:
    if state['feedback'] == 'correct' or state['iteration'] >= state['max_iteration']:
        return END          # END is just the string '__end__'
    return 'make_guess'     # go back -> this edge is what makes it a loop


# 4. Graph
graph = StateGraph(GuessState)

graph.add_node('make_guess', make_guess)
graph.add_node('check_guess', check_guess)

graph.add_edge(START, 'make_guess')
graph.add_edge('make_guess', 'check_guess')
graph.add_conditional_edges('check_guess', should_continue)

# 5. Compile and execute
workflow = graph.compile()

initial_state = {
    'secret': 73,
    'low': 1,
    'high': 100,
    'iteration': 0,
    'max_iteration': 100,
    'history': [],
}

output_state = workflow.invoke(initial_state, {'recursion_limit': 200})  # 100 rounds x 2 steps = 200, plus some margin
print('Guesses :', output_state['history'])
print('Result  :', output_state['feedback'], 'in', output_state['iteration'], 'tries')
# Guesses : [50, 75, 62, 68, 71, 73]
# Result  : correct in 6 tries
```
