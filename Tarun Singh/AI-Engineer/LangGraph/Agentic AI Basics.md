---
tags:
  - AIEngineer
---

## Agentic AI Key Characteristics

### Autonomous

- Types of Autonomy
	- Autonomy in decision makin
	- Autonomy in execution
	- Autonomy in tool usage
- Autonomy can be controlled by
	- Permission Scope
	- Human-in-the-loop (HITL)
	- Guardrails / Policies
	- Override Controls (Allow user to stop, pause and change the agents behavior at any time)
### Goal Oriented
- Goals acts as a campus for Autonomy 
- Goals can come with constraints and need to be stored in core memory
### Planning
- Planning is the agent's ability to break down a high-level goal hto a structured sequence of actions or subgoals and decide the best path to achieve the desired Outcome.
- Done in 3 steps:
  - Generating multiple candidate plans
  - Evaluate each pian
  - Select the best plan with the help of
	- Human-in-the-loop input (e.g., "Which Of these options do you prefer?" )
	- A pre-programmed policy (e.g., "Favor low-cost channels first")
### Reasoning
  - Reasoning required at both the stages Planning and Execution
### Adaptability
- Adaptability is the agenes ability to modify its plans, strateg•es. or actions in response to unexpected conditions — all while staying aligned with the goals
### Context Awareness
- Types of context
  - The original goal
  - Progress till now + Interaction History
  - Environment state
  - Tool Responses
  - User specific perference
  - Policy guardrails
- Implemented through memory
  - Short term memory
  - Long term memory
### Key Components
- Brain
	- If the aganet is llm based then brain will be that LLM
	- Task of brain
	  - Goal Interpretation
	  - Planning
	  - Reasoning
	  - Tool Selection
	  - Communication
- Orchestrator
	- The Orchestrator = do the execution of the plan
	- We build the orchestrator via langGraph
- Tools
- Memory
- Supervisor
	- Human in the loop
	- Gurdrails

## Core Concepts

### LLM Workflows
- Prompt chaining (normal chain in langchain prompt->llm->prompt->llm)
- Routing = Making conditional
- Parallelization = Parallel chains
- Evalutor Optimizer = 2 LLMs goes back and forth till the output is kinda acceptable. (Like GAN works)
### Graph, Nodes and Edges
  - Node
    - Represent a task
    - A python function
    - Input of all the nodes is a dict (Mostly state)  and also the output
  - Edges
    - Flow of the graph
    - Types
      - Sequential
      - Parallel
      - Conditional
      - Loop
### State
- The state is the shared memory that flows through your workflow. It hold all the data being passed between nodes as your graph runs.
- All the nodes can make changes in this state
### Reducers
- Define how updates from nodes are applied to the shared state
- Each key in the state can have its own reducer, which determines whether new data replaces,merges or adds to the existing value.
### Execution Model
- Graph definition
	- Define Nodes and Edges
- Compilation
	- It check the graph structure and prepare it for execuition
- Invocation
	- In this step we give our first starting node to invoke function -> from there our graph execution get started.
- Halting Condition
	- Execution steps when
		  - No nodes are active
		  - No message are in transit