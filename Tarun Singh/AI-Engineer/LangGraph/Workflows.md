---
tags:
  - AIEngineer
  - workflow
---

## Sequential Workflows

```Python 
from langgraph.graph import StateGraph, START, END # start and end are kinda dummy node showing from here the program start and end
from typing import TypedDict

# Define the state first
class BMIState(TypedDict):
  weight_kg: float
  height_m: float
  bmi: float
  category: str

#making the function which actully calculate the bmi
# Each node take state as input and return sate as output
def calculate_bmi(state: BMIState) -> BMIState:
  weight=state['weight_kg']
  height= state['height_m']
  state['bmi']= round(weight/ (height*height),2)
  return state

def label_bmi(state: BMIState)->BMIState:
  bmi=state['bmi']
  if bmi <18.5 :
    state['category'] = 'Underweight'
  elif 18.5<=bmi<25:
    state['category'] = 'Normal'
  elif 25<=bmi<30:
    state['category'] = 'Overweight'
  else:
    state['category'] = 'Obese'
  return state

# Deine the graph
graph = StateGraph(BMIState)

# Add nodes
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('label_bmi', label_bmi)

# Add edges
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi', END)

# compile the graph
workflow = graph.compile()

# execute the graph
initial_state= {
    'weight_kg':100,
    'height_m': 1.80
}

output_state = workflow.invoke(initial_state)
print(output_state)
```
## Parallel Workflows

##### Note 1 - In Node we cant return complete state.
- In parallel-node-functions we can not send the complete state as return
- (unlike the sequential chain where we can)
- why becuase langGrpah will thing all the key-value pair present in the state is being edited out simultaneously by all the parallel node at once
- which will cuase issue
- so we do the partial update by sending only the dict -> with intended/edited key-value pair
##### Note 2 - Here we also use reducer function
- when we are calculating multiple vales in parallel and need to store in a list we need to use reducer function
- For example we have 5 parallel nodes and each node returns a single value, we need to store all those 5 value in a list -> then we need reducer function  

```Python 


```
## Conditional Workflows

## Iterative Workflows

