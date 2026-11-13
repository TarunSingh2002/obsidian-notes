---
tags:
  - AIEngineer
  - langchain
  - runnables
---

# Runnables

- Types of Runnables
    - Task specific runnables
    - Runnable primitives

- Task Specific Runnables
    - Definition: These are core LangChain components that have been converted into Runnables SO they can be used in pipeliræs.
    - Examples:
        - ChatOpenAI - Runs an LLM model.
        - PromptTemplate - Formats prompts dynamically.
        - Retriever - Retrieves relevant documents.

- Runnable primitives
    - Definition: These are fundamental building blocks for structuring execution logic in Al workflows
    - Examples:
        - RunnableSequence - Runs steps in order ( | operator).
        - RunnableParallel — Runs multiple steps simultaneously.
        - Runnablemap — Maps the same input across multiple functions.
        - RunnableBranch — Implements conditional execution (if-else logic).
        - RunnableLambda — Wraps custom Python functions into Runnables.
        - RunnablePassthrough - just forwards input as output (acts as a piqqehoider).

- RunnableSequence (| operator)
    - RunnableSequence is a sequential chain of runnables in LangChain that executes each step one after another, passing the output of one step as the input to the next.
    - It is useful when you need to compose multiple runnables together in a structured workflow.

- RunnableParallel
    - Help us in generating a parallel chains

- RunnablePassthrough
    - RunnablePassthrough is a special Runnable primitive that simply returns the input as output without modifying it.
    
- RunnableLambda
    - RunnableLambda is a runnable primitive that allows you to apply custom Python functions within an Al pipeline.
    - It acts as a middleware between different Al components, enabling preprocessing, transformation, API calls, filtering, and post-processing in a LangChain workflow.

- RunnableBranch 
    - Implements conditional execution (if-else logic).

## RunnablePassthrough

- Example:- we make a sequential chain where we aree generating a joke on text and then giving that joke to ai to generate the explaination -> so at the end we only get the explaintion not the joke. RunnablePassthrough fix this issue

- Solution
![[langchain-runnables-3.png]]

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

joke_gen_chain = prompt1 | model | parser

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': prompt2 | model | parser
})

final_chain = joke_gen_chain | parallel_chain

print(final_chain.invoke({'topic':'cricket'}))
```

## RunnableLambda

- Exmaple:- we are giving the llm a text and asking it to generate a joke on it, ands at last we are making python function to count the words in the output text

```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

joke_gen_chain = prompt | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({'topic':'AI'})

print(result)
```

## RunnableParallel

- RunnableParallel is a composition primitive that runs multiple runnables in parallel on the same input, producing a dictionary of outputs. It is essentially a mapping from string keys to runnables. When you invoke a RunnableParallel, LangChain runs each mapped runnable concurrently and returns a dict with the same keys, where each value is the output of the corresponding runnable

```Python
from langchain_core.runnables import RunnableParallel, RunnableLambda
# Define simple runnables
r1 = RunnableLambda(lambda x: x + 1)
r2 = RunnableLambda(lambda x: x * 2)

# Parallel mapping: two runnables on the same input
map_chain = RunnableParallel({"plus_one": r1, "times_two": r2})
result = map_chain.invoke(10)
print(result)  # -> {"plus_one": 11, "times_two": 20}
```

## RunnableAssign

- It tak dict as input from a RunnableParallel (called the mapper).
- RunnableAssign is a convenient wrapper for augmenting a dict with new keys
- it merges the mapper’s output into the input dictionary

```Python
from langchain_core.runnables.passthrough import RunnableAssign, RunnableParallel
from langchain_core.runnables.base import RunnableLambda

r1 = RunnableLambda(lambda x: {"added": x["input"] + 10})

mapper = RunnableParallel({"add_step": r1})
# result = mapper.invoke(5)
# print(result)  -> {'add_step': {'added': 15}}

runnable_assign = RunnableAssign(mapper)
output= runnable_assign.invoke({"input": 5})

print(output)
```
