---
tags:
  - AIEngineer
  - langchain
  - chains
---

# Chains

## How i am passing output of one step and input to another step with the control over the name of variable

- help of RunnableAssign and RunnableParallel
- In LCEL, the output of one step becomes the input to the next. Because each runnable (including parallel runnables and assign-runnables) outputs a dictionary, by chaining with the pipe operator (|) we implicitly feed that dictionary to the next step.

### IF the output of First model is single string which is going in secound model

- Code Explaination
    - ``` RunnableParallel({"Story":story_chain}).invoke({"character_name": "Alice"})```  
    output-> {'Story':"Story...."}
    - ```RunnableAssign(RunnableParallel({"Story":story_chain})).invoke({"character_name": "Alice"})``` 
    output-> {'Story':"Story....", 'character_name': 'Alice'}
    - now chain 2 willl get this output as input dictonary-> {'Story':"Story....", 'character_name': 'Alice'} and the same will continue

```Python
from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.passthrough import RunnableAssign

# Define the prompts
prompt1 = PromptTemplate.from_template("Write a short story about {character_name}.")
prompt2 = PromptTemplate.from_template("Summarize this story:\n{Story}")
prompt3 = PromptTemplate.from_template(
    "Given character {character_name} and story summary: {summary}, tell a joke."
)

# Initialize the Google Gemini LLM (requires your Google API key)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")

# Build sub-chains for each step
story_chain   = prompt1 | llm | StrOutputParser()
summary_chain = prompt2 | llm | StrOutputParser()
joke_chain    = prompt3 | llm | StrOutputParser()

# Wrap each in a RunnableParallel mapping to a key, then in a RunnableAssign
chain1 = RunnableAssign(RunnableParallel({"Story":story_chain}))
chain2 = RunnableAssign(RunnableParallel({"summary":summary_chain}))
chain3 = RunnableAssign(RunnableParallel({"joke":joke_chain}))

pipeline = chain1 | chain2 | chain3

# Run the pipeline
result = pipeline.invoke({"character_name": "Alice"})
print(result["Story"])    # The story generated about Alice
print("###########################################################################################")
print(result["summary"])  # The summary of that story
print("###########################################################################################")
print(result["joke"])     # A joke involving Alice and the summary
```

### If the output of the 1 model is a dict(conatining 2 elemnets) and we only wnat to shre one element as input to another prompt, how can we do that?

    - use RunnableLambda to extract bothe the emelmets from the input of the first model

```Python
from langchain_core.runnables import RunnableLambda
extract = RunnableLambda(lambda x: {
    **x, # getting the old values 
    """
    But if you want to get some specific cvalues only then do this -->
    "old_value_1": x["old_value_1"],  # Recopy the old value
    "old_value_2": x["old_value_2"],  # Recopy the old value
    """
    "element1": x["output_from_1_model"]["element1"],
    "element2": x["output_from_1_model"]["element2"]
})
chaina = prompt | structured_model | extract
```

## Sequential Chains

### Example1

```Python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()
```

### Example2

- Critical Rules for Variable Binding
    - If next component expects one input variable, previous output is directly mapped

- Note in the given exapmple the text in prompt2 will be the output of parser

```Python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()
```

## Parallel Chains

```Python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()

model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()
```

## Conditional Chains

```Python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()
```
