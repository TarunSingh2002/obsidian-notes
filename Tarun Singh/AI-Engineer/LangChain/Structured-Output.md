---
tags:
  - AIEngineer
  - langchain
  - structured-output
---

# Structured Output

- There are 2 types of model
   - who can generate structure output like (gpt) -> use with_structure_output method
   - who can not enerate structure output -> use output_parsers

## with_structure_output

![[langchain-structured-output-2.png]]
- 3 differnt ways to define the output parameters
  - TypedDict
  - Pydantic
  - json_schema

- 2 Different ways to get the output

```python

with_structured_output("text", method="function_calling/json_mode/json_schema")
```

### TypedDict
- Return python dictionary
- TypedDict is a way to.define a dictionary in Python where you specify what keys and values
should exist. It helps ensure that your dictionary follows a specific structure.

Corn of TypedDict
- The return datatype is not guaranteed

```Python
from typing import TypedDict, Annotated, Optional, Literal

# schema
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("text input of user")

print(result['name'])
```

### Pydantic

- Return Pydantic object
- Pydantic is a data validation and data parsing library for Python. It ensures that the data you work with is correct, structured, and type-safe.

Pros
- It tries to understand which data type is coming and also performs typecasting if possible. For example, if an integer is expected but it comes as a string like `'32'`, Pydantic will automatically convert it to `32`.
- Field function: description, default value, constraints, and regex expression.

```
intFiled: Optional[float] = Field(
    default=0,gt=0,lt=10,
    description="a integer filed which will be greater than or equal to 0 and less than or equal to 10"
)
```

```Python
from typing import Optional, Literal
from pydantic import BaseModel, Field

class Review(BaseModel):

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Text by user""")
print(result.name)
```

### Json_schema

- Return Python Dictionary
- Here it is necessary to give these fileds
     - title, type, required(will contain the key name which is not optional) and properties

```Python
# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""user text""")

print(result)
```

## output_parsers

- Output Parsers in LangChain help convert raw LLM responses into structured formats like
JSON, CSV, Pydantic models, and more. They ensure consistency, validation, and ease Of use in
applications.

- Most popular output parsers
    - StrOutputParser:- it take model output and return string
    - Jsonoutput:- it return json output but can not enforce any parameters and proper json format
    - structureoutput:- skipped
    - pydnaticoutput:- skipped

### StrOutputParser

```Python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
```

### Jsonoutputparser

```Python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
```
