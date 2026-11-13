---
tags:
  - AIEngineer
  - langchain
  - prompt
---

## Prompt template

- Where we can use a prompt 
  - Single Message interaction with AI with no chat history [Case 1]
  - Single Message interaction with AI with chat history [Case 2]
- Types of prompt in [Case 1] and [Case 2]:-
  - Static Prompts
  - Dynamic Prompts

### Types of Prompt template in [Case 1]

#### Static Prompts

```Python
# Hugging face api chat models
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India") # static prompt
print(result.content)
```

#### Dynamic Prompt

```Python
from langchain_core.prompts import PromptTemplate
template= PromptTemplate(
    template="""
    Question: {question_text},
    Answer {answer_text}.
    """,
    input_variables=['question_text', 'answer_text'],
    validate_template=True
)
question="What is the capital of India"
answer= "this question step by step"
chain = template | model 
result= chain.invoke({
    "question_text": question,
    "answer_text": answer
})
print(result.content)
```

### Types of Prompt template in [Case 2]


- Types of messages (used in chat history):-
  - System messages: The first message we send to ai (like - you are a helpful assistant)
  - Human messages:- Messages send by human
  - AI messages:- Messages send by AI

- Message Placeholder:- A MessagesPlaceholder in LangChain is a spectal placeholder used inside a
ChatPromptTemplate to dynamically insert chat history or a list of messages at runtime.

#### Static

```Python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)
```

#### Dynamic

```Python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'), # system message 
    ('human', 'Explain in simple terms, what is {topic}') # human message
])

prompt = chat_template.invoke({'domain':'cricket','topic':'whit ball'})
```

#### Message Placeholder

- Example:- For a chatbot with chathistory feature, once the chat is completed we save the chat history in a db and when user comes again the chat history will be loaded first form db to message placeholder and then the user new query with previous history is given to the model!!

```Python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'), # system message 
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', 'Explain in simple terms, what is {topic}') # human message
])

chat_histry =[]   #get it from db

chat_template.invoke({'domain':'cricket','topic':'whit ball', 'chat_history':chat_histry})
```
