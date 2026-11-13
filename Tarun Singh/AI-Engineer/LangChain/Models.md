---
tags:
  - AIEngineer
  - langchain
  - models
---

# Models

- There are 2 types of it - language models and embedding models
 - language models are also 2 types llm models and chat models
 - LLMs are general purpose models but chat models are special purpose models
 - A llm model can do text generation, text summerization, code generation, question answering, etc
 - But a chat models can take sequence of messages as input and return chat messages as output (they also have momery)
 -Langchain is moving towards chat bots and slowly moving away from the llms
 ![[langchain-models-1.png]]

## LLMs
 - Working with openai only

```Python
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
my_key = os.getenv("MY_OPENAI_KEY")

llm = OpenAI(model='gpt-3.5-turbo-instruct', openai_api_key=my_key)

result = llm.invoke("What is the capital of India")

print(result)
```

## ChatModels

[What is ChatModels and its standard parameters (like temperature..)](https://python.langchain.com/docs/concepts/chat_models/)

[How to use all paid models (like openai) with langchain](https://python.langchain.com/docs/integrations/chat/)

  - [Chtagpt model info](https://platform.openai.com/docs/models)
  - [Gemini models info](https://ai.google.dev/gemini-api/docs/models/gemini)
  - [Hugging face models](https://huggingface.co/models)

Code

 - OpenAI chat models
 - Anthropic chat models
 - Google geminia chat models
 - Hugging face api chat models
 - Hugging face local chat models

Standerd Parameters
- **model**:- The name or identifier of the specific AI model you want to use (e.g., "gpt-3.5-turbo" or "gpt-4").

- **temperature**:- Controls the randomness of the model's output. A higher value (e.g., 1.0) makes responses more creative, while a lower value (e.g., 0.0) makes them more deterministic and focused.

- **timeout**:- The maximum time (in seconds) to wait for a response from the model before canceling the request. Ensures the request doesn’t hang indefinitely.

- **max_tokens**:- Limits the total number of tokens (words and punctuation) in the response. This controls how long the output can be.

- **stop**:- Specifies stop sequences that indicate when the model should stop generating tokens. For example, you might use specific strings to signal the end of a response.

- **max_retries**:- The maximum number of attempts the system will make to resend a request if it fails due to issues like network timeouts or rate limits.

- **api_key**:- The API key required for authenticating with the model provider. This is usually issued when you sign up for access to the model.

- **base_url**:- The URL of the API endpoint where requests are sent. This is typically provided by the model's provider and is necessary for directing your requests.

- **rate_limiter**:- An optional BaseRateLimiter to space out requests to avoid exceeding rate limits. See rate-limiting below for more details.

```Python
# OpenAI chat models
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
my_openai_key = os.getenv("MY_OPENAI_KEY")

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10, api_key=my_openai_key
)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)
```

```Python
# Anthropic chat models
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()
my_anthropic_key = os.getenv("MY_ANTHROPIC_KEY") # In your .env: MY_ANTHROPIC_KEY=abc...

model = ChatAnthropic(model='claude-3-5-sonnet-20241022', api_key=my_anthropic_key)

result = model.invoke('What is the capital of India')

print(result.content)
```

```Python
# Google geminia chat models
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
my_google_key = os.getenv("MY_GOOGLE_KEY")  # In your .env: MY_GOOGLE_KEY=def...

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=my_google_key
)

result = model.invoke('What is the capital of India')

print(result.content)
```

```Python
# Hugging face api chat models
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
my_hf_key = os.getenv("MY_HUGGINGFACE_KEY")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=my_hf_key
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
```

```Python
# Hugging face local chat models
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
```

## Embedding Models

[What is Embedding Models and its standard methods (like embed_documents and embed_query)](https://python.langchain.com/docs/concepts/embedding_models/)

[How to use all paid models and their integration with langchain](https://python.langchain.com/docs/integrations/text_embedding/)

  - [Chtagpt model info]()
  - [Hugging face models]()

code
  - Embedding Openai query
  - Embedding Openai Docs
  - Embedding Hugging face Docs and query by local
  - Embedding Hugging face Docs and query by API

```Python
# Embedding Openai query
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
my_openai_key = os.getenv("MY_OPENAI_KEY")  # In your .env: MY_OPENAI_KEY=123...

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32, api_key=my_openai_key)

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))
```

```Python
# Embedding Openai Docs
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
my_openai_key = os.getenv("MY_OPENAI_KEY")  # In your .env: MY_OPENAI_KEY=123...

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32, api_key=my_openai_key)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

print(str(result))
```

```Python
# Embedding Hugging face Docs and query by local
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(str(embedding.embed_query("Hello, world!")))
print(str(vector))
```

```Python
# Embedding Hugging face Docs and query by API
from langchain_community import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
my_huggingface_api_token = os.getenv("MY_HUGGINGFACE_API_TOKEN")  # In your .env: MY_HUGGINGFACE_API_TOKEN=your_token_here

embedding = HuggingFaceInferenceAPIEmbeddings(
    model_id="sentence-transformers/all-MiniLM-L6-v2",  # example Hugging Face model
    api_token=my_huggingface_api_token
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

query_embedding = embedding.embed_query("Hello, world!")

print("Query Embedding:", query_embedding)
print("Document Embeddings:", result)
```
