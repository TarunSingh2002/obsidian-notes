---
tags:
  - AIEngineer
  - langchain
  - setup
---

# LangChain

- [Complete How-To Documentation](https://python.langchain.com/docs/how_to/)
- [Complete API Reference](https://python.langchain.com/api_reference/)

## Keys management and basic setup

1. Create a `.env` file.
2. Store your API keys like this:
   ```plaintext
   MY_KEY=1234567890
```

```Python
!pip install langchain langchain_huggingface langchain-community transformers torch langchainhub langchain-openai
```

```Python
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    print(os.getenv('MY_KEY'))
```

## Topics

- [[Prompt-Template]]
- [[Models]]
- [[Structured-Output]]
- [[Chains]]
- [[Runnables]]
- [[Document-Loaders]]
- [[Text-Splitters]]
- [[Vector-Stores]]
- [[Retrievers]]
- [[RAG]]
