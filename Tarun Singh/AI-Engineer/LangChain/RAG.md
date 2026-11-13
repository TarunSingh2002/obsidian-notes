---
tags:
  - AIEngineer
  - langchain
  - rag
---

# RAG

- Note at the end we did not give the vectore embedding to the llm. we give the text associatd to that vectore embeding to the llm. vectore embedding only used for searching the document according to the user query

### Step 0 - basic imports

```Python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
```

### Step 1.1 - Indexing (Document Ingestion)

```Python
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")
```

### Step 1.2 - Indexing (Text Splitting)

```Python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
```

### Step 1.3 & 1.4 - Indexing (Embedding Generation and Storing in Vector Store)

```Python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
```

### Step 2 - Retrieval

```Python
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
```

### Step 3 - Augmentation

```Python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
```

```Python
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
```

```Python
question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
```

```Python
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
context_text
```

```Python
final_prompt = prompt.invoke({"context": context_text, "question": question})
```

### Step 4 - Generation

```Python
answer = llm.invoke(final_prompt)
print(answer.content)
```

### Step 5 - Building a Chain

```Python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
```

```Python
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
```

```Python
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
```

```Python
parallel_chain.invoke('who is Demis')
```

```Python
parser = StrOutputParser()
```

```Python
main_chain = parallel_chain | prompt | llm | parser
```

```Python
main_chain.invoke('Can you summarize the video')
```
