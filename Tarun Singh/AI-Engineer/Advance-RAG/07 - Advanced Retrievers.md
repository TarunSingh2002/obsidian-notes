---
tags:
  - AIEngineer
  - rag
  - retrievers
---

# 07 - Advanced Retrievers

> Lessons: Contextual Compression & Parent Document Retriever · Self Query & Multi Query Retriever
> Code: `11_advanced_retrievers/notebooks/` (each one = LangChain classic + `custom_*` version)
> All four live in `langchain_classic` (support ends Dec 2026) → the course also builds each one by hand.

| Retriever | Problem it fixes | Extra cost |
|-|-|-|
| **Contextual Compression** | retrieved chunks carry noise (5 topics, query needs 1) | LLM call per doc |
| **Parent Document** | small chunks = good search, big chunks = good context | a doc store |
| **Self-Query** | query has hard constraints (price < 60k, year ≥ 2005) | LLM call to parse |
| **Multi-Query** | generic query → narrow coverage | LLM call + N retrievals |

---

## 1. Contextual Compression: retrieve, then trim
- Base retriever (any) → **compressor** keeps only the query-relevant text.

| Compressor | Does | Limit |
|-|-|-|
| `LLMChainExtractor` | LLM extracts relevant text **inside** each doc, drops empty ones | 1 LLM call per doc |
| `EmbeddingsFilter` | drops docs below a similarity threshold | can't trim inside a doc |
| `DocumentCompressorPipeline` | chain them: **cheap filter first, then LLM** | order matters |

```Python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (LLMChainExtractor, EmbeddingsFilter,
                                                               DocumentCompressorPipeline)

llm = ChatOpenAI(model='gpt-5-mini', temperature=0)      # 0: extract as-is, don't rephrase
embedding_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.73)
llm_extractor = LLMChainExtractor.from_llm(llm)
pipeline = DocumentCompressorPipeline(transformers=[embedding_filter, llm_extractor])

retriever = ContextualCompressionRetriever(base_compressor=pipeline,
                                           base_retriever=vectorstore.as_retriever(search_kwargs={'k': 3}))
docs = retriever.invoke('How is CRISPR enabling personalized medicine?')
```
- Demo: 3 docs (medicine + quantum + AI) → only the CRISPR sentences of the medicine doc survive.

### Custom version
```Python
from typing import Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    'Given the following question and context, extract any part of the context *AS IS* that is '
    'relevant to answer the question. If none of the context is relevant return NO_OUTPUT.\n\n'
    'Remember, *DO NOT* edit the extracted parts of the context.\n\n'
    '> Question: {question}\n> Context:\n>>>\n{context}\n>>>\nExtracted relevant parts:')   # copied from LangChain's source


class CustomLLMExtractor:
    def __init__(self, llm):
        self.chain = EXTRACT_PROMPT | llm | StrOutputParser()

    def compress_documents(self, documents, query):
        compressed = []
        for doc in documents:
            result = self.chain.invoke({'question': query, 'context': doc.page_content}).strip()
            if result and result != 'NO_OUTPUT':
                compressed.append(Document(page_content=result, metadata=doc.metadata))
        return compressed


class CustomContextualCompressionRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    base_compressor: Any

    def _get_relevant_documents(self, query, *, run_manager=None):
        docs = self.base_retriever.invoke(query)                       # 1. retrieve
        return self.base_compressor.compress_documents(docs, query)    # 2. compress
```

---

## 2. Parent Document Retriever: search small, answer big
- **Search on small child chunks** (sharp embeddings), **send the big parent chunks** to the LLM (enough info).
- Indexing: parent splitter → each parent gets a uuid → doc store `{id: parent}` → child splitter → each child's metadata has `parent_doc_id` → children into the vector store.
- Retrieval: top-k children → parent ids (**dedupe**) → fetch parents.
```Python
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore, LocalFileStore, create_kv_docstore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
docstore = InMemoryStore()                      # or create_kv_docstore(LocalFileStore('./parents')) to persist

retriever = ParentDocumentRetriever(vectorstore=Chroma(collection_name='pdr', embedding_function=embeddings),
                                    docstore=docstore,
                                    child_splitter=child_splitter,
                                    parent_splitter=parent_splitter,
                                    search_kwargs={'k': 3})
retriever.add_documents(documents)              # 5 docs → 11 parents, 51 children
results = retriever.invoke(query)               # 3 children → 2 parents (two shared a parent)
```
- Persisting the doc store alone isn't enough: Chroma also needs a `persist_directory`.

### Custom version
```Python
import uuid


class CustomParentDocumentRetriever(BaseRetriever):
    vectorstore: Any
    docstore: Any
    child_splitter: Any
    parent_splitter: Any
    id_key: str = 'parent_doc_id'
    k: int = 3

    def add_documents(self, docs):
        for parent in self.parent_splitter.split_documents(docs):
            parent_id = str(uuid.uuid4())
            children = self.child_splitter.split_documents([parent])
            for child in children:
                child.metadata[self.id_key] = parent_id        # link child → parent
            self.vectorstore.add_documents(children)
            self.docstore.mset([(parent_id, parent)])          # like dict[parent_id] = parent

    def _get_relevant_documents(self, query, *, run_manager=None):
        child_docs = self.vectorstore.similarity_search(query, k=self.k)
        parent_ids = []
        for child in child_docs:
            pid = child.metadata.get(self.id_key)
            if pid is not None and pid not in parent_ids:      # dedupe, keep order
                parent_ids.append(pid)
        parents = []
        for doc in self.docstore.mget(parent_ids):
            if doc is not None:
                parents.append(doc)
        return parents
```

---

## 3. Self-Query Retriever: natural language → semantic query + metadata filter
- "5 laptops under ₹60,000 with a bigger screen" → semantic `laptops` + filter `price < 60000, screen > 15`.
- Embeddings can't enforce "< 60000" (price lives in metadata, not in the embedded text).
- LLM (knows the metadata schema) → structured query → translator → Chroma filter → filtered similarity search.
- Chroma filter syntax: `{'$and': [{'genre': {'$eq': 'sci-fi'}}, {'year': {'$gte': 2005}}]}` (`$eq $ne $gt $gte $lt $lte`, `$and $or`).
```Python
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator

metadata_field_info = [
    AttributeInfo(name='genre', description='The genre of the movie. One of: action, sci-fi, drama, comedy', type='string'),
    AttributeInfo(name='year', description='The year the movie was released', type='integer'),
    AttributeInfo(name='rating', description='The movie rating', type='float'),
    AttributeInfo(name='director', description='The director of the movie', type='string'),
]
retriever = SelfQueryRetriever.from_llm(llm=llm,                        # temperature 0
                                        vectorstore=vectorstore,
                                        document_contents='Brief plot description of movies',
                                        metadata_field_info=metadata_field_info,
                                        structured_query_translator=ChromaTranslator(),
                                        enable_limit=True)             # "recommend 2 movies" → 2
docs = retriever.invoke('What are some sci-fi movies released on and after 2005?')
```
- gpt-5-mini parsed the filter wrong (The Matrix 1999 slipped in) → **gpt-5 fixed it**. Wrong results? Print the parsed filter first, then try a stronger model.
- No constraint in the query → just a normal similarity search.
- List allowed values in descriptions, so the LLM uses your exact labels.

### Custom version (Pydantic + hand-built filter)
```Python
from typing import Optional
from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    field: str = Field(description='The metadata field name to filter on')
    value: str | int | float = Field(description='The value to filter by')
    operator: str = Field(default='eq', description='Comparison operator: eq, ne, gt, gte, lt, lte')


class SelfQuerySchema(BaseModel):
    query: str = Field(description="The semantic search query extracted from the user's query")
    filters: Optional[list[MetadataFilter]] = Field(default=None, description='Metadata filters. None if no filters apply.')


query_chain = prompt | llm.with_structured_output(SelfQuerySchema)
# prompt (system): "You are a query parser for a movie database. Fields: genre (string), year (integer) ...
#                   Operators: eq, ne, gt, gte, lt, lte. Add filters ONLY if the user explicitly states a constraint."

OP_MAP = {'eq': '$eq', 'ne': '$ne', 'gt': '$gt', 'gte': '$gte', 'lt': '$lt', 'lte': '$lte'}


class CustomSelfQueryRetriever(BaseRetriever):
    vectorstore: Any
    query_chain: Any

    def _build_chroma_filter(self, filters):
        conditions = []
        for f in filters:
            op = OP_MAP.get(f.operator, '$eq')
            conditions.append({f.field: {op: f.value}})
        if len(conditions) == 1:
            return conditions[0]
        return {'$and': conditions}

    def _get_relevant_documents(self, query, *, run_manager=None):
        parsed = self.query_chain.invoke({'query': query})
        print(parsed)                                          # debug: see the parsed query + filters
        chroma_filter = None
        if parsed.filters:
            chroma_filter = self._build_chroma_filter(parsed.filters)
        return self.vectorstore.similarity_search(parsed.query, k=3, filter=chroma_filter)
```
- Duplicate results? You probably re-ran the add-documents cell on the same collection. Check the store, not the retriever.

---

## 4. Multi-Query Retriever: 1 generic query → N specific queries
- "How can I boost my health?" → physical / mental / immunity queries → retrieve each → **merge + dedupe**.
```Python
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

llm = ChatOpenAI(model='gpt-5-mini', temperature=0.3)          # a little randomness for varied rewrites (≤ 0.5)
retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                                         llm=llm,
                                         include_original=True)  # also search the original query
docs = retriever.invoke('How are modern technologies improving human health?')   # 3 × 3 = 9 → 6 unique
```

### Custom version
```Python
class QueriesSchema(BaseModel):
    queries: list[str] = Field(description='List of 3 alternative versions of the question')


prompt = ChatPromptTemplate.from_template(
    'You are an AI language model assistant. Your task is to generate 3 different versions of the given '
    'user question to retrieve relevant documents from a vector database. By generating multiple '
    'perspectives on the user question, your goal is to help the user overcome some of the limitations '
    'of distance-based similarity search.\n\nOriginal question: {question}')
query_chain = prompt | llm.with_structured_output(QueriesSchema)


class CustomMultiQueryRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    query_chain: Any

    def _get_relevant_documents(self, query, *, run_manager=None):
        queries = self.query_chain.invoke({'question': query}).queries
        queries.append(query)                                  # = include_original
        seen = set()
        unique_docs = []
        for q in queries:
            for doc in self.base_retriever.invoke(q):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
        return unique_docs
```

## Rules
- Search small, answer big (Parent Doc).
- Clean the context: cheap embedding filter → then LLM extractor (Compression).
- Hard constraints → metadata filters, not embeddings (Self-Query).
- Don't make users write better queries: rewrite them (Multi-Query).
- Temperature: **0** for extracting / parsing, **0.2-0.3** for rewrites.
