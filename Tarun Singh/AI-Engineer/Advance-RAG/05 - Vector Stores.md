---
tags:
  - AIEngineer
  - rag
  - vectorstore
  - chroma
---

# 05 - Vector Stores (ChromaDB)

> Lesson: What are Vector Stores, Their Importance & CRUD Operations
> Code: `04_vector_stores/notebooks/` (chromadb_crud_operations, chromadb_load_existing_db, chromadb_pdf_pipeline)

## Why
- Doc embeddings don't change between queries → **embed once, persist, reuse**. Only the query is embedded per request.
- Each record = **ID + embedding + text + metadata**:
  - embedding → similarity search / ranking only
  - text → goes into the prompt
  - metadata → citations
- Supports **CRUD** (keep the knowledge base fresh).

## Indexing: exact vs ANN (Approximate Nearest Neighbour)
| | Exact (brute force) | IVF (clustering) | HNSW (graph) |
|-|-|-|-|
| Searches | every vector, O(n) | centroids → vectors of the best cluster(s) | greedy walk: sparse top layer → dense base layer |
| Accuracy | perfect | can miss a neighbour in another cluster | can miss unconnected neighbours |
| Use | small stores (~1000) | large | large (**Chroma default**) |
- **IVF**: k-means → centroids. 100k vectors, 20 clusters → 20 + ~5000 comparisons instead of 100k (~20× fewer). New docs join the nearest centroid.
- **HNSW**: "six degrees of separation". Layer 0 = all vectors linked to their neighbours, upper layers = random subsets. The query enters at the top, moves greedily to closer nodes, then drops a layer. 1M vectors → hundreds to ~1000 comparisons.
- ANN trades a little accuracy for a big speed-up. Approximate by design.

## Chroma
- Open source, free, `langchain-chroma`. Others: FAISS, Pinecone, Milvus, Qdrant, Weaviate (same LangChain interface).
- **Collection** = a folder / table. Always name it.
- No `persist_directory` → in RAM (lost on restart). With it → SQLite (text, ids, metadata) + one binary folder per collection.

```Python
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = Chroma(collection_name='demo_2',
                      embedding_function=embeddings,
                      persist_directory='chroma_langchain_db')

# CREATE: pass Documents, Chroma embeds them
documents = []
for item in document_examples:
    documents.append(Document(id=str(uuid4()),                 # unique id (uuid4 = no collisions)
                              page_content=item['text'],
                              metadata={'topic': item['topic'], 'doc_number': item['doc_number']}))
ids = vector_store.add_documents(documents)                     # returns the ids

# READ
raw = vector_store.get(include=['embeddings', 'metadatas', 'documents'])   # dict, embeddings = numpy (10 x 1536)
docs = vector_store.get_by_ids(ids[:2])                                     # list of Document
results = vector_store.similarity_search('How does RAG help an LLM?', k=3)
results = vector_store.similarity_search_with_score('How does RAG help an LLM?', k=4)   # (doc, score)

# UPDATE: same id, new content (re-embedded)
vector_store.update_documents(ids=ids_to_update, documents=updated_documents)

# DELETE
vector_store.delete(ids=ids_to_delete)
```
- ⚠ Chroma's default score is a **distance** (squared L2): **lower = closer**.
- **Update, don't add**: leave policy changed 10 → 12 days; adding a new doc keeps both → contradictory answers. Overwrite the same ID.

## Reload an existing store (new session / new app)
```Python
vector_store = Chroma(collection_name='demo_2',                        # same name
                      embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'),  # same model
                      persist_directory='chroma_langchain_db')         # same folder
```
- Migrate = copy the folder.

## End-to-end ingestion (PDF → Chroma)
```Python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = PyPDFLoader(str(pdf_path)).load()                        # 6 pages
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)  # 88

vector_store = Chroma.from_documents(documents=chunks,          # create + fill in one step
                                     embedding=embeddings,
                                     collection_name='rag-pipeline',
                                     persist_directory='chroma_langchain_db')

for doc, score in vector_store.similarity_search_with_score('How do AI agents use tools and memory?', k=2):
    print(round(score, 4), doc.metadata.get('page_label'), doc.page_content[:100])
```
- `Chroma(...)` + `add_documents` → a store you keep updating. `from_documents` → one-shot build.

## Rules
- Embed once, persist, reuse.
- Same embedding model when building **and** reloading.
- Unique stable IDs; update by ID; delete what's obsolete.
