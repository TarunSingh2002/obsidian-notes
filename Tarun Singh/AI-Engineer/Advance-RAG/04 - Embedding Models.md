---
tags:
  - AIEngineer
  - rag
  - embeddings
---

# 04 - Embedding Models

> Lessons: Vector Embeddings, Embedding Space and Distance Metrics · Dimensionality, Proprietary vs Open Source
> Code: `03_embeddings/notebooks/` (openai_embeddings, ollama_embeddings)

## Why "contextual" embeddings
| Stage | Example | Problem |
|-|-|-|
| Classical | BoW, TF-IDF | sparse, keyword-only (still useful → BM25 / hybrid search) |
| Word embeddings | Word2Vec, GloVe | 1 vector per word: "river **bank**" = "money **bank**" |
| **Contextual** | BERT, OpenAI, EmbeddingGemma, Sentence Transformers | dense + context-aware ✅ |
- "Login issue" should match "password recovery" chunks → needs **contextual** meaning, not keyword match.

## Embedding space (intuition)
- Similar meaning → close together (clusters), point in the same **direction**.
- Query → **same model** → lands in a neighbourhood → nearest chunks = retrieved.

## Similarity metrics
| | Euclidean | **Cosine** ⭐ | Dot product |
|-|-|-|-|
| Measures | straight-line distance | angle | sum of element products |
| Range | unbounded | **[-1, 1]** | unbounded |
| Better = | lower | higher | higher |
| Magnitude matters? | yes | **no** | yes |
- Cosine = `A·B / (|A| |B|)`. 1 = same, 0 = unrelated, -1 = opposite direction.
- **Normalized vectors (OpenAI, etc.) → cosine = dot product** (same ranking, cheaper). OpenAI and Ollama both recommend cosine.

## Dimensionality (384 / 768 / 1024 / 1536 / 3072)
| More dims | |
|-|-|
| ✅ | finer nuance (sick vs casual vs childcare leave stay separate) |
| ❌ | storage grows linearly: 2M vectors × 384 dims × 4 bytes ≈ 3 GB → 1536 dims ≈ 12 GB |
| ❌ | slower similarity (~4× the ops for 1536 vs 384), more network cost |
- Quality **plateaus**. A well-trained 768 model can beat a weak 1536 one. **Test on your data and pick the balanced point.**

## Proprietary vs open source
| | Proprietary (OpenAI API) | Open source (Ollama / HF, self-hosted) |
|-|-|-|
| Pros | no infra, managed, scaling, support | no per-call cost, **privacy**, no rate limits, offline, fine-tunable |
| Cons | per-call cost, data leaves you, rate limits, **vendor lock-in** (switching = re-embed everything) | GPUs / RAM, DevOps, manual updates |
- Sensitive data (medical, legal) → lean open source. Low volume → API is cheaper. Very high volume → self-host wins.

## Code
```Python
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings

load_dotenv()

docs = PyPDFLoader(file_path='../Openclaw_Research_Report.pdf').load()        # 34 pages
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)  # 273

texts = []
for doc in chunks:
    texts.append(doc.page_content)        # embedders want strings, not Documents

# OpenAI: text-embedding-3-small (1536) / -large (3072)
embedder = OpenAIEmbeddings(model='text-embedding-3-large')
query_vec = embedder.embed_query(text=query)            # 1D, len 3072
doc_vecs = embedder.embed_documents(texts=texts)        # 273 x 3072
small_256 = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=256)   # custom size

# Open source via Ollama:  ollama pull embeddinggemma   (768 dims, ~621 MB)
embedder = OllamaEmbeddings(model='embeddinggemma')
doc_vecs = embedder.embed_documents(texts=texts)        # 273 x 768
```
- Native `ollama.embed(model=..., input=...)` works too, but it has no `embed_documents` → **LangChain vector stores fail with it**. Use the LangChain classes, so swapping models = changing one line.

## Rules
- **Same embedding model** for chunks and queries.
- Cosine by default (dot product if normalized).
- Don't chase dimensions; model quality matters more.
