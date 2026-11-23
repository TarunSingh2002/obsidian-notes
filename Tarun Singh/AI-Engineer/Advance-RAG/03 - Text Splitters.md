---
tags:
  - AIEngineer
  - rag
  - langchain
---

# 03 - Text Splitters (Chunking)

> Lessons: What are Text Splitters, Text & Length Based Splitters · Advanced Text Splitter Techniques
> Code: `02_text_splitters/notebooks/` (character, recursive, document, semantic, llm_based)

## Why chunk
- Without chunking, top-5 out of 5 whole docs = **everything**. Chunking makes retrieval **selective**.
- Embedding = fixed size (e.g. 512 numbers) → 10,000 words into 512 numbers loses meaning; ~1000 words loses much less.
- Even page-wise PDFs need splitting: 1 page ≠ 1 topic.
- `pip install langchain-text-splitters`

| Method | In | Out |
|-|-|-|
| `split_text` | string | list of strings |
| `split_documents` | list of Document | list of Document (metadata kept) ← use in pipelines |
| `create_documents` | list of strings | list of Document |

## 5 strategies (pick top-down, stop when chunks look good)
| Strategy | Splits by | Use when | Cost |
|-|-|-|-|
| 1. Length (`CharacterTextSplitter`) | char / token count | quick baseline | cuts words, mixes topics |
| 2. **Recursive** (`RecursiveCharacterTextSplitter`) | `\n\n` → `\n` → space → char | **default for most text** | sizes vary |
| 3. Document structure | format separators (code, md, json, html) | structured files | can make huge chunks |
| 4. Semantic (`SemanticChunker`) | embedding similarity between sentences | topics shift irregularly | embedding cost, needs tuning |
| 5. LLM-based | LLM decides boundaries | small, complex text where 4 fails | most expensive, non-deterministic |

## 1. Length-based
```Python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10,
                                 length_function=len,
                                 separator=' ')          # '' char, ' ' word, '\n' line, '\n\n' paragraph
chunks = splitter.split_text(text)

# token-based (same tokenizer as the LLM)
token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base', chunk_size=50, chunk_overlap=5)   # sizes now in TOKENS
```
- `chunk_size` is an **upper bound only if a separator exists** before it. Otherwise the chunk grows past it + a warning → tune `chunk_size` and `separator` **together**.
- **Overlap** = repeat the tail of a chunk at the start of the next (fixes cut words). It's a patch: better to split on words or bigger units.

## 2. Recursive ⭐ (start here)
```Python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = splitter.split_documents(documents)
```
- Tries paragraph → line → word → char only until pieces fit, then **merges** neighbours back up to `chunk_size`.
- Overlap rule of thumb: **~10% of chunk_size**, go to 20% if results are poor.

## 3. Document structure (still recursive, different separators)
```Python
from langchain_text_splitters import (RecursiveCharacterTextSplitter, Language,
                                      RecursiveJsonSplitter, MarkdownHeaderTextSplitter)

# code: keeps classes / functions together
py_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                           chunk_size=400, chunk_overlap=50)
code_chunks = py_splitter.split_text(python_code)
# separators: ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']

# json: never breaks nesting
json_splitter = RecursiveJsonSplitter(max_chunk_size=200)
chunks = json_splitter.split_json(json_data=json_data)           # dicts (split_text → JSON strings)
# lists aren't split by default → convert_lists=True

# markdown: by headers, no size / overlap. Returns Documents, headers go to metadata
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[('#', 'Header_1'), ('##', 'Header_2'), ('###', 'Header_3')],
    strip_headers=False)                                         # keep the header in the text
md_chunks = md_splitter.split_text(markdown_text)
```
- A unit bigger than `chunk_size` still gets split (a `return` line ended up alone).
- Big chunks? Chain a second splitter after it.

## 4. Semantic chunking
- Sentences → embed each → cosine similarity of **neighbours** → similarity drops below the threshold = new chunk. No size, no separator.
```Python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

chunker = SemanticChunker(embeddings=OpenAIEmbeddings(),
                          breakpoint_threshold_type='percentile',   # or standard_deviation, interquartile
                          breakpoint_threshold_amount=90)            # a tuning knob, not a similarity score
chunks = chunker.split_text(text)
```
- Demo (AI + pasta + climate text): never got the ideal 3 chunks cleanly. **Trial and error.**

## 5. LLM-based chunking (one structured-output call)
```Python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class Chunk(BaseModel):
    chunk_text: str
    summary: str


class Chunker(BaseModel):
    chunks: list[Chunk]


llm_chunker = ChatOpenAI(model='gpt-5-mini').with_structured_output(schema=Chunker)
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an expert Text Chunker. Split the text at natural topic boundaries '
               'without changing it. For each chunk also write a 1-2 line summary.'),
    ('human', 'Split the given text into chunks \nText: {text}'),
])
response = (prompt | llm_chunker).invoke({'text': text})

docs = []
for c in response.chunks:
    docs.append(Document(page_content=c.chunk_text, metadata={'summary': c.summary}))
```
- Got the perfect 3 chunks. Keep the input **small** (the context window and lost-in-the-middle problems apply again).

## Rules
- Start **recursive**. Go to semantic, then LLM, only if chunks are bad.
- **One topic per chunk.** Always **print and inspect** your chunks.
