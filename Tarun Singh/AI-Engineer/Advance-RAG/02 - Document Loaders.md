---
tags:
  - AIEngineer
  - rag
  - langchain
---

# 02 - Document Loaders

> Lessons: What are Document Loaders & How They Work · Text, CSV and PDF Loader · JSON, Web, Recursive Web Loader
> Code: `01_document_loaders/notebooks/` (text, pdf, csv, json, web)

## Idea
- Loader = **translator**: any format (txt, pdf, csv, json, html...) → **one format**: `Document`.
- 3 steps: **load** (file → memory) → **parse** (understand the structure) → **output** `Document`.
- `Document(page_content=str, metadata=dict)`. Metadata (source, page...) = source attribution later.
- One loader per format (each has its own parser). All are in `langchain-community`.
- Knowledge **source** = your raw files. Knowledge **base** = the vector DB built from them.

## Same pattern for every loader
```Python
from pathlib import Path
from langchain_community.document_loaders import TextLoader

file_path = Path('../knowledge-source/transformers.txt')
print(file_path.exists())            # sanity check

loader = TextLoader(file_path)        # encoding='utf-8' if emojis / special chars
documents = loader.load()             # ALWAYS a list of Document
print(len(documents))                 # 1 (whole txt file = 1 Document)
print(documents[0].page_content)
print(documents[0].metadata)          # {'source': '...'}
```

## PDF loaders (pick by need, test on your PDF)
| Loader | Good at | Watch out |
|-|-|-|
| `PyPDFLoader` | clean page-wise text, simple | image OCR barely worked in the demo |
| `PDFMinerLoader` | text + **image text via OCR** + table text | slower, default mode is `single` |
| `PDFPlumberLoader` | **richest metadata** | words can merge (missing spaces) |
| `PyMuPDFLoader` | detailed metadata | |
| Mathpix | math-heavy PDFs | |

```Python
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser

loader = PyPDFLoader(file_path.as_posix(), mode='page')   # 1 Document per page (use this)
documents = loader.load()

# images + tables: PDFMiner + OCR
miner = PDFMinerLoader(file_path.as_posix(),
                       mode='page',                        # set it, default is 'single'
                       extract_images=True,
                       images_parser=RapidOCRBlobParser(), # or TesseractBlobParser
                       images_inner_format='html-img')     # image text wrapped in <img> tags
documents = miner.load()

# get one page by metadata
for doc in documents:
    if doc.metadata['page'] == 7:        # 0-based index. page_label = printed page number
        print(doc.page_content)
        break
```
- `mode='single'` → whole PDF in 1 Document, loses per-page metadata. Prefer `'page'`.
- `html-img` beats markdown markers: PDF text rarely contains HTML tags, so image text stays separable.

## CSV: 1 row = 1 Document
```Python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path=file_path,
                   source_column='Industry',                                    # metadata['source'] = this column
                   metadata_columns=['Website', 'Founded', 'Number of employees'],  # moved out of the text
                   content_columns=['Description'])                             # only this gets embedded
documents = loader.load()    # 1000 rows → 1000 Documents
```
- Column names must match the header exactly.
- Rule: **embed only meaningful text**, move IDs / attributes into metadata.

## JSON: `jq_schema` says where the records are
```Python
from langchain_community.document_loaders.json_loader import JSONLoader


def metadata_func(record, metadata):
    # record = the full JSON object, metadata = default (source, seq_num)
    metadata['product_name'] = record['productName']
    metadata['category'] = record['category']
    metadata['price'] = record['price']
    del metadata['seq_num']
    return metadata


loader = JSONLoader(file_path=file_path.as_posix(),
                    jq_schema='.products[]',        # each product object = 1 Document
                    content_key='Description',      # this key → page_content
                    metadata_func=metadata_func)
documents = loader.load()
```
- `.products` → error (a list, not a string). `.products[]` → each object. `.products[].Description` → just the text.
- Needs `pip install jq`.

## Web
```Python
from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader

loader = WebBaseLoader(web_paths=[url_1, url_2, url_3])   # 1 URL = 1 Document (uses bs4)
documents = loader.load()                                 # metadata: source, title, language

loader = RecursiveUrlLoader(url=base_url, max_depth=2)    # root + all child links (raw HTML)
```
- Set the `USER_AGENT` env var if it warns.
- Recursive loading of 222 pages took ~6 min and held them all in memory → use `lazy_load()`.

## `load()` vs `lazy_load()`
```Python
docs = loader.lazy_load()        # generator: returns instantly, loads nothing yet
counter = 0
for doc in docs:                 # loads ONE doc per iteration
    if counter == 20:
        break
    counter += 1
    print(doc.page_content[:300])
```
- `load()` = full list in memory (small sources). `lazy_load()` = one at a time (big / unknown sources).
- `load_and_split()` is deprecated → use text splitters.
