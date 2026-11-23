---
tags:
  - AIEngineer
  - rag
  - multimodal
---

# 15 - Multi-Modal RAG (Text + Images)

> Lesson: Multi Modal RAG
> Code: no repo folder (code from the course notes). Data: the CRAG research paper PDF (text + charts + tables).

## Problem
- Real docs (PDFs, PPTs) mix text, images and tables.
- **OCR on a chart** → a flat list of numbers and labels with no structure (which number belongs to which bar?) → bad embedding → bad retrieval.
- A **text embedding model can't embed pixels**. Two problems:
  1. **Representation**: how to turn an image into a vector?
  2. **Retrieval**: a text-query vector must be comparable with that image vector (same space).

## Two strategies
| | Strategy 1: VLM captions ⭐ common | Strategy 2: CLIP (unified embeddings) |
|-|-|-|
| Ingestion | VLM writes a detailed description → embed it like text | CLIP embeds text and images into **one shared space** |
| VLM needed | ingestion **and** generation | generation only |
| Metadata for images | the **image path** (not the description) | the image path |
| Risks | VLM cost / hallucination, prompt quality | query must be phrased like a caption |
| Change to the pipeline | small (prompt + VLM) | swap the embedding model |
- Always send the **original image** (loaded from its path) to a **VLM** at generation time. The description / embedding is only a search key.
- Two separate embedding models (text + image) don't work: their vector spaces aren't comparable.

## CLIP in 3 lines
- Image encoder (ViT) + text encoder, trained **jointly** on ~400M (image, caption) pairs.
- **Contrastive**: pull matching pairs together, push mismatched pairs apart (without the push → **representational collapse**).
- N×N similarity matrix, cross-entropy on rows **and** columns (symmetric) → image→text and text→image both work.

## Loading (both strategies): `unstructured`
```Python
from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(PDF_PATH,
                            mode='elements',            # each element = its own Document
                            strategy='hi_res',          # better image extraction (needs Tesseract installed)
                            extract_images_in_pdf=True)
elements = loader.load()      # 254 elements: NarrativeText, Title, ListItem, Table (7), Image (6), ...

text_docs = []
image_elements = []
for el in elements:
    if el.metadata.get('category') == 'Image':
        image_elements.append(el)              # metadata['image_path'] = the extracted PNG
    else:
        text_docs.append(el)
```
- It OCRs text-heavy figures, but keeps real charts as image files.
- Tables: OCR loses structure. Treating them as images is usually safer. (Here `unstructured`'s table text worked fine.)

## Strategy 1 code (VLM captions)
```Python
import base64
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

vlm = ChatOpenAI(model='gpt-5-mini')       # vision-capable


def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def caption_image(image_path):
    messages = [
        SystemMessage(content=IMAGE_CAPTION_SYSTEM_PROMPT),   # image type, ALL visible text/labels, data points,
                                                              # trends, subject, layout. No vague language.
        HumanMessage(content=[
            {'type': 'text', 'text': 'Describe this image extracted from a document.'},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encode_image(image_path)}'}},
        ]),
    ]
    return vlm.invoke(messages).content


caption_docs = []
for el in image_elements:
    caption_docs.append(Document(page_content=caption_image(el.metadata['image_path']),
                                 metadata=el.metadata))     # keeps image_path + category='Image'

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
all_docs = splitter.split_documents(text_docs + caption_docs)     # 257 text + 20 caption chunks
vector_store = Chroma.from_documents(filter_complex_metadata(all_docs),   # Chroma: only str/int/float/bool metadata
                                     OpenAIEmbeddings(model='text-embedding-3-small'))
retriever = vector_store.as_retriever(search_kwargs={'k': 6})
```

### Build the multimodal prompt
```Python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def build_messages(inputs):
    docs = inputs['docs']
    question = inputs['question']
    text_parts = []
    image_content = []
    seen_paths = set()
    for d in docs:
        if d.metadata.get('category') == 'Image':
            path = d.metadata.get('image_path')
            if path and path not in seen_paths:           # dedupe: many caption chunks → 1 image
                seen_paths.add(path)
                image_content.append({'type': 'image_url',
                                      'image_url': {'url': f'data:image/jpeg;base64,{encode_image(path)}'}})
        else:
            text_parts.append(d.page_content)

    messages = prompt.format_messages(context='\n\n'.join(text_parts), question=question)
    if image_content:
        messages[-1] = HumanMessage(content=[{'type': 'text', 'text': messages[-1].content}] + image_content)
    return messages


chain = ({'docs': retriever, 'question': RunnablePassthrough()}
         | RunnableLambda(build_messages)
         | vlm | StrOutputParser())
chain.invoke('How does Self-CRAG compare with Self-RAG in the line chart?')
```
- Result: correct chart answer (the gap widens as retrieval accuracy drops) and correct table numbers (CRAG 27.2 TFLOPs/token, 0.512 s).

## Strategy 2 code (CLIP)
```Python
from langchain_experimental.open_clip import OpenCLIPEmbeddings

clip = OpenCLIPEmbeddings(model_name='ViT-B-32', checkpoint='laion2b_s34b_b79k')   # small, CPU-friendly
store = Chroma(embedding_function=clip, collection_name='multimodal')

store.add_documents(filter_complex_metadata(text_docs))            # text → CLIP text encoder

image_uris = []
image_metadatas = []
for el in filter_complex_metadata(image_elements):
    image_uris.append(el.metadata['image_path'])
    image_metadatas.append(el.metadata)
store.add_images(uris=image_uris, metadatas=image_metadatas)       # images → CLIP image encoder (page_content = path)

retriever = store.as_retriever(search_kwargs={'k': 6})             # one search finds text AND images
```
- Phrase image queries **like a caption** ("how generation accuracy changes as retrieval accuracy drops...").

## Checklist
- Separate text and images at loading. Don't OCR charts.
- Generation needs a **VLM**. Handle 3 cases: text + images / text only / nothing → "I don't know".
- Image metadata: path + page + source (+ optional short caption).
- Return the **original** text / image, never the search-only description.
