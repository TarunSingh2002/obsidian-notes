---
tags:
  - AIEngineer
  - rag
  - graph-rag
  - neo4j
---

# 14 - Graph RAG

> Lessons: Graph RAG Theory · Graph RAG Code
> Code: `09_graph_rag/` (`01_ingestion.ipynb`, `02_retrieval.ipynb`), Neo4j AuraDB Free

## Why
- "Which company of Elon Musk is headquartered in Austin?" = **multi-hop** (Musk → companies → HQ).
- Traditional RAG: chunks are **independent**, scored one by one → it can't stitch facts across chunks. Structural, can't be tuned away.
- Graph = **nodes (entities)** + **edges (relationships)**. Relationships are stored explicitly → walk them (**hops**).

| | Vector DB | Graph DB (Neo4j) |
|-|-|-|
| Stores | chunk text + embedding + metadata | nodes (text, type, source chunk, embedding) + edges (type, start id, end id) |
| Relationships | ❌ | ✅ explicit |
| Query language | similarity search | **Cypher** (the LLM writes it for you) |

## Ingestion
```
load → split → LLM extracts entities + relationships per chunk (NER-like, temp 0) → build graph in Neo4j → embed nodes
```

## Retrieval (theory)
1. LLM extracts entities from the query (Elon Musk, Austin), with the **graph schema** as context.
2. Embed the entity → semantic search over node embeddings → **start node**.
3. **Traverse** (BFS preferred: broad coverage): Musk → founded → Tesla / SpaceX / xAI → HQ → Austin / Hawthorne.
4. Result → plain-text context → prompt → LLM: "Tesla".
- Semantic search finds **where to start**, traversal finds **what to collect**.

## Setup (Neo4j Aura)
- Create an AuraDB Free instance → **save the password (shown once)**. Private data → run Neo4j in Docker locally.
- `.env`: `NEO4J_URI=neo4j+s://...` (not `bolt://`), `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `OPENAI_API_KEY`.
- Test in the Aura console: `MATCH (n) RETURN count(n)` → 0 on an empty DB.

## Ingestion code
```Python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector

load_dotenv()
llm = ChatOpenAI(model='gpt-5-mini', temperature=0)          # deterministic extraction
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

docs = PyPDFLoader('data/elon_musk.pdf').load()
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)  # 14 (small = tighter extraction)

graph = Neo4jGraph(url=os.environ['NEO4J_URI'],
                   username=os.environ['NEO4J_USERNAME'],
                   password=os.environ['NEO4J_PASSWORD'])

graph_docs = LLMGraphTransformer(llm=llm).convert_to_graph_documents(chunks)   # ~7 min for 14 chunks!
# e.g. (Elon Musk)-[BORN_IN]->(Pretoria, South Africa), (Elon Musk)-[HAS_OCCUPATION]->(Engineer)

graph.add_graph_documents(graph_docs,
                          include_source=True,       # chunk = Document node, linked by MENTIONS edges
                          baseEntityLabel=True)      # extra __Entity__ label (indexed, faster)
# demo: 99 nodes, 227 relationships

vector_index = Neo4jVector.from_existing_graph(embedding=embeddings,
                                               url=os.environ['NEO4J_URI'],
                                               username=os.environ['NEO4J_USERNAME'],
                                               password=os.environ['NEO4J_PASSWORD'],
                                               index_name='elon_musk_chunks',
                                               node_label='Document',
                                               text_node_properties=['text'],
                                               embedding_node_property='embedding')
```

## Retrieval code (GraphCypherQAChain)
- Schema → LLM writes Cypher → runs it on Neo4j → QA prompt turns the result into an answer.
```Python
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

graph.refresh_schema()            # labels + relationship patterns, e.g. (:Organization)-[:HEADQUARTERED_IN]->(:Location)

CYPHER_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or node labels that are not provided.
Schema:
{schema}

Cypher syntax rules:
1. When matching multiple relationship types with |, only the FIRST type gets a colon prefix:
   Correct: (n)-[:TYPE1|TYPE2|TYPE3]->(m)
2. When a node could have one of several labels, use label union syntax: MATCH (n:Organization|Company)
3. Organization and place names are stored title-cased. Always compare case-insensitively:
   WHERE toLower(n.id) = toLower("SpaceX")
4. Person names may be partial (e.g. "Musk"). Match with CONTAINS:
   WHERE toLower(p.id) CONTAINS 'elon' AND toLower(p.id) CONTAINS 'musk'

Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_chain = GraphCypherQAChain.from_llm(llm=llm,
                                           graph=graph,
                                           verbose=True,                      # shows generated Cypher + context
                                           allow_dangerous_requests=True,     # required
                                           cypher_prompt=PromptTemplate(input_variables=['schema', 'question'],
                                                                        template=CYPHER_TEMPLATE))
print(cypher_chain.invoke({'query': 'Where is Spacex HQ located?'})['result'])   # Hawthorne, California
```
- The custom prompt exists because the default one broke on Neo4j 5+ syntax and on title-cased IDs (`SpaceX` → `Spacex`).

| Multi-hop question | Path | Answer |
|-|-|-|
| Product released by the org whose board Musk left? | Person-LEFT_BOARD→Org-RELEASED→Product | ChatGPT |
| Where is Musk's partner Shivon Zilis a director? | Person-PARTNER-Person-DIRECTOR→Org | Neuralink |
| City of the solar company Tesla acquired? | Org-ACQUIRED→Org-HEADQUARTERED_IN→Location | San Mateo |
```
MATCH (p:Person)-[:LEFT_BOARD]->(org:Organization)-[:RELEASED]->(prod:Product)
WHERE toLower(p.id) CONTAINS 'elon' AND toLower(p.id) CONTAINS 'musk'
RETURN DISTINCT prod.id AS product
```
- Note: the demo finds start nodes by **text match in Cypher**, not by embedding search (the vector index isn't used in retrieval).

## Limitations / when to use
- Cost: 1 LLM call per chunk + 1 embedding per entity. Grows linearly with the corpus.
- Graph quality = extraction quality (errors compound into wrong answers).
- Fails on simple / general queries (no entities to extract).

| Use Graph RAG | Use traditional RAG |
|-|-|
| interconnected entities, multi-hop, cross-doc relationships | plain text, similarity is enough, independent docs, cheap / fast ingestion |

## Rules
- Pick the store by the **shape of the question**.
- Prototype on a small slice, **inspect extracted nodes / edges** before scaling.
- Give the LLM the schema, read the generated Cypher (`verbose=True`).
