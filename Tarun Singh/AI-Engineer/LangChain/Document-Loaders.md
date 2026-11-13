---
tags:
  - AIEngineer
  - langchain
  - document-loaders
---

# Document Loaders

- Most Popular Document loaders
    - TextLoader
    - PyPDFLoader
    - WebBaseLoader
    - CSVLoader
- Document loaders are components in LangChain used to load data from various sources into a standardized format (usually as Document objects), which can then be used for chunking, embeddin , retrieval, and generation.
- The Datatype of the document is list
- A sample Document format 

```
Document(
    page_content = "The actual text content",
    matadata={"source": "filename.pdf",...}
)
```

- Lazy Load vs Load

![[langchain-document-loaders-4.png]]

## TextLoader

- TextLoader is a simple and commonly used document loader in LangChain that reads plain text (.txt) files and converts them into LangChain Document objects.

```Python
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()  # loader.lazy_load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))
```

## PyPDFLoader

- not very good with scanned PDF or complex layout Pdf

- better alternative for pdf loader
    - pdf with tables/columns - PDFPlumberLoader
    - scanned/image pdfs - unstructuredPDFLoader or AmazonTextractPDFLoader
    - Image data/layouts - PyMuPDFLoader
    - want best structure extraction - UnstructuredPDFLoader

```Python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load() # loader.lazy_load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)
```

## DirectoryLoader

- DirectoryLoader is a document loader that lets you load multiple documents from a directory
(folder) of files.
- glob
    - "**/*.txt" -> All .txt file 
    - "*.pdf" -> ALL pdf file
    - "data/*.csv" -> ALL .csv files in the data/ folder
    - "**/*" -> all files

```Python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()  # loader.lazy_load()

for document in docs:
    print(document.metadata)
```

## WebBaseLoader

- WebBaseLoader is a document loader in LangChain used to load and extract text content from web pages (URLs).
- It uses BeautifulSoup under the hood to parse HTML and extract visible text.
- When to Use:
    - For blogs, news articles, or public websites where the content is primarily text-based and static.
- Limitations:
    - Doesn't handle JavaScript-heavy pages well (use SeleniumURLLoader for that).
    - Loads only static content (w at s in the HTML, not what loads after the page renders).

```Python
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url1 = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
url2 = "https://www.flipkart.com/"
url = [url1, url2]
loader = WebBaseLoader(url)

docs = loader.load() # loader.lazy_load()


chain = prompt | model | parser

print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))
```

## CSVLoader

- CSVLoader is a document loader used to load CSV files into LangChain Document obJects — one per row. by default.

```Python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')

docs = loader.load() # loader.lazy_load()

print(len(docs))
print(docs[1])
```
