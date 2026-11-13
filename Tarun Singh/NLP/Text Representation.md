---
tags:
  - NLP
  - text-representation
  - word-embeddings
---
## Techniques
- One Hot Encoding (OHE)
- Bag of Words (BoW)
- N-grams (Uni-grams, Bi-grams, ...)
- TF-IDF
- Word2Vec

---

## Basic Terminology
- **Corpus (C)**: all the words combined together
- **Vocabulary (V)**: the unique words present in the dataset
- **Document (D)**: a single row of text data
- **Word (W)**

### Example data

| Document | Text                    |
| -------- | ----------------------- |
| D1       | people watch campusx    |
| D2       | campusx watch campusx   |
| D3       | people write comment    |
| D4       | campusx write comment   |

- Corpus = `people watch campusx campusx watch campusx people write comment campusx write comment`
- Vocabulary = `people watch campusx write comment` → V = 5

---

## One Hot Encoding (OHE)
- Converts each word into a V-dimensional vector (V = vocabulary size)
- Not used in practice
- Example: D1 = "people watch campusx", V = 5
	- OHE for the vocabulary
		- ![[NLP - ohe vocabulary.png]]
	- OHE representation of D1
		- ![[NLP - ohe D1.png]]
- **Advantages**
	- Easy to implement
- **Disadvantages**
	- Sparsity
	- A fixed input size is required
	- At prediction, if a new word comes, it breaks the model. This problem is called **OOV (out of vocabulary)**
	- Doesn't capture semantic meaning

---

## Bag of Words (BoW)
- Works best for text classification problems
- **Working**
	1. First, create the vocabulary
	2. Then, for each row, store the occurrence count of each word
	- ![[NLP - bag of words example.png]]
- **Core idea**
	- Texts of the same category have the same words with similar frequency
	- The order of words doesn't matter here
	- Word context doesn't matter, but it's still able to capture some semantic meaning
- **Code**
	```python
	from sklearn.feature_extraction.text import CountVectorizer

	vectorizer = CountVectorizer()

	X_train_bow = vectorizer.fit_transform(df['c1']).toarray()
	```
- **Advantages**
	- Simple and intuitive
	- OOV resolved: it only looks for vocabulary words in the test data, so new words in the test data are simply ignored
	- Able to capture semantic meaning (a little bit, not very much)
		- Each data point acts as a vector, and similar vectors are closer to each other
	- No need for a fixed input size
- **Disadvantages**
	- Sparsity
	- Ordering is ignored
	- Semantic meaning isn't reliable
		- Example: "i like food" and "i dont like food" share most words, so their vectors are very close, but they mean opposite things
- **Hyperparameters**
	- `binary`: bool, default `False`
		- If `True`, all non-zero counts are set to 1
		- `binary=True` works better for sentiment analysis
	- `max_features`: int, default `None`
		- Used to remove rare words
		- Ex → `max_features=10` keeps only the 10 most frequent words in the corpus

---

## N-grams
- Here the vocabulary is made of multiple consecutive words. The rest is similar to Bag of Words
- Note: Bag of Words is a special case of N-grams → `ngram_range=(1, 1)` = unigrams = Bag of Words
- **Types of N-grams**
	- **Unigrams (1-gram)**: single words. "I love NLP" → `["I", "love", "NLP"]`
	- **Bigrams (2-gram)**: pairs of consecutive words. "I love NLP" → `["I love", "love NLP"]`
	- **Trigrams (3-gram)**: triplets of consecutive words. "I love NLP" → `["I love NLP"]`
	- **N-grams (n-gram)**: any sequence of n consecutive words
- **Code**
	```python
	from sklearn.feature_extraction.text import CountVectorizer

	vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=5000)

	X_train_ngm = vectorizer.fit_transform(df['c1']).toarray()
	```
- **Hyperparameter**
	- `ngram_range`: default `(1, 1)`
		- `(1, 2)` means the vocabulary includes both single words and word pairs (unigrams + bigrams)
- **Advantages**
	- Captures semantic meaning better than OHE and Bag of Words
- **Disadvantages**
	- Sparsity
	- Not efficient for large datasets (the vocabulary grows very fast)
	- Still can't capture semantic meaning properly

---

## TF-IDF (Term Frequency - Inverse Document Frequency)
- Gives different weightage to different words
- Everything stays the same as Bag of Words, but instead of the raw frequency, we calculate the TF-IDF value for each word
- **Components**
	- **Term Frequency (TF)**: measures how frequently a term appears in a document
		$$TF(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$
	- **Inverse Document Frequency (IDF)**: measures how important a term is
		$$IDF(t) = \log\left(\frac{\text{Total number of documents in corpus } D}{\text{Number of documents containing term } t}\right)$$
		- Why take the log?
			- Without the log, using just N/n: for a very rare word, IDF = 1000/1 = 1000
			- TF is always between 0 and 1
			- So in TF × IDF, the huge IDF would dominate and TF would have no significant impact. The log scales IDF down
	- **TF-IDF = TF × IDF**
- **Code**
	```python
	from sklearn.feature_extraction.text import TfidfVectorizer

	vectorizer = TfidfVectorizer()

	x_train_tfidf = vectorizer.fit_transform(df['c1']).toarray()
	```
- **Disadvantages**
	- Sparsity
	- Can't capture semantic meaning properly

---

## Word2Vec
- A deep-learning-based word embedding technique
- **Advantages**
	- Captures the semantic meaning of words
	- The word vectors are **dense** and small in dimension
- **Intuition**
	- Word2Vec uses a shallow neural network to create features (like 300), then gives each word a numeric value for each feature. Based on these values, it decides how similar words are
	- Once we have the vector for each word, a sentence's vector is built by combining (averaging) the vectors of its words
- **Ways to use it**
	- Use a pre-trained model → if your data and the pre-trained data share ~80% of their vocabulary
	- Train a Word2Vec model on your own dataset → if you have enough data, this will outperform pre-trained models
- **Types**
	- **CBOW (Continuous Bag of Words)**
		- The model predicts the target word from its context words. E.g. given the context "The cat sat on the ___", it predicts "mat"
		- Used when data is small
	- **Skip-gram**
		- The model predicts the context words from a target word. E.g. given "cat", it predicts words like "the", "sat", "on"
		- Used when data is large

### Code: train your own model
- Install and import
	```python
	!pip install gensim

	import gensim
	from nltk import sent_tokenize
	from gensim.utils import simple_preprocess
	from gensim.models import Word2Vec
	```
- Dataset used (IMDB reviews)
	- ![[NLP - imdb review dataset.png]]
- Model creation
	```python
	# CBOW
	model = Word2Vec(vector_size=300, window=5, min_count=1, sg=0)
	# Skip-gram
	model = Word2Vec(vector_size=300, window=5, min_count=1, sg=1)

	# 'vector_size' is the size of the word vectors
	# 'window' is the maximum distance between the current and predicted word within a sentence
	# 'min_count' ignores all words with total frequency lower than this
	# 'sg' defines the training algorithm: 0 for CBOW, 1 for Skip-gram
	```
- Tokenization
	```python
	story = []
	for doc in df['review']:
	    raw_sent = sent_tokenize(doc)
	    for sent in raw_sent:
	        story.append(simple_preprocess(sent))
	```
- Create the vocabulary from the data
	```python
	model.build_vocab(story)
	```
- Train
	```python
	model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
	```
- Turn each document into one vector (average of its word vectors)
	```python
	import numpy as np
	from tqdm import tqdm

	def document_vector(doc):
	    # remove out-of-vocabulary words
	    doc = [word for word in doc.split() if word in model.wv.index_to_key]
	    return np.mean(model.wv[doc], axis=0)

	X = []
	for doc in tqdm(df['review'].values):
	    X.append(document_vector(doc))

	X = np.array(X)
	```

### Code: pre-trained model
```python
import gensim
from gensim.models import Word2Vec, KeyedVectors

!pip install wget
!wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)
```

---
- Previous: [[Text Pre-processing]]
