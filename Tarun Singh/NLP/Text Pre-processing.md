---
tags:
  - NLP
  - text-preprocessing
---
## Lowercasing
- On a column
	```python
	df['c1'] = df['c1'].str.lower()
	```

---

## Removing
- To remove something, we use regular expressions
- Use a regex builder tool to make regular expressions
- **HTML tags**
	```python
	import re

	def remove_html_tags(text):
	    pattern = re.compile('<.*?>')
	    return pattern.sub(r'', text)

	df['c1'] = df['c1'].apply(remove_html_tags)
	```
- **URLs**
	```python
	def remove_url(text):
	    pattern = re.compile(r'https?://\S+|www\.\S+')
	    return pattern.sub(r'', text)

	df['review'] = df['review'].apply(remove_url)
	```

---

## Removing Punctuation
- Punctuation characters (`string.punctuation`)
	```
	!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
	```
- Code
	```python
	import string

	exclude = string.punctuation

	def remove_punc(text):
	    return text.translate(str.maketrans('', '', exclude))

	df['c1'] = df['c1'].apply(remove_punc)
	```

---

## Chat Word Treatment
- Expand chat slang into full words (e.g. *gn* → *good night*, *imo* → *in my opinion*)

---

## Spelling Correction
```python
from textblob import TextBlob

def spelling_correction(text):
    textBlb = TextBlob(text)
    return textBlb.correct().string

df['c1'] = df['c1'].apply(spelling_correction)
```

---

## Removing Stop Words
- If you want to do parts of speech tagging, then **don't** remove stop words
```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word not in stop_words:
            new_text.append(word)
    return " ".join(new_text)

df['c1'] = df['c1'].apply(remove_stopwords)
```

---

## Handling Emojis
There are 2 ways:
- **Remove them**
	```python
	import re

	def remove_emoji(text):
	    emoji_pattern = re.compile("["
	                               u"\U0001F600-\U0001F64F"  # emoticons
	                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
	                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	                               u"\U00002702-\U000027B0"
	                               u"\U000024C2-\U0001F251"
	                               "]+", flags=re.UNICODE)
	    return emoji_pattern.sub(r'', text)

	df['c1'] = df['c1'].apply(remove_emoji)
	```
- **Replace them with their meaning**
	```python
	import emoji
	import re

	def replace_emojis_with_meanings(text):
	    # First, demojize the text to get :emoji_name: format
	    demojized_text = emoji.demojize(text)

	    # Use regex to find all :emoji_name: patterns and replace them with readable text
	    def convert_match(match):
	        # Get the matched emoji name without colons
	        emoji_name = match.group(1)
	        # Replace underscores with spaces
	        readable_name = emoji_name.replace('_', ' ')
	        return readable_name

	    # Replace all :emoji_name: with readable text
	    readable_text = re.sub(r':([a-zA-Z0-9_]+):', convert_match, demojized_text)
	    return readable_text

	df['c1'] = df['c1'].apply(replace_emojis_with_meanings)
	```

---

## Stemming and Lemmatization

### Stemming
- Stemming is the process of reducing inflection in words to their root form, mapping a group of words to the same stem even if the stem itself is not a valid word in the language
	- walking, walks, walked → walk
- That's why we don't use stemming when the text has to be shown to the user. In that case we use **lemmatization**
- Code
	```python
	from nltk.stem.porter import PorterStemmer

	ps = PorterStemmer()

	def stem_words(text):
	    return " ".join([ps.stem(word) for word in text.split()])

	df['c1'] = df['c1'].apply(stem_words)
	```

### Lemmatization
- Lemmatization also gets the root of a word, but it makes sure the root word is a valid word in the language
- Lemmatization is slower than stemming
- Code
	```python
	import nltk
	from nltk.stem import WordNetLemmatizer

	wordnet_lemmatizer = WordNetLemmatizer()

	sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."
	punctuations = "?:!.,;"
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [word for word in sentence_words if word not in punctuations]

	print("{0:20}{1:20}".format("Word", "Lemma"))
	for word in sentence_words:
	    print("{0:20}{1:20}".format(word, wordnet_lemmatizer.lemmatize(word, pos='v')))
	```

---

## POS Tagging
- **Theory**
	- Part-of-speech (POS) tagging labels the words in a sentence with their part-of-speech category, such as nouns, verbs, adjectives, adverbs, etc.
	- ML techniques used for POS tagging: Hidden Markov Models or Conditional Random Fields
- **Used in**
	- Named entity recognition
	- Chatbots
	- Word sense disambiguation
		- Example: "I ran to the store because I ran out of milk" → here *ran* has 2 meanings
- **Code**
	```python
	!pip install spacy

	import spacy

	nlp = spacy.load('en_core_web_sm')

	doc = nlp(u"I will google about facebook")

	doc.text            # 'I will google about facebook'
	doc[-1]             # facebook
	doc[2].pos_         # 'VERB'
	doc[2].tag_         # 'VB'
	spacy.explain('VB') # 'verb, base form'
	```

---

## Tokenization
- Breaking a text document into smaller parts. These can be words, sentences, etc. (mostly words)
- Code
	```python
	import spacy

	nlp = spacy.load('en_core_web_sm')

	doc4 = nlp("I am going to visit delhi!")

	for token in doc4:
	    print(token)
	```

---
- Part of: [[NLP Pipeline]] · Next: [[Text Representation]]
