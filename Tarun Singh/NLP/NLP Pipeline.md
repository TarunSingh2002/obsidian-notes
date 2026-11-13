---
tags:
  - NLP
  - pipeline
---
## Basics
- The NLP pipeline is the set of steps followed to build end-to-end NLP software
- Steps
	1. Data acquisition
	2. Text preparation
	3. Feature engineering
	4. Modelling
	5. Deployment
- Points to remember
	- It's not universal
	- DL pipelines are slightly different
	- The pipeline is not linear (you often go back to earlier steps)

---

## 1. Data Acquisition
There could be 3 scenarios:
- **You have the data**
	- Data is in a CSV file
	- Data is in a database → ask the data engineer
	- You have very little data → use data augmentation
- **Others have the data**
	- Public dataset
	- Web scraping
	- Fetch data from an API
- **No one has the data**

---

## 2. Text Preparation
- **Text cleanup**
	- Remove spelling mistakes
	- HTML tag cleaning
	- Change emojis to machine-understandable text
- **Basic pre-processing**
	- Basics
		- Tokenization
			- Sentence tokenization
			- Word tokenization
	- Optional
		- Stop word removal
		- Stemming
		- Removing digits, punctuation
		- Changing text to upper or lower case
		- Language detection (used if you're making a translator)
- **Advanced pre-processing**
	- Parts of speech tagging
	- Parsing
	- Coreference resolution
- Details: [[Text Pre-processing]]

---

## 3. Feature Engineering
- Encoding the words to numbers
- Details: [[Text Representation]]

---

## 4. Modelling
- **Model building**
	- Heuristic approaches → used if data is very little
	- ML algos → used if data is little
		- Sometimes ML algos are used together with a heuristic-based approach
	- DL algos → used if data is large
		- Transfer learning is also used here
	- Cloud API
- **Evaluation**
	- Intrinsic evaluation
	- Extrinsic evaluation

---

## 5. Deployment
- Deployment
- Monitoring
- Model update

---
- Previous: [[Intro to NLP]]
