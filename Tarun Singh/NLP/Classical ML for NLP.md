---
tags:
  - NLP
  - machine-learning
  - text-classification
---
## Lesson 05 — NLP with classical ML
- Goal of this lesson: build a **real, usable text classifier** with classical ML, evaluate it properly, and see that Logistic Regression is just a 1-layer neural network in PyTorch
- You need from before: [[Text Pre-processing]] and [[Text Representation]] (TF-IDF)

---

## Do people still use ML for NLP? (Yes, but for specific jobs)
- **Where it is still used in real companies**
	- Spam filters, ticket / email routing ("this is a billing complaint"), simple sentiment, intent detection, language detection
	- When data is small (a few hundred to a few thousand labelled rows)
	- When it must run on CPU, very fast, very cheap (thousands of texts per second on a laptop)
	- When you must explain the decision ("we flagged this because of the words *refund* and *fraud*")
- **The rule every good NLP engineer follows**
	- Always build a **TF-IDF + Logistic Regression baseline first** (takes 5 minutes)
	- Then try a transformer (Part 4). If the transformer is only 1-2% better but 100x slower and costlier, you ship the ML model
- **Where ML is NOT used anymore**
	- Translation, summarization, chatbots, question answering, text generation → all transformers / LLMs now
	- Tasks where word order and context matter a lot ("not bad at all" is positive)

---

## Which ML models actually matter
| Model | Use it? | Why |
|---|---|---|
| **Logistic Regression** | ✅ Main one | Fast, strong, gives probabilities, easy to explain. Your default |
| **Linear SVM** (`LinearSVC`) | ✅ | Often equal or slightly better than LR on text. No probabilities by default |
| **Multinomial Naive Bayes** | ✅ | Super fast, great with very little data, classic for spam |
| Random Forest / XGBoost on TF-IDF | ⚠️ Rarely | TF-IDF has 10k–100k sparse columns, tree models are slow and usually worse here |
| HMM, CRF | ❌ Name only | Old way of POS tagging / NER. Transformers replaced them |
| LDA (topic modelling) | ❌ Name only | Finds "topics" in documents without labels. Rarely used now |

- **Why linear models win on text:** TF-IDF gives a very wide, sparse matrix (many columns, mostly zeros). Linear models are made for exactly this. Each word just gets one weight: positive weight → pushes toward class 1, negative → toward class 0

---

## The full recipe (IMDB sentiment, 50k reviews)
- Dataset: "IMDB Dataset of 50K Movie Reviews" on Kaggle, columns `review`, `sentiment`

### Step 1 — Load and lightly clean
- For ML on TF-IDF, keep cleaning **light**: lowercase, remove HTML and URLs. That is usually enough
- ⚠️ **Do not remove stop words for sentiment.** The stop word list contains *not*, *no*, *nor*. "not good" becomes "good" → wrong meaning
```python
import re
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('IMDB Dataset.csv')
df['label'] = (df['sentiment'] == 'positive').astype(int)   # positive -> 1, negative -> 0

def clean(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)                     # html tags like <br />
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)     # urls
    return text

df['review'] = df['review'].apply(clean)

# stratify -> train and test keep the same positive/negative ratio
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
```

### Step 2 — Build a Pipeline (vectorizer + model together)
- **Why a `Pipeline`?** The vectorizer must learn its vocabulary and IDF values from the **train data only**. If you fit TF-IDF on all data, test information leaks into training (= data leakage, your score looks better than reality). Pipeline does this correctly for you, and you save one object instead of two
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2),   # words + word pairs ("not good")
                              min_df=2,             # drop words seen in < 2 documents (typos, rare names)
                              max_df=0.9,           # drop words in > 90% of documents (useless)
                              sublinear_tf=True)),  # use 1 + log(tf): a word 10 times is not 10x more important
    ('clf', LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)      # raw text goes in directly!
```
- The important `TfidfVectorizer` knobs

| Parameter | What it does | Good start |
|---|---|---|
| `ngram_range` | which N-grams to include | `(1, 2)` |
| `min_df` | ignore very rare words (int = count, float = fraction) | `2` to `5` |
| `max_df` | ignore words that are in almost every document | `0.9` |
| `max_features` | keep only the top N most frequent terms (controls memory) | `50000` |
| `sublinear_tf` | log scaling of counts, almost always helps | `True` |
| `stop_words` | `'english'` removes stop words | leave `None` for sentiment |

### Step 3 — Evaluate (not only accuracy!)
```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
print(confusion_matrix(y_test, y_pred))
```
- On real IMDB you should get **around 89–91% accuracy**. That is a strong number for 1 minute of training on CPU
- **Which metric to look at?**
	- **Accuracy** → only when classes are balanced (IMDB is 50/50, so fine here)
	- **Precision** of a class → "when the model says spam, how often is it really spam?" Care about it when a false alarm is costly (good email going to spam)
	- **Recall** of a class → "of all real spam, how much did we catch?" Care about it when missing one is costly (fraud, toxic content)
	- **F1** → balance of both. Use it (or macro F1 for many classes) when classes are imbalanced
- Confusion matrix: rows = real class, columns = predicted class. Off-diagonal numbers are the mistakes

### Step 4 — Predict on new text
```python
new_reviews = ["What a brilliant movie, I loved it", "Total waste of time. Boring."]
print(model.predict(new_reviews))         # [1 0]
print(model.predict_proba(new_reviews))   # [[P(neg), P(pos)], ...]
```
- Note: the pipeline does the TF-IDF transform for you. But your **`clean()` function is outside** the pipeline, so call it on new text too (or the model sees text in a different form than in training)

### Step 5 — Compare the 3 main models
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

candidates = {
    'Naive Bayes':         MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM':          LinearSVC(),
}

for name, clf in candidates.items():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ('clf', clf),
    ])
    pipe.fit(X_train, y_train)
    print(f"{name:20s} F1 = {f1_score(y_test, pipe.predict(X_test)):.4f}")
```
- Usually: SVM ≈ LR > NB. But NB trains in a blink and is very good with tiny data

### Step 6 — Tune hyperparameters
- In a pipeline, a parameter name is `stepname__paramname` (two underscores)
- `C` in Logistic Regression / SVM = **inverse** of regularization. Small C → stronger regularization (simpler model). Same idea as `weight_decay` in your PyTorch notes, but flipped
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df':      [1, 2, 5],
    'clf__C':             [0.1, 1, 10],
}

search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)        # mean cross-validation F1
best_model = search.best_estimator_
```
- You can also use Optuna here exactly like in your PyTorch notes

### Step 7 — Look inside the model (explainability)
- Logistic Regression gives every word / word pair **one weight**. Sort the weights and you see what the model learned
```python
import numpy as np

vectorizer = best_model.named_steps['tfidf']
clf        = best_model.named_steps['clf']

words   = vectorizer.get_feature_names_out()
weights = clf.coef_[0]                 # shape: (number_of_terms,)
order   = np.argsort(weights)

print("Most NEGATIVE words:", words[order[:15]])
print("Most POSITIVE words:", words[order[-15:]])
```
- You will see things like *worst, waste, awful, boring* on one side and *excellent, great, perfect, loved* on the other. If you see junk (like a movie name), that tells you something about your data

### Step 8 — Save and load
```python
import joblib

joblib.dump(best_model, 'sentiment_tfidf_lr.joblib')     # saves vectorizer + model together
loaded = joblib.load('sentiment_tfidf_lr.joblib')
loaded.predict([clean("this film was amazing")])
```

---

## Bridge to deep learning: the same model in PyTorch
- **Big idea:** Logistic Regression = `nn.Linear(n_features, 1)` + sigmoid + `BCEWithLogitsLoss`. That's it. It is a neural network with **zero hidden layers**
- So the TF-IDF matrix is just the input features `X`, exactly like the tabular data in your cheat sheet
- One problem: the TF-IDF matrix is **sparse** (mostly zeros). 40,000 rows × 20,000 columns as dense float32 = 3.2 GB. So we keep it sparse and only make **one row dense** inside `__getitem__`
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. features (fit on train only!)
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
X_train_sparse = tfidf.fit_transform(X_train)     # scipy sparse matrix
X_test_sparse  = tfidf.transform(X_test)

# 2. dataset: keep sparse, make one row dense when asked
class TfidfDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X = X_sparse
        self.y = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        row = torch.tensor(self.X[i].toarray()[0], dtype=torch.float32)
        return row, self.y[i]

train_loader = DataLoader(TfidfDataset(X_train_sparse, y_train), batch_size=64,  shuffle=True)
test_loader  = DataLoader(TfidfDataset(X_test_sparse,  y_test),  batch_size=256, shuffle=False)

# 3. model = logistic regression
class LogisticRegressionTorch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)     # one weight per word + one bias
    def forward(self, x):
        return self.linear(x)                      # raw logit, no sigmoid here

model_pt  = LogisticRegressionTorch(X_train_sparse.shape[1]).to(device)
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_pt.parameters(), lr=1e-2, weight_decay=1e-5)  # weight_decay ~ the "C" of sklearn

# 4. train + evaluate (your normal loop)
for epoch in range(10):
    model_pt.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model_pt(x).squeeze(1)     # (B, 1) -> (B,) to match y
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model_pt.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = (torch.sigmoid(model_pt(x).squeeze(1)) > 0.5).float()
            correct += (preds == y).sum().item()
    print(f"epoch {epoch+1} | loss {total_loss/len(train_loader):.4f} | "
          f"test acc {correct/len(test_loader.dataset):.4f}")
```
- You should land very close to the sklearn number. Same model, different library
- **Want an ANN instead?** Just change the model. Everything else stays the same
```python
class TfidfMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),          # high dropout: 20k inputs overfit fast
            nn.Linear(256, 1),
        )
    def forward(self, x):
        return self.net(x)
```
- Honest result: on TF-IDF features, an MLP is usually only a little better than LR. **Why?** TF-IDF already threw away word order and meaning. The model can't learn what the input doesn't contain. That is exactly why Part 3 changes the **input** (embeddings instead of TF-IDF), not just the model

---

## Multi-class? Same code
- For 3+ classes (e.g. sports / politics / tech) the sklearn code is **identical**, `LogisticRegression` handles it automatically
- In PyTorch: `nn.Linear(n_features, num_classes)` + `nn.CrossEntropyLoss()` + labels as `torch.long` (as in your cheat sheet)
- Imbalanced classes: `LogisticRegression(class_weight='balanced')`, and look at macro F1, not accuracy

---

## Key takeaways
- TF-IDF + Logistic Regression is the **baseline you always build first**
- Use a `Pipeline` so the vectorizer only learns from train data
- Light cleaning only, and don't remove *not* for sentiment
- Look at precision / recall / F1, not only accuracy
- Logistic Regression is a 0-hidden-layer neural network. The limit is the **input** (TF-IDF loses order and meaning), which is the motivation for embeddings in the next part

---

## Exercise (do it, 30–45 min)
1. Download the IMDB dataset from Kaggle and run Steps 1–8. Write down your test accuracy
2. Run it once with `stop_words='english'`. Does it get better or worse? Why? (Hint: print whether `'not'` is in `vectorizer.get_feature_names_out()`)
3. Change `ngram_range` to `(1, 1)` and compare. What do bigrams add?
4. Run the PyTorch version and compare its accuracy to sklearn's

---
- Previous: [[Text Representation]] · Next: Text → Tensors in PyTorch (lesson 06, coming next) · Roadmap: [[NLP Course Roadmap]]
