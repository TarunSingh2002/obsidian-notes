---
tags:
  - deepLearning
  - pytorch
  - transformer
---
# Making a Transformer from Scratch with PyTorch

> [!info] How this note works
> One chapter at a time. Each chapter = **idea → tiny PyTorch code → exercise** you run in `transformer.ipynb`.
> Everything builds only on [[CheatSheet-PyTorch]].
> We build a **GPT style** model (reads text, predicts the next character), train it, and make it write.

## Chapter map

| Ch | Topic | After it you can |
| -- | ----- | ---------------- |
| 1 | The big picture | explain what a transformer does, why it replaced RNNs, and its 3 families |
| 2 | Text → numbers: encoding techniques | one-hot, bag of words, TF-IDF, word / char / subword tokenizers (BPE, WordPiece, SentencePiece) |
| 3 | Embeddings (`nn.Embedding`) | turn each token id into a learnable vector |
| 4 | Sequence data and the `(B, T, C)` shape | build `(input, target)` batches for next-token prediction |
| 5 | Baseline model (bigram) + training loop | train a model that looks at only 1 previous character |
| 6 | Dot product = similarity, softmax = weights | measure how related two vectors are |
| 7 | Attention by hand + the causal mask | let each token average over the tokens before it |
| 8 | Self-attention head (Query, Key, Value) | write one attention head as an `nn.Module` |
| 9 | Multi-head attention | run several heads in parallel and combine them |
| 10 | Positional encoding | tell the model *where* each token sits |
| 11 | Feed-forward, residual connections, LayerNorm | understand the "glue" around attention |
| 12 | The Transformer block | stack it all into one reusable block |
| 13 | Full model + training on real text | train your own mini GPT |
| 14 | Generating text (sampling, temperature) | make it write new text |
| 15 | Encoder vs decoder, cross-attention | see how BERT, GPT and the original transformer differ |

> [!tip] After this note
> BERT/GPT variants → Hugging Face fine-tuning → distillation → LoRA.

---

## Chapter 1 — The Big Picture

### 1.1 What problem does a transformer solve?
- A transformer is a neural network that takes a **sequence** (words, characters, image patches, audio frames…) and produces a sequence or a prediction.
- Its most famous job is **language modelling**: given some text, predict the **next token**.

```
"I love"          ──▶  model  ──▶  "pizza"   (most likely)
"I love pizza"    ──▶  model  ──▶  "<end>"
```

> [!abstract] The one sentence to remember
> ChatGPT is a transformer doing **next-token prediction in a loop**: predict one token, append it to the input, predict again.

- **Next-token prediction is just multi-class classification**, which you already know:
	- classes = every token in the vocabulary (e.g. ~50,000 classes for GPT-2)
	- last layer outputs **logits** of shape `[vocab_size]`
	- loss = `nn.CrossEntropyLoss()` with the true next token as the class index
	- prediction = `softmax` → pick a token

### 1.2 Why not a normal ANN?
Your ANN from the cheat sheet takes a **fixed** number of features (e.g. 30 columns). Text breaks that:

| Problem with text | Why an ANN struggles |
| ----------------- | -------------------- |
| Variable length | "Hi" is 1 word, a paragraph is 200 |
| Order matters | "dog bites man" ≠ "man bites dog" |
| Context changes meaning | "bank" of a river vs a money "bank" |
| Long-range links | "The **keys** that I left on the table **are** gone" → "are" depends on "keys", 7 words earlier |

### 1.3 Before transformers: RNNs and LSTMs
- An **RNN** reads one token at a time and carries a "memory" vector (hidden state) forward.

```
 h0 ─▶[RNN]─h1─▶[RNN]─h2─▶[RNN]─h3─▶ ...
        ↑         ↑         ↑
      "The"     "cat"     "sat"
```

- Two big problems:
	1. **Sequential = slow.** Step 3 needs h2, which needs h1. You cannot process the words in parallel, so GPUs sit idle.
	2. **Forgets long context.** Everything must squeeze through one hidden vector, and gradients shrink as they travel back through many steps (**vanishing gradients**). LSTM/GRU reduce this but do not remove it.

### 1.4 The breakthrough: "Attention Is All You Need" (2017)
- Paper by Vaswani et al. (Google). Idea: **drop recurrence completely, use only attention.**
- **Attention** = every token looks at the other tokens directly and pulls in the information it needs.

```
"The animal didn't cross the street because it was too tired"

 "it" looks at every word and scores how relevant each one is:
   animal ████████████  0.62
   tired  ███           0.11
   street ██            0.08
   ...    (scores sum to 1 -> that is a softmax)
 "it" becomes a weighted mix of those words -> now "it" knows it means the animal
```

- What this fixes:

| | RNN / LSTM | Transformer |
| - | ---------- | ----------- |
| Processing | one token after another | **all tokens at once** (parallel, GPU friendly) |
| Path between 2 tokens | n steps apart | **1 step** (direct connection) |
| Long-range memory | fades | strong (up to the context length) |
| Cost vs length `T` | grows **linearly** | grows **quadratically**, `O(T²)`, because every token looks at every token |

> [!warning] The trade-off
> The `O(T²)` cost is why every model has a **maximum context length** (e.g. GPT-2: 1024 tokens). Doubling the input makes attention ~4× more expensive.

### 1.5 The whole pipeline (GPT style)
This is the map for the rest of the note. Every box is a chapter.

```
 "I love pi"                                   raw text
     │
 [Tokenizer]                      Ch 2          text -> token ids
     │   [12, 54, 7, 9]
 [Token Embedding]                Ch 3          id -> vector of size d_model
     +
 [Positional Encoding]            Ch 10         adds "where am I in the sentence"
     │   shape (T, d_model)
 ┌───▼──────────────────────────┐
 │  Masked Multi-Head Attention │ Ch 6-9        tokens talk to each other
 │  + Add & LayerNorm           │ Ch 11
 │  Feed-Forward (MLP)          │ Ch 11         each token thinks on its own
 │  + Add & LayerNorm           │
 └───┬──────────────────────────┘  × N blocks   Ch 12
     │
 [Linear  d_model -> vocab_size]                logits for every possible next token
     │
 [Softmax]                                      probabilities
     │
   "z"   (next token)                           Ch 13-14
```

> [!abstract] Two kinds of work inside a block
> **Attention** = tokens exchange information *with each other* (communication).
> **Feed-forward** = each token processes what it collected *on its own* (computation).
> A block does both; we stack many blocks.

### 1.6 Training vs generating
- **Training is parallel.** One sentence gives many training examples at once, because at every position the target is simply the next token:

```
 tokens :  I    love   pizza  <end>
 input  :  I    love   pizza
 target :  love pizza  <end>

 position 1: sees "I"              -> must predict "love"
 position 2: sees "I love"         -> must predict "pizza"
 position 3: sees "I love pizza"   -> must predict "<end>"
```

- A **causal mask** stops position 2 from peeking at "pizza" (that would be cheating). Ch 7.
- Feeding the real previous tokens during training (not the model's own guesses) is called **teacher forcing**.
- **Generating is sequential (autoregressive).** Predict 1 token → append → feed everything back in → repeat until `<end>` or a length limit.

### 1.7 The three families

| Family | Sees | Trained to | Good at | Examples |
| ------ | ---- | ---------- | ------- | -------- |
| **Encoder-only** | whole sentence, both directions | fill in masked words ("I [MASK] pizza") | understanding: classification, NER, search embeddings | BERT, RoBERTa, DistilBERT |
| **Decoder-only** | only the past (causal mask) | predict the next token | generating text, chat, code | GPT-2/3/4, LLaMA, Claude |
| **Encoder-decoder** | encoder reads the input fully, decoder writes the output | map one sequence to another | translation, summarisation | original Transformer, T5, BART |


### 1.8 Vocabulary you will see everywhere

| Term | Meaning | Original paper (base) |
| ---- | ------- | --------------------- |
| **token** | one unit of text (word, sub-word or character) | — |
| **vocab_size** | number of distinct tokens the model knows | ~37,000 |
| **context length / block_size** (`T`) | max tokens the model sees at once | — |
| **d_model** (`C`, embedding dim) | size of the vector that represents each token | 512 |
| **head** | one independent attention calculation | 8 heads |
| **d_k** | size of each head = `d_model / n_heads` | 64 |
| **d_ff** | hidden size inside the feed-forward layer (usually `4 × d_model`) | 2048 |
| **N / n_layers** | number of stacked blocks | 6 |
| **logits** | raw scores over the vocab before softmax | — |

> [!tip] Where the parameters live
> Almost all parameters are in `nn.Linear` layers (attention projections + feed-forward) and the embedding table. A transformer is mostly **matmuls + softmax + normalization**, things you have already used.

### 1.9 Key takeaways
- A transformer turns tokens into vectors, lets them **exchange information through attention**, processes each one with a small MLP, repeats N times, then classifies the next token.
- It beat RNNs because it is **parallel** and connects any two tokens **directly**; the price is `O(T²)` attention and a fixed context length.
- Training: all positions in parallel with a causal mask. Generation: one token at a time.
- Encoder-only (BERT) understands, decoder-only (GPT) generates, encoder-decoder (T5) transforms one sequence into another.


---

## Chapter 2 — Text → Numbers (Encoding Techniques)

A neural network only does maths on tensors, so text must become numbers first. That is **two separate questions**:

| Question | Name | Example |
| -------- | ---- | ------- |
| 1. How do I **split** text into units? | **tokenization** | `"unhappy"` → `["un", "happy"]` |
| 2. How do I **represent** each unit (or a whole document) as numbers? | **encoding / vectorization** | `"happy"` → `id 4821` → `[0.12, -0.8, …]` |

### 2.1 The classic encodings (before deep learning)

**Integer (label) encoding** — give each word an id.
```
vocab = {"I": 0, "love": 1, "pizza": 2}      "I love pizza" -> [0, 1, 2]
```
- Problem: the numbers suggest an order and a size ("pizza" = 2 × "love"?), which means nothing. Fine as an **index**, bad as a **feature**.

**One-hot encoding** — a vector of size `vocab_size`, all 0 except a 1 at the word's id.
```
I     -> [1, 0, 0]
love  -> [0, 1, 0]
pizza -> [0, 0, 1]
```
- Problems:
	- **huge and sparse**: 50,000-word vocab = 50,000 numbers per word, 49,999 of them zero
	- **no meaning**: the dot product of any two different words is 0, so "cat" is as far from "dog" as from "pizza"

**Bag of Words (BoW)** — one vector per **document**: count how many times each vocab word appears.
```
vocab = [dog, bites, man, the]
"the dog bites the man"  -> [1, 1, 1, 2]
"the man bites the dog"  -> [1, 1, 1, 2]     <- identical! order is lost
```
- sklearn: `CountVectorizer`. Simple and a decent baseline for text classification.
- **N-grams** add some local order by also counting pairs/triples (`"dog bites"`, `"bites man"`), but the vocab explodes.

**TF-IDF** — BoW, but each count is re-weighted by how **rare** the word is across documents.
```
TF-IDF(word, doc) = TF(word, doc) × log( N / DF(word) )

TF = how often the word appears in this doc
N  = number of docs,  DF = number of docs that contain the word
```
- "the" appears in every doc → `log(N/N) = 0` → ignored. A rare word like "transformer" gets a high weight.
- sklearn: `TfidfVectorizer`. Still **sparse, order-blind and meaning-blind**.

**Dense word embeddings (Word2Vec 2013, GloVe 2014, FastText 2016)** — each word gets a **learned, small, dense** vector (e.g. 300 numbers) where similar words end up close together.
```
king  - man + woman ≈ queen
cos_sim(cat, dog) high,  cos_sim(cat, pizza) low
```
- Word2Vec learns them by predicting a word from its neighbours (CBOW) or neighbours from a word (Skip-gram). FastText also uses character n-grams, so it can build a vector for unseen words.
- Limitation: **static**, one vector per word whatever the context. "bank" (river) and "bank" (money) get the same vector.
- **Transformers fix this**: they start with an embedding per token (Ch 3), then attention turns it into a **contextual** embedding that depends on the surrounding words.

| Technique | Unit | Dense? | Keeps order? | Captures meaning? | Used today for |
| --------- | ---- | ------ | ------------ | ----------------- | -------------- |
| Integer | word | — | — | ❌ | **as the index** into an embedding table |
| One-hot | word | ❌ sparse | ❌ | ❌ | small categorical features |
| Bag of Words / N-grams | document | ❌ | ❌ / a bit | ❌ | quick baselines |
| TF-IDF | document | ❌ | ❌ | ❌ (only importance) | search, classical ML baselines |
| Word2Vec / GloVe / FastText | word | ✅ | ❌ | ✅ static | light NLP, historical step |
| **Transformer embeddings** | token | ✅ | ✅ (with positions) | ✅ **contextual** | everything modern |

> [!abstract] What a transformer actually uses
> **Tokenizer → integer ids → `nn.Embedding` lookup → learned dense vectors.**
> Mathematically, an embedding lookup = one-hot vector × weight matrix, just done as a fast table lookup instead of a matmul (Ch 3).

### 2.2 Tokenization: choosing the unit

**Word-level** — split on spaces and punctuation.
- ✅ each token carries real meaning, sequences are short
- ❌ vocab gets huge (every name, typo, plural, verb form)
- ❌ **OOV (out-of-vocabulary)**: an unseen word becomes `[UNK]`, its information is lost
- ❌ "run", "runs", "running" are 3 unrelated ids

**Character-level** — every character is a token.
- ✅ tiny vocab (~100), no OOV ever
- ❌ sequences are ~4–5× longer → attention cost `O(T²)` grows a lot, and context fills up fast
- ❌ one character carries little meaning, so the model must learn spelling first

**Subword-level** — common words stay whole, rare words split into frequent pieces. **What every modern model uses.**
```
"the"          -> ["the"]
"unhappiness"  -> ["un", "happiness"]   or  ["un", "happ", "iness"]
"tokenization" -> ["token", "ization"]
```
- ✅ small, fixed vocab (30k–130k) and **no OOV** (worst case it falls back to characters/bytes)
- ✅ shared pieces generalise: "-ization", "un-", "-ing"

> [!note] Our mini GPT will use **character-level** tokens: the tokenizer is 5 lines, so we can focus on the transformer. Real models use subwords.

### 2.3 Subword algorithms

#### BPE — Byte Pair Encoding (GPT-2/3/4, LLaMA, RoBERTa)
Start from characters, then **repeatedly merge the most frequent adjacent pair** into a new token until the vocab reaches the target size.

```
Corpus word counts:  low ×5, lower ×2, newest ×6, widest ×3     (_ marks end of word)

start   : l o w _ | l o w e r _ | n e w e s t _ | w i d e s t _
merge 1 : e + s  -> es    (6 + 3 = 9 times)
merge 2 : es + t -> est
merge 3 : est + _ -> est_
merge 4 : l + o  -> lo
merge 5 : lo + w -> low
...
result  : low_ | low e r _ | n e w est_ | w i d est_
```
- **Training** the tokenizer = learning the ordered list of merges. **Encoding** new text = applying those merges in the same order.
- **Byte-level BPE** (GPT-2 onward) starts from the 256 possible **bytes** instead of characters → any text, emoji or language can be encoded, so `[UNK]` is impossible.

#### WordPiece (BERT, DistilBERT)
- Same build-up idea as BPE, but picks the pair that most increases the **likelihood** of the data, roughly `count(ab) / (count(a) × count(b))`. It prefers pairs that appear together more than chance, not just frequent ones.
- Pieces that continue a word get a `##` prefix:
```
"unaffable" -> ["un", "##aff", "##able"]
```

#### Unigram LM (T5, ALBERT, XLNet)
- Goes the **opposite way**: start with a big vocab, then repeatedly **remove** the tokens whose removal hurts the data likelihood least.
- Can give several possible splits with probabilities (useful as data augmentation, "subword regularization").

#### SentencePiece (a library, not an algorithm)
- Implements **BPE or Unigram** directly on raw text, treating the **space as a normal character** `▁`. No language-specific pre-splitting, so it works for Chinese/Japanese with no spaces, and decoding is perfectly reversible.
```
"Hello world" -> ["▁Hello", "▁world"]
```
- Used by T5, LLaMA 1/2, ALBERT, XLNet.

| Model | Tokenizer | Vocab size |
| ----- | --------- | ---------- |
| BERT-base | WordPiece | 30,522 |
| GPT-2 | byte-level BPE | 50,257 |
| T5 | SentencePiece (Unigram) | 32,000 |
| LLaMA 2 | SentencePiece (BPE) | 32,000 |
| LLaMA 3 | tiktoken (byte-level BPE) | 128,256 |

### 2.4 Vocab size is a trade-off

| Bigger vocab | Smaller vocab |
| ------------ | ------------- |
| ✅ shorter sequences (cheaper attention, more text fits in the context) | ✅ smaller embedding table and output layer |
| ❌ embedding table and final `Linear` are both `vocab_size × d_model` → more parameters | ❌ longer sequences |
| ❌ rare tokens get few training examples | ✅ every token seen often |

- Rule of thumb for English with GPT tokenizers: **1 token ≈ 4 characters ≈ ¾ of a word**.

### 2.5 Special tokens

| Token | Purpose | Seen in |
| ----- | ------- | ------- |
| `[PAD]` | fill shorter sequences so a batch is rectangular | all |
| `[UNK]` | unknown token (word-level / WordPiece) | BERT |
| `[CLS]` | first token; its final vector represents the whole sentence for classification | BERT |
| `[SEP]` | separates two sentences | BERT |
| `[MASK]` | the hidden word BERT must predict | BERT |
| `<bos>` / `<eos>` | beginning / end of sequence | many |
| `<\|endoftext\|>` | document separator and end-of-text | GPT-2 |

> [!warning] Padding needs an attention mask
> Sentences in a batch have different lengths → pad them to the same length (your `collate_fn` from the cheat sheet does this). The model also gets an **attention mask** (`1` = real token, `0` = pad) so attention ignores the pads. Ch 7.

### 2.6 The full tokenizer pipeline
```
raw text
  │ 1. normalization      lowercase? Unicode cleanup, strip accents (BERT-uncased)
  │ 2. pre-tokenization   split on spaces/punctuation (skipped by SentencePiece)
  │ 3. model              BPE / WordPiece / Unigram -> subword tokens
  │ 4. post-processing    add [CLS] [SEP] / <bos> <eos>
  ▼
token ids  [101, 7592, 2088, 102]      -> tensor of dtype torch.long
```
- `encode`: text → ids. `decode`: ids → text. A good tokenizer is **lossless**: `decode(encode(x)) == x`.
- A model and its tokenizer are a pair: ids only mean something for the tokenizer the model was trained with.

### 2.7 Tokenizer quirks (where LLM weirdness comes from)
- `"hello"` and `" hello"` (leading space) are **different tokens** in GPT tokenizers.
- The model sees tokens, not letters, so tasks like "count the r's in strawberry" or reversing a word are hard.
- Numbers split unevenly (`"12345"` might become `"123"`, `"45"`), which hurts arithmetic.
- Non-English text often needs **more tokens** for the same meaning, so it is slower and costs more context.

### 2.8 In code

**Character-level tokenizer** (what our mini GPT will use)
```python
import torch

text = "hello transformer"

chars = sorted(set(text))                      # the vocabulary
vocab_size = len(chars)                        # 12
stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
itos = {i: ch for ch, i in stoi.items()}       # int -> string

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

ids = encode("hello")
print(ids)                                     # [4, 2, 5, 5, 8]
print(decode(ids))                             # hello

data = torch.tensor(encode(text), dtype=torch.long)   # ids are indices -> long
```

**One-hot in PyTorch**
```python
import torch.nn.functional as F
F.one_hot(torch.tensor([4, 2, 5]), num_classes=vocab_size)   # shape (3, 12)
```

**BPE training from scratch** (the whole algorithm)
```python
from collections import Counter

def get_pair_counts(words):
    # words: {('l','o','w','_'): 5, ...}  ->  how often each adjacent pair appears
    pairs = Counter()
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_pair(words, pair):
    # replace every occurrence of the pair with one merged symbol
    merged = {}
    for symbols, freq in words.items():
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        merged[tuple(new)] = freq
    return merged

corpus = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3
words = {tuple(w) + ("_",): f for w, f in Counter(corpus.split()).items()}

for step in range(1, 7):
    pairs = get_pair_counts(words)
    best = max(pairs, key=pairs.get)           # most frequent pair
    words = merge_pair(words, best)
    print(f"merge {step}: {best[0]} + {best[1]} -> {best[0] + best[1]}")
```

**A real tokenizer** (`pip install tiktoken`)
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Transformers are unbelievable")
print(ids)
print([enc.decode([i]) for i in ids])          # see how it was split
print(enc.n_vocab)                             # 50257
```

### 2.9 Key takeaways
- Two steps: **tokenize** (split into units) → **encode** (units → numbers).
- One-hot, BoW and TF-IDF are sparse and have no notion of meaning; Word2Vec/GloVe give meaning but are **static**; transformers give **contextual** vectors.
- Word-level has OOV and huge vocabs, char-level has very long sequences, **subword** (BPE / WordPiece / Unigram) is the balance every modern model uses.
- BPE merges the most frequent pair; WordPiece merges the most "likely" pair (`##`); Unigram prunes a big vocab; SentencePiece is the library that runs BPE/Unigram on raw text with `▁` for spaces.
- Token ids are **indices** (`torch.long`) into an embedding table, which is the next chapter.
