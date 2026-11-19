---
tags:
  - NLP
  - roadmap
---
## How this course works
- One lesson = one note in this folder. Read them **in order**, top to bottom
- Every lesson has: simple explanation → PyTorch / Python code → "what is actually used in real jobs" → small exercise
- I skip things you will never use. If something is only "good to know the name", I say so and move on
- You already know (from [[CheatSheet-PyTorch]]): tensors, autograd, `nn.Module`, `Dataset` / `DataLoader`, the training loop, `BCEWithLogitsLoss` / `CrossEntropyLoss`, Adam, BatchNorm, Dropout, GPU, save/load, Optuna. So I will **not** re-teach those, I will build on them

---

## ▶ Where to start
1. Start with **Part 1** below. These are your 4 existing notes. They are short and correct, read them as they are (my small "teacher notes" for each are in the table)
2. Then go to [[Classical ML for NLP]] (lesson 05). This is the first new lesson
3. Tick the box when you finish a lesson

---

## Part 1 — Basics (already written, ~1 hour)

| # | Lesson | Teacher notes: what to focus on |
|---|---|---|
| 01 | [[Intro to NLP]] | Just read once. Only remember the 3 approaches: heuristic → ML → DL |
| 02 | [[NLP Pipeline]] | Remember the 5 steps. "Modelling" depends on data size: little data → ML, lots of data → DL / transfer learning |
| 03 | [[Text Pre-processing]] | Lowercase, HTML / URL / punctuation removal and tokenization are **must know**. Spelling correction (TextBlob) is very slow, almost nobody uses it. **Big point:** for transformers (Part 4) you do almost **none** of this cleaning, the tokenizer handles raw text. Heavy cleaning is mostly for classical ML |
| 04 | [[Text Representation]] | BoW, N-grams and **TF-IDF are the important ones** (TF-IDF is used in lesson 05). OHE is only for understanding. Word2Vec: understand the idea "words become dense vectors where similar words are close", that idea is the base of `nn.Embedding` in Part 3 |

- [ ] 01 · [ ] 02 · [ ] 03 · [ ] 04

---

## Part 2 — NLP with classical ML (only what is used)

| # | Lesson | What you will be able to do |
|---|---|---|
| 05 | [[Classical ML for NLP]] | Build a real text classifier (TF-IDF + Logistic Regression / Naive Bayes / Linear SVM), evaluate it, read which words it learned, and rebuild the same model in PyTorch |

- Things I am **skipping on purpose**: HMM and CRF (old way of POS / NER tagging, transformers replaced them), LDA topic modelling (know the name only)
- [ ] 05

---

## Part 3 — Deep learning for NLP (before transformers)

| # | Lesson | What you will be able to do |
|---|---|---|
| 06 | Text → Tensors in PyTorch | Build a vocabulary, turn words into ids, padding, `collate_fn`, `nn.Embedding` |
| 07 | ANN for NLP | Yes, ANNs work for text! Average the word embeddings → MLP (this is how fastText works). Full PyTorch training |
| 08 | RNN, LSTM, GRU (in words only) | Understand why they were made, what problem they had (slow, forget long text), and why transformers replaced them. No heavy maths, very little code |

- [ ] 06 · [ ] 07 · [ ] 08

---

## Part 4 — Transformers in detail

| # | Lesson | What you will be able to do |
|---|---|---|
| 09 | Subword tokenization | Why words are split into pieces (BPE, WordPiece), how a Hugging Face tokenizer works, special tokens, attention mask |
| 10 | Attention from zero | Query, Key, Value, scaled dot-product attention, masking. Written in PyTorch line by line |
| 11 | The Transformer architecture | Multi-head attention, positional encoding, feed forward, residual + LayerNorm, encoder and decoder blocks. Built in PyTorch |
| 12 | 3 families of transformers | Encoder only (BERT), decoder only (GPT), encoder-decoder (T5). Which one for which task |
| 13 | Train a transformer from scratch | Train a small transformer text classifier and a tiny GPT that writes text, both in pure PyTorch |
| 14 | Hugging Face essentials | `AutoTokenizer`, `AutoModel`, `pipeline`, `datasets`, loading and saving models |
| 15 | Fine-tune BERT (classification) | Fine-tune a pretrained model on your own data, with your own PyTorch loop **and** with `Trainer` |
| 16 | Fine-tune for other tasks | Token classification (NER) and question answering / summarization, the practical way |

- [ ] 09 · [ ] 10 · [ ] 11 · [ ] 12 · [ ] 13 · [ ] 14 · [ ] 15 · [ ] 16

---

## Part 5 — Knowledge distillation (complete)

| # | Lesson | What you will be able to do |
|---|---|---|
| 17 | Distillation idea + ANN distillation | Teacher → student, soft labels, temperature, KL divergence loss. Distill a big ANN into a small ANN in PyTorch |
| 18 | Distilling a transformer | Make a small BERT learn from a big BERT (logits, hidden states, attention). How DistilBERT was made |
| 19 | Quantization and pruning (quick) | The other 2 ways to make models small and fast, so you know when to use what |

- [ ] 17 · [ ] 18 · [ ] 19

---

## Part 6 — LLMs and fine-tuning (complete)

| # | Lesson | What you will be able to do |
|---|---|---|
| 20 | How LLMs work | Next-token prediction, pretraining, context length, decoding (greedy, temperature, top-k, top-p) |
| 21 | The LLM training stages | Pretraining → SFT (instruction tuning) → preference tuning (RLHF / DPO). What each stage changes |
| 22 | Full fine-tuning a small LLM (SFT) | Dataset format, chat templates, masking the prompt in the loss, training loop, memory tricks (mixed precision, gradient accumulation, gradient checkpointing) |
| 23 | LoRA and QLoRA | Fine-tune a big LLM on one GPU. Write LoRA by hand in PyTorch first, then use `peft` |
| 24 | Preference tuning with DPO | Teach the model "this answer is better than that one" |
| 25 | LLM distillation | Use a big LLM to teach a small one (generated data + logit distillation) |
| 26 | Evaluate, merge and serve | Evaluate a fine-tuned model, merge LoRA weights, run it fast for users |

- [ ] 20 · [ ] 21 · [ ] 22 · [ ] 23 · [ ] 24 · [ ] 25 · [ ] 26

---
- Back to [[NLP-Navigation]]
