---
tags:
  - AIEngineer
  - llm-eval
---

# 05 - Eval Methods: Programmatic, Human, LLM-as-a-Judge + Reference-based vs Reference-free

> Video: https://www.youtube.com/watch?v=uQFLY8rQVYA
> **Eval method** = the mechanism that decides whether an output is good. It takes an output and produces a **judgment**.
> Simply put: **who runs** the eval pipeline?

| Method                         | Who judges         | Pros                          | Cons                        |
| ------------------------------ | ------------------ | ----------------------------- | --------------------------- |
| **Programmatic / deterministic** | code             | cheap, fast, exact            | only for exact / checkable outputs |
| **Human**                      | people             | most **reliable** judgment, handles nuance | **costly**, doesn't scale  |
| **Model-graded (LLM-as-a-judge)** | another LLM     | nuance + scale + cheap        | must be validated against humans |

> Note: building the golden dataset is a separate activity (usually done by humans). The method = who **executes** the eval.

---

## 1. Programmatic: evaluating a RAG retriever (component level)
- **Target:** the retriever of the CampusX RAG chatbot. **Task:** does it fetch the relevant docs?
- **Metric: Recall@k** = out of all the correct docs that exist, how many are in the top-k?
  - Q: "ML course prerequisites and duration?" Correct docs = `{1001, 1003}`. Retrieved (k=5) = `{1001, 1002, 1004, 1005, 1006}` → found 1 of 2 → **recall = 0.5**
  - Range 0 to 1 (ideal = 1).
- **Golden dataset:** 50-100 questions (easy, hard, edge cases, random) + a human marks which doc IDs hold the answer.
- **Run:** send each question **only to the retriever**, compute recall per question, **average** → e.g. 67%.
- **Improve the retriever:** better embedding model, **query expansion** (LLM rewrites/expands the query first), bigger `k`, add a **reranker**.
- Other retriever metrics exist too (precision: how many fetched docs were useless; ranking order).

```Python
golden = [
    {'question': 'How long is the ML course?', 'relevant_ids': ['1001']},
    {'question': 'ML course prerequisites and duration?', 'relevant_ids': ['1001', '1003']},
]

k = 5
total_recall = 0
for row in golden:
    retrieved_ids = retriever_search(row['question'], k)   # your retriever
    found = 0
    for doc_id in row['relevant_ids']:
        if doc_id in retrieved_ids:
            found = found + 1
    recall = found / len(row['relevant_ids'])
    total_recall = total_recall + recall

print('Recall@k:', total_recall / len(golden))
```

---

## 2. Human: chatbot helpfulness (application level)
- **Target:** the whole CampusX chatbot. **Task:** is it **helpful**? (helpful = accurate + right tone + complete)
- No standard metric → define a **rubric** (business specific), 1-5:
  - **5** = correct, complete, right tone · **3** = partially helpful · **1** = not helpful at all
- **Dataset:** only **questions** (50-100: "How long is ML course?", "Fee for DL course?", "Refund if I drop out?", "Pay in instalments?"). No correct answers.
- **Run:** question → chatbot → answer → a **human grader** scores it with the rubric → average = helpfulness score.
- **Multiple graders?** If graders often **disagree** → the **rubric is ambiguous** → refine it. If they agree → the rubric is clear.

### 5 ways humans evaluate LLM systems
1. **Direct grading / rating** (above)
2. **Red teaming**: a group deliberately attacks the system to find where it breaks → findings go back to the dev team
3. **A/B testing**: 2 versions live, users rate them → roll out the winner
4. **Golden dataset & rubric creation**
5. **Human-in-the-loop**: grey-area / complex cases are passed to a human

---

## 3. LLM-as-a-Judge: UPSC Mains answer grader
- **Setup:** CampusX UPSC sells mock Mains tests (subjective answers). 10,000 students → too many subject experts to pay. A platform offers an **LLM that grades** answers using your rubrics. Must evaluate: does it grade **like a human expert**?
- **Success criterion:** LLM marks ≈ human expert marks.
- **Rubric per question** (from an expert), e.g. Q1 "Ethical governance is impossible without administrative accountability. Discuss (15 marks)":
  - covers ethical governance & accountability · explains the link · gives mechanisms · cites examples · balanced conclusion
- **Golden dataset:** `answer_id, question_id, student answer, human_marks` (50-100 answers graded by an expert with the rubric).
- **Run:** for each answer → the judge LLM gets question + marks + rubric + student answer → returns dimensions covered, total marks, a 1-sentence reason.
- **Metric: MAE (mean absolute error)** between LLM marks and human marks.
  - MAE = 2.3 → the LLM deviates from the human by ±2.3 marks on average. **Goal → bring it close to 0.**
  - Improve with: a better LLM, a better system prompt, a clearer rubric.

### Judge prompt (idea)
```
You are grading a UPSC Mains answer against an evaluation rubric.

Question: {question}   (Marks: {max_marks})
Rubric dimensions:
{rubric}

Aspirant's answer:
{answer}

For each dimension, decide whether the answer genuinely addresses it, then allocate marks.
Do NOT reward verbosity, keyword stuffing or confident claims without substance.
Reward structure, relevant examples and balanced argumentation.

Return: dimensions_covered, total_marks, one-sentence justification.
```

```Python
total_error = 0
for row in golden:
    llm_marks = judge_llm(row['question'], row['rubric'], row['answer'])   # LLM-as-a-judge
    error = abs(llm_marks - row['human_marks'])
    total_error = total_error + error

mae = total_error / len(golden)
print('MAE:', mae)   # closer to 0 = grades like a human
```

---

## Reference-based vs Reference-free
| | **Reference-based** | **Reference-free** |
|-|-|-|
| What | a **known correct answer** (or key points) is written **in advance** for each test case → compare the output to it | **no predefined correct answer** → judge the output on its own against criteria / a rubric (a general scale, not a per-item answer) |
| Examples | Retriever eval (correct doc IDs known), UPSC (human marks known) | Chatbot helpfulness (only questions + rubric) |
| Test | Does the golden dataset have the correct answer? **Yes** → reference-based | **No** → reference-free |

## Remember
- 3 methods: code / human / LLM. Use code when possible, humans when it's nuanced and small, an LLM judge for nuance at scale.
- Retriever → Recall@k (programmatic). Helpfulness → rubric (human). Grading like an expert → LLM judge + MAE vs human.
- Validate an LLM judge **against human labels** before trusting it.
- Next: offline vs online evals.
