---
tags:
  - AIEngineer
  - llm-eval
---

# 03 - How to Evaluate an LLM Application: The Complete Workflow

> Video: https://www.youtube.com/watch?v=Pv4mkG2K_s8
> The **same loop** is used for everything: a simple classifier, RAG, complex agents.

## Example app: Zomato email router
- Zomato gets tons of customer emails. Goal: **auto-classify** each one → route it to the right team.
  - `billing` → billing team, `technical` → tech team, `general` → customer support
- System: **1 LLM + a system prompt** ("you are a customer agent, read the email, return the category"). Built in 10 minutes.
- Deploy it directly? **No → evaluate first.**

## The workflow (10 steps, a loop)
```
1 Define task & target
2 Define success criteria (+ metric)
3 Build dataset (golden dataset)
4 Define eval method (code / human / LLM)
5 Run the system on the dataset
6 Evaluate results (get the score)
7 Analyze results (where does it fail?)
8 Improve the system ──┐
   └── back to 5 (iterate) until satisfied
9 Deploy
10 Monitor → production failures → add to golden dataset → back to 5
```

| Step | Zomato example |
| ---- | -------------- |
| **1. Task & target** | target = the whole workflow. Task = classification: does it classify correctly? |
| **2. Success criteria** | correct classification → metric = **accuracy** (100 emails, 90 right → 90%) |
| **3. Dataset** | `email text → expected label`. "My card was charged twice" → billing. "App crashes on login" → technical. "What are your hours?" → general. Usually **50-500 rows**, ideally from **real past emails**, labelled **by humans**. This is the **golden dataset**. |
| **4. Eval method** | who compares output vs expected? **Automated (code)** / **Human** / **LLM**. Here: a simple Python accuracy check (no need to pay a human or an LLM). |
| **5. Run** | send every dataset row through the system → get predictions |
| **6. Evaluate** | Python computes accuracy → e.g. 80% |
| **7. Analyze** | why are the 20% wrong? confusing system prompt (billing vs technical)? weak / small model? |
| **8. Improve** | tweak the prompt → re-run on the **same** dataset → 90%. Bigger LLM → 95% → manager happy |
| **9. Deploy** | when good enough |
| **10. Monitor** | production failures (an email routed wrong) → **add it to the golden dataset** → re-evaluate. The dataset gets richer over time. Forever, as long as the app is live. |

### Simple code idea (steps 5-6, automated method)
```Python
golden_dataset = [
    {'email': 'My card was charged twice', 'expected': 'billing'},
    {'email': 'The app crashes on login', 'expected': 'technical'},
    {'email': 'What are your hours?', 'expected': 'general'},
]

correct = 0
for row in golden_dataset:
    predicted = classify_email(row['email'])   # your LLM system
    if predicted == row['expected']:
        correct = correct + 1

accuracy = correct / len(golden_dataset)
print('Accuracy:', accuracy)
```

## Choosing the eval method
| Output type                        | Method                                            |
| ---------------------------------- | ------------------------------------------------- |
| Label / number (classification)    | **Code**: exact match, cheap, fast                |
| Long text (chatbot paragraphs)     | Code can't judge semantic meaning → **Human** (accurate but costly) or **LLM** (the middle ground) |

## Who finds wrong outputs in production?
- A feedback process. E.g. a billing email goes to the tech team → the customer says "I need billing" → the tech team **flags** it → it goes into the dataset.

## ⭐ One app → many evals
- One LLM app usually has **several evals**, not one. E.g. RAG:
  - an eval for the **retriever**
  - an eval for the **embedding model**
  - an eval for the **full RAG flow** (answers)
  - an eval for **latency**

## Remember
- Task → criteria/metric → golden dataset → method → run → evaluate → analyze → improve → **loop** → deploy → monitor → failures back into the dataset.
- **Repeatable** = the same golden dataset after every change → you can compare versions.
- Eval method: code (exact outputs), human (costly), LLM (scalable for text).
