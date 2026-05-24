---
tags:
  - machineLearning
  - Algo
---
## Difference between Multinomial Naive Bayes and Categorical Naive Bayes

| Categorical Naive Bayes                                          | Multinomial Naive Bayes                                          |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| It assumes that each feature follows a categorical distribution. | It assumes that each feature follows a categorical distribution. |
| usage - Recommendation system, Medical diagonsis                 | usage - Document classification, Document classification         |
## Intuition 
- ![[Pasted image 20260501014352.png]]
- ![[Pasted image 20260501014413.png]]
- ![[Pasted image 20260501014435.png]]

## Maths for naive bayes
- ![[Pasted image 20260501013557.png]]
- ![[Pasted image 20260501013623.png]]
- ![[Pasted image 20260501013653.png]]
- ![[Pasted image 20260501013715.png]]

## Code

```Python
# For categorical input columns only
from sklearn.naive_bayes import MultinomialNB

# For numerical input columns only
from sklearn.naive_bayes import GaussianNB

#For a mix of numerical and categorical input columns
# use either GaussianNB or MultinomialNB
```