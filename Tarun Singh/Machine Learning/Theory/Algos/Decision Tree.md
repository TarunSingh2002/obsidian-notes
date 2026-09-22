---
tags:
  - Algo
  - machineLearning
---
## Basics 
- Giant structure of nested if-else condition
- Div = Over-fits a lot, Prone to error on imbalance data set
- Working -> skipped -> A video explanation would be better
## Decision Tree For Regression, Intuition
- ![[Pasted image 20260424004150.png]]
- ![[Pasted image 20260424004227.png]]
- ![[Pasted image 20260424004254.png]]
- ![[Pasted image 20260424004314.png]]
## Hyper Parameters
- **max_depth:** int, default=None , The maximum depth of the tree, cause over-fitting if very high or under-fitting if very low
- **criterion:** {“gini”, “entropy”, “log_loss”}, default=”gini”,  + {“squared_error”, “friedman_mse”, “absolute_error”, “poisson”}, default=”squared_error” The function to measure the quality of a split
- **splitter:** {“best”, “random”}, default=”best”, The strategy used to choose the split at each node
- **min_samples_split:** int or float, default=2, The minimum number of samples required to split an internal node
- **min_samples_leaf:** int or float, default=1, The minimum number of samples required to be at a leaf node
- **max_features:** int, float or {“sqrt”, “log2”}, default=None, The number of features to consider when looking for the best split
- **random state**
- **max_leaf_nodes:** int, default=None
- **min_impurity_decrease:** float, default=0.0, A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
```Python
from sklearn.tree import DecisionTreeRegressor, DecisonTreeClassifier
```


