---
tags:
  - machineLearning
  - Algo
---
## Basics
- Types
	- Single LR
	- Multiple LR
	- Polynomial LR (Used for non linear data)
- Loss Function - MSE (Generally)
- Assumptions
	- Linear relation between input (each column/feature) and output 
	- No multi-collinearity (No co-relation between input column/feature - it should be independent features)
	- Normality of residual (residual=y_test-y_pred, should be normally distributed) 
	- Homoscedasticity (Homoscedasticity mean having a same spread for x-axis and y-axis)
	- No Autocorrelation of errors

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2,include_bias=True) # degree polynomial regression + bias
X_train_trans = poly.fit_transform(X_train) 
X_test_trans = poly.transform(X_test)
# then apply LR

from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge=Ridge(alpha=2) # alpha = Regularization strength # used when all column imp
lasso=Lasso(alpha=2) # used when all column are not imp
elasticNet=ElasticNet(alpha=2, l1_ratio=0.1) # mixing parameter for l1,l2 # used when we dont know all column are imp or not
```

## Regularization
- It reduce the variance or overfitting
- Adding a penalty term in loss function
- Type
	- Ridge (L2)
		- You apply this on dataset where you know all columns are not important for predictions
		- Geometric intuition
			- ![[Pasted image 20260421234643.png]]
			- ![[Pasted image 20260421234711.png]]
	- lasso (L1)
		- You apply this on dataset where you don't know all columns are not important for predictions
		- here new term added to loss function = α( |w1| + |w2| + .....|wn|)
		- Lasso regression help in feature selection and dimensionality reduction how ??
			- If we keep increasing the value of α -> wight become zero for some features/column
	- elastic-net
		- You apply this on dataset where you don't know all columns are important or not for predictions
		- Combination of ridge and lasso
		- Intuition
			- ![[Pasted image 20260421235623.png|569]]

## Algorithm Implementation from Scratch 

### Simple Linear Regression
- Goal - find the value of m (slop) anf b (y-intersept), 2 approches
	- Closed form solution = Ordinary Least Square
	- Non-Closed form solution = Gradient Decent
- 


#Algo 