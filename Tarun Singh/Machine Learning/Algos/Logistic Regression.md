---
tags:
  - machineLearning
  - Algo
---
## Basics 
- Classification Algo
- used when data is linearly separable
- One way to implement is 'Perceptron Trick'
- Loss Function = Sigmoid for binary class classification and softmax for multi class classification 
- Binary class classification Geometrical Intuition
	- ![[Pasted image 20260423000949.png]]
	- ![[Pasted image 20260423001010.png]]
	- ![[Pasted image 20260423001037.png]]
- Multi class classification Geometrical Intuition
	- ![[Pasted image 20260423001102.png]]
	- ![[Pasted image 20260423001125.png]]
## Hyper Parameters
- **l1_ratio:** 0 for l2 regularization, 1 for l1, and 0 to 1 for elasticnet
- **C:** it is (1/λ) , λ-> studied in regularization, smaller values = stronger Regularization
- **fit_intercept:** True/False , bias term add or not
- **class_weight:** dict/none/'balanced', default value = none
	- dict : {class_label1: weight, class_label2: weight}
	- The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
- **random_state** 
- **solver:** {‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
- **max_iter**int, default=100, Maximum number of times gradient decent/solver run

```Python 
from sklearn.linear_model import LogisticRegression
l= LogisticRegression(multi_class='multinomial') # for multi class

# polynomial 
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2,include_bias=True) X_trf = poly.fit_transform(X)
```

## Algorithm Implementation from Scratch