---
tags:
  - machineLearning
---
## Types
- Type 1
	- Supervised
		- We have both input and output data 
		- type
			- Regression 
			- Classification
	- Unsupervised
		- We only input data no output data
		- Type
			- Clustering
			- Dimensionlity Reduction
			- Anomoly Detection
			- Assosiation rule learning
	- SemiSupervised
		- Half supervised and half unsupervised learning
		- Example -> Google photos
	- Reinforcement Learning 
		- No Data 
		- Model learn by interacting with the environment
		- For Example: Self driving car
- Type 2
	- Batch / offline Learning
	- Online Learning
## General Steps
- Frame the problem
- Get the data 
- Data processing
- EDA
- Feature Engineering and selection
- Model Training
- Model Testing
## Tensor
- A Data Structure 
- Rank = number of axis = Number of Dimensions
- Types  
	- 0D Tensor / Scalar
	- 1D Tensor / Vector
	- 2D Tensor / Matrix
	- 3D Tensor 
	- 4D Tensor
	- 5D Tensor
```python
# Scalar
a=np.array(4)
a.ndim # 0

# Vector
a= np.array([1,2,3])

# Matrix
a= np.array( [[1,2,3], [4,5,6]] )

# 3D Tensor
a= np.array( [ [[1,2,3],[4,5,6]] , [[7,8,9],[10,11,12]] ] )
```

## Types of data
- Numerical Data
	- Discrete Numerical Data: 1,2 ..
	- Continuous Numerical Data: 0.1, 3.2, 6.7 .. 
- Categorical Data
	- Nominal Categorical Data: City name, male/female
	- Ordinal Categorical Data: rating(5/4/3/2/1), grade(A/B/C/D)
## Curse Of Dimensionality
There are only certain number of dimensions (features/columns) which can give you best resulting model, further increase in dimensions will result in inaccuracy in model
## Bias Variance Trade off
- Bias
	- model is too simple to capture real patterns
	- Leads to underfitting
	- High error in both train and test data
- Variance
	- how much the model’s predictions change with different training data
	- Leads to overfitting
	- Low train error but high test error
- In a model we can either have (low bias and high variance) or have (high bias and low variance) -> we tried to achieve a model with low-bias and low-variance
- Ways to get low-bias and low-variance model
	- Regularization = Reduce variance
	- Bagging = Reduce variance
	- Boosting = Reduces bias (primarily), can also affect variance