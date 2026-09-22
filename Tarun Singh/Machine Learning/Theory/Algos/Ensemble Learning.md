---
tags:
  - Algo
  - machineLearning
---
## Basics
- Type
	- Voting Ensemble
	- Bagging Ensemble, Example Random Forest
	- Boasting, Example Adaboasting, Gradient boasting, Xgboost
	- Stacking
- Disadvantages
	- computation increased
- Advantages
	- improved performance
	- decrease bias and variance
	- Robustness
## Voting
- Assumptions
	- It is important that all the model should be idependent/dis-similar
	- Each model should have Minimum accuracy of 0.51
- Prediction
	- classification
		- Hard voting
			- Mode ![[Pasted image 20260503180207.png]]
		- Soft voting
			- ![[Pasted image 20260503180232.png]]
	- regression = mean
```Python 
from sklearn.ensemble import VotingClassifier, VotingRegressor
clf1 = LogisticRegression() 
clf2 = RandomForestClassifier() 
clf3 = KNeighborsClassifier()
estimators = [('lr',clf1),('rf',clf2),('knn',clf3)]
vc=VotingClassifier(estimators=estimators,voting='hard/soft')
```

## Bagging
- Basics
	- Bagging = Bootstrapping(Get random set of data from a dataset with replacement) + Aggregation(mean=regression, mode=classification) 
- Types
	- Bagging = row sampling with replacement
	- Pasting = row sampling without replacement
	- Random Subspaces = column sampling (with and without replacement)
	- Random Patches = row+column sampling (with or without replacement)
```Python 
# Classification 
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

# bagging classifier
bag = BaggingClassifier(
	base_estimator=DecisionTreeClassifier(), 
	n_estimators=500, 
	max_samples=0.5, # row sampling [each model get 0.5 size of complete data]
	bootstrap=True, # row replacement
	random_state=42)
# Pasting classifier
bag = BaggingClassifier(
	base_estimator=DecisionTreeClassifier(), 
	n_estimators=500, 
	max_samples=0.25, 
	bootstrap=False, 
	random_state=42, 
	verbose = 1, 
	n_jobs=-1)
# Random Subspaces
bag = BaggingClassifier(
	base_estimator=DecisionTreeClassifier(), 
	n_estimators=500, 
	max_samples=1.0, 
	bootstrap=False,
	max_features=0.5, # column sampling
	bootstrap_features=True/False, # column replacement
	random_state=42)
# Random Patches
bag = BaggingClassifier(
	base_estimator=DecisionTreeClassifier(), 
	n_estimators=500, 
	max_samples=0.25, 
	bootstrap=True, 
	max_features=0.5, 
	bootstrap_features=True, 
	random_state=42)
```

## Random Forest
- Random = Bagging, Forest = Group of decision Trees
- Fully grown decision tree = Low bias + high variance
- Difference between Bagging and Random Forest
	- Bagging
		- Support different models like LR, SVM, KNN etc
		- Model level column sampling
	- Random Forest 
		- Decision Tree model only
		- Node level column sampling
### Hyper Parameter 
- **n_estimators:** int, default=100
- **max_features:** {“sqrt”, “log2”, None}, int or float, default=”sqrt”
- **bootstrap:** bool, default=True
- **max_samples:** int or float ,default=None
```Python 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
```
## Bagging vs Boosting

| Bagging                                    | Boosting                                   |
| ------------------------------------------ | ------------------------------------------ |
| Use models with Low bias and high variance | Use models with high bias and low variance |
| Parallel learning possible                 | Sequential Learning                        |
| Base model weightage is equal              | Base model weightage is not equal          |

## Adaboost

### Basics
- Adaboost is a stage wise additive method 
- Adaboost is build by adding multiple weak learners
- All weak learners are added stage by stage
- Weak Learner
	- A machine learning model who has very less accuracy (around 50 %)
	- Adaboosts mostly use decision stumps as a group of weak learners
- Decision stumps are weak learners, decision tree with maximum depth =1
### Maths behind adaboost
#### Training and Prediction
- ![[Pasted image 20260503224225.png]]
- ![[Pasted image 20260503224253.png]]
#### Step by Step intuition 
- ![[Pasted image 20260503224446.png]]
- ![[Pasted image 20260503224507.png]]
- ![[Pasted image 20260503224527.png]]


### Code
- **estimator:** object, default=None (none = decision Tree classifier)
- **n_estimators:** int, default=50
- **learning_rate:** float, default=1.0
- **algorithm:** {‘SAMME’, ‘SAMME.R’}, default=’SAMME.R’
```Python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
```
## Stacking

**Core Intuition** 
- first we train multiple models(called Base models) on data, then we give these base model output to a new model(called meta model) as input with data output column
### How to avoid Over-fitting
- There is very high probability that the complete model start over fitting
- 2 Methods
	- **Blending/Hold out method**
		- Not supported by sklearn
		- Core intuition
			- Divide the data into 2 parts (Data training and Data test)
			- Divide the training data into 2 parts (D1 and D2)
			- use D1 to train base models
			- Then base models do prediction for D2 Y_pred1 , Y_pred2 , Y_pred3
			- These Y_pred1 , Y_pred2 , Y_pred3 and output column used for training meta model
	- **Stacking/K-Fold method**
		- K = number of pieces you want to divide your data
		- ![[Pasted image 20260504083736.png]]
		- ![[Pasted image 20260504083756.png]]
### Multi Layer Blending 
- ![[Pasted image 20260504084429.png]]
- ![[Pasted image 20260504084446.png]]


### Code
```Python
# classification , regression
estimators = [ 
	('rf', RandomForestClassifier(n_estimators=10, random_state=42)), 
	('knn', KNeighborsClassifier(n_neighbors=10)),
	('gbdt',GradientBoostingClassifier()) 
]
from sklearn.ensemble import StackingClassifier, StackingRegressor 
clf = StackingClassifier( 
	estimators=estimators, 
	final_estimator=LogisticRegression(),
	cv=10 # cv = cross validation = act as k for k fold mean
)
```
