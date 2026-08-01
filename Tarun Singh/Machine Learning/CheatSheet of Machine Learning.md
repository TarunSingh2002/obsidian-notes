---
tags:
  - machineLearning
  - cheatSheet
---
## Basic Commands 

```python
# Making Tensors
a=np.array(4)
a.ndim # 0
# Vector
a= np.array([1,2,3])
# Matrix
a= np.array( [[1,2,3], [4,5,6]] )
# 3D Tensor
a= np.array( [ [[1,2,3],[4,5,6]] , [[7,8,9],[10,11,12]] ] )

# Read csv
import pandas as pd 
df = pd.read_csv('abc.csv')

# basic Eda commands
df.info()
df.describe()

# Train Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, radom_state=0.3)

# To get number of unique categories 
df['column'].nunique()
# To get to know number of occurrence of each unique categories
df['column'].value_counts()

#cross validataion 
from sklearn.model_selection import cross_val_score
cross_val_score(model,x_train,y_train,cv=5, scoring='accuracy' ).mean()
# other scoring name here https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names

# To get to know what percentage of null values present in each column
df.isnull().mean()*100
```

## Feature Engineering

### Feature Scaling
- Types :
	- Standardization
		- Use when we depended on gradient decent (like ANN, linear regression or logistic regression) and when the algo use distance between points like K-Mean, K-Nearest-Neighbours 
	- Normalization
		- MinMaxScaling: 
			- used when values in between a fix range, may effect the values, squeeze the outliers
		- RobustScaling:
			- used when we have a lot of outliers
		- MaxAbsScaling:
			- used when we have a lot of sparse values
		- Mean Scaling:
			- not supported  by sklearn
- ```python
  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
  ```
### Encoding of a Categorical column
- Types
	- Ordinal Encoding:
		- Used with Ordinal Input categorical columns
	- One Hot Encoding:
		- Used with nominal categorical column
		- For a column with n categories it will create n-1 new columns. -> that's why not good for a categorical column with many categories 
	- Label Encoding
		- Used with Output categorical column
```python
# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder(drop='first', sparse=False, dtype=np.int32)

# Label Encoding
from sklearn.preproessing import LabelEncoder

# Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=['Poor','Average','Good'])
```

### Pipelines & ColumnTransformer

> [!abstract] TL;DR / Note / IMP - Hacks
> 
> - **Pipeline** = sequential vertical stack. Data flows top → bottom.
> - **ColumnTransformer** = parallel horizontal branches. Different columns go different ways, then concatenated.
> - Nest them: Pipelines inside ColumnTransformers inside a Pipeline. This is normal.
> - Always feed **raw DataFrames** into the pipeline. Never bypass it.
> - pipline/column_trans.set_output(transform='pandas')  -> this will return dataframe as output
> - ('removing_irrelevent_columns', 'drop', ['c1','c2'] # this will drop irrelevent column dataframe with the help of column transformer  --> 
#### `Pipeline` — Sequential
Data flows through **top to bottom**. Step N receives the output of step N−1.

```
    input
      ↓
    step1  (transformer)
      ↓
    step2  (transformer)
      ↓
    step3  (model / transformer)
      ↓
    output
```

> [!tip] Rule A Pipeline **inherits the methods of its last step**.
> 
> - Last step = transformer → pipeline has `fit_transform(x_train)`, `transform(x_test)'
> - Last step = model → pipeline has `fit(x_train,y_train)`, `predict(x_test)`

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('step1', SomeTransformer()),
    ('step2', AnotherTransformer()),
    ('step3', SomeModel()),
])
pipe.set_output(transform='pandas')   # very imp
```

#### `ColumnTransformer` — Parallel Branches
Data is **sliced by column** into independent branches. Each branch runs on its slice, then all outputs are **concatenated side by side**.

```
              input (DataFrame)
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
     cols[a,b]  cols[c,d]   cols[e,f]
        ↓           ↓           ↓
      trf1        trf2        trf3
        ↓           ↓           ↓
        └───── concatenate ─────┘
                    ↓
                 output
```

> [!warning] Not sequential Each branch sees the **original input** for its columns. `trf2` does **not** see `trf1`'s output. They run independently.

```python
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ('name1', SomeTransformer(), ['col1', 'col2']),
        ('name2', OtherTransformer(), ['col3']),
    ],
    remainder='passthrough'   # or 'drop' or another transformer
)
```

Example:
```python
# Input columns: [age, city, salary, gender]
ColumnTransformer([
    ('ohe', OneHotEncoder(), ['city', 'gender']),
], remainder='passthrough')
```

Output columns, in this order:
1. `city_delhi`, `city_mumbai`, ... ← from OHE on `city`
2. `gender_M`, `gender_F` ← from OHE on `gender`
3. `age`, `salary` ← from `remainder='passthrough'`

The original `city` and `gender` are **gone**. They were consumed and replaced by their OHE output.


- If you list the same column twice:

```python
ColumnTransformer([
    ('trf1', Transformer1(), ['a']),
    ('trf2', Transformer2(), ['a']),
], remainder='passthrough')
```

- `trf1` gets the **original** `a`.
- `trf2` also gets the **original** `a` (NOT trf1's output).
- Output has: `trf1(a)`, `trf2(a)` — column `a` appears twice, transformed two different ways.
- Original `a` does **not** come through `remainder` (it was "used").
#### Pipeline inside a column Transformer

```python
numeric_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, ['age', 'fare']),   # ← pipeline as a branch
], remainder='drop')
```



#### Debug in isolation
Before wiring into a pipeline, test the transformer alone:
```python
fe = FeatureEngineering()
result = fe.fit_transform(X_train)
print(result.head())
print(result.columns.tolist())
```
#### 8.4 Useful inspection
```python
pipeline.get_feature_names_out()           # output column names
pipeline.set_output(transform='pandas')    # return DataFrames
pipeline.named_steps['preprocess']         # access a specific step
pipeline['preprocess']                     # shorthand
```

### Normally Distribution Of Data
These function normally distribute the data of column
- Type
	- Power Transformer
		- Box Cox transform: not applicable to values <=0 , special case of Yoe johnson
		- Yoe johnson transform
	- Function Transformer
		- Log: Right skew data, not work on negative values
		- Reciprocal
		- square: Left skew data
		- square root
```python
from sklearn.preprocessing import PowerTransformer, FunctionTransformer

# yoe-johnson 
PowerTransformer()
# box cox 
PowerTransformer(method='box-cox')
```

### Missing value handling
- Types of missing values:
	- MCAR (Missing Completely At Random): Absolutely random value missing.
	- MAR (Missing At Random): 
	- MNAR (Missing Not At Random):
- Types of way we can handle missing values
	- Remove row (CCA - Complete Case Analysis)
	- Impute:
		- Uni-variate 
			- Types
				- Numerical column
					- Mean/median
					- End of the distribution
					- Arbitrary value imputation for numerical column
					- Random value imputation (no supported by sklearn)
					- Missing Indicator - i skipped
				- Categorical column
					- Mode
					- Arbitrary value imputation for categorical
					- Random value imputation (no supported by sklearn)
					- Missing Indicator - i skipped 
		- Multi-variate: 
			- Types
				- Knn impute
				- iterative impute algo - mice
- Complete Case Analysis (CCA):
	- Used when missing data is MCAR + If Missing data is less then 5 % + Distribution of the variable should remain the same.
- Mean/Median/Mode
	- Used when missing data is MCAR + If normal distribution = use mean else median + missing data is 5% or less + change in co-relation.
	- It can change existing distribution and introduce outliers.
	- New column standard deviation, co-relation, distribution should remain the same ans also check for outliers.
- Arbitrary Value Imputation for numerical column + Categorical column
	- This arbitrary value represents the null value.
	- Used when value is MNAR (Missing not at random)
- End of distribution
	- Similarly like Arbitrary Value Imputation, but we just put other extreme values of the column
	- Used when value is MNAR (Missing not at random)

```python 
# applying CCA
df['column1'].dropna() # column wise
df.dropna() # data frame wise

# Mean/Median/mode/constant/arbitrary value
from sklearn.impute import SimpleImputer
s= SimpleImputer(strategy='mean/median/constant/most_frequent', fill_value='used when stategy is contant')

from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors=3,weights='distance/uniform')

from sklearn.impute import IterativeImputer
imp_mean = IterativeImputer(random_state=0)
```

### Handling Outliers
Algo which calculate the weight get effected by the outliers the most
- Types of way to handle outliers
	- Trimming
	- Capping
	- others
		-  Discretization 
		- Treating them as Missing value
- Types of way to detect the outliers
	- Z Score Treatment
	- Box Plot/Inter-Quartile Proximity rule
	- Percentile based approach
```python
# Z-Score Method
upperlimit = df['col1'].mean() + 3*df['col1'].std()
lowerlimit = df['col1'].mean() - 3*df['col1'].std()
# Trimig 
new_df = df[ (df['col1'] < upper_limit) & (df['col1'] > lower_limit) ]
# Capping
df['col1'] = np.where( 
	df['cgpa']>upper_limit, upper_limit, 
	np.where( 
		df['cgpa']<lower_limit, lower_limit, df['cgpa'] 
	)
)

# Interquartile proximity rule
per75 = df['col1'].quantile(0.75)
per25 = df['col1'].quantile(0.25)
iqr=per75-per25
upperlimit=per75+1.5*iqr
lowerlimit=per25-1.5*iqr

# Percentile based approach -> my assumption of 0.1 will edge case
upperlimit=df['col1'].qunatile(0.99)
lowerlimit=df['col1'].qunatile(0.1)
```

### Encoding/Handling Numerical Data
- If a column have a very long range, - you may need to convert it too categorical column
- Types
	- Discretization(Binning): Making Intervals
		- Types
			- unsupervised binning
				- Equal Width/uniform binning
				- Equal frequency/Qunatile binning
				- Kmean binning: used when data is present in clusters
			- custom binning: Custom logic based binning
			- supervised binning: Example Decision Tree Binning
	- Binarization: convert a numerical column into 1 or 0
```python
from sklearn.preprocessing import KBinsDiscretizer
kbin=KBinsDiscretizer(n_bin=10, strategy='uniform/quantile/kmeans', encoder='onehot/ordinal/onehot-dense')

from sklearn.preprocessing import Binarizer
b=binarize(threshold=0.5, copy=False)
```


### Handling Date And Time Columns
```python
# Date
# Converting a object column to datetime datatype
date['date'] = pd.to_datetime(date['date'])

# Extract year
date['date'] = date['date'].dt.year

# Extract month
date['date'] = date['date'].dt.month

# Extract month-name
date['date'] = date['date'].dt.month_name()

# Extract Day
date['date'] = date['date'].dt.day

# Extract Day of week
date['date'] = date['date'].dt.dayofweek

# Extract Day of week name
date['date'] = date['date'].dt.day_name()

# is weekend
date['date'] = np.where(date['date'].dt.day_name().isin(['Sunday', 'Saturday']), 1,0)

# Extract week of the year
date['date'] = date['date'].dt.week

# Extract Quarter
date['date'] = date['date'].dt.

# Extract semenster
date['date'] = date['date']

# Time
# Extract hour, min, sec
time['hour'] = time['date'].dt.hour 
time['min'] = time['date'].dt.minute 
time['sec'] = time['date'].dt.second
time['time'] = time['date'].dt.time

```


### Imbalance Dataset
- Problem 
	- Bias towards majority class
	- Some Metrics are not reliable for imbalance dataset, for example accuracy
- Types
	- Under-sampling
	- Oversampling
		- Div= duplication of data cause over-fitting
	- SMOTE (Synthetic Minority Oversampling Technique)
		- Create new points for minority class instead of duplication
		- Div = not applicable on categorical dataset, computational heavy, Depended on neighbors values, sensitive to outliers, no gureente of coorect points
	- Ensemble Methods
		- BalancedRandomForestClassifier + BalancedBaggingClassifier
	- Cost Sensitive Learning
		- Applying class weights (assign more weights to minority class)
		- Applying custom loss function 
```python 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, RUSBoostClassifier

oversample = RandomOverSampler(_random_state_=1)
X_res, y_res = oversample.fit_resample(X, y)

classifier = BalancedRandomForestClassifier(random_state=42)

#class weights
model = LogisticRegression(class_weight={0:5,1:1})
```
### Making your own column transformer

```python 
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns_to_drop, errors="ignore")        



# how to use it
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), ["age", "salary"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
pipeline = Pipeline(steps=[
    ("drop_ids", ColumnDropper(columns_to_drop=["user_id", "transaction_id"])),
    ("preprocess", preprocessor),
    ("model", SomeModel()),
])

```
## Matrics

### Regression Metrics

| Metric | Formula | Unit | Outlier robust? | Note |
| ------ | ------- | ---- | --------------- | ---- |
| **MSE** | $\frac{1}{n}\sum (y_i-\hat{y}_i)^2$ | squared unit of y | ❌ | differentiable -> good as a loss function |
| **MAE** | $\frac{1}{n}\sum \|y_i-\hat{y}_i\|$ | same as y | ✅ | not differentiable at 0 |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y_i-\hat{y}_i)^2}$ | same as y | ❌ | MSE brought back to y's unit |
| **R2** | $1-\frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}$ | none (0-1) | — | how much better than just predicting the mean |
| **Adj R2** | $1-\frac{(1-R^2)(n-1)}{n-p-1}$ | none | — | punishes useless columns |

- **R2** aka "coefficient of determination" / "goodness of fit", closer to 1 = better
	- Div = adding **any** new feature (even an irrelevant one) increases R2 -> that's why Adjusted R2 exists
- **Adjusted R2** only goes up if the new feature actually helps (n = data points, p = number of input columns)
```Python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = lr.predict(X_test)

print("MAE",  mean_absolute_error(y_test,y_pred))
print("MSE",  mean_squared_error(y_test,y_pred))
print("RMSE", np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2",   r2_score(y_test,y_pred))

# Adjusted R2
r2 = r2_score(y_test,y_pred)
n = X_test.shape[0]   # number of data points
p = X_test.shape[1]   # number of input columns
adj_r2 = 1 - ((1-r2)*(n-1)/(n-p-1))
```

### Classification Metrics
- Confusion Matrix (binary)

|                | Predicted 0         | Predicted 1         |
| -------------- | ------------------- | ------------------- |
| **Actual 0**   | TN                  | FP (Type 1 error)   |
| **Actual 1**   | FN (Type 2 error)   | TP                  |

- **FP / Type 1** = predicted positive, actually negative -> a false alarm
- **FN / Type 2** = predicted negative, actually positive -> **this is about missing it**

| Metric | Formula | Question it answers | Use when |
| ------ | ------- | ------------------- | -------- |
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | overall how many did I get right | balanced dataset only |
| **Precision** | $\frac{TP}{TP+FP}$ | when model says positive, how often right? | minimise **FP** -> spam classifier |
| **Recall / TPR** | $\frac{TP}{TP+FN}$ | out of all real positives, how many did I find? | minimise **FN** -> cancer detector |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P+R}$ | harmonic mean of the two | FP and FN equally important |

- All of these have to be **maximised**
- **Accuracy Div** = misleading on imbalanced data + does not tell you *which* type of error you are making
- Precision ⬆ = Recall ⬇ (the threshold trade off)
- Multi class -> `average=` param
	- **macro** = simple average over classes (all classes matter equally) -> use for **nominal** categories
	- **weighted** = average weighted by class support -> use for **ordinal** categories / imbalanced data
	- **micro** = global count of TP/FP/FN (equals accuracy in single label problems)
- **ROC-AUC**
	- Curve of **TPR** $\frac{TP}{TP+FN}$ on y-axis vs **FPR** $\frac{FP}{FP+TN}$ on x-axis, plotted at every threshold
	- AUC = area under it -> 1.0 = perfect, 0.5 = random guessing, <0.5 = worse than random
	- Threshold independent -> tells how well the model **separates** the classes
	- Needs `predict_proba()` not `predict()`
	- For heavily imbalanced data prefer **Precision-Recall AUC** over ROC-AUC
```Python
# binary class classification
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score,
                             precision_score, f1_score, classification_report,
                             roc_auc_score, roc_curve)
y_pred = lr.predict(X_test)

print("Accuracy",         accuracy_score(y_test,y_pred))
print("Confusion matrix", confusion_matrix(y_test,y_pred))
print("Precision - ",     precision_score(y_test,y_pred))
print("Recall - ",        recall_score(y_test,y_pred))
print("F1 score - ",      f1_score(y_test,y_pred))

# everything in one go
print(classification_report(y_test,y_pred))

# multi class classification
# use macro when categories are nominal and weighted when categories are ordinal
precision_score(y_test,y_pred,average='weighted')
recall_score(y_test,y_pred,average='weighted')
f1_score(y_test,y_pred,average='weighted')

precision_score(y_test,y_pred,average='macro')
recall_score(y_test,y_pred,average='macro')
f1_score(y_test,y_pred,average='macro')

# ROC-AUC -> needs probabilities
y_prob = lr.predict_proba(X_test)[:,1]
print("ROC-AUC", roc_auc_score(y_test, y_prob))
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
```

## Hyper Parameter Tunning
[[HyperParamterTunning]]

## Algos

### Linear Regression
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
### Logistic Regression
- **l1_ratio:** 0 for l2 regularization, 1 for l1, and 0 to 1 for elasticnet
- **C:** it is (1/λ) , λ-> studied in regularization, smaller values = stronger Regularization
- **fit_intercept:** True/False , bias term add or not
- **class_weight:** dict/none/'balanced', default value = none
	- dict : {class_label1: weight, class_label2: weight}
	- The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
- **random_state** 
- **solver:** {‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
- **max_iter**int, default=100, Maximum number of times gradient decent/solver run
```python
from sklearn.linear_model import LogisticRegression
l= LogisticRegression(multi_class='multinomial') # for multi class

# polynomial 
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2,include_bias=True) X_trf = poly.fit_transform(X)
```

### Decision Tree
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

### Gradient Decent
- Optimizing technique -> give it a differentiable function, it returns the minima
- Types
	- **Batch:** whole data in one go per update, slow on big data, smooth convergence
	- **Stochastic (SGD):** 1 row per update, fast + noisy, can escape local minima
	- **Mini Batch:** batch of rows per update, the practical middle ground
- Used by = Linear Regression, Logistic Regression, ANN -> so **scale the data (Standardization)**
- learning rate too high = overshoot/diverge, too low = very slow convergence
```Python
from sklearn.linear_model import SGDRegressor, SGDClassifier
sgd = SGDRegressor(
	loss='squared_error',      # 'hinge'/'log_loss' for SGDClassifier
	learning_rate='invscaling', # 'constant','optimal','adaptive'
	eta0=0.01,                 # initial learning rate
	max_iter=1000,
	penalty='l2'               # 'l1','elasticnet',None -> regularization
)
```

### K Nearest Neighbors
- Lazy learner -> stores points at train time, all the work happens at predict time (slow prediction)
- Working: distance of point P to all points -> pick k closest -> majority vote (classification) / mean (regression)
- Keep **k odd** to avoid ties. Low k = over-fitting, high k = under-fitting
- **Must standardize** the data (distance based)
- Limitations: large data, high dimensional data (curse of dimensionality), outliers, imbalanced data, features with different scales
- **n_neighbors:** int, default=5, the k
- **weights:** {'uniform','distance'}, default='uniform', 'distance' = closer neighbours count more
- **metric:** default='minkowski' , **p:** 1 = manhattan, 2 = euclidean
- **algorithm:** {'auto','ball_tree','kd_tree','brute'}, default='auto'
```Python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
knn = KNeighborsClassifier(n_neighbors=3)

# choosing the best k
scores=[]
for i in range(1,16):
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train,y_train)
	scores.append(accuracy_score(y_test, knn.predict(X_test)))
plt.plot(range(1,16),scores)
```

### Naive Bayes
- Based on Bayes theorem with the "naive" assumption -> all the features are independent of each other
- Very fast, works well on text/document classification and high dimensional data
- Which one to use
	- **GaussianNB:** numerical input columns
	- **MultinomialNB:** count/categorical input columns -> document classification
	- **CategoricalNB:** categorical input columns -> recommendation system, medical diagnosis
	- **BernoulliNB:** binary(0/1) input columns
	- Mix of numerical + categorical -> use either GaussianNB or MultinomialNB
- **alpha:** float, default=1.0, Laplace/additive smoothing -> handles the zero probability problem
- **fit_prior:** bool, default=True, learn class prior probabilities or assume uniform
```Python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB
nb = MultinomialNB(alpha=1.0)
```

### SVM
- Finds the hyperplane with the **maximum margin** between classes; points on the margin = support vectors
- Hard margin = no misclassification allowed, Soft margin = allows some misclassification (C controls it)
- **Kernel trick** = project data into higher dimension to make non-linear data linearly separable
- **Scale the data** before using SVM
- **C:** float, default=1.0, inverse of regularization -> small C = wide margin/more errors allowed (under-fit), large C = narrow margin (over-fit)
- **kernel:** {'linear','poly','rbf','sigmoid'}, default='rbf'
- **gamma:** {'scale','auto'} or float, default='scale', how far a single point's influence reaches -> high gamma = over-fitting
- **degree:** int, default=3, only for kernel='poly'
- **epsilon:** float, default=0.1, SVR only -> the tube inside which no penalty is given
```Python
# classification
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# regression
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
```

### Ensemble Learning
- Types: Voting, Bagging (ex- Random Forest), Boosting (ex- Adaboost, Gradient Boosting, XGBoost), Stacking
- Advantage = better performance, less bias and variance, robustness | Disadvantage = computation increases

| Bagging                                    | Boosting                                   |
| ------------------------------------------ | ------------------------------------------ |
| Use models with Low bias and high variance | Use models with high bias and low variance |
| Parallel learning possible                 | Sequential Learning                        |
| Base model weightage is equal              | Base model weightage is not equal          |

#### Voting
- All base models should be **independent/dis-similar** and each should have minimum accuracy of 0.51
- classification = hard voting (mode) / soft voting (average of probabilities) , regression = mean
```Python
from sklearn.ensemble import VotingClassifier, VotingRegressor
estimators = [('lr',LogisticRegression()),('rf',RandomForestClassifier()),('knn',KNeighborsClassifier())]
vc = VotingClassifier(estimators=estimators, voting='hard') # 'soft'
```

#### Bagging
- Bagging = Bootstrapping (random sample of data **with replacement**) + Aggregation (mean = regression, mode = classification)
- Types
	- **Bagging** = row sampling with replacement
	- **Pasting** = row sampling without replacement
	- **Random Subspaces** = column sampling (with/without replacement)
	- **Random Patches** = row + column sampling
```Python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
bag = BaggingClassifier(
	estimator=DecisionTreeClassifier(),
	n_estimators=500,
	max_samples=0.5,          # row sampling
	bootstrap=True,           # row replacement -> False = Pasting
	max_features=0.5,         # column sampling -> Random Subspaces/Patches
	bootstrap_features=False, # column replacement
	random_state=42, n_jobs=-1)
```

#### Random Forest
- Random = Bagging, Forest = group of Decision Trees | fully grown tree = low bias + high variance
- Difference from Bagging: only Decision Trees + column sampling happens at **node level** (bagging = model level)
- **n_estimators:** int, default=100
- **max_features:** {'sqrt','log2',None}, int or float, default='sqrt'
- **bootstrap:** bool, default=True , **max_samples:** int or float, default=None
- plus all the Decision Tree params (max_depth, min_samples_split, ...)
```Python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
```

#### Adaboost
- Stage wise additive method -> weak learners (mostly **decision stumps**, max_depth=1) added one by one
- Each stage increases the weight of misclassified rows, and each model gets its own weightage (alpha)
- **estimator:** object, default=None (None = Decision Tree stump)
- **n_estimators:** int, default=50 , **learning_rate:** float, default=1.0
- **algorithm:** {'SAMME','SAMME.R'}, default='SAMME.R'
```Python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
```

#### Stacking
- Train multiple **base models** on data -> feed their predictions as input to a **meta model** along with the y column
- Over-fitting is the main risk, 2 ways to avoid it
	- **Blending / Hold out:** split train into D1 & D2, train base models on D1, predict D2, train meta model on those predictions (not supported by sklearn)
	- **Stacking / K-Fold:** cross-validated out-of-fold predictions feed the meta model (this is what sklearn does)
```Python
from sklearn.ensemble import StackingClassifier, StackingRegressor
estimators = [
	('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
	('knn', KNeighborsClassifier(n_neighbors=10)),
	('gbdt', GradientBoostingClassifier())
]
clf = StackingClassifier(
	estimators=estimators,
	final_estimator=LogisticRegression(),
	cv=10   # cv acts as the k of k-fold
)
```

### Gradient Boosting
- Sequential stage wise addition -> every next model is trained on the **residual/gradient (loss)** of the previous one
- Prediction = initial guess (mean/log-odds) + learning_rate * (sum of all the tree outputs)

| Adaboost                                      | Gradient Boosting                                         |
| --------------------------------------------- | --------------------------------------------------------- |
| Use decision stumps which have max depth of 1 | Use decision trees which have max depth in between (8-32) |
| we assign a different weight to each model    | we assign a single learning rate for each model           |

- **n_estimators:** int, default=100 , **learning_rate:** float, default=0.1 -> trade off between the two
- **max_depth:** int, default=3
- **subsample:** float, default=1.0, <1.0 = Stochastic Gradient Boosting
- **loss:** 'squared_error'/'absolute_error'/'huber' (reg) , 'log_loss'/'exponential' (clf)
```Python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
gb_reg = GradientBoostingRegressor(
	n_estimators=100,
	learning_rate=0.1,
	max_depth=3,
	random_state=42
)
```

### XGBoost
- Optimized gradient boosting. Why it is better
	- **Performance:** regularised learning objective, handles missing values, sparsity aware split finding, tree pruning, efficient split finding (weighted quantile sketch + approximate tree learning)
	- **Speed:** GPU support, distributed computing, cache awareness, parallel processing, optimized data structure, out of core computing
	- **Flexibility:** cross platform, multiple languages, integration with other libraries, supports all kinds of ML problems
- **n_estimators / learning_rate (eta) / max_depth:** the main 3 knobs
- **reg_lambda (L2), reg_alpha (L1), gamma:** regularization -> gamma = minimum loss reduction needed to make a split
- **subsample:** row sampling , **colsample_bytree:** column sampling
- **scale_pos_weight:** for imbalanced dataset
- **early_stopping_rounds:** stop when validation score stops improving
```Python
from xgboost import XGBClassifier, XGBRegressor
xgb = XGBClassifier(
	n_estimators=100,
	learning_rate=0.1,
	max_depth=3,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_lambda=1,
	random_state=42
)
xgb.fit(X_train, y_train)
```
