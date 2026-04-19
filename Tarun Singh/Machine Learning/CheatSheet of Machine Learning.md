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
### Column Transformer
```python
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(transformers = [
	('tnf1', OrderinalEncoder(), ['col1', 'col2']),
	('tnf2', _, ['col3', 'col4']),
	('tnf3', _, ['col5', 'col6']),
	], remainder='passthrough/drop')
""" 
Note -> here instead of column name we can also give column index [0,1]
"""
```
### Pipeline
```python
from sklearn.pipeline import Pipeline,make_pipeline

# trandformer 1
trf1 = ColumnTransformer( transformers=[ 
		('impute_age',SimpleImputer(),[2]), 
		('impute_embarked',SimpleImputer(),[6]) 
	],remainder='passthrough')
	
# trandformer 2
trf2 = ColumnTransformer( transformers=[ 
		('ohe',OneHotEncoder(),[1,2])
	],remainder='passthrough')

trf3 = DecisionTreeClassifier()

pipe = Pipeline([ 
		('trf1',trf1), 
		('trf2',trf2), 
		('trf3',trf3),
	])
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
### Normal Distribution - QQ Plot

### Matrics

### Questions?
- how to find a column following a normal distribution or some other distribution ? - for normal 'qq plot'?
- Changing the numerical column distribution to normal -> helpful in which kind of algos
- what is mcar, mar, mnar
- when we are doing missing value imputation -> how can we find out what kind of missing data we have -> like mcar, mar, mnar
- What kind of imputation technique we need to apply if we find out what kind of missing value we have like -> mcar, mar, mnar
- How can i do the compariosn of before and after of the both numerical and categorical column to find out does it not cause in -> chaneg in distribution or ration right? 
- Multivritae imputer will get applied to all column missing value right we cant apply then on one column right? and how this iterative imputer work in short ok -> and ist imp parameter also most imp paramter to know.
- Th complete outlier deteiction and handling part -> i ahve a lot questions like what if i do not have any indepth knowlege of the column and i know it is not nomally distributed then only one way -> interQunatile proximity based approach left right? -> now -> but for all the column is it correct to just remove the extreme value on the basis of this?? -> and also the traetmnt of those value need pandas code which will never fit in the pipeline of the sklearn right? is the correct way of handling the outliers? (like not having them in pipeline)
- How to knwo when to convert a numerical column to categorical column ? when they have lot of extremen values?? like no range?