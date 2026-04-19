---
tags:
  - machineLearning
---
## Types

- Feature Transformation
	- Missing value imputation
	- Encoding/Handling of categorical column
	- Outliers Detection and Handling
	- Feature Scaling
- Feature Construction
- Dimensionality Reduction 
	- Feature Selection 
	- Feature Extraction
## Feature Scaling
- Changing the value of a numerical column to use a common scale, without distorting difference in the range of values or losing information
- By which i mean -> removing the units like Kg, meter etc
- Types
	- Standardisation (Z-score normalisation)
		- Here mean=0 , std = 1, value lies in-between -1 to 1
		- Formula $z = \frac{x - \mu}{\sigma}$ here μ = mean, σ = standard deviation
		- No effect on outliers
		- Use when we depended on gradient decent (like ANN, linear regression or logistic regression) and when the algo use distance between points like K-Mean, K-Nearest-Neighbours
		- How it look graphically 
		  ![[Pasted image 20260409210918.png]]  
		  to ![[Pasted image 20260409210943.png]]
	- Normalisation
		- Types
			- MinMaxScaling
				- value lies in between 0 to 1
				- Formula $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ 
				- Squeeze the impact of the outliers, because It may cause change in data
				- Use when the value lies in between a fix range (like marks of student from 0 to 100)
				- How it look graphically
				   ![[Pasted image 20260409213024.png|533]]
			- Mean Normalization
				- value lies in between -1 to 1
				- Formula $x' = \frac{x - \mu}{x_{\max} - x_{\min}}$
				- Not supported by sklearn
				- used when you wanted centred data
			- MaxAbsScaling
				- formula $x' = \frac{x}{|x_{\max}|}$
				- used  when we have a lot of zeros/sparse data
			- RobustScaling
				- formula $x' = \frac{x - Q_1}{Q_3 - Q_1}$
				- used when we have a lot of outliers
- ```Python 
  # standerdization 
  from sklearn.preprocessing import StandardScaler
  standardscaler= StandardScaler()
  standardscaler.fit(x_train)
  standardscaler.transform(x_test)

  # normalisation - MinMaxScaling
  from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler
  ```



## Encoding of a Categorical column
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

## Column Transformer
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

## Pipeline
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

## Normally Distribution Of Data
- These function normally distribute the data of column
- Type
	- Power Transformer
		- Box Cox transform
		- Yoe johnson transform
	- Function Transformer
		- Log
		- Reciprocal
		- square
		- square root
- Log transform = Right skew data, not work on negative values
- Square transform = Left skew data
- Box cox is a special case of Yoe johnson transform and also box cox transformer not applicable to value <=0
```python
from sklearn.preprocessing import PowerTransformer, FunctionTransformer

# yoe-johnson 
PowerTransformer()
# box cox 
PowerTransformer(method='box-cox')
```

## Missing Value Imputation
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
- KNN Impute:
	- KNN algo used
	-  Adv = Accurate , Div = Slow+Memory usage increase
- Iterative Impute Algo (MICE): 
	- used when data is MAR
	- Adv = Accurate , Div = Slow+Memory usage increase
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

## Outlier Detection
- Algo which calculate the weight get effected by the outliers the most
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
- Z score Treatment
	- Used when data is normally distributed or close to normally distributed
	- value must be in (mean ± 3 * SD)
	- ![[Pasted image 20260413005643.png]]
- Box Plot/Inter-Quartile Proximity rule
	- ![[Pasted image 20260413005811.png]]
- Percentile based approach
	- In this method we Assuming a percentile like For MAX=99 and For Min = 1 and now saying that if i fond a value more then MAX and value less the MIN , i will treat them as a outlier
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

## Encoding Numerical Data
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
## Handling Date And Time Columns
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

## Feature Extraction 
- Example = PCA (Unsupervised Machine Learning Algo)
## Imbalance Dataset
- Problem 
	- Bias towards majority class
	- Some Metrics are not reliable for imbalance dataset, for example accuracy
- Types
	- Under-sampling
		- ![[Pasted image 20260414004542.png]]
		- Div = loss of data 
	- Oversampling
		- ![[Pasted image 20260414005611.png]]
		- Div= duplication of data cause over-fitting
	- SMOTE (Synthetic Minority Oversampling Technique)
		- Create new points for minority class instead of duplication
		- ![[Pasted image 20260414010120.png]]
		- In SMOTE a minority class data point is first selected, then its k-nearest neighbors are identified and finally a new synthetic data point is generated between the selected point and one of its neighbors.
		- Div = not applicable on categorical dataset, computational heavy, Depended on neighbors values, sensitive to outliers, no gureente of coorect points
	- Ensemble Methods
		- BalancedRandomForestClassifier + BalancedBaggingClassifier
			- ![[Pasted image 20260414011128.png|569]]
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