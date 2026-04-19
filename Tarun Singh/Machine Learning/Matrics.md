---
tags:
  - machineLearning
---
## Types 
- Classification Metrics
- Regression Metrics
## Regression Metrics
- Types
	- Loss functions
		- MSE
		- MAE
		- RMSE
	- R2 Score
	- Adjusted R2 Score
- MSE (Mean Square Error):
	- $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
	- Div = This is not robust to outlier
	- The unit of the value/loose comes from this method is same as unit of output column of the regression problem but the value is squared
- MAE (Mean Absolute Error):
	- $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
	- Robust to outliers
	- Div = Here we use modules function which is not differentiable at 0
	- The unit of the value/loose comes from this method is same as unit of output column of the regression problem
- Root Mean Square Error:
	- $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
	- The unit of the value/loose comes from this method is same as unit of output column of the regression problem
	- Div = This is not robust to outlier
- R2 Score:
	- $R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$
	- Also know as 'Coefficient of determination" and "goodness of fit"
	- ![[Pasted image 20260419214440.png]]
	- You calculate how much your regression model are better then mean
	- Value lies in between 0 - 1
	- Div = If you add more and more input feature , even if they are irrelevant your R2 score will start increasing. that's why we use Adjusted R2 score
	- we have to make sure the value always towards 1 (if you get 1 = perfect model)
- Adjusted R2 Score:
	- $R^2_{\text{adj}} = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)$ n= number of data point, p = number of columns/independent features
	- If you add more and more input feature , even if they are irrelevant your R2 score will start increasing. that's why we use Adjusted R2 score
```Python
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
y_pred = lr.predict(X_test)

print("MAE",mean_absolute_error(y_test,y_pred))
print("MSE",mean_squared_error(y_test,y_pred))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2",r2_score(y_test,y_pred))

# Adjusted r2 score 
p=df.shape[1] # number of column/indpendent features
n=df.shape[0] # number of data point
r2_score_adju = 1 - ((1-r2)*(40-1)/(40-1-1))
```
## Classification Metrics
- Types
	- Accuracy
	- Confusion Matrix
	- Precision 
	- Recall
	- Roc-Auc Curev
- Accuracy:
	- Div = Misleading for imbalance dataset + does not tell the type of error
	- $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} =  \frac{Number-of-correct-prediction}{Number-of-total-prediction}$
- Confusion Matrix:
	- Binary Class Classification
		- ![[Pasted image 20260419224500.png]]
		- False Negative (Type 2 error): Model predicted negative but actual outcome was positive. This is about Missing.
		- False Positive (Type 1 error): Model predicted positive but actual outcome was negative. 
	- Multi Class Classification
		- ![[Pasted image 20260419225256.png]]
- Precision 
	- For minimising the FP - Email spam classifier (Example)
	- When the model says **positive**, how often is it right?
	- Binary Class Classification
		- $\text{Precision} = \frac{TP}{TP + FP} = \frac{Total-Correct-Positive-Prediction}{Total-Positive-Prediction}$
- Recall / True Positive Rate (TPR)
	- For minimising the FN - Cancer Detector (Example)
	- Out of all truly positive, how much did I find?
	- Binary Class Classification
		- $\text{Recall} = \frac{TP}{TP + FN}  = \frac{Total-Correct-Positive-Prediction}{Total-Actual-Positive-Prediction}$
- F1 Score
	- When FP and FN equally important
	- Binary Class Classification
		- $F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- Multi - Class - Classification with Precision, Recall, F1 score
	- ![[Pasted image 20260420004402.png]]
	- ![[Pasted image 20260420004428.png]]
	- ![[Pasted image 20260420004451.png|641]]  
	- Types
		- macro precision
		- weighted precision
		- macro f1_score
		- weighted f1_score
		- macro recall
		- weighted recall
- AUC ROC Curve
	- 

```Python 
#binary class classification
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

y_pred = lr.predict(X_test)

print("Accuracy", accuracy_score(y_test,y_pred))
print("Confusion matrix", confusion_matrix(y_test,y_pred))
print("Precision - ",precision_score(y_test,y_pred1)) 
print("Recall - ",recall_score(y_test,y_pred1))
print("F1 score - ",f1_score(y_test,y_pred1))

# multi class classification 
# use macro when categories are nominal and use weighed when categories are ordinal
precision_score(y_test,y_pred1,average='weighted')
recall_score(y_test,y_pred1,average='weighted')
f1_score(y_test,y_pred1,average='weighted')

precision_score(y_test,y_pred1,average='macro')    
recall_score(y_test,y_pred1,average='macro')    
f1_score(y_test,y_pred1,average='macro')
```