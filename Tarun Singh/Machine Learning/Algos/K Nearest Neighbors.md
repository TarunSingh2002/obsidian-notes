
## Intuition 
- can be applicable for nd data
- Working
	- for a point P, get all the distance from p and rest of the point
	- get the k closest points from p
	- in the end do prediction on the basis of majority count (because we are using majority count -> we avoid even k values)
	- For example -> if majority of k points are 1 -> then the output will be 1
- How to select k= the number of neighbours = Try multiple knn model

## Basics
- It is advised to standardise your data if you are using knn
- It is lazy learning technique
	- Because we just store points while training and most of work done while prediction    
	- prediction become slow
- for very high value of k lead to under fitting
- for very low value of k lead to over fitting
- Limitation
	- Not good for large data because it is a lazy learning technique
	- Not good for high dimentional data because of curse of dimensionality
	    - for very high dimensional data(like 500 features) , distance concept becomes quite un reliable
	- Not work good for data with outliers
	- Not work good for Imbalanced data set
	- Not work good for data set that have features with very large scale difference
	    - that's why standardization is suggested
## code
- ![[Pasted image 20260504015108.png]]
```Python
# choosing the best model
for i in range(1,16): 
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train,y_train) 
	y_pred = knn.predict(X_test) 
	scores.append(accuracy_score(y_test, y_pred))
import matplotlib.pyplot as plt 
plt.plot(range(1,16),scores)

# Single model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
```