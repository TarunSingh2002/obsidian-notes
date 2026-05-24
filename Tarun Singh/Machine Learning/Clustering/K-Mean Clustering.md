
## Working
- Here k mean = Number of cluster
- ![[Pasted image 20260504011508.png]]
- ![[Pasted image 20260504011526.png]]
- ![[Pasted image 20260504011601.png]]
- ![[Pasted image 20260504011641.png]]
- How to find number of clusters = generally the number of cluster is less then 20

## Code
### Elbow Method
```Python 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11): 
	km = KMeans(n_clusters=i) 
	km.fit_predict(df) 
	wcss.append(km.inertia_)
plt.plot(range(1,11),wcss) # Now we know Number of cluster
```
### Clustering
```Python 
X = df.iloc[:,:].values 
km = KMeans(n_clusters=4) 
y_means = km.fit_predict(X)
```