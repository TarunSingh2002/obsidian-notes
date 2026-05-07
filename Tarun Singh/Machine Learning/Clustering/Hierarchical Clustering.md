- Types
	- Agglomerative clustering
	- Divisive clustering

## Agglomerative clustering

- Not applicable for the large data set
### Logic
- ![[Pasted image 20260504015436.png]]
- ![[Pasted image 20260504015500.png]]
- ![[Pasted image 20260504015526.png]]
- ![[Pasted image 20260504015546.png]]
- ![[Pasted image 20260504015603.png]]
- ![[Pasted image 20260504015626.png]]
### Code
- **n_clusters:** The number of clusters to find.
- **linkage:** {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
- **distance_threshold**: float, default=None
	- The linkage distance threshold at or above which clusters will not be merged.
```Python 
# Designing Dendograms
import scipy.cluster.hierarchy as shc 
plt.figure(figsize=(10, 7)) 
plt.title("Customer Dendograms") 
dend = shc.dendrogram(shc.linkage(data, method='ward'))

# Clustering
from sklearn.cluster import AgglomerativeClustering 
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
labels_=cluster.fit_predict(data)
```