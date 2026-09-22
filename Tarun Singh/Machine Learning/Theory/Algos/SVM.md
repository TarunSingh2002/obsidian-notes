## SVM Classification

- ![[Pasted image 20260504091529.png]]
- ![[Pasted image 20260504091552.png]]
## Code

```Python
# classification 
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# regression
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
```