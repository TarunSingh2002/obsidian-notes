## Working 

- Sequential stage wise addition
- Gradient boosting uses the gradient (loss) of model as a input to the its next model and it goes on 
- ![[Pasted image 20260504012818.png]]
- ![[Pasted image 20260504012836.png]]
- ![[Pasted image 20260504012925.png]]
## Adaboost Vs Gradient Boosting

| Adaboost                                      | Gradient Boosting                                         |
| --------------------------------------------- | --------------------------------------------------------- |
| Use decision stumps which have max depth of 1 | Use decision trees which have max depth in between (8-32) |
| we assign a different weight to each model    | we assign a single learning rate for each model           |
## Code

```Python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier 

gb_reg = GradientBoostingRegressor(
	n_estimators=100, 
	learning_rate=0.1, 
	max_depth=3, 
	random_state=42
)
```