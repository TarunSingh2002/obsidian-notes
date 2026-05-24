---
tags:
  - machineLearning
---
### Types
- Optuna
- Grid Search CV
- Random Search CV
### Optuna-Basics
- Use Bayesian Search
- Key Terms
	- Study
		- Collection if trails
	- Trial
		- In each trial we try different value of hyper-parameters
	- Trial Parameters
		- Each trial has a different value for hyper-parameter which we called Trial Parameter
	- Objective Function
		- It take hyper-parameter as input and return a value (like accuracy, loss, or other metrics) that optuna tries to optimize
	- Sampler
		- A algo whci suggest which hyper - parameter should be evaluated next, by default optuna use TPE (Tree-structured-Extimator)
- Documentation links for each function used

| Functions           | Link                                                                                                                    |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| study.optimize()    | [Link](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize) |
| optuna.create_study | [Link](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html)                            |
| trial               | [Link](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)                             |

### Optuna-Basics-For-ML
#### Example
```Python
# !pip install optuna

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Define the objective function
def objective(trial):
	# Suggest values for the hyperparameters
	n_estimators = trial.suggest_int('n_estimators', 50, 200)
	max_depth = trial.suggest_int('max_depth', 3, 20)
	# Create the RandomForestClassifier with suggested hyperparameters
	model = RandomForestClassifier(
	n_estimators=n_estimators,
	max_depth=max_depth,
	random_state=42
	)
	# Perform 3-fold cross-validation and calculate accuracy
	score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
	return score # Return the accuracy score for Optuna to maximize
	
	
# Create a study object and optimize the objective function

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler()) # We aim to maximize accuracy

study.optimize(objective, n_trials=50) # Run 50 trials to find the best hyperparameters

print(f'Best trial accuracy: {study.best_trial.value}')
print(f'Best hyperparameters: {study.best_trial.params}')



# Train the model with the best aramter form optuna
from sklearn.metrics import accuracy_score
# Train a RandomForestClassifier using the best hyperparameters from Optuna
best_model = RandomForestClassifier(**study.best_trial.params, random_state=42)
# Fit the model to the training data
best_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = best_model.predict(X_test)
# Calculate the accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
# Print the test accuracy
print(f'Test Accuracy with best hyperparameters: {test_accuracy:.2f}')
```

#### Some Functions Used basic Details
```Python
# Trial 
category=  ["linear", "poly", "rbf"] # / [True, False], [1,2,3] 
kernel = trial.suggest_categorical("kernel",)

suggest_int("n_layers", 1, 3) # -> 1 | 2 | 3
trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True) # -> float in [1e-5, 1e-2] (log scale)
trial.suggest_float("dropout", 0.0, 0.5) # -> float in [0.0, 0.5]



# Create Study - 
study = optuna.create_study(direction='maximize/minimize', sampler=optuna.samplers.TPESampler())

```

#### Optuna Visualizations
```Python 
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice, plot_contour, plot_param_importances

plot_optimization_history(study).show()
plot_parallel_coordinate(study).show()
plot_slice(study).show()
plot_contour(study).show()
plot_param_importances(study).show()
```

### Optuna-Multiple-ML-Models

#### Example
```Python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Define the objective function for Optuna

def objective(trial):

	# Choose the algorithm to tune
	classifier_name = trial.suggest_categorical('classifier', ['SVM', 'RandomForest', 'GradientBoosting'])
	if classifier_name == 'SVM':
		# SVM hyperparameters
		c = trial.suggest_float('C', 0.1, 100, log=True)
		kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
		gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
		model = SVC(C=c, kernel=kernel, gamma=gamma, random_state=42)
	elif classifier_name == 'RandomForest':
		# Random Forest hyperparameters
		n_estimators = trial.suggest_int('n_estimators', 50, 300)
		max_depth = trial.suggest_int('max_depth', 3, 20)
		min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
		min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
		bootstrap = trial.suggest_categorical('bootstrap', [True, False])
		model = RandomForestClassifier(
			n_estimators=n_estimators,
			max_depth=max_depth,
			min_samples_split=min_samples_split,
			min_samples_leaf=min_samples_leaf,
			bootstrap=bootstrap,
			random_state=42
			)
	elif classifier_name == 'GradientBoosting':
	# Gradient Boosting hyperparameters
	n_estimators = trial.suggest_int('n_estimators', 50, 300)
	learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
	max_depth = trial.suggest_int('max_depth', 3, 20)
	min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
	min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
	model = GradientBoostingClassifier(
		n_estimators=n_estimators,
		learning_rate=learning_rate,
		max_depth=max_depth,
		min_samples_split=min_samples_split,
		min_samples_leaf=min_samples_leaf,
		random_state=42
		)
	# Perform cross-validation and return the mean accuracy
	score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
	return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```
