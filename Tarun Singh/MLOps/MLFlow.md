---
tags:
  - MLOps
---
## Run vs Experiment 
- A experiment can have multiple runs
- A project can have multiple experiment

## Basics Commands of MLFlow
```Python 
pip install mlflow
mlflow ui
```

## Basics code of MLFlow
```Python
import mlflow
import mlflow.sklearn

# this added at the top level 
mlflow.set_experiment('your-experiment-name')
# mlflow.autolog()



# this added where you wnat to log somthing
with mlflow.start_run():
	# make the model like rf=randomforest(), then rf.fit(), then predict, 
	# find accuracy
	mlflow.log_metric('accuracy',accuracy)   # here we can just give dict also
	mlflow.log_param('max_depth', max_depth) # here we can just give dict also
	mlflow.log_param('n_estimator', n_estimator)
	mlflow.log_artifact('a.png') # a.png is the path of the artifact
	
	tags={
	'a':'tage1',
	'b':'tag2'
	}
	mlflow.set_tags(tags)
	
	mlflow.sklearn.log_model(rf, 'random forest model')
```

## MLFlow multi-file 
- Your **first stage** (data loading) creates the run and writes `run.info.run_id` into a small file like `run_id.txt`
- Every **later stage** (preprocessing, training, evaluation) reads that file and opens `mlflow.start_run(run_id=that_id)` — all their params, metrics, and artifacts land in the **same single run**

```Python 
# In the first stage:
with mlflow.start_run() as run:
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

# In every later stage:
run_id = open("run_id.txt").read().strip()
with mlflow.start_run(run_id=run_id):   # resumes the SAME run
```

## MLFlow - HyperParameter tunning

```Python 
import mlflow

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

mlflow.set_experiment('breast-cancer-rf-hp')
with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)
    # log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])
    
    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    # Log params
    mlflow.log_params(best_params)
    # Log metrics
    mlflow.log_metric("accuracy", best_score)
```

## Remote - MLFlow server - dagshub

```Python
pip install dagshub
```

```Python
import dagshub
dagshub.init(... # you will find this command, once you set up the repo of dagshub
dagshub.set_tracking_url("url") # you will get this url form the repo of dagshub 
```

#### Dagshub flow
- add the repo to dagshub
- get the tracking link and code from their also
- Do the pip install and use the code