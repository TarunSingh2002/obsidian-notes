---
tags:
  - MLOps
---
## Basics And Important Points
- It's like Git, but for **data files, models, and experiments**. Git is bad at tracking huge files (CSVs, images, model weights). DVC solves this — it tracks _where_ your data is, while Git tracks _what changed_.
- Git tracks your code + `.dvc` files. DVC tracks your actual data.
- We only need to create a new branch in git not in dvc
- **When Data Changes**: always run first the dvc add and commit and then do the git add+commit+push
- **One git commit = a snapshot of all your `.dvc` files = exactly one hash per data file = one version of each file.**
- For a new data file => dvc add (create hash)+dvc commit+dvc push but for a already tracked data file => dvc commit(update the hash)+dvc push  

---
## DVC Commands

```Python
# setup
pip install dvc 
pip install dvc[s3] 
pip install dvc[gdrive]

#Starting
git init
dvc init
dvc remote add -d myremote <fooldername/path>
dvc remote add -d myremote gdrive://<folder-id> # set up remote
git commit -m "init dvc" # DVC creates a .dvc/ folder (like .git/)

# Add a data file
dvc add data/train.csv # DVC creates a pointer file: data/train.csv.dvc
dvc commit 
git add data/train.csv.dvc
git commit -m "add training data"
dvc push

# other commands of dvc and git 
git checkout <hashid>
dvc push
dvc pull
dvc checkout  # restore data after git checkout

# Adding a tag in git --> Tag point to specific commit
git tag -a v1.0 -m "Entered Raw Data"
git push --tag
# Fetching a tag
git checkout v1.0
dvc pull
```


---
## DVC Pipeline
- Steps: Data ingestion -> Data Preprocessing -> Feature Engineering -> model training -> model evaluation
- Used to written in dvc.yaml file
- A dvc.lock file automatically created
- For a Pipeline 2 files get created **params.yaml** and **dvc.yaml**
- dvc.yaml contains 4 things
	- cmd = Commands to run the module
	- deps = Dependencies (code/data)
	- params = Store Parameters
	- outs = Output paths  (DVC Store these files)
	- metrics = Experiment tracking (DVC Store these files)
```YAML
# dvc.yml file
stages:
  prepare:
    cmd: python prepare.py
    deps:                  # 👀 DVC WATCHES these (if changed → re-run stage)
      - data/raw.csv        # previos file output [note - not previos file] 
      - src/prepare.py      # this file is it self depended on it self
    outs:                        # 📦 DVC TRACKS 
      - data/prepared.csv
    params:                      # ⚙️ DVC reads these from params.yaml
      - data_ingestion.test_size
    metrics:                     # 📊 DVC TRACKS
      - metrics/scores.json 
        
        
# params.yaml 
data_ingestion:
  test_size: 0.20

feature_engineering:
  max_features: 35

model_building:
  n_estimators: 22
  random_state: 2
```
-  **Logic** 
	- **We write the params in parameter file and in .py file we the params.yaml file and use the parameters** 
	- **we write the pipeline in dvc.yml file**  
```Python
# DVC Pipeline Flow
dvc init
dvc repro # run pipeline (smart) 
git add -A
git commit -m ""
git push origin <branch-name>
dvc push

# All the 
dvc dag # visualize pipeline graph

# how to load params value
import yaml
def load_yaml(yaml_path:Path) -> dict:
	with open(yaml_path, 'r') as file:
		params = yaml.safe_lod(file)
	return params
""" 
params['data_ingestion']['test_size'] # 0.20
"""
```


---
## Experiment Tracking

- Code changes for experiment tracking in emodel_evaluation.py file (where model matrices come)

```Python 
from dvclive import Live
import yaml

# code to track the matrics with parameters in the evaluation function
with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy', accuracy_score(y_test, y_pred))
    live.log_metric('precision', precision_score(y_test, y_pred))
    live.log_metric('recall', recall_score(y_test, y_pred))
	# saving the parameters also , here params store all value of params.yaml
	live.log_params(params)
```

- Commands to start the experiment tracking

```Python 
# DVC Pipeline Flow
dvc init
dvc exp run  # instead of dvc repro
git add -A
git commit -m ""
git push origin <branch-name>
dvc push

# All the 
dvc dag # visualize pipeline graph```
```

---

## Important Point -> Git loses DVC Data

### Idea

Git only saves the files that you **commit**.  
So if `dvc.lock` changes but you do **not** commit it, that version is not saved in Git history.

DVC tracks pipeline state through files like:

- `dvc.yaml`
- `dvc.lock`

Git stores only the committed versions of these files.

---
#MLOps