---
tags:
  - machineLearning
  - workflow
  - decisionGuide
---

# The ML Project Decision Guide (Regression & Classification)
---
## PART 0

Every supervised project is the same 7 phases, always in this order:

```
1. FRAME      → What am I predicting? Regression or classification? What's the metric?
2. PEEK       → shape, dtypes, head, target distribution. NO cleaning yet.
3. SPLIT      → train_test_split FIRST. Everything below is learned on train only.
4. EDA        → understand columns: univariate → bivariate (vs target) → multivariate.
5. PREPROCESS → missing → outliers → encode → scale → transform. (fit on train only)
6. MODEL      → dumb baseline → linear → tree/ensemble. Compare with CV.
7. TUNE       → Optuna on the best 1–2 models. Final eval on test ONCE.
```
---
## PART 1 — FRAME the problem (5 minutes, no code)

Answer these before touching pandas:

- **Target column?** What am I predicting.
- **Regression or classification?** Is the target a continuous number (price, temperature) or a category (churn yes/no, species)?
- **What metric?** This decides everything downstream.

Concrete metric defaults (don't overthink it):

| Problem | Default metric | When to switch |
|---|---|---|
| Regression | **RMSE** (penalizes big errors) | Use **MAE** if outliers in target shouldn't dominate; **R²** for "how much variance explained" |
| Balanced classification | **accuracy** + look at confusion matrix | — |
| Imbalanced classification | **F1** or **ROC-AUC**, never accuracy | Use **recall** if missing a positive is costly (fraud, disease); **precision** if false alarms are costly |


---

## PART 2 — PEEK at the data (look, don't touch)

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")   # one consistent look for every project

df = pd.read_csv("data.csv")
df.shape                 # how many rows/cols — sets expectations
df.head()                # eyeball actual values
df.info()                # dtypes + non-null counts (missing data hint #1)
df.isnull().mean().mul(100).sort_values(ascending=False)
df.duplicated().sum()
```

What you are looking for here (just noting it, not fixing):
- Columns with huge missing %.
- Duplicated rows
- Shape

---

## PART 3 — SPLIT now (before EDA, before cleaning)

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if classification else None   # stratify keeps class ratio in both splits
)
```
---

## PART 4 — EDA: the systematic three-pass method

## PART 5 — PREPROCESSING: the fixed order (this is your flowchart)

Here is the canonical order. **Do it in this sequence every time.** The reasons for the order are baked in.

```
   ┌─────────────────────────────────────────────────────────────┐
   │  STEP A. Drop junk columns (IDs, constants, leaky columns)    │
   └─────────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────────┐
   │  STEP B. Handle MISSING values   (impute)                     │
   │     • do this BEFORE outliers, because outlier math (mean,    │
   │       quantiles) breaks on NaNs                               │
   └─────────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────────┐
   │  STEP C. Handle OUTLIERS    (cap; rarely trim)                │
   │     • only matters for linear/distance models; skip for trees │
   └─────────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────────┐
   │  STEP D. ENCODE categoricals  (ordinal / one-hot)             │
   └─────────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────────┐
   │  STEP E. TRANSFORM skew (log / power)  — optional, linear only│
   └─────────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────────┐
   │  STEP F. SCALE  (StandardScaler) — linear/distance only       │
   └─────────────────────────────────────────────────────────────┘
```

And the crucial meta-rule: **all of B–F live inside a `ColumnTransformer` + `Pipeline`** so they auto-fit on train and apply to test/CV with zero leakage. (More on the one exception — outliers — below.)

### STEP B — Missing values: the decision rule

You asked: "there are so many ways to fill nulls, which one?" Here's the concrete rule. Don't agonize — pick by column type and missing %.

```
Is the column missing > ~40% of values?
   └ YES → drop the column (too little signal).
   └ NO  ↓
Is it NUMERIC or CATEGORICAL?
   ├ NUMERIC →  SimpleImputer(strategy='median')      ← median is the safe default
   │            (median beats mean: robust to skew & outliers)
   └ CATEGORICAL → SimpleImputer(strategy='most_frequent')
                   or strategy='constant', fill_value='Missing'
                   (use 'Missing' when absence might itself be meaningful)
```

That covers 95% of cases concretely. Save KNNImputer / IterativeImputer (MICE) for when simple imputation visibly hurts your CV score — they're slower and rarely change much for a beginner project.

**About MCAR / MAR / MNAR (your question):** these are *theoretical* labels for *why* data is missing:
- **MCAR** (Missing Completely At Random): missingness unrelated to anything. Safe to impute simply or drop rows.
- **MAR** (Missing At Random): missingness depends on *other observed* columns (e.g. income missing more for one job type). Impute, optionally add a "was-missing" indicator flag.
- **MNAR** (Missing Not At Random): missingness depends on the *unseen value itself* (e.g. high earners hide income). Here the *fact of missingness* is information → add a missing-indicator column.

The honest truth: **you usually can't prove which one you have** from the data alone. So the practical move is: add a binary "missing flag" column when missingness is substantial (`SimpleImputer(add_indicator=True)`), impute the value with median/mode, and **let cross-validation tell you if it helped.** Don't get stuck trying to philosophically classify it.

**How to check imputation didn't wreck the column:** compare describe() and a histogram before vs after. The mean/median and overall shape should stay similar; if imputing collapses all variance into one spike, reconsider.

### STEP C — Outliers: the honest, concrete answer

You had a great, frustrated question here. Let me answer all of it directly:

**First: do you even need to handle outliers?**
- **Tree-based models (DecisionTree, RandomForest, XGBoost): basically ignore outliers.** They split on order, not magnitude. → **If you're using trees, skip outlier handling entirely.**
- **Linear models, KNN, SVM, anything with distance or gradients:** outliers can dominate. → handle them.

So outlier handling is *conditional on model family*, which is why it's not always done.

**Second: which detection method, when I don't know the column deeply?**
```
Is the column roughly normal (QQ plot ~ straight)?
   └ YES → Z-score method (cap beyond mean ± 3·std)
   └ NO  → IQR method (cap beyond Q1−1.5·IQR and Q3+1.5·IQR)   ← your safe default
```
The **IQR method is your default** when you don't know the distribution, exactly as you guessed. It doesn't assume normality.

**Third: trim or cap?** **Cap (clip), don't trim.** Trimming deletes rows and shrinks your data; capping keeps the row but pulls the extreme value to the boundary. Capping is the more concrete, less destructive default.

**Fourth — your sharpest question: "is it correct to just blindly cap extremes on every column? and the pandas code won't fit in a sklearn pipeline!"**

Two-part answer:
1. **Don't blindly cap every column.** Only cap a column if (a) you're using a linear/distance model AND (b) the boxplot/QQ showed real extreme values. A column that's already well-behaved doesn't need it. Also never cap a column where the extreme values are the *point* (e.g. fraud amounts in fraud detection).
2. **It CAN go in a pipeline** — you were right that raw pandas `np.where` can't, but the fix is a transformer that does the clipping. Cleanest concrete option:

```python
from sklearn.preprocessing import FunctionTransformer

def iqr_clip(X):
    X = pd.DataFrame(X)
    for c in X.columns:
        q1, q3 = X[c].quantile(0.25), X[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        X[c] = X[c].clip(lo, hi)
    return X.values

outlier_capper = FunctionTransformer(iqr_clip)   # drop this into your ColumnTransformer
```

This is leakage-safe enough for a project, and it lives *inside* the pipeline so you never touch the test set manually. (Purists compute the bounds during `.fit`; for a learning project the above is fine and concrete.)

### STEP D — Encoding (this is already correct on your cheat sheet)

```
Categorical column type?
   ├ ORDINAL (has natural order: Low<Med<High) → OrdinalEncoder(categories=[[...ordered...]])
   ├ NOMINAL, few categories (≤ ~10)           → OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
   └ NOMINAL, high cardinality (many uniques)   → don't one-hot (explodes columns).
                                                  Use target/frequency encoding, or for trees just OrdinalEncoder.
```
`handle_unknown='ignore'` matters: it stops your pipeline crashing when the test set has a category the train set never saw.

The target column for **classification** gets `LabelEncoder` (or just leave it if already 0/1) — encode y separately, outside the feature pipeline.

### STEP E — Transform skew (optional, linear models only)

Trees don't care about skew → skip for trees. For linear models, if Pass-1 EDA flagged `skew > 1`:
- right-skew, all-positive column → log: `FunctionTransformer(np.log1p)` (`log1p` handles zeros).
- general fix → `PowerTransformer(method='yeo-johnson')` (works with negatives and zeros, unlike box-cox). **Yeo-Johnson is your concrete default** because it always works.

### STEP F — Scale (linear/distance models only)

```
Model is linear / KNN / SVM / uses gradient descent?  → StandardScaler()   ← default
Model is tree-based (RF, XGBoost, DecisionTree)?      → no scaling needed.
```
`StandardScaler` is your one default. Reach for `RobustScaler` only if a column still has heavy outliers after capping, `MinMaxScaler` only if an algorithm specifically needs a 0–1 range. Don't overthink it — StandardScaler 95% of the time.

### Putting B–F together (the concrete pipeline skeleton)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    # ('outlier', outlier_capper),     # add only for linear/distance models
    ('scale',  StandardScaler()),       # drop this line for tree models
])

categorical_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe',    OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
])

preprocess = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols),
], remainder='drop')

model = Pipeline([
    ('prep', preprocess),
    ('algo', None),   # plug a model in here
])
```

Now swapping models is one line, and there is zero leakage because the whole thing fits on train.

---

## PART 6 — MODELING: baseline → linear → trees (always compare with CV)

Never trust a single train/test number. Use cross-validation on the training set to compare models.

```python
from sklearn.model_selection import cross_val_score

def evaluate(algo, scoring):
    model.set_params(algo=algo)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
    print(f"{algo.__class__.__name__}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**The ladder — climb it in order:**

**Regression** (`scoring='neg_root_mean_squared_error'`):
```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

evaluate(DummyRegressor(strategy='mean'), ...)   # the "I learned nothing" baseline
evaluate(LinearRegression(), ...)                 # is the relationship linear?
evaluate(Ridge(), ...)                            # linear + regularization
evaluate(RandomForestRegressor(random_state=42), ...)
evaluate(XGBRegressor(random_state=42), ...)
```

**Classification** (`scoring='f1'` or `'roc_auc'` if imbalanced, else `'accuracy'`):
```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

evaluate(DummyClassifier(strategy='most_frequent'), ...)
evaluate(LogisticRegression(max_iter=1000), ...)
evaluate(RandomForestClassifier(random_state=42), ...)
evaluate(XGBClassifier(random_state=42), ...)
```

**How to read the ladder:**
- If you can't beat the **Dummy** baseline, something is broken (leakage, wrong metric, useless features).
- If **linear ≈ trees**, the relationship is mostly linear → keep the simpler linear model (faster, interpretable). This empirically confirms what your EDA scatterplots suggested.
- If **trees ≫ linear**, the relationship is non-linear → go with the ensemble.
- High CV **std** = unstable model / too little data.

This *is* the answer to "how do I decide which model works best" — you don't decide by intuition, you let the CV ladder decide, with EDA setting your expectation.

**Imbalanced classification** (from your cheat sheet): the clean concrete default is `class_weight='balanced'` inside LogisticRegression/RandomForest — try that *before* reaching for SMOTE. It's one parameter and no extra library.

---

## PART 7 — TUNE the winner (Optuna) and final test

Tune only your **top 1–2 models** from the ladder — not all of them.

- Put the **whole pipeline** inside the Optuna objective and use `cross_val_score` inside it, so tuning is still leakage-free.
- After tuning, retrain the best pipeline on **all** of `X_train`, then call `.predict(X_test)` **exactly once** and report that number. That single test score is your honest estimate of real-world performance. If you tweak things after seeing the test score, you've contaminated it.

You already know Optuna, so I'll stop here — the only new discipline is *tune inside CV, touch test once.*

---

## The one-page mental flowchart (print this)

```
FRAME    → reg or clf? pick metric.
PEEK     → shape, info, describe, %missing, plot target.
SPLIT    → train_test_split(stratify if clf). Train-only from here.
EDA
  Pass1 univariate → hist+box (num), countplot (cat), read skew/kurt/QQ
  Pass2 bivariate  → feature vs TARGET → linear-looking? trees-looking?
  Pass3 multivar   → corr heatmap → drop redundant (>0.9)
  → write a 5-line plan.
PREPROCESS (fixed order, inside Pipeline)
  A drop junk → B impute (median/most_frequent) → C cap outliers (IQR, linear only)
  → D encode (ordinal/onehot) → E power-transform skew (linear only) → F scale (linear only)
MODEL    → dummy → linear → RF → XGB, compare via 5-fold CV on chosen metric.
TUNE     → Optuna on top model (CV inside). Predict test ONCE. Done.
```

---

## Quick answers to your leftover cheat-sheet questions

- **Find if a column is normal?** QQ plot + skew/kurt near 0. Formal tests unnecessary for ML.
- **Skew-fixing helps which algos?** Linear/logistic regression (and anything assuming linearity). Useless for trees.
- **Which imputer for MCAR/MAR/MNAR?** You rarely can prove the type. Default: median (num) / most_frequent (cat) + add a missing-indicator when missingness is heavy. Let CV judge.
- **Compare column before/after imputation?** `describe()` + histogram before vs after; mean/median/shape should be stable.
- **Does IterativeImputer apply to all columns?** It models each column from the others, so yes it operates across the whole numeric frame, not one column in isolation. Key param: `max_iter`. Use only if simple imputation underperforms.
- **Outliers without deep column knowledge?** IQR method + cap (clip), and only for linear/distance models. Don't cap blindly; don't cap meaningful extremes.
- **Outlier handling in a pipeline?** Yes — wrap the clip logic in a `FunctionTransformer` (code in Step C).
- **When to convert numeric → categorical (binning)?** Three triggers: (1) the relationship to target is step-like not smooth (EDA scatter shows plateaus), (2) you want a linear model to capture non-linearity cheaply, (3) the raw scale is noisy and only the *range* matters (e.g. age → age-group). Use `KBinsDiscretizer(strategy='quantile')` as the default. Note trees do their own binning, so this mainly helps linear models.
- **Do I need p-tests / t-tests?** For predictive ML, almost never. They belong to inferential stats / A-B testing. The only ML cameo is optional statistical feature selection (`SelectKBest` with chi2/ANOVA).
- **Multi-stage stacking in sklearn?** Use `StackingRegressor` / `StackingClassifier` — pass a list of base `estimators` and a `final_estimator`; for deeper stacks, a StackingClassifier can itself be a base estimator inside another one.
