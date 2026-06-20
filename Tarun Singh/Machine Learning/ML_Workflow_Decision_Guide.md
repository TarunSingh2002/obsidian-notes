---
tags:
  - machineLearning
  - workflow
  - decisionGuide
---

# The ML Project Decision Guide (Regression & Classification)

> The goal of this doc is NOT to teach you new tools. You already have the tools (your cheat sheet proves it). The goal is to give you a **fixed order of operations** and a **decision rule for every fork in the road**, so that on every new Kaggle dataset you stop asking "what do I do now?" and instead run a checklist.
>
> One library for all visuals: **seaborn** (with `matplotlib.pyplot` for layout). Pick concrete defaults. Move fast.

---

## PART 0 — The 30,000-foot view (memorize this)

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

The single most important rule, the one that fixes 80% of beginner anxiety:

> **Split before you explore or clean. Fit every transformer on `X_train` only, then `.transform()` the test set.** Otherwise you leak information from the test set and your scores are a lie.

This also answers a hidden question you had: "what order — null fill or outliers first?" The order is fixed (Part 5). You don't decide it per project; you follow it.

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

Write the answer down. Everything you do later is "does this improve my chosen metric on cross-validation?"

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
df.describe().T          # numeric summary: min/max/mean/std — spot weird ranges
df.describe(include='object').T   # categorical summary: unique counts, top value
df.isnull().mean().mul(100).sort_values(ascending=False)   # % missing per column
```

What you are looking for here (just noting it, not fixing):
- Columns that are secretly the wrong dtype (a number stored as object, a date stored as object).
- Columns with huge missing %.
- ID-like columns (unique per row) → drop them, they have no predictive value.
- The **target's** shape (next).

**Look at the target immediately:**

```python
# Regression target
sns.histplot(df['target'], kde=True)
plt.show()
df['target'].skew()       # > 1 or < -1 means strongly skewed → consider log later

# Classification target
df['target'].value_counts(normalize=True)   # class balance → imbalance check
sns.countplot(x='target', data=df)
plt.show()
```

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

From here on, **you only look at `X_train` / `y_train`** when deciding what to do. The test set does not exist until the very end.

> You *can* explore the full `df` for pure curiosity, but every **decision** (the median to impute, the outlier cap, the scaler's mean) must be computed from train only — which is exactly why sklearn transformers have `.fit()` (learn from train) and `.transform()` (apply to both). This is the whole reason we use the Pipeline in your cheat sheet.

---

## PART 4 — EDA: the systematic three-pass method

This is the part you said scares you. Here's the fix: EDA is **always three passes, in this order.** You don't improvise.

### Pass 1 — Univariate (one column at a time): "What is this column?"

Split your columns into two lists first:

```python
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
```

**For each numeric column**, run the same 4 things:

```python
col = 'age'
print(X_train[col].describe())
print("skew:", X_train[col].skew(), " kurtosis:", X_train[col].kurt())

fig, ax = plt.subplots(1, 2, figsize=(10,4))
sns.histplot(X_train[col], kde=True, ax=ax[0])   # shape of distribution
sns.boxplot(x=X_train[col], ax=ax[1])            # outliers at a glance
plt.show()
```

**For each categorical column:**

```python
col = 'city'
print(X_train[col].value_counts())
print("cardinality:", X_train[col].nunique())     # how many categories
sns.countplot(y=col, data=X_train,
              order=X_train[col].value_counts().index)
plt.show()
```

**Reading the statistics (your direct question — what do these numbers mean):**

- **Mean vs Median:** if mean ≫ median, the column is right-skewed (a few big values pulling the average up). If they're close, it's roughly symmetric.
- **Skewness** (`.skew()`):
  - ≈ 0 → symmetric (good for linear models).
  - **> +1 → right/positive skew** (long tail to the right; e.g. income, prices). Candidate for **log transform**.
  - **< −1 → left/negative skew** (long tail to the left). Candidate for **square** transform.
  - Between −1 and +1 → fine, leave it.
- **Kurtosis** (`.kurt()`, this is *excess* kurtosis in pandas):
  - ≈ 0 → tails like a normal curve.
  - **> 0 → heavy tails / sharp peak** → expect more outliers than normal.
  - **< 0 → light tails / flat** → fewer extreme values.
  - Practical use: high kurtosis = pay extra attention to outlier handling for this column.
- **Standard deviation:** spread. A column whose std is near 0 is almost constant → it carries no information → drop candidate.

**"How do I know if a column is normally distributed?"** Three signals, in increasing rigor:
1. The histogram looks like a bell.
2. `skew` near 0 **and** `kurt` near 0.
3. The **QQ plot** is a straight diagonal line:

```python
import scipy.stats as stats
stats.probplot(X_train[col], dist="norm", plot=plt)
plt.show()
# Points hug the 45° line → normal. Points curl off at the ends → skewed/heavy tails.
```

You generally **don't need a formal normality test (Shapiro, etc.)** for ML — normality is not required by tree models at all, and linear models care about residual normality, not feature normality. Use the QQ plot as a visual gut-check and move on. (This answers your "do I need p-tests / t-tests" question: **for predictive ML, almost never.** Hypothesis tests belong to inferential statistics / A-B testing, not to building a Kaggle predictor. The one place a test sneaks in is feature selection — e.g. chi-square or ANOVA F-test via `SelectKBest` — and even there it's optional.)

### Pass 2 — Bivariate (each feature vs the TARGET): "Does this column help predict y?"

This is the highest-value pass. It tells you which features matter and hints at which model family will win.

**Numeric feature vs target:**

```python
# Regression target → scatter
sns.scatterplot(x='feature', y='target', data=pd.concat([X_train, y_train], axis=1))
plt.show()

# Classification target → compare distributions per class
df_tr = pd.concat([X_train, y_train], axis=1)
sns.boxplot(x='target', y='feature', data=df_tr)      # or sns.kdeplot with hue
plt.show()
```

**Categorical feature vs target:**

```python
# Regression target → mean of y per category
sns.barplot(x='cat_feature', y='target', data=df_tr)
plt.show()

# Classification target → stacked proportion
pd.crosstab(df_tr['cat_feature'], df_tr['target'], normalize='index').plot(kind='bar', stacked=True)
plt.show()
```

**What you're learning here — and this answers your big question "how do I decide linear vs tree?":**

- In the **scatter (numeric vs regression target)**: do the points fall roughly along a **straight line**? → linear models (LinearRegression / Ridge / Lasso) will do well. Is it a **curve, plateau, or step**? → tree-based models (DecisionTree → RandomForest → GradientBoosting/XGBoost) will do better because they capture non-linearity automatically.
- If a **boxplot of feature-per-class** shows the boxes clearly separated → that feature is a strong predictor.
- If relationships look like clean lines/planes → **linear**. If they look like thresholds ("price jumps after 2000 sqft"), interactions, or anything non-monotonic → **trees**.

You don't have to *guess* in the end — you'll test both families in Part 6. EDA just tells you what to *expect* so you're not flying blind.

### Pass 3 — Multivariate (columns vs each other): "Are features redundant? Any structure?"

```python
# Correlation heatmap (numeric columns)
corr = X_train[num_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.show()
```

**Reading correlation:**
- Value runs −1 to +1. Near **+1** = move together, near **−1** = move oppositely, near **0** = no *linear* relationship (there could still be a non-linear one — correlation only catches straight-line relationships).
- **Two features correlated > ~0.9 with each other** = multicollinearity. They're saying the same thing. For **linear models** this is harmful (unstable coefficients) → drop one or use Ridge. For **tree models** it's mostly harmless but you can still drop one for simplicity.
- A feature **highly correlated with the target** = a strong predictor, good news.

`sns.pairplot(df_tr, hue='target')` is the "show me everything" version for small datasets (≤ ~6 numeric columns); skip it when you have many columns (too slow, unreadable).

> **End of EDA, write a 5-line summary to yourself:** which columns look predictive, which look useless, which are skewed, which have outliers, which have missing values, and whether relationships look linear or non-linear. That summary *is* your preprocessing plan.

---

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
