---
tags:
  - machineLearning
  - EDA
  - deepDive
---
---
# The Ultimate EDA Guide

```python
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
import scipy.stats as stats
sns.set_theme(style="whitegrid")

TARGET = 'price'                                  # ← set once per project
df_tr = pd.concat([X_train, y_train], axis=1)     # the one dataframe we plot from
plot_df = df_tr.sample(min(len(df_tr), 20_000), random_state=42)  # for scatter plots only
```
---

# PART 1 — What EDA actually is

EDA answers exactly **three questions**, in this order:

1. **Identity** — what _is_ each column? (its true type, its shape, its quality) → _univariate_ = "one column at a time"
2. **Signal** — does each column relate to the **target**, and what _shape_ is that relationship? → _bivariate_ = "two columns at a time (feature vs target)"
3. **Structure** — do columns repeat each other or team up? → _multivariate_ = "many columns together"

The whole guide is 9 steps:
```
STEP 0  Find every column's TRUE type
STEP 1  First-contact quality scan (duplicates, fake values, impossible values)
STEP 2  Study the TARGET deeply
STEP 3  Univariate — one column at a time, by type
STEP 4  Missing values — diagnose (treatment happens later, in preprocessing)
STEP 5  Bivariate — every feature vs the target
STEP 6  Multivariate — features vs each other, interactions
STEP 7  Leakage hunt — the "too good to be true" check
STEP 8  Write the report
```

---

# PART 2 — STEP 0: Find every column's TRUE type

- Type
	- Continuous numeric
	- Discrete numeric
	- Ordinal categorical
	- Nominal categorical

| Special case                     | What it is                                              | What to do                                                                           |
| -------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Binary**                       | exactly 2 values (yes/no, 0/1)                          | simplest case; just check the balance (§5.4)                                         |
| **High-cardinality categorical** | nominal with tons of labels (50+ cities, product names) | normal categorical analysis, but special encoding later (never one-hot)              |
| **Datetime**                     | dates/times                                             | convert, break into parts (year, month, weekday…), then re-classify the parts (§5.5) |
| **ID column**                    | unique (or nearly) per row: user_id, order_no           | zero predictive value → drop                                                         |
| **Constant column**              | one single value everywhere                             | zero information → drop                                                              |


---

# PART 3 — STEP 1: First-contact quality scan

**Why now:** broken values poison every statistic and plot you'll make later. One row with age = 999 silently wrecks the mean, the std, the histogram, the outlier bounds — everything. Ten minutes here saves days.

### 3.1 Exact duplicate rows

```python
print("duplicated rows:", X_train.duplicated().sum())
```

A few duplicates in real data can be legitimate (two identical orders). Hundreds of them usually mean a data-collection bug → drop them (`X_train.drop_duplicates()`), and note it in your report.

### 3.2 Impossible values

Think for 30 seconds per numeric column: _what values are physically impossible?_ Then check:

```python
X_train.describe().T[['min', 'max']]          # scan the extremes of every column
(X_train['age'] < 0).sum()                    # targeted checks for the suspicious ones
(X_train['age'] > 120).sum()
(X_train['discount_pct'] > 100).sum()
```

Impossible values are usually **disguised missing values** (next) or unit errors (a height of 5.9 among 170-ish values = feet among centimeters). Action: convert to NaN, or fix the unit — and write it down.

### 3.3 Missing values in disguise (very common, very missed)

Not every hole is a NaN. Real datasets hide missing values as innocent-looking codes: `0`, `-1`, `99`, `999`, `9999`, `''` (empty string), `'?'`, `'unknown'`, `'NA'`, `'none'`, `'null'`, a single space.

```python
SUSPECTS = [-1, 99, 999, 9999, '', ' ', '?', 'unknown', 'Unknown', 'NA', 'N/A', 'none', 'null']
for c in X_train.columns:
    hits = X_train[c].isin(SUSPECTS).sum()
    if hits:
        print(f"{c}: {hits} suspect values")
```

⚠ **Judge per column.** `0` is a legitimate value for "number of children" but impossible for "height". `-1` is fine for "temperature" but a code for "salary". When you confirm a code means "missing", convert it to a real NaN so every later step treats it correctly:

```python
X_train['salary'] = X_train['salary'].replace({-1: np.nan, 999999: np.nan})
X_train['city']   = X_train['city'].replace({'?': np.nan, 'unknown': np.nan})
```

### 3.4 Text columns secretly blocking numbers

```python
converted = pd.to_numeric(X_train[col], errors='coerce')   # non-numbers become NaN
bad = X_train[col][converted.isna() & X_train[col].notna()]
print(bad.unique()[:10])    # shows exactly WHICH text values block the column
```

Typical finds: `'unknown'`, `'$1,200'`, `'12 kg'`. Clean them (`.str.replace('$','').str.replace(',','')` etc.), convert, and the column returns to its true numeric life.

### 3.5 Constants and IDs

Already flagged by `classify_columns()` — drop them now so they stop cluttering every later loop.

---

# PART 4 — STEP 2: The TARGET (study it before any feature)

The target defines the project. Its shape picks your metric, can change what you train on, and sets warnings for later.

### 4.1 Regression target (continuous)

```python
sns.histplot(x=y_train, kde=True); plt.show()
print("skew:", y_train.skew().round(2), "| min:", y_train.min(), "| max:", y_train.max())
```

|you see|it means|action|
|---|---|---|
|roughly a bell|friendly target|nothing special|
|strong right skew (skew > 1) — typical for prices, incomes|big values dominate the error|train on `np.log1p(y)`, convert predictions back with `np.expm1(pred)`; often a big win for linear models|
|hard bounds (e.g. everything in [0, 1] → it's a rate/probability)|model can predict outside the bounds|clip final predictions: `np.clip(pred, 0, 1)`|
|a huge spike at one value (e.g. tons of exact 0s)|two behaviors mixed ("bought nothing" vs "bought some amount")|note it; models will struggle right there; advanced fix = two-stage model, but just knowing is enough for now|

### 4.2 Classification target (categories)

```python
print(y_train.value_counts(normalize=True))
sns.countplot(x=y_train); plt.show()
```

|minority class share|verdict|actions|
|---|---|---|
|≥ ~40%|balanced|accuracy is fine|
|~20–40%|mild imbalance|prefer F1 / ROC-AUC; watch the confusion matrix|
|< ~20%|imbalanced|F1 or ROC-AUC (never accuracy), `stratify=y` in the split, `class_weight='balanced'` as the first fix|

Multiclass: print the share of _every_ class; classes with a handful of rows can't be learned — consider merging them or accept poor performance there.

---

# PART 5 — STEP 3: Univariate — one column at a time, by type
## 5.1 CONTINUOUS columns — the deep dive

The standard cell (same for every continuous column):

```python
for i in numerical_continuous_column:
    print("column name =>", i)
    print(x_train[i].describe())
    print("skew =>", x_train[i].skew())
    print('kurtosis =>', x_train[i].kurt())
    fig, ax = plt.subplots(2,2, figsize=(11,7))
    sns.histplot(x=x_train[i], kde=True ,ax=ax[0,0], bins = 20)
    sns.boxplot(x=x_train[i], ax=ax[0,1])
    stats.probplot(x_train[i], dist="norm", plot=ax[1,0])
    plt.tight_layout()
    plt.show()
```

### 5.1.1 Reading `describe()` — a fully worked example

Imagine a `salary` column in a 10,000-row dataset returns:

```
count      9,800
mean      58,000
std       32,000
min           -5
25%       38,000
50%       51,000
75%       70,000
max      900,000
```

Read it line by line:

|line|value|what it tells you|
|---|---|---|
|count|9,800 of 10,000|**200 missing values** → note for Step 4|
|mean vs 50%|58,000 vs 51,000|mean > median → **right skew** (a few big salaries pull the average up)|
|std|32,000|judged only against the scale — see 5.1.3|
|min|−5|**impossible** → disguised missing or error → back to Step 1|
|max|900,000|a very long right tail → outliers exist; is 900k real (a CEO) or an error? eyeball that row|
|25%/50%/75%|38k/51k/70k|the middle half of people earn 38–70k; if quartiles land on suspiciously round repeated numbers, the column may be discrete in disguise|

### 5.1.2 Mean vs Median — the honest-average story

Ten people: nine earn 50k, one earns 5,000,000.

- **Mean** = (9×50k + 5M) / 10 = **545,000** ← "the average person here earns half a million" — a lie.
- **Median** (the middle person when sorted) = **50,000** ← the truth.

**Rule:** the mean is dragged by extreme values; the median is not. When mean and median disagree a lot, the data is skewed toward the side of the mean, and the **median is the number to trust**.

### 5.1.3 Standard deviation — explained from zero (read slowly, once, forever)

**What std is:** one number answering _"on average, how far do values sit from the mean?"_

- `[10, 10, 10, 10]` → everyone exactly at the mean → std = 0.
- `[0, 20, 0, 20]` → same mean (10), but everyone sits 10 away from it → std = 10. Bigger std = values more spread out. That's the whole idea.

**The trap:** std is measured **in the column's own units**. So "std = 5" by itself is meaningless:

- for salaries (values around 50,000) → std 5 = everyone earns virtually the same → almost constant
- for exam scores out of 100 → std 5 = normal, healthy spread
- for number of children (values 0–6) → std 5 = insanely spread (impossible, actually)

Same number, three opposite meanings. **Never judge std alone. Always compare it to the column's own scale.** Two rulers:

**Ruler 1 — std ÷ range** (range = max − min). This turns std into a scale-free score:

```python
r = X_train[col]
ratio = r.std() / (r.max() - r.min())
```

|std ÷ range|plain meaning|picture|
|---|---|---|
|~0.00 – 0.05|values almost all the same (nearly constant)|`▁▁█▁▁`|
|~0.10 – 0.20|bunched around the middle, bell-like|`▁▂▅█▅▂▁`|
|~0.25 – 0.35|spread out flat/evenly across the whole range|`▅▅▅▅▅▅▅`|
|~0.40 – 0.50|values pile at the two ENDS (0.5 is the mathematical maximum)|`█▁▁▁▁▁█`|

So on a 0-to-1 column, std ≈ 0.29 doesn't mean "small" — it means _the column is spread out as evenly as it possibly can be_. And std ≈ 0.03 on a 0-to-1 column means "nearly constant". Same numbers, opposite verdicts on different scales — the ruler is what decides, never the raw std.

**Ruler 2 — std ÷ mean** (the _coefficient of variation_, CV) — for comparing spread **across different columns**. Only valid when the column is all-positive.

Example: house prices (mean 500,000, std 150,000 → CV = 0.30) vs customer age (mean 40, std 12 → CV = 0.30). The raw stds differ by a factor of 10,000, yet both columns are _equally_ variable relative to their own size.

**The "can I drop this column?" test — done right.** A column is a drop candidate only when it barely varies — when almost every row holds the SAME value, because then it cannot help the model tell rows apart. Test **value dominance**, never raw std:

```python
top_share = X_train[col].value_counts(normalize=True, dropna=False).iloc[0]
# top_share > 0.99  → 99%+ of rows share one value → drop candidate
# nunique == 1      → constant → drop
```

One caveat before deleting a 99%-one-value column: glance at its bivariate behavior first (Step 5). Occasionally the rare 1% perfectly flags something important (e.g. a rare "fraud_alert" flag). The check costs one groupby; do it, then delete with a clear conscience.

### 5.1.4 Skewness — explained from zero

A **tail** is the thin stretched-out side of a distribution. **Skew is named after the side the TAIL is on** (not the side the peak is on — this confuses everyone at first):

```
RIGHT (positive) skew:  ▂█▅▃▂▁▁▁   peak left, tail stretches RIGHT
                                   examples: income, house prices, hospital bills
LEFT (negative) skew:   ▁▁▁▂▃▅█▂   peak right, tail stretches LEFT
                                   examples: scores on an easy exam, age at retirement
```

|`.skew()`|verdict|action (for LINEAR models only — trees ignore skew completely)|
|---|---|---|
|−0.5 … +0.5|symmetric|nothing|
|0.5 … 1 (or −1 … −0.5)|mild|usually nothing|
|**> +1**|strong right skew|log1p or Yeo-Johnson transform|
|**< −1**|strong left skew|square or Yeo-Johnson transform|

### 5.1.5 Kurtosis — simplified to what you actually use

Kurtosis is a **"surprise tails" score**: compared to a bell curve, does this column produce **more** extreme values or **fewer**? (pandas `.kurt()` is centered so a perfect bell = 0.)

|`.kurt()`|meaning|what you do|
|---|---|---|
|around 0|tails like a normal bell|nothing|
|clearly positive (> +1)|MORE extreme values than a bell — sharp peak, long tails|expect real outliers → go stare at the **boxplot**|
|clearly negative (< −1)|almost NO extreme values — flat top, box-like, or several separate humps|go stare at the **histogram**: is it flat? multiple humps? only a few distinct bars (a category in disguise)?|

That's it. **You never act on kurtosis directly** — it's a pointer telling you which plot to inspect more carefully.

### 5.1.6 Histogram — the reading manual

A histogram chops the value range into buckets ("bins") and shows how many rows fall in each. Default bins are fine; blocky → try `bins=50`, spiky/noisy → try `bins=20`.

|shape|picture|plain meaning|everyday example|action|
|---|---|---|---|---|
|Bell|`▁▂▅█▅▂▁`|symmetric, normal-ish|adult heights|nothing|
|Right skew|`▂█▅▃▂▁▁▁`|long right tail; mean > median|income|log1p / YJ for linear|
|Left skew|`▁▁▁▂▃▅█▂`|long left tail|easy-exam scores|square / YJ for linear|
|Uniform / flat|`▅▅▅▅▅▅▅`|every value about equally common|lottery digits; synthetic Kaggle features|nothing — perfectly fine; **NOT droppable**|
|Bimodal (two humps)|`▂█▂▁▂█▂`|two hidden groups mixed together|heights of men+women combined|find the column that separates the humps (Step 5/6); consider a "which group" flag|
|Spike + tail|`█▂▁▂▂▁▁`|one special value (often 0) plus a distribution|yearly medical spend (many people spend 0)|add binary `is_zero` feature; log1p the rest|
|Comb / gaps|`█▁█▁█▁█`|only certain values occur|anything rounded or coded|it's discrete in disguise → reroute to §5.2|
|Wall at an edge|`█▅▃▂▁`|values piled against a hard floor/cap|percentages at 0 or 100; sensor maxed out|note the bound; it's a property, not an error|
|Isolated island|`▅█▅▁▁▁▁▂`|main mass + a small far-away cluster|heights: 170-ish cm plus a clump at 5.9 (feet!)|investigate: unit error? special segment?|

### 5.1.7 Boxplot — the reading manual

```
   fliers      whisker             box              whisker      fliers
   o  o    |------------[ Q1 ═══╪═══ Q3 ]------------|       o o o
                               median
   box       = the middle 50% of values (Q1 to Q3; its width = IQR)
   whisker   = stretches to the furthest point within 1.5×IQR of the box
   flier (o) = any point beyond the whiskers
```

Reading:

- **Median line off-center**, or **one whisker much longer** → skew, on the long side.
- **Fliers are NOT automatically errors.** "Beyond 1.5×IQR" is just a drawing convention; on skewed or heavy-tailed data, plenty of perfectly real values land there. A flier is an _invitation to look at the row_, never a deletion order.
- **Weakness:** a boxplot cannot show two humps — a bimodal column and a bell can draw the same box. If kurtosis was very negative or the histogram looked lumpy, confirm with `sns.violinplot(data=X_train, x=col)` (a violin = boxplot + shape).

### 5.1.8 QQ plot — optional, 60 seconds

Answers one question: "how close is this column to a normal bell curve?" (Useful for choosing z-score vs IQR outlier bounds later; that's about it.)

```python
stats.probplot(X_train[col], dist="norm", plot=plt); plt.show()
```

|pattern|meaning|
|---|---|
|points hug the diagonal line|~normal|
|right end curls up (J shape)|right skew|
|left end curls down|left skew|
|both ends peel away (S shape)|heavier/lighter tails than normal|
|staircase / steps|few distinct values → discrete in disguise|

You do **not** need formal normality tests (Shapiro, etc.): predictive ML doesn't require normal features, and on 100k+ rows those tests reject everything anyway. The QQ glance is plenty.

---

## 5.2 DISCRETE count columns

Do **not** use `describe` + histogram as your main tools here — with few distinct values they mislead. The main tool is `value_counts`; the main plot is `countplot`.

```python
col = 'num_children'
print(X_train[col].value_counts(normalize=True).sort_index())   # share of each level
sns.countplot(data=X_train, x=col); plt.show()
```

Look for three things:

1. **How many levels, and are there gaps?** (values 0,1,2,3,5 but never 4 → why? data bug or real?)
2. **Rare levels.** Any level holding < ~1% of rows is too thin to trust or to one-hot encode.
3. **Grouping plan for rare high levels.** For ordered counts, merge from the top with `clip`:

```python
# before:  0:40%  1:35%  2:15%  3:7%  4:2%  5:0.6%  6:0.3%  7:0.1%
X_train[col] = X_train[col].clip(upper=4)
# after:   0:40%  1:35%  2:15%  3:7%  4:3%      ← "4" now means "4 or more"
```

Group because levels are **rare** — never because "std looks small". Note: tree models are fine with the raw ungrouped count; grouping matters mainly (a) when you'll one-hot for a linear model and (b) so that per-level statistics in Step 5 are trustworthy.

If a count column has _many_ levels (say 0–200 with most values present) → it behaves like a continuous column → analyze it with §5.1 instead.

---

## 5.3 CATEGORICAL columns (nominal & ordinal)

```python
col = 'city'
print("cardinality:", X_train[col].nunique())           # cardinality = number of distinct labels
print(X_train[col].value_counts(normalize=True).head(15))
sns.countplot( y=x_train[col] )
plt.show()
```

Four checks, every time:

**1. Cardinality → future encoding.** ≤ ~10 labels → one-hot later is fine. Dozens+ → high-cardinality: plan frequency/target encoding, or plain OrdinalEncoder if using trees. (Decision executes in preprocessing; EDA's job is to write the plan.)

**2. Rare labels → merge plan.** Labels under ~1% each are noise for models and for statistics:

```python
share = X_train[col].value_counts(normalize=True)
rare = share[share < 0.01].index
X_train[col] = X_train[col].where(~X_train[col].isin(rare), 'Other')
```

**3. Dirty labels.** `'NYC'`, `'nyc '`, `'New York'` are three labels to pandas and one city in reality. Quick probe: if cleaned cardinality < raw cardinality, you have dirt:

```python
raw = X_train[col].nunique()
clean = X_train[col].str.strip().str.lower().nunique()
print(raw, "->", clean)          # differ? → apply .str.strip().str.lower(), map synonyms
```

**4. Natural order?** If yes (Low/Med/High; S/M/L) → it's ordinal → **write the order down right now** — you'll hand it to `OrdinalEncoder(categories=[[...]])` in preprocessing, and you don't want to rediscover it then.

---

## 5.4 BINARY columns

```python
print(X_train[col].value_counts(normalize=True))
```

One number to check: the balance. A 60/40 or even 90/10 binary is fine. A 99.5/0.5 binary is a near-constant → drop candidate — **but** peek at its bivariate first: if the rare 0.5% strongly predicts the target, it's a keeper (rare flags are sometimes gold).

---

## 5.5 DATETIME columns

A raw date is useless to a model; its **parts** are useful. Convert, decompose, then send each part back through Step 0 classification:

```python
d = pd.to_datetime(X_train['listed_date'])
X_train['year']       = d.dt.year                       # → discrete/ordinal
X_train['month']      = d.dt.month                      # → ordinal, cyclic (12 sits next to 1)
X_train['day']        = d.dt.day
X_train['dayofweek']  = d.dt.dayofweek                  # 0=Mon … 6=Sun
X_train['is_weekend'] = (d.dt.dayofweek >= 5).astype(int)   # → binary
X_train['quarter']    = d.dt.quarter
X_train['week']       = d.dt.isocalendar().week.astype(int)
X_train['hour']       = d.dt.hour                       # if time exists
```

Also check the date **range and gaps** (`d.min(), d.max()`) — a missing chunk of months means the data isn't what you think. (Cyclic encoding with sin/cos exists for month/hour; optional, later.)

---

# PART 6 — STEP 4: Missing values — diagnosis

(Treatment — imputers, pipelines — happens in preprocessing. EDA's job is the **plan**.)

```python
miss = X_train.isnull().mean().mul(100).sort_values(ascending=False)
print(miss[miss > 0])
```

Answer three questions per affected column:

**Q1 — How much is missing?**

|% missing|default plan|
|---|---|
|< 5%|simple impute (median / most_frequent), move on|
|5 – 40%|impute + consider a missing-indicator (see Q3)|
|> ~40–50%|drop-the-column candidate (mostly hole, little signal) — unless Q3 says the hole itself is signal|

**Q2 — Where is it missing?** Is missingness concentrated in some group?

```python
# does missingness in 'salary' depend on another column, e.g. job type?
print(X_train.groupby('job_type')['salary'].apply(lambda s: s.isnull().mean()))
```

If yes (e.g. salary missing mostly for 'freelancer') → the holes are not random; imputing with one global median is crude — group-wise thinking, or at least an indicator, is warranted.

**Q3 — Does the HOLE itself carry signal?** The most valuable check. Plain question: _do rows with a hole behave differently on the target?_

```python
for c in X_train.columns[X_train.isnull().any()]:
    print(c)
    print(df_tr.groupby(X_train[c].isnull())[TARGET].agg(['mean', 'count']), "\n")
```

If the target mean clearly differs between has-hole and no-hole rows → the _fact of absence_ is information (people who hide their income are different from people who report it) → plan `SimpleImputer(add_indicator=True)` so the model receives that fact. If the means match → plain imputation loses nothing.

(These three questions are the practical version of the MCAR/MAR/MNAR theory: Q2≈"MAR?", Q3≈"MNAR?". You never need to prove the label — you just run the two checks and act.)

---

# PART 7 — STEP 5: Bivariate — every feature vs the TARGET (the heart of EDA)

This is where two things get decided: **which features matter**, and **which model family to expect to win**.

### 7.0 The chooser

|Feature type ↓ / Target →|**Continuous target (regression)**|**Class target (classification)**|
|---|---|---|
|Continuous|fixed scatter + **binned-mean line** (§7.1)|per-class KDE / boxplot (§7.3)|
|Discrete / binary / ordinal / nominal|groupby table + boxplot + pointplot (§7.2)|normalized crosstab bars (§7.4)|

Why the chooser exists: **a scatter plot only works when BOTH variables are continuous and the points are visible.** A feature with few distinct values draws vertical stripes (range visible, signal invisible); 100k+ points draw a solid blob (density invisible). Both failures have dedicated fixes below.

---

## 7.1 Number column vs numeric target — the simple way

**Same job as 7.2.** There, the groups already existed (night / dim / daylight) and you compared their averages. Here there are no groups — so **we cut them ourselves**: slice the feature into 20 pieces, take the average target inside each piece, and compare.
### The code

```python
df_tr= pd.concat([x_train,y_train], axis=1)
small = df_tr.sample(10000, random_state=42) #m 10k rows only to avoide overplotting scatter plot
TARGET='accident_risk'
tmp = df_tr.copy()

for i in numerical_continuous_column:
	# picture only (optional, take no decision from it) - sample so it's fast
	# alpha = contorl transpracy, more overlpped point, darket at area will be
	# s = control the size of the point, small point less overlaps  
	sns.scatterplot(data=small, x=i, y=TARGET, alpha=0.05, s=8); plt.show()

	# THE MAIN PLOT - use FULL data here (more rows per slice = smoother line)
	tmp = df_tr.copy()
	tmp['bin'] = pd.cut(tmp[i], bins=20)
	m = tmp.groupby('bin', observed=True)[TARGET].agg(['mean', 'count']).reset_index()
	m['mid'] = m['bin'].apply(lambda b: b.mid)
	sns.lineplot(data=m, x='mid', y='mean', marker='o'); plt.show()

	# the numbers (eye test + number test together)
	gap = m['mean'].max() - m['mean'].min()
	score = gap / y_train.std()
	print(f"gap = {gap:.3f}   gap/std = {score:.2f}")
	print(m)
```

The line plot is your **boxplot equivalent** — the one plot you actually decide from. The scatter is just "let me see the dots". Skip it if it confuses you.

#### Step 1 — is Feature useful?
Look at the line in line plot, then check the printed number. Both should agree:
- line looks **flat** + score under 0.1 → **weak** → write it down, move on.
- line clearly **goes up or down** + score over 0.3 → **useful** → go to Step 2.
- score in the middle → helps a little → keep it, don't build anything special.

### Step 2 — HOW does line-plot line move? 

**Shape 1 — straight ramp** (steady climb or steady fall)

```
mean                              
0.50 |                    ● ●     
0.40 |              ● ●          
0.30 |        ● ●                
0.20 |  ● ●                      
     +--------------------------- feature
```

Each step is about the same size. Table check: differences between rows are similar (0.21 → 0.25 → 0.29 → 0.33...). → **Action: nothing. Keep the column as it is.**

**Shape 2 — flat, then a JUMP**

```
mean
0.50 |              ● ● ● ●      
0.40 |                           
0.30 |  ● ● ● ●                  
0.20 |                           
     +--------------------------- feature
              ↑ jump here
```

Table check: small differences, small differences, then **one big step** (0.30 → 0.31 → 0.38, a jump 3-4× bigger than the usual step), then normal again. → **Action: make a yes/no flag at the jump point:** `df['big_feature'] = (df[col] >= 0.5).astype(int)`

**Shape 3 — hill (up then down) or valley (down then up)**

```
mean                              
0.50 |        ● ● ●              
0.40 |     ●       ●             
0.30 |  ●             ●   ●      
     +--------------------------- feature
```

Table check: the values rise for a while, then **turn around** and fall (or the reverse). One clear turn, not tiny wobbles. → **Action: trees will handle it. For a linear model, add `feature²`.**
	- Why does squaring help for a hill shape?
		- Because a linear model can ONLY draw a straight line. Give it one column, and its best try at a hill is a straight line through the middle — badly wrong at both ends and in the center.
		- Now here's the trick. A hill needs the prediction to first rise, then fall. A straight line can't do that... but `x²` **bends**. So when the model gets both `x` and `x²`, it can mix them:
		- prediction = a·x + b·x²
		- With `b` negative, that formula makes a ∩ shape (rises, peaks, falls). With `b` positive, a U shape. The model picks `a` and `b` itself during training.
### Ignore the wobbles
Small up-down bumps are noise, not shape. Check the `count` column — slices with fewer rows wobble more. Ask: _does the line turn around ONCE and clearly (real shape), or does it jiggle up-down-up-down (noise)?_ Jiggle = ignore. Read the big trend.

---

## 7.2 Discrete / categorical feature × continuous target

**What we are doing (one line):** check — do the groups have **different target averages**?.
Different target averages→ the column helps predict. Same target averages→ it doesn't. 
That's the whole analysis.
Tiny example. Guessing a student's exam score:
- their **city**: average score A = 65, B = 66, C = 65 → all same → city tells you nothing → weak.
- did they **study**: yes = 80, no = 40 → big difference → very useful.
```Python 
df_tr= pd.concat([x_train,y_train], axis=1)
TARGET='accident_risk'
temp = categorical_ordinal_column+categorical_norminal_column+binary_column
for i in temp:
    tab= df_tr.groupby(i)[TARGET].agg(['mean', 'median', 'count'])
    print(tab)
    gap = tab['mean'].max() - tab['mean'].min()
    print('how much target normally moves',gap / y_train.std())
    # 2) the distribution per level
    sns.boxplot(data=df_tr, x=i, y=TARGET)
    plt.show()
    print('-'*100)
```
### Step 0 — trust check (5 seconds)
Look at `count`. An average made from very few rows is luck, not truth.
- Every group must have a sizable number of count.
- For Example 3 groups exist group 1 contains 1k rows, group 2 contains 1.2k rows and 3rd group contains 10 rows that mean 3 group mean is not trust worthy
### Step 1 — the eye test (boxplot only)
- Boxes sit at **different heights** → column **HELPS** → go to Step 2.
- All boxes at the **same height** → column is **WEAK** → write "weak", move on. (Keep weak columns for now. Dropping is an optional experiment later — model scores will tell you.)
### Step 1.5 — the tie-breaker (only when your eyes can't decide)
Sometimes it looks "maybe a little different?" and you're stuck in boxplot comparison. Turn the eye test into a number:

```python
gap = tab['mean'].max() - tab['mean'].min()   # biggest average − smallest average
print(gap / y_train.std())                    # y_train.std() = how much the target normally moves
```

- under **0.1** → basically the same → weak
- over **0.3** → really different → helps
- in between → helps a little
### Step 2 — WHERE is the difference? (free bonus: new columns)
- **One group** clearly above/below the rest → make a yes/no column for it (winter is high → `is_winter`).
- Ordered groups, and values **jump after a point** → make a yes/no column (`rating >= 4`).
- **All groups** clearly different → nothing to create; just keep the column.
### Special case — a column with MANY groups (like 40 cities)

Sort the groups by their average first, then the pattern becomes visible:

```python
order = df_tr.groupby(col)[TARGET].mean().sort_values().index
sns.boxplot(data=df_tr, x=col, y=TARGET, order=order); plt.xticks(rotation=90)
```

### Step 2.5 Feature-Engineering-part — you made a new column out of this
- **Merging categories by behavior:** if groups a and b have the same box, c is different, d is different → make one new column with 3 groups: `a+b` / `c` / `d`. Real technique, works. Two safety rules: merge only when the boxes truly sit on top of each other, and only when the groups are **big** — small same-looking groups might be luck, and then you'd be building a feature out of noise. After making it, same test: old vs new vs both.
- This kind of new column helps for linear model as trees find jumps by themselves
## 7.3 Continuous feature × class target (classification)

```python
# one curve per class; common_norm=False draws each class with its own area = 1,
# so a big class can't visually drown a small one
sns.kdeplot(data=df_tr, x='age', hue=TARGET, common_norm=False); plt.show()

# same info as boxes (better when there are many classes)
sns.boxplot(data=df_tr, x=TARGET, y='age'); plt.show()
```

**Reading = separation.** The further apart the per-class curves/boxes sit, the more this feature alone can tell the classes apart. Fully overlapping → weak alone (it might still help in combination — trees find combinations). One class's curve showing two humps → a hidden subgroup inside that class → hunt in Step 6.

---

## 7.4 Categorical feature × class target

```python
ct = pd.crosstab(df_tr['contract_type'], df_tr[TARGET], normalize='index')
ct.plot(kind='bar', stacked=True); plt.show()
print(df_tr['contract_type'].value_counts())     # counts — mandatory, again
```

`normalize='index'` makes every bar sum to 1, so you compare **proportions** fairly between big and small categories. Bars with clearly different color splits → predictive feature. Identical splits everywhere → weak. A dramatic split in a 30-row category → means nothing (tiny-group lie again).

---

## 7.5 The payoff — what bivariate tells you about MODEL CHOICE

Collect the shapes you found:

- Mostly **straight lines / clean level-shifts** → expect **linear models** (Linear/Ridge/Logistic) to do well → keep them, they're fast and interpretable.
- **Steps, curves, U-shapes, threshold effects** → expect **tree ensembles** (RandomForest, XGBoost) to win — they carve thresholds natively.
- You'll try both families anyway (the CV ladder in the workflow guide decides). EDA's job is to write the **prediction** in your report — "expect trees > linear because of the step in feature X" — so that when CV confirms it, you've learned pattern→outcome, and when CV surprises you, you investigate. That loop is how intuition is built.

---

# PART 8 — STEP 6: Multivariate — features vs each other

### 8.1 Twin columns (Question 1: "Do I have twin columns?)

```python
num_cols = ['feature1', 'feature2']
corr = X_train[num_cols].corr()          # add method='spearman' for the rank version
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.show()
```

- **|corr| > ~0.9 between two FEATURES** = they carry the same information (e.g. `area_sqft` and `num_rooms`). Drop any one of them
- Correlation value can go from -1 to 1
- Applied between 2 numerical columns.

### 8.2 Teaming up (Question 2: "Do two features team up?) 

**Teaming up, plain words:** the effect of feature A on the target _depends on_ feature B. Classic example: being a smoker raises insurance charges a little for the young — and enormously for the old. Age alone and smoker alone don't tell that story; the **combination** does.
- For example - creating a new column
```Python
  df['combined_new_column'] = df[ df['feature1']=='a'  & df['feture2'] == 'b']
```

- code
```python
tmp = df_tr.copy()
tmp['age_bin'] = pd.cut(tmp['age'], bins=5)
pivot = tmp.pivot_table(index='smoker', columns='age_bin',
                        values=TARGET, aggfunc='mean', observed=True)
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='coolwarm'); plt.show()
```
#### How to read it (the only skill here)

**Compare rows. Does every row tell the same story?**
No teaming up — every row goes low→high the same way, one row just sits higher:
```
            25     35     45     60     70
daylight   0.24   0.23   0.23   0.41   0.41
dim        0.24   0.23   0.24   0.41   0.41
night      0.40   0.41   0.41   0.64   0.60     <- higher, but SAME pattern
```
→ additive → do nothing.
Teaming up — one row breaks the pattern:
```
            25     35     45     60     70
daylight   0.24   0.23   0.23   0.41   0.41
night      0.40   0.41   0.41   0.85   0.84     <- explodes only at high speed
```
→ interaction → build a combined column.
Also counts as an interaction if a row goes the **opposite** direction from the others (one row falls where the others rise).

Only needed for **linear models** — trees find interactions by themselves.

---

# PART 9 — STEP 7: The leakage hunt (too-good-to-be-true check)

**Leakage** = a feature that contains information you would **not have at prediction time** — often a disguised copy of the answer. A leaky feature makes your CV scores look amazing and your real-world performance garbage; it is the most expensive mistake in applied ML.

**The time-machine test.** For every suspicious column ask: _"At the exact moment I'd make this prediction in real life, would I already know this value?"_

- Predicting loan default at approval time; column = `num_missed_payments` (collected during the loan) → time-machine **FAIL** → drop.
- Predicting churn this month; column = `account_closed_date` → FAIL → drop.
- Predicting house sale price; column = `last_sold_price` of the _same sale_ → FAIL. (Previous sale years ago → fine.)

**EDA red flags that scream leakage:**

- a feature with |correlation| > ~0.95 with the target;
- a category whose target is 100% one class across thousands of rows;
- a numeric feature that is an arithmetic transform of the target (target = price, feature = price_per_sqft × area).

When a flag fires: investigate what the column _means_ and _when it gets recorded_. If it fails the time-machine test → drop it and celebrate — you just saved the whole project.

---

# PART 10 — STEP 8: Write the report (the actual deliverable)

### 10.1 The observation → action master map

|you observed|you do|
|---|---|
|nunique == 1, or top value share > 99%|drop column (after one bivariate glance at the rare 1%)|
|nunique == n_rows on an identifier|ID → drop|
|numeric dtype but nunique ≤ ~15|reclassify: discrete/ordinal → analyze via value_counts & boxplots, not scatter|
|number-looking text ('$1,200', 'unknown')|clean → convert → reclassify|
|impossible values / suspicious codes (0, −1, 999)|convert to NaN (per-column judgment)|
|mean ≫ median, skew > 1 (continuous)|log1p / Yeo-Johnson **for linear models only**|
|std ÷ range ≈ 0.29, kurt ≈ −1.2, flat histogram|uniform-ish → fine as-is; **not droppable**|
|bimodal histogram|hunt the splitting variable; consider a group flag|
|spike at 0 + tail|add `is_zero` flag; log1p the rest|
|level/label share < 1%|merge → "Other" (nominal) or "k+" via clip (ordered)|
|dirty labels ('NYC' vs 'nyc ')|strip/lower/map, re-check cardinality|
|ordinal order exists|write the order down for OrdinalEncoder|
|missing > 40–50%|drop-column candidate|
|missing rows' target mean differs from rest|impute with `add_indicator=True`|
|binned-mean line = step at t|add flag `feature > t`; expect trees > plain linear|
|binned-mean line = U shape|trees, or add feature² for linear|
|\|Spearman\| ≫ \|Pearson\||monotonic curve → transform for linear, or trees|
|both ≈ 0 but binned line shows shape|trust the line, not the coefficients|
|group mean built on count < ~100|distrust; merge levels, re-check|
|feature-pair corr > 0.9|drop one for linear; note importance-splitting for trees|
|pivot heatmap corner lights up|interaction → trees, or explicit A×B feature for linear|
|feature corr with target > 0.95 / 100%-pure category|leakage suspect → time-machine test|
|target skew > 1 (regression)|train on log1p(y), expm1 back|
|target bounded (e.g. [0,1])|clip predictions to the bounds|
|minority class < 20%|F1/ROC-AUC, stratify, class_weight='balanced'|

### 10.2 The report template + a filled example

Template (10–15 lines, every project):

```
DATASET: <name>        ROWS×COLS: <r>×<c>        DUPLICATES: <n> (<action>)
TARGET: <col> — <type>, <shape/balance>, <bounds> → metric <...>, <target transform?>
COLUMN VERDICTS (one line each):
  <col>: <true type> | <shape note> | <missing%> | <quality issue> | → <action>
RELATIONSHIPS: strongest = <...>; shapes = <linear/step/curve/U>; interactions = <...>
REDUNDANCY: <pairs > 0.9> → <dropping which>
LEAKAGE: <suspects & verdicts>
PREPROCESS PLAN: impute <...> | encode <...> | transform <...> | scale <linear only>
MODEL EXPECTATION: <linear should suffice / non-linear signals → expect trees to win>
```

Filled example (made-up house-prices project):

```
DATASET: house_prices        ROWS×COLS: 21,000 × 12      DUPLICATES: 14 (dropped)
TARGET: price — continuous, right skew 2.1 → RMSE, train on log1p(price), expm1 back
COLUMN VERDICTS:
  area_sqft: continuous | right skew 1.4 | 0% miss | → YJ for linear; near-linear vs target
  bedrooms: discrete 1–9 | levels 8,9 = 0.4% | → clip to 7+; means rise then flatten
  condition: ordinal Poor<Fair<Good<Excellent | → OrdinalEncoder with that order
  city: nominal, 42 labels, 30 of them <1% | → merge rare to 'Other'; big mean gaps remain
  built_year: numeric | → derive age = 2026 − built_year; mild downward curve vs price
  has_pool: binary, 7% yes | → keep; pool homes ≈ +18% mean price
  listing_id: ID | → dropped
  last_sold_price: corr 0.97 with target → time-machine FAIL → dropped (leakage)
RELATIONSHIPS: area (linear), city (strong), condition (monotonic)
REDUNDANCY: area_sqft vs num_rooms corr 0.93 → dropping num_rooms for linear runs
LEAKAGE: last_sold_price dropped; others pass
PREPROCESS PLAN: no imputation needed | OHE city(after merge) + Ordinal condition | YJ area | scale for linear only
MODEL EXPECTATION: mostly linear + one interaction (city×area corner lit) → Ridge close; XGB slightly ahead
```

### 10.3 The prediction habit (how intuition is actually built)

The last line of the report is a **prediction**. Then the CV ladder (workflow guide, Part 6) grades it. Prediction confirmed → you learned a pattern→outcome pair. Prediction wrong → gold: go back and find what you missed. Five datasets of this loop teach more than fifty tutorials.

---

# PART 11 — The classic mistakes (read before every project until they're reflexes)

1. **Judging std in absolute terms.** Std lives in the column's units. Use std÷range or CV. Droppability = value dominance, never "std looks small".
2. **Trusting the dtype.** int64 can hide categories (rating, zip); object can hide numbers ('$1,200'). True type comes from nunique + eyeballing values.
3. **Scatter plots for discrete features.** Few unique x-values → stripes → range visible, signal invisible. Groupby + boxplot + pointplot.
4. **Concluding from an overplotted blob.** 100k points hide density. Alpha, 2D hist, binned-mean line.
5. **Treating boxplot fliers as errors.** They're just "beyond 1.5×IQR". On skewed data they're real values. Fliers = look, not delete.
6. **"Correlation ≈ 0, so no relationship."** Pearson misses curves; both miss U-shapes. The binned-mean line is the arbiter.
7. **Trusting group means without counts.** A mean over 30 rows is a rumor. Print count next to every mean, always.
8. **Missing values hiding as 0 / −1 / 999 / 'unknown'.** Hunt them in Step 1 or every later statistic is poisoned.
9. **Deleting before understanding.** Every drop needs a written reason tied to evidence (constant, ID, duplicate-of, leaky).
10. **Transforming features for tree models.** Skew fixes, scaling, outlier capping — trees don't care. Save the effort.
11. **Making preprocessing decisions from the full dataset.** Explore freely; but every threshold/median/cap is computed on train (inside the pipeline).
12. **Ignoring a too-good feature.** 0.97 correlation with the target feels like victory; it's usually leakage. Time-machine test.
13. **EDA with no written output.** If it changed no decision in the report, the plot was decoration.

---

# PART 12 — FAQ (the questions every beginner hits)

**Q: Do I need hypothesis tests (t-test, chi-square, p-values) in EDA?** Mostly no. Those belong to _inferential_ statistics ("is this difference real in the population?" — A/B tests, science papers). Predictive ML has its own judge: cross-validation. The one small cameo is optional statistical feature _selection_ (`SelectKBest` with chi2/ANOVA) — and even that is optional. Visual evidence + groupby tables + CV cover you.

**Q: What if my dataset has 100+ columns?** Don't hand-plot them. Strategy: (1) `classify_columns()` for all; (2) quality scan for all (it's all loops); (3) univariate as a skim-loop (§5.6) — stop only at weird ones; (4) **screen before deep bivariate**: compute Pearson+Spearman vs target for all numerics and groupby-mean spread for all categoricals (optionally MI), rank features, then do the deep bivariate treatment only for the top ~15–20 plus anything strange. Depth where it pays; loops everywhere else.

**Q: How many histogram bins?** Default first. Looks blocky → `bins=50`. Looks spiky/noisy → `bins=20`. Bins change the look, not the truth — when in doubt, try two settings.

**Q: How long should EDA take?** First projects: several hours — normal and correct, you're building the reflexes. After ~5 projects: 30–60 minutes for a medium dataset, because every step is a reflex and only the _findings_ differ.

**Q: EDA before or after the train/test split?** Split first. Explore train. Every _decision number_ (medians, caps, encodings, merges) must come from train only — that's the leakage rule from the workflow guide, and it's why all treatment later lives inside a Pipeline.

**Q: What if I find… nothing? All flat lines, no separations?** That's a finding. Write "weak individual features" in the report, set expectations low, and still run the model ladder — trees sometimes find interactions no single-feature plot shows. If even trees barely beat the dummy baseline, the features genuinely lack signal, and the honest next step is better features (feature engineering / more data), not fancier models.

**Q: Histogram vs KDE — which one?** They're the same information; the KDE is just a smoothed outline of the histogram. `sns.histplot(kde=True)` gives both — use that and don't think about it again.

---

# PART 13 — The one-page checklist (print me)

```
SETUP     □ split done  □ TARGET set  □ df_tr built  □ plot_df sampled
STEP 0    □ classify_columns() → assign every column a TRUE type (eyeball the ambiguous)
STEP 1    □ duplicates  □ impossible values  □ disguised missing (0/-1/999/'unknown')
          □ text-blocked numbers  □ constants & IDs dropped
STEP 2    □ target: shape/balance, skew, bounds → metric + target-transform decision
STEP 3    □ continuous → describe+skew/kurt+hist+box(+QQ) → shape catalog → plan
          □ discrete   → value_counts+countplot → rare-level clip plan
          □ categorical→ cardinality, rare→'Other', dirty labels, record ordinal orders
          □ binary     → balance   □ datetime → decompose & re-classify parts
          □ one written line per column
STEP 4    □ % missing  □ missingness vs other columns  □ missingness vs TARGET (indicator?)
STEP 5    □ use the CHOOSER:  cont×cont → binned-mean line (+2D hist) + Pearson & Spearman
          □ disc/cat×cont → groupby(mean,median,COUNT) + boxplot + pointplot
          □ cont×class → per-class KDE/boxplot   □ cat×class → normalized crosstab
          □ note the SHAPE of every strong relationship (line/curve/step/U)
STEP 6    □ corr heatmap → note >0.9 twins   □ pivot heatmap on top features → interactions
STEP 7    □ leakage: time-machine test on any too-good feature
STEP 8    □ fill the report template  □ write the MODEL EXPECTATION prediction
```

Run this on five datasets. On the fifth, you'll notice you no longer need the checklist — that's the plan working.