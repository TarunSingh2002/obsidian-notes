---
tags:
  - machineLearning
  - EDA
  - deepDive
---








### 7.0 The chooser

| Feature type ↓ / Target →             | **Continuous target (regression)**          | **Class target (classification)** |
| ------------------------------------- | ------------------------------------------- | --------------------------------- |
| Continuous                            | fixed scatter + **binned-mean line** (§7.1) | per-class KDE / boxplot (§7.3)    |
| Discrete / binary / ordinal / nominal | groupby table + boxplot + pointplot (§7.2)  | normalized crosstab bars (§7.4)   |

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