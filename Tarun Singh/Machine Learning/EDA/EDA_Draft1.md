---
tags:
  - machineLearning
  - EDA
  - deepDive
---






# PART 8 — STEP 6: Bivariate

### 7.0 The chooser

| Feature type ↓ / Target →             | **Continuous target (regression)**          | **Class target (classification)** |
| ------------------------------------- | ------------------------------------------- | --------------------------------- |
| Continuous                            | fixed scatter + **binned-mean line** (§7.1) | per-class KDE / boxplot (§7.3)    |
| Discrete / binary / ordinal / nominal | groupby table + boxplot + pointplot (§7.2)  | normalized crosstab bars (§7.4)   |

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





