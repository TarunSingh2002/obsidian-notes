
## Create a DataFrame
```python
pd.DataFrame({f'getNthHighestSalary{N}': [None]})
```

## Select Columns
- `df[['a', 'b']]` → gives you a DataFrame with 2 columns
- `df['a']` → gives you a single column as a Series

---

## merge
```python
pd.merge(left, right, how='left', left_on='managerId', right_on='id', suffixes=['_a', '_b'])
```
- Instead of `left_on` and `right_on`, you can also use `on=['key1', 'key2']`
```python
s = pd.merge(s, o, on='sales_id', how='left', indicator=True, suffixes=('', '_right'))
```

## concat
- Combines 2 or more DataFrames vertically
	```python
	pd.concat([df1, df2])
	```

---

## .duplicated()
- Works on a DataFrame like this
	```python
	df.duplicated(keep=False, subset=['email'])
	```
- Parameters
	- `subset`
		- Column label or iterable of labels, optional
		- Only considers certain columns for identifying duplicates. Uses all columns by default
	- `keep`: `{'first', 'last', False}`, default `'first'`
		- Determines which duplicates (if any) to mark
		- `first`: mark duplicates as `True` except for the first occurrence
		- `last`: mark duplicates as `True` except for the last occurrence
		- `False`: mark all duplicates as `True`

## .unique()
- Works on a Series

---

## Sort a DataFrame
```python
df.sort_values(by='column_name')
df.sort_values(by='column_name', ascending=False)
df.sort_values(by=['col1', 'col2'], ascending=[True, False])
```

## Rename a Column
```python
# Renames 'old_name' to 'new_name'
df = df.rename(columns={'old_name': 'new_name'})
```

## Round
```python
df['a'] = df['a'].round(2)
```

## Date
```python
df['trans_date'].dt.strftime('%Y-%m')
```

---

## Loop over a DataFrame
```python
for index, row in df.iterrows():
    print('-' * 100)
    print(f"Row Label: {index}")
    print(row)
    print(type(row))  # Series

    df.at[index, 'column_name'] = new_value  # like 1
    """
    ----------------------------------------------------------------------------------------------------
    Row Label: 0
    id                               1
    recordDate     2015-01-01 00:00:00
    temperature                     10
    Name: 0, dtype: object
    <class 'pandas.core.series.Series'>
    ----------------------------------------------------------------------------------------------------
    """
```

---

## Null Values
- The standard way to filter for null values uses 2 functions: `isna()` and `notna()`
	- `.isna()` and `.notna()` → the Series way
	- `pd.isna()` and `pd.notna()` → the single value way
	```python
	df2 = df2[df2['id'].isna()]
	```
- Inside an `apply` function, use `pd.notna(row['column name'])`
	```python
	def func(row):
	    if not pd.notna(row['manager_id']) or row['manager_id'] in emp_id:
	        return False
	    return True

	df['manager_exist'] = df.apply(func, axis=1)
	```

---

## apply
```python
def calculate_average(row):
    return (row['A'] + row['B'] + row['C']) / 3

# Apply the function to each row and create a new 'Average' column
df['Average'] = df.apply(calculate_average, axis=1)
```
- `axis`
	- `0` or `'index'`: apply the function to each column
	- `1` or `'columns'`: apply the function to each row

---

## groupby
- **Named aggregations**
	```python
	s.groupby('product_id') \
	 .agg(
	     min_date=('sale_date', 'min'),
	     max_price=('price', 'max')
	 ) \
	 .reset_index()
	```
	- Other aggregations you can use
		- `'nunique'`
		- `'first'` → works for both str and int, takes the first value in the group
		- `'mean'`
		- `'count'`, `'sum'`
	- `groupby(..., dropna=True/False)` → `True` by default
- **Custom aggregation functions** (lambda functions are also allowed)
	```python
	def func1(amounts):
	    return amounts[df.loc[amounts.index, 'is_even'] == 'odd'].sum()

	df = df.groupby(['transaction_date']).agg(
	    transaction_date=('transaction_date', 'first'),
	    odd_sum=('amount', func1),
	    even_sum=('amount', func2)
	)
	```
	- This performs different aggregation functions on multiple columns
- **Count rows per group**
	```python
	freq = df.groupby('customer_number').size().reset_index(name='frequency')
	result = freq[freq['frequency'] == freq['frequency'].max()][['customer_number']]
	```
- **Mean per group**
	```python
	df2 = df.groupby('project_id')['experience_years'].mean().round(2).reset_index(name='average_years')
	```

---

## String Functions
```python
df['name'].str.contains('date', na=False)
df['name'].str.startswith('da', na=False)
df['name'].str.endswith('te', na=False)
```

---

## np.where
- **On a DataFrame**
	```python
	df['has_start'] = np.where(df['dna_sequence'].str.startswith('ATG'), 1, 0)
	df['has_stop'] = np.where(df['dna_sequence'].str.endswith(('TAA', 'TAG', 'TGA')), 1, 0)
	df['has_atat'] = np.where(df['dna_sequence'].str.contains('ATAT'), 1, 0)
	df['has_ggg'] = np.where(df['dna_sequence'].str.contains('GGG'), 1, 0)
	```
- **On an array**
	- Example 1: choose values
		```python
		import numpy as np

		arr = np.array([10, 20, 30, 40, 50])

		# If an element is > 25, keep it; otherwise, replace it with -1
		result = np.where(arr > 25, arr, -1)
		print(result)
		# Output: [-1 -1 30 40 50]
		```
	- Example 2: find indices
		```python
		import numpy as np

		arr = np.array([5, 12, 3, 22, 8])

		# Locate indices where elements are greater than 10
		indices = np.where(arr > 10)
		print(indices)
		# Output: (array([1, 3]),)
		```
- See also [[Numpy]]
