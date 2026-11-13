# Numpy

## Definition
- NumPy is a low-level library written in C for high-level mathematical functions
- It supports one data structure: the **N-d (NumPy) array**

## NumPy Array Benefits over a Python List
- Faster
- Convenient
- Less memory consuming

---

## NumPy Array

### Properties
- Homogeneous
- Can only hold numbers / strings
- Fixed item size

### Import
```python
import numpy as np
```

### Creating
- **Way 1**: from a list
	```python
	arr1 = np.array([1, 2, 3])
	arr2 = np.array([[1, 2], [3, 4]])
	```
- **Way 2**: `arange`
	- `arr6 = np.arange(10)`
		- ![[Numpy - arange 10.png]]
	- `arr6 = np.arange(5, 16)`
		- ![[Numpy - arange 5 16.png]]
	- `arr6 = np.arange(5, 16, 1)`
		- ![[Numpy - arange 5 16 1.png]]
- **Copy**
	```python
	arr3 = arr1.copy()
	```
- **Others**
	- Way 3: `arr3 = np.zeros((2, 3))`
		- ![[Numpy - zeros.png]]
	- Way 4: `arr4 = np.ones((3, 3))`
		- ![[Numpy - ones.png]]
	- Way 5: `arr5 = np.identity(5)`
		- ![[Numpy - identity.png]]
	- Way 6: `arr7 = np.linspace(10, 20, 10)`
		- ![[Numpy - linspace.png]]

---

## Attributes
For these examples:
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([[1, 2], [3, 4]])
```

| Attribute  | What it tells you    | `arr1`  | `arr2`   |
| ---------- | -------------------- | ------- | -------- |
| `shape`    | size of each dimension | `(3,)`  | `(2, 2)` |
| `ndim`     | number of dimensions | `1`     | `2`      |
| `size`     | number of items      | `3`     | `4`      |
| `itemsize` | bytes per item (int32 = 4) | `4` | `4`      |
| `dtype`    | data type            | `int32` | `int32`  |

- For a 3-dimensional array, `shape` looks like `(2, 2, 2)`
	- First 2 → number of 2-D arrays
	- Second 2 → number of rows
	- Third 2 → number of columns

---

## Functions and Methods

### type
```python
type(arr1)   # numpy.ndarray
```

### astype(): convert the data type
```python
arr9.astype('float')   # change the datatype to float
```

### reshape(): change the shape of your array
- `arr10 = np.arange(24)`
	- ![[Numpy - arange 24.png]]
- `arr10.reshape(6, 4)`
	- ![[Numpy - reshape 6 4.png]]
- `arr10.reshape(3, 2, 4)`
	- ![[Numpy - reshape 3 2 4.png]]
- The only thing to keep in mind: the array's size must equal the product of the new shape

### max / min / sum / mean ...
- With `arr10` reshaped to `(6, 4)`:
	```python
	arr10.max()         # 23
	arr10.max(axis=1)   # array([ 3,  7, 11, 15, 19, 23])  → max of each row
	arr10.max(axis=0)   # array([20, 21, 22, 23])          → max of each column
	```
	- `axis=0` works down the rows, so you get one result per column
	- `axis=1` works across the columns, so you get one result per row
- Similarly: `min()`, `sum()`, `mean()`, `std()`, `np.sin(arr10)`, `np.median(arr10)`, `np.exp(arr10)`, `np.sqrt(arr10)`

### transpose(): transpose of a matrix
- `arr10.transpose()`
	- ![[Numpy - transpose.png]]

### sort
```python
np.sort(arr)
```

### argmax / argmin
```python
np.argmax(arr)   # index of the maximum number
np.argmin(arr)   # index of the minimum number
```

### ravel(): convert a higher-dimension array to a 1-D array
- `arr10.ravel()`
	- ![[Numpy - ravel.png]]

### hstack() and vstack()
- Arrays
	```python
	arr11 = np.arange(6).reshape(2, 3)
	arr12 = np.arange(6).reshape(2, 3)
	```
	- ![[Numpy - stack input arrays.png]]
- `np.hstack((arr11, arr12))`
	- ![[Numpy - hstack.png]]
- `np.vstack((arr11, arr12))`
	- ![[Numpy - vstack.png]]

### random
```python
np.random.random()           # a random float, like 0.78
np.random.uniform(8, 10)     # a random float between 8 and 10, like 8.097
np.random.uniform(8, 10, 7)  # 7 random floats between 8 and 10
# array([9.52066276, 9.6231581 , 9.88588586, 9.21188173, 9.05646   , 8.37958654, 8.01195448])
np.random.randint(7, 10)     # a random integer from 7 to 9, like 7
```

### Splitting
- *Skipped*

---

## Indexing, Slicing and Iteration
- Array used (`arr10.reshape(6, 4)`)
	- ![[Numpy - reshape 6 4.png]]

### Indexing + Slicing
- `arr[2]`
	- ![[Numpy - index arr 2.png]]
- `arr[-1]`
	- ![[Numpy - index arr -1.png]]
- `arr[1][2]` → `6`
- `arr[:, 2]` → all rows, column 2
	- ![[Numpy - slice all rows col 2.png]]
- Extract this
	- ![[Numpy - slice extract 1.png]]
	- `arr[2:4, 1:3]`
- Extract this
	- ![[Numpy - slice extract 2.png]]
	- `arr[4:6, 2:4]`
	- `arr[4:, 2:]`

### Iteration
- **Row by row**
	```python
	for i in arr10:
	    print(i)
	```
	- ![[Numpy - iterate rows.png]]
- **Element by element** (`nditer` goes through every single item)
	```python
	for i in np.nditer(arr10):
	    print(i)
	```
	- ![[Numpy - nditer.png]]

---

## Operations
- Arrays used
	```python
	arr1 = np.array([1, 2, 3, 4, 5])
	arr2 = np.array([2, 3, 4, 5, 6])
	```
- `arr1 - arr2`
	- ![[Numpy - array subtraction.png]]
	- Similarly `arr1 * arr2`, `arr1 + arr2`, etc.
- `arr1 * 2`
	- ![[Numpy - array times 2.png]]
- `arr1 > 3`
	- ![[Numpy - array greater than 3.png]]
- Dot product
	```python
	arr1.dot(arr2)
	```

---

## Fancy Indexing
- Array used
	- ![[Numpy - fancy indexing array.png]]
- Get rows 0, 3 and 5
	- `arr10[[0, 3, 5]]`
	- ![[Numpy - fancy indexing rows.png]]
- Get columns 0, 2 and 3
	- `arr10[:, [0, 2, 3]]`
	- ![[Numpy - fancy indexing cols.png]]

## Indexing with Boolean Arrays
- Array used
	- ![[Numpy - fancy indexing array.png]]
- Get the values greater than 5 and smaller than 11
	- `arr10[(arr10 < 11) & (arr10 > 5)]`
	- ![[Numpy - boolean indexing.png]]
- Set every value greater than 5 and smaller than 11 to 0
	```python
	arr10[(arr10 < 11) & (arr10 > 5)] = 0
	```

---

## Graphs using NumPy
- *Skipped*

## Broadcasting
- *Skipped*

---
- See also [[Pandas]]
