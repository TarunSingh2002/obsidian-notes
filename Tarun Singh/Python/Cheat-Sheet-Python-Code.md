

## List
- Get an element's index
	```python
	arr.index(value)
	```
- Delete an element at a particular index
	```python
	arr.pop(index)
	```
- Insert at a particular index
	```python
	arr.insert(index, value)
	```

## String
```python
a.startswith("abc")
a.endswith("abc")
"abc" in a      # contains (str has no .contains())
```
- Character ↔ number
	```python
	ord('a')   # 97
	chr(97)    # 'a'
	```

## Dictionary (map)
- Delete a key-value pair
	```python
	del my_dict["age"]
	```

## Set
```python
a = set()   # note: a = {} creates an empty dict, not a set
a.add("d")
```
- Assigning by index is not possible
	```python
	s[1] = "Hello"   # error
	```

## Tuple
- Can't be changed once the values are set (immutable)

---

## Stack
- There's no stack type in Python, so use a list
	- push = `list.append(x)`
	- pop = `list.pop()`

## Deque
- Double ended queue
```python
from collections import deque

q = deque()
q.append(1)       # right
q.appendleft(5)   # left
q.pop()           # right
q.popleft()       # left
q[0], q[-1]       # peek both ends
```

---

## Heap
- Python's `heapq` only has a **min heap** (Python 3.14 adds max-heap functions)
- Functions
	- `heapq.heappush(heap, x)`: adds `x` into the heap and keeps heap order
	- `heapq.heappop(heap)`: removes and returns the smallest element
	- `heapq.heapify(list)`: converts a normal list into a heap
	- `heapq.heappushpop(heap, x)`: pushes `x`, then pops the smallest element
	- `heapq.heapreplace(heap, x)`: pops the smallest element, then pushes `x`
- **Single value heap**
	```python
	import heapq

	heap = []
	heapq.heappush(heap, 5)
	heapq.heappush(heap, 2)
	heapq.heappush(heap, 8)

	print(heap)                 # internal heap structure
	print(heapq.heappop(heap))  # 2
	print(heapq.heappop(heap))  # 5
	```
- **Multiple values in a heap**
	```python
	heapq.heappush(heap, (priority, value))
	```
	```python
	heap = []
	heapq.heappush(heap, (2, "B"))
	heapq.heappush(heap, (5, "A"))
	heapq.heappush(heap, (8, "C"))
	# [(2, "B"), (5, "A"), (8, "C")]
	```
- **Max heap**
	- `heapq` doesn't directly give a max heap, so implement one by negating the priority
	```python
	import heapq

	heap = []
	heapq.heappush(heap, -5)
	heapq.heappush(heap, -2)
	heapq.heappush(heap, -8)

	print(-heapq.heappop(heap))  # 8
	print(-heapq.heappop(heap))  # 5
	print(-heapq.heappop(heap))  # 2
	```

## SortedList (sortedcontainers)
```python
from sortedcontainers import SortedList

a = SortedList([1, 2, 3, 4, 5])
print(a)   # SortedList([1, 2, 3, 4, 5])

a.add(6)
print(a)   # SortedList([1, 2, 3, 4, 5, 6])

a.remove(5)

a.clear()
print(a)   # SortedList([])
```

## Built-in Binary Search
```python
import bisect

i = bisect.bisect_left(arr, x)
j = bisect.bisect_right(arr, x)
```

## Multi-Source BFS
- *To be added*

---

## Hacks
- **Sort by the 1st value, then by the 2nd value if the 1st is the same**
	```python
	d = {
	    "a": [2, 3],
	    "b": [1, 5],
	    "c": [2, 1],
	    "d": [1, 2]
	}

	sorted_items = sorted(d.items(), key=lambda x: (x[1][0], x[1][1]))
	# [('d', [1, 2]), ('b', [1, 5]), ('c', [2, 1]), ('a', [2, 3])]
	```
	- For the first ascending and the second descending, use `key=lambda x: (x[1][0], -x[1][1])`
	- `sorted()` on `d.items()` already returns a list
