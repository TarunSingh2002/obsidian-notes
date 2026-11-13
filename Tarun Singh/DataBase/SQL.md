---
tags:
  - database
  - sql
  - cheatSheet
---
# SQL Cheat Sheet

## Types of Joins
- **Left Join** (or Left Outer Join)
	- ![[SQL - left join.png]]
- **Right Join** (or Right Outer Join)
	- ![[SQL - right join.png]]
- **JOIN** (or Inner Join)
	- ![[SQL - inner join.png]]
- **Full Outer Join** (or Full Join)
	- ![[SQL - full outer join.png]]
- **Left Exclusive Join**
	- ![[SQL - left exclusive join.png]]
- **Right Exclusive Join**
	- ![[SQL - right exclusive join.png]]
- **Full Outer Exclusive Join**
	- ![[SQL - full outer exclusive join.png]]
- **Cross Join** (or Cartesian Join)
	- ![[SQL - cross join.png]]
- **Self Join**
	- ![[SQL - self join.png]]

---

## SQL Order of Execution
SQL usually runs in this order:
1. `FROM`
2. `WHERE`
3. `GROUP BY`
4. `HAVING`
5. `SELECT`
6. `ORDER BY`

- Example
	```sql
	SELECT mode_of_payment, COUNT(*) AS total
	FROM payments
	WHERE amount > 100
	GROUP BY mode_of_payment
	HAVING COUNT(*) > 2;
	```
	```sql
	SELECT column, AGG_FUNC(column)
	FROM table
	WHERE condition
	GROUP BY column
	HAVING condition
	ORDER BY column;
	```

---

## Aggregation Functions
- `COUNT`
- `MIN`
- `MAX`
- `AVG`
- `SUM`
- **Important:** apart from `COUNT`, they return `NULL` if there are no rows to aggregate

---

## GROUP BY
- `GROUP BY` → makes buckets (groups)
	```sql
	SELECT class, COUNT(*) AS frequency FROM courses GROUP BY class;
	```
- For more than 1 column
	```sql
	GROUP BY column1, column2
	```
- `GROUP_CONCAT`: join strings into `a,b,c`
	```sql
	GROUP_CONCAT(text_column) AS combined_text
	```
	- Advanced example
		```sql
		GROUP_CONCAT(DISTINCT product ORDER BY product) AS products
		```

## HAVING
- Used with `GROUP BY` to filter the groups

---

## DISTINCT
```sql
SELECT DISTINCT Department
FROM Employees;
```

## LIMIT and OFFSET
```sql
LIMIT 1 OFFSET 2
```
- `LIMIT 1` returns only 1 row
- `OFFSET 2` skips the first 2 rows

## ROUND
```sql
ROUND(number, decimal_places)
```

## NULL → 0
```sql
COALESCE(SUM(distance), 0)
```

## A fixed value in a whole column
```sql
SELECT 'a' AS column1 FROM table1;
```

---

## CASE
- `CASE` is SQL's version of if-else logic
	```sql
	CASE
	    WHEN condition1 THEN result1
	    WHEN condition2 THEN result2
	    ...
	    ELSE default_result
	END
	```
- Example
	```sql
	SELECT
	    x,
	    y,
	    z,
	    CASE
	        WHEN x + y > z AND x + z > y AND y + z > x
	        THEN 'Yes'
	        ELSE 'No'
	    END AS triangle
	FROM Triangle;
	```

---

## Date Time
- `EXTRACT`
	```sql
	EXTRACT(DAY FROM BirthTime)
	```
	- Works with `YEAR`, `MONTH`, `DAY`, `HOUR` and `MINUTE`
- Compare the year of a datetime field
	```sql
	EXTRACT(YEAR FROM timestamp_column) = 2020
	```
- `DATE_ADD`
	```sql
	DATE_ADD(w2.recordDate, INTERVAL 1 DAY)
	```

---

## UNION
- The `UNION` operator combines the result sets of two or more `SELECT` queries into a single output. It removes duplicate rows and returns only unique records (use `UNION ALL` to keep duplicates)
- Example
	```sql
	SELECT city FROM Table1
	UNION
	SELECT city FROM Table2;
	```

---

## Window Functions
- ![[SQL - window function 1.png]]
- ![[SQL - window function 2.png]]
- ![[SQL - window function 3.png]]
- **Aggregate function + PARTITION BY**
	- ![[SQL - window agg partition by.png]]
	- **Aggregate function + ROWS BETWEEN**
		- ![[SQL - window agg rows between.png]]
		- Frame bounds can also be
			- `CURRENT ROW`
			- `1 PRECEDING`
			- `1 FOLLOWING`
- **Ranking functions**
	- ![[SQL - window ranking functions.png]]
- **Value functions**
	- ![[SQL - window value functions.png]]
- Example: running total ("sum till now")
	```sql
	SELECT
	    Turn,
	    ID,
	    Name,
	    Weight,
	    SUM(Weight) OVER (
	        ORDER BY Turn
	        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
	    ) AS TotalWeight
	FROM MyTable
	ORDER BY Turn;
	```

---

## CTE
- Much cleaner: define the join once and use it twice
	```sql
	WITH temp AS (
	    SELECT title.worker_ref_id, title.worker_title, worker.salary
	    FROM title
	    LEFT JOIN worker ON title.worker_ref_id = worker.worker_id
	)
	SELECT worker_title AS best_paid_title
	FROM temp
	WHERE salary = (SELECT MAX(salary) FROM temp);
	```

---

## Hacks
- **Subquery in WHERE**
	```sql
	WHERE salary < 30000
	  AND manager_id NOT IN (SELECT DISTINCT employee_id FROM employees)
	```
	```sql
	WHERE (customer_id, order_date) IN (
	    SELECT customer_id, MIN(order_date)
	    FROM delivery
	    GROUP BY customer_id
	)
	```
- **COUNT + GROUP BY**
	```sql
	COUNT(DISTINCT column_name)
	COUNT(CASE WHEN condition THEN 1 END)
	```
	- Example
		```sql
		COUNT(CASE WHEN action = 'confirmed' THEN 1 END)
		```
	- Careful: `COUNT(CASE WHEN action = 'confirmed' THEN 1 ELSE 0 END)` counts **every** row, because `0` is not `NULL`. Use `SUM(...)` with `ELSE 0` instead
