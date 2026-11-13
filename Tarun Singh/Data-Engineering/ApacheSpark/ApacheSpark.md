---
tags:
  - dataEngineering
  - apache-spark
  - pyspark
  - big-data
---
# Apache Spark

## Basics
- Apache Spark is an open-source computing engine and set of libraries designed for **parallel data processing on a computer cluster**
	- We use a group of machines called a **cluster** (Driver + Workers), and each machine is called a **node**
- Apache Spark **does not store data**
	- But it can connect to different data sources → S3, JDBC/ODBC, Azure Storage, HDFS

### Why distributed computing
- Vertical scaling leads to low availability issues and has a theoretical limit (there's only so far you can increase RAM and CPU cores)

### What is Big Data
- The "Big Data" problem refers to the challenges of storing, processing, and analyzing datasets that are too big for traditional tools
- **The 3 V's**
	- **Volume**
		- The sheer size of the data (terabytes, petabytes, or more)
		- Too large, so it won't fit on one machine
	- **Velocity**
		- The speed at which data is generated and must be handled
		- Like 1 TB of data per second
	- **Variety**
		- The range of data types and sources: structured (tables), semi-structured (JSON, XML), and unstructured (text, images, video)
- In practice, solving Big Data problems means combining:
	- Scalable storage (e.g. HDFS or cloud object storage)
	- Distributed compute (e.g. Spark, Flink)
	- Scalable ingestion (e.g. Kafka)
	- All while ensuring data quality, security, and governance

---

## Spark vs Hadoop
- Hadoop was used before Apache Spark
- ![[ApacheSpark - spark vs hadoop.png]]
- Hadoop uses **disk**, Apache Spark uses **RAM** (in-memory computation)

---

## Features
- In-memory computation
- Lazy evaluation
- Fault tolerant
- Partitioning
- Batch processing and stream processing with the same API

---

## Architecture
![[ApacheSpark - architecture.png]]
- Apache Spark architecture follows the **Master-Slave** concept
- Driver + Workers = cluster of nodes

### Components
- **Resource Manager**
	- It is the Master
	- It allocates and manages the resources for the work
- **Workers / Executors**
	- It is a resource
- **Driver**
	- It is a resource
	- It manages the Workers / Executors

### Flow
1. **Job Submission**
	- You hand Spark your application code plus resource hints (e.g. "I need 1 driver, 3 executors, X GB each")
	- Spark packages this into an application request and hands it to the cluster's Resource Manager
2. **Application Master Container Allocation**
	- ![[ApacheSpark - application master container.png]]
	- The Resource Manager allocates a dedicated container for your app. This becomes the **Application Master (AM) Container**
	- Inside that container, Spark launches the Driver process
	- Both the **PySpark Main process** and the **JVM Main process** live together inside the AM container, and together they form your Spark **Driver**
	- Spark is written in Scala/Java, so a Python wrapper is used to convert the Python code to Scala/Java
3. **Driver Startup (in the AM Container)**
	- The Driver (your app's "brain") spins up inside the AM Container
	- It creates its SparkContext / SparkSession
4. **Executor Request**
	- From within the AM Container, the Driver tells the Resource Manager: "Please launch N executors of size X each"
5. **Executor Allocation**
	- The Resource Manager launches those executor containers across the available worker nodes
	- Each container becomes one Executor process
	- A worker node contains only a **JVM** if your application doesn't use a UDF (User Defined Function). If your program uses a UDF, the worker node contains **Python + JVM**
6. **Task Scheduling & Coordination**
	- The Driver (in the AM Container) divides your job into tasks and assigns them to executors
	- It monitors progress, handles retries, and orchestrates any data shuffles
7. **Execution & Data Exchange**
	- Executors run tasks: reading input, performing transformations, writing output
	- They report status back to the Driver and exchange shuffle data with each other
8. **Job Completion & Teardown**
	- Once all tasks finish, the Driver signals "Job done"
	- The Resource Manager tears down the executor containers, then the AM Container (Driver) itself is cleaned up
	- All resources are returned to the cluster pool

---

## Spark Session
- **SparkContext** and **SparkSession** are the two main entry points for writing Spark applications
- Under the hood, a SparkSession contains a SparkContext (and replaces SQLContext and HiveContext)
- The Driver node connects to the Cluster Manager with the help of the SparkContext

### Code
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
         .appName("MyApp") \
         .master("yarn") \
         .config("spark.some.config", "value") \
         .getOrCreate()

# the underlying SparkContext
sc = spark.sparkContext
```

### Lifecycle
- `builder.getOrCreate()` either returns an existing session or spins up a new one (with its SparkContext)
- On exit, `spark.stop()` shuts down both the SparkContext and all executors

### What it provides
- **DataFrame & Dataset API**
	- Read from CSV/Parquet, run SQL queries, perform MLlib operations, and stream data, all from one object
- **Catalog & Metadata**
	- Manage temporary views, global tables, and functions
- **Configuration**
	- Set all your Spark properties (memory, shuffle partitions, serializers) in one place
- **Access to Low-Level APIs**
	- If you need RDDs, just call `spark.sparkContext.parallelize(...)`

---

## Lazy Evaluation & Actions

### Lazy Evaluation
- **What it is**
	- Spark doesn't immediately execute your transformation calls (like `map`, `filter`, `select`, `join`). Instead, it builds up a **logical plan** (a recipe) of all the steps you've asked for
- **Why it matters**
	- **Optimization**: Spark can analyze the full pipeline and reorder or combine operations (predicate push-down, pipeline fusion) before doing any real work
	- **Fault Tolerance**: it tracks how to recompute each step from the original data (its **lineage**), so if a partition or executor fails, it rebuilds only what's needed

### Actions
- **What they are**
	- Actions are the calls that trigger execution of the plan you've built. They return results to the driver or write data out
	- Common actions
		- `collect()`
		- `count()`
		- `show()`
		- `display()`
		- `write.parquet(...)`
		- `take(n)`
- **What happens on an action**
	1. Spark takes your accumulated logical plan
	2. It optimizes it into a **physical plan** (via the **Catalyst optimizer**). This physical plan runs as a **Spark job**
	3. It schedules tasks across executors
	4. Data flows through your transformations, and the requested result is produced

---

## Databricks Basics
- In Databricks you don't need to create the Spark session or the cluster yourself
