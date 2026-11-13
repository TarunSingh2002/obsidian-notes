---
tags:
  - database
  - mongodb
  - nosql
  - pymongo
---
### Links
- [Documentation](https://www.mongodb.com/docs/)
- [Tutorial Playlist](https://youtube.com/playlist?list=PLd3UqWTnYXOlzZmysFvd13g7W23voyuK9&si=UbMxpe6a-vGCNVvA)
- [PyMongo Documentation](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/)
- [PyMongo API Level Documentation](https://pymongo.readthedocs.io/en/stable/api/index.html)
- [Database Level Operations API](https://pymongo.readthedocs.io/en/4.11/api/pymongo/database.html#pymongo.database.Database.create_collection)
- [Collection Level Operations API](https://pymongo.readthedocs.io/en/4.11/api/pymongo/collection.html#pymongo.collection.Collection.with_options)

---

## Basics
- Based on JavaScript
- MongoDB is **case sensitive**
- Internally uses Mozilla's **SpiderMonkey** JavaScript engine
- MongoDB drivers for Python applications: **PyMongo**, **Motor**
- Data formats: **JSON**, **BSON**, **EJSON**

### Document Database / NoSQL Database
- A NoSQL database contains multiple **collections**
- Each collection contains multiple **documents**
- Structure
	- MongoDB → multiple databases → multiple collections → multiple documents → field and value pairs (Binary JSON format, **BSON**)

#### Comparison to a relational database

| MongoDB    | Relational DB |
| ---------- | ------------- |
| Collection | Table         |
| Document   | Row           |
| Field      | Column        |

- **Documents**
	- Stored in JSON-like BSON format
	- Each document has an auto-generated unique `_id`
	- Schema is **flexible (schema-less)**, so documents in the same collection can have different fields. Best practice is still to give related data a similar structure.
	- Document size limit is **16MB**
- **Fields**
	- Not every document needs to include the same fields

### MongoDB Shell/Client vs MongoDB Server
- Both are JavaScript-based applications
- **MongoDB Server**
	- Stores the data in the database
	- Server can be local or remote (MongoDB Atlas)
- **MongoDB Shell/Client**
	- Performs CRUD operations on the MongoDB server
	- It has a predefined object called `db`

---

## MongoDB Atlas
- Architecture
	- Organization → Clusters → Projects → Database
- Setup
	1. Create a project
	2. Go to Project → Security section → **Database Access**
		- Add a new user with password access
			- Give a name
			- Give a password
			- Give Built-in Role → **Atlas admin**
			- Then add the user
	3. Go to Project → Security section → **Network Access**
		- Add current IP address
		- Allow access from anywhere
	4. Create a cluster
		- You will get multiple options to connect to your cluster, like Compass, Shell, website, Driver (Python = PyMongo)
		- If you don't see those options, go to Clusters → click **Connect**
		- Connect with Shell / Compass / PyMongo accordingly
	5. Then create a Database and then a Collection

---

## Shell Basic Commands

### Server Related Commands
- Start the server
	```bash
	mongod --dbpath "your\location\data\db"
	```
	- Note: the first time, a bunch of files get created in the `db` folder
- Start the MongoDB shell
	```bash
	mongosh
	```

### show
```js
show profile
show users
show roles
show log <type>
show logs
```

### Connecting PyMongo with Atlas
```python
from pymongo import MongoClient

uri = "mongodb+srv://<username>:<password>@cluster0.y04v9yz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)

# Close the client
client.close()
```

---

## Database and Collections

### Database
- **Show databases**
	```js
	show dbs
	```
- **Create / use a database**
	```js
	use <db name>
	```
	- PyMongo
		```python
		# way 1
		database = client.get_database("sample_mflix")
		# way 2
		database = client["sample_mflix"]
		```
- **Drop database**
	```js
	db.dropDatabase()
	```

### Collection
- **Show collections**
	```js
	show tables
	show collections
	```
	- PyMongo
		```python
		collection_list = database.list_collections()
		```
- **Get an existing collection** (PyMongo)
	```python
	# way 1
	movies = database.get_collection("movies")
	# way 2
	movies = database["movies"]
	```
- **Create a collection**
	```js
	db.createCollection("<collection name>")
	```
	- PyMongo
		```python
		database = client["test_database"]
		database.create_collection("example_collection")
		```
- **Drop a collection**
	```js
	db.<collection name>.drop()
	```
	- PyMongo
		```python
		collection = database["test_collection"]
		collection.drop()
		```

### Validations
- **Creation**
	```js
	db.createCollection("<collection name>", {
	    validator: {
	        $jsonSchema: {
	            required: ['name', 'price', 'author'],
	            properties: {
	                name: {
	                    bsonType: 'string',
	                    description: 'must be a string and required'
	                },
	                price: {
	                    bsonType: 'number',
	                    description: 'must be a number and required'
	                },
	                author: {
	                    bsonType: 'array',
	                    description: 'must be an array and required',
	                    items: {
	                        bsonType: 'object',
	                        required: ['name', 'email'],
	                        properties: {
	                            name: {
	                                bsonType: 'string',
	                                description: 'must be a string and required'
	                            },
	                            email: {
	                                bsonType: 'string',
	                                description: 'must be a string and required'
	                            }
	                        }
	                    }
	                }
	            }
	        }
	    },
	    validationAction: 'error'
	});
	```
	- Here we create a collection with required fields: `name`, `price` and `author`
	- `validationAction` can be `warn` or `error`
		- `warn`: only a warning appears, and the change is still saved
		- `error`: the write is rejected, so only a string value is accepted for `name` and only a number for `price`
	- `description` is what the user sees in the warning or error
- **Update**
	```js
	db.runCommand({ collMod: "<collection name>", validator: <new validator> })
	```

---

## CRUD Operations

### Create
- **Create 1 document**
	```js
	db.<collection name>.insertOne({
	  name: "abc", age: 23,
	  isabc: { new: 1, new2: 2 },
	  def: [true, false, true]
	})
	```
- **Create many documents**
	```js
	db.<collection name>.insertMany([{}, {}, {}, {}], { ordered: false })
	```
	- `ordered: false`: if an error occurs while inserting some of the documents, the rest still get added to the collection
	- Write concern: *skipped*

### Read
- Notes
	- `find()` returns a **cursor** (pointer) over the matching documents, and `findOne()` returns a **single document**
	- In the empty case, `find()` returns an empty cursor and `findOne()` returns `null`
	- If a field holds a list, MongoDB automatically searches inside that list
- **Find all documents**
	```js
	db.<collection name>.find()
	db.<collection name>.find({ condition1, condition2 })
	db.<collection name>.find({ age: 22, name: "Tarun" })
	```
- **Find one document**
	```js
	db.<collection name>.findOne({ name: "Ram" })
	```
	- PyMongo
		```python
		query = { "title": "Back to the Future" }
		movie = movies.find_one(query)
		```
- **Find in a nested document**
	```js
	db.<collection name>.find({ "a.b.c": true })
	```
	- Here `"a.b.c"` is the field path, and it must be written in quotes like this

#### Projection
- Projection means selecting only the specific fields you want from a document instead of fetching the whole document
- Syntax
	```js
	db.student.find(query, { field1: 1, field2: 1 })
	```
	- `0` means exclude
	- `1` means include
	- A query can look like `{ age: { $lt: 20, $gt: 10 } }`
- Note
	- You can't mix inclusion and exclusion, so `a: 1, b: 0` is not allowed
	- The only exception is `_id`, so `{ _id: 0, a: 1 }` works

#### Complex find() Queries
- **Finding with a condition**
	```js
	db.<collection name>.find({ age: { $lt: 12 } })              // age less than 12
	db.<collection name>.find({ age: { $in: [289, 128, 768] } }) // age is one of these values
	```
- **Cursor methods**
	- `count()`
		```js
		db.<collection name>.find().count()
		```
	- `limit()`
		```js
		db.<collection name>.find().limit(2)
		```
	- `skip()`
		```js
		db.<collection name>.find().limit(20).skip(10)
		```
		- `skip(10)` skips the first 10 documents, then `limit(20)` shows the next 20
	- `sort()`
		```js
		db.<collection name>.find().limit(10).skip(9).sort({ age: 1 })
		```
		- `1` for ascending order, `-1` for descending order
- **Loop**
	```js
	db.<collection name>.find().forEach(doc => {
	    // access elements through doc._id
	})
	```
- **Complex expressions**: `$expr`
	- The `$expr` operator lets you use aggregation expressions inside a query
	- `{ $expr: { operator: ['$field', value] } }`
	- Example 1
		```js
		db.<collection name>.find({ $expr: { $gt: ['$price', 1324] } })
		```
	- Example 2: find documents where (quantity × price) is greater than target
		```js
		db.student.find({ $expr: { $gt: [{ $multiply: ['$quantity', '$price'] }, '$target'] } })
		```

#### $all and $elemMatch
- **$all**
	- Selects documents where the field's value is an array that contains **all** the specified elements
	- Example: return documents whose `comments` array has both users (Alice and Vinod)
		![[MongoDB - all example.png]]
		```js
		db.<collection name>.find({ 'comments.user': { $all: ['Alice', 'Vinod'] } })
		```
- **$elemMatch**: *skipped*

### Update / Add
- Updates the field if it exists, or creates a new field if it doesn't
- **In a single document**
	```js
	db.<collection name>.updateOne(
	  { name: "Ram" },
	  { $set: { idCards: { card1: true, card2: false } } }
	)
	```
- **In many documents**
	```js
	db.<collection name>.updateMany(
	  { age: 12 },
	  { $set: { abc: [1, 2, 3] } }
	)
	```
- **Arrays**
	- `$push`: add an element
		```js
		db.student.updateOne(
		  { filter },
		  { $push: { arrayField: "new element" } }
		)
		```
	- `$pop`: remove the last element (`1`) or the first element (`-1`)
		```js
		db.student.updateOne(
		  { filter },
		  { $pop: { arrayField: 1 } }
		)
		```
	- `$set`: update an embedded document inside an array
		```js
		db.student.updateOne(
		  { filter },
		  { $set: { "arrayField.$.text": "new element" } }
		)
		```
		- Example: set the `text` value in the `comments` array to "Awesome" where `_id` = 7 and user name = "Alice"
			![[MongoDB - update array element example.png]]
			```js
			db.student.updateOne(
			  { _id: 7, 'comments.user': 'Alice' },
			  { $set: { "comments.$.text": "Awesome" } }
			)
			```
			- Note: here `$` acts as the index of the array element matched by the filter

### Removing / Renaming Fields
- **Remove a field**
	```js
	db.<collection_name>.updateOne(
	  { filter },
	  { $unset: { fieldname: 1 } }
	)
	```
- **Rename a field**
	```js
	db.<collection_name>.updateOne(
	  { filter },
	  { $rename: { oldFieldName: "newFieldName" } }
	)
	db.user.updateMany({}, { $rename: { "a": "b" } });
	```

### Delete Documents
```js
db.<collection_name>.deleteOne({ filter })
db.<collection_name>.deleteMany({ filter })
```

---

## Operators

### Comparison Operators

| Operator    | Meaning                  |
| ----------- | ------------------------ |
| `$eq`       | equal to                 |
| `$ne`       | not equal to             |
| `$gt`       | greater than             |
| `$gte`      | greater than or equal to |
| `$lt`       | less than                |
| `$lte`      | less than or equal to    |
| `$in`       | in                       |
| `$nin`      | not in                   |
| `$multiply` | `{ $multiply: ['$quantity', '$price'] }` |
| `$add`      | addition                 |

- Example
	```js
	db.<collection name>.find({ age: { $lt: 12 } })  // age less than 12
	```

### Logical Operators
- Syntax: `{ $operator: [{ condition1 }, ..., { conditionN }] }`
- `$and`, `$or`, `$not`, `$nor`
- Example
	```js
	db.<collection name>.find({ $and: [{ price: { $gt: 100 } }, { name: 'D' }] })
	```

### Element Operators
- **$exists**
	- `{ field: { $exists: <boolean> } }`
	- Example
		```js
		db.<collection name>.find({ age: { $exists: true } })
		```
	- Question: find documents where the `price` field is present and its value is greater than 1200
		```js
		db.<collection name>.find({ price: { $exists: true, $gt: 1200 } })
		```
- **$type**
	- `{ field: { $type: "<bson-data-type>" } }`
	- Data types
		- string
		- number
			- long
			- int
			- double
		- decimal
		- dbPointer
		- object
		- array
		- bool
		- date
		- null
		- etc.
	- Example
		```js
		db.<collection name>.find({ price: { $type: "int" } })
		```
- **$size**
	- Matches documents where the size of an array field matches a specified value
	- `{ field: { $size: <array-length> } }`
	- Example
		```js
		db.student.find({ comments: { $size: 2 } })
		```

---

## Indexes

### Theory
- Indexes are specialized data structures that speed up data retrieval in MongoDB
- They store the data in a more searchable format
- A collection can have multiple indexes, and they are stored separately from the collection
- An index can be made **unique**, so duplicate values can't be entered
- `_id` is a default index
- **When to use**
	- If you often search, sort, or filter by a specific field, an index on that field lets MongoDB find the matching documents much faster
	- Example: in a `users` collection you often search by `email`. An index on `email` speeds up those queries, and you can even make it unique.
- **When not to use**
	- The field gets a large number of write operations
	- Small collections
	- Rarely used fields
	- Fields with only a few unique values (e.g. a Boolean field), because many documents share the same value
	- Complex queries that need to search on multiple fields
- **Types**
	- Single field indexes
	- Compound indexes
	- Text indexes

### Commands
- **Single field index**
	```js
	db.<collection name>.createIndex({ field: 1 })
	```
	- `1` creates the index in ascending order, `-1` in descending order
- **Compound index**
	```js
	db.<collection name>.createIndex({ fieldOne: 1, fieldSecond: 1 })
	```
	- Sorting happens on the first field first, then on the second field
- **Unique index**
	```js
	db.<collection name>.createIndex({ email: 1 }, { unique: true })
	```
- **Get info about the existing indexes in a collection**
	```js
	db.<collection name>.getIndexes()
	```
- **Delete an index**
	```js
	db.<collection name>.dropIndex({ field: 1 })
	db.<collection name>.dropIndex("Index_name")
	```

---

## Aggregations

### Syntax
```js
db.collection.aggregate(pipeline, options)
```
- `pipeline` is a list of stages (queries)

### $match
- Works the same as `find()`
	```js
	db.<collection name>.aggregate([
	  { $match: { gender: "male" } }
	])
	```
	![[MongoDB - aggregate match.png]]

### $group
- The `_id` field in the group stage specifies the field the documents are grouped by
- **Storing some fields from the documents in the group**
	```js
	db.<collection name>.aggregate([
	  { $group: { _id: "$age", NewField1: { $push: "$OldFieldName" } } }
	])
	```
	![[MongoDB - aggregate group push field.png]]
	- `NewField1` will be an array
	- `_id` stores the distinct group values
- **Storing the complete document in the group**
	```js
	db.<collection name>.aggregate([
	  { $group: { _id: "$age", NewField1: { $push: "$$ROOT" } } }
	])
	```
	![[MongoDB - aggregate group push ROOT.png]]
	- `_id` stores the distinct group values
	- `$$ROOT` holds the complete root document
	- `NewField1` will be an array of documents
- **Accumulator operators**
	- `$avg`
	- `$sum`

### Pipeline
- What a single document looks like
	![[MongoDB - pipeline sample document.png]]
- Example: count the male teachers per age and sort by count in descending order
	```js
	db.teachers.aggregate([
	  { $match: { gender: "male" } },
	  { $group: { _id: "$age", numberOfTeachers: { $sum: 1 } } },
	  { $sort: { numberOfTeachers: -1 } }
	])
	```
