# MongoDB使用及与Django交互指南

### 目录

1. Django install for windows
2. MongoDB操作指南
3. pymongo连接Django与MongoDB
4. MongoDB的游标使用
5. 对关键词的模糊搜索功能探索



### <span style='color:brown'>推荐参考指南：</span>

- [Basic MongoDB Operations in Python](https://www.mongodb.com/developer/quickstart/python-quickstart-crud/)





## 1、Django  install for  windows

参考资料：

- [How to install Django on Windows](https://docs.djangoproject.com/en/4.0/howto/windows/)
- [Setting up a virtual environment for your Django Project](https://codesource.io/setting-up-a-virtual-environment-for-your-django-project/)
- [windows10设置环境变量PATH](https://jingyan.baidu.com/article/8ebacdf02d3c2949f65cd5d0.html)

**Install Django on Windows:**

1. Install python

   检查python版本：

   ```shell
   python --version
   ```

2. About pip

   设置虚拟环境：

   ```shell
   py -m venv project-name
   ```

   激活环境：

   ```shell
   project-name\Scripts\activate.bat
   ```

3. Install Django

   ```shell
   py -m pip install Django
   ```

> <center>NOTE</center>
>在使用pip时，可能存在无法安装问题，需要注意VPN设置问题。



## 2、MongoDB操作指南

- mongosh 交互工具
- [mongosh命令行操作指南](https://docs.mongodb.com/manual/reference/mongo-shell/)

```shell
# 显示已有的数据表
show dbs

# 切换使用不同的数据库
use dataname

# 显示collections--集合
show collections

# 删除数据集
db.dropDatabase()

# 创建新的数据库
use appdb

# 添加数据
db.users.insertOne({name : 'Mick'})

# 查看数据库详情
db.users.find()

# MongoDB 显示所有集合中的所有内容
show dbs 
use <db name>
show collections
#选择你的收藏，并输入以下内容，以查看该收藏的所有内容
db.collectionName.find()


# 退出mongodb
exit
```

### **遇到的问题：**

1. vs code无法切换到虚拟环境问题

   **问题描述：**

   > 无法加载文件 C:\Users\DH\Desktop\cs\rename.ps1，因为在此系统上禁止运行脚本。有关详细信息，请参阅 https:/go.microsoft.com/fwlink/?LinkID=135170 中的 about_Execution_Policies。
   >
   > \+ CategoryInfo : SecurityError: (:) []，ParentContainsErrorRecordException
   >
   > \+ FullyQualifiedErrorId : UnauthorizedAccess

   **原因分析：**

   - [网址参考](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.2)

   计算机上启动 Windows PowerShell 时，执行策略是： Restricted（默认设置）；

   Restricted 执行策略不允许任何脚本运行；AllSigned 和 RemoteSigned 执行策略可防止 Windows PowerShell 运行没有数字签名的脚本。

   **解决方案：**

   - 查看计算机上的现用执行策略，打开PowerShell 然后输入 **get-executionpolicy**；
   - 以管理员身份打开PowerShell 输入 **set-executionpolicy remotesigned**，选择Y。

2. 内网工程代码运行异常问题

   - 问题描述：[gethostbyaddr() raises UnicodeDecodeError in Python 3 [duplicate\]](https://stackoverflow.com/questions/25948687/gethostbyaddr-raises-unicodedecodeerror-in-python-3)
   - 参考链接
     - [解决方法](https://stackoverflow.com/questions/25948687/gethostbyaddr-raises-unicodedecodeerror-in-python-3)
     - [具体操作](https://support.microsoft.com/en-us/windows/rename-your-windows-10-pc-750bc75d-8ff8-e99a-b9dc-04dff566ae74)--改写hostname(事业部-XXX，改成：duoyi)



## 3、pymongo连接Django与MongoDB

参考资料：

- https://docs.mongodb.com/drivers/pymongo/

#### Connect to MongoDB Atlas

```python
import pymongo

# Replace the uri string with your MongoDB deployment's connection string.
conn_str = "mongodb+srv://<username>:<password>@<cluster-address>/test?retryWrites=true&w=majority"

# set a 5-second connection timeout
client = pymongo.MongoClient(conn_str, serverSelectionTimeoutMS=5000)

try:
    print(client.server_info())
except Exception:
    print("Unable to connect to the server.")
```

#### Stable API

您可以使用从 MongoDB Server 5.0 版和 PyMongo 驱动程序 3.12 版开始的稳定 API 功能。使用 Stable API 功能时，您可以更新驱动程序或服务器，而不必担心 Stable API 涵盖的任何命令的向后兼容性问题。

```python
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Replace <connection string> with your MongoDB deployment's connection string.
conn_str = "<connection string>"

# Set the version of the {+stable-api+} on the client.
client = pymongo.MongoClient(conn_str, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
```

### Use PyMongoDB fetching data

- [参考](https://www.geeksforgeeks.org/how-to-fetch-data-from-mongodb-using-python/)

1. find

   1. find one

      ```python
      import pymongo
      client = pymongo.MongoClient("mongodb://localhost:27017/")
      # Database Name
      db = client["database"]
      # Collection Name
      col = db["GeeksForGeeks"]
      x = col.find_one()
      print(x)
      ```

   2. find all

      ```python
      import pymongo
      client = pymongo.MongoClient("mongodb://localhost:27017/")
      # Database Name
      db = client["database"]
      # Collection Name
      col = db["GeeksForGeeks"]
      x = col.find()
      for data in x:
      	print(data)
      ```

   3. find only special filed

      ```python
      import pymongo
      client = pymongo.MongoClient("mongodb://localhost:27017/")
      # Database Name
      db = client["database"]
      # Collection Name
      col = db["GeeksForGeeks"]
      # Fields with values as 1 will
      # only appear in the result
      x = col.find({},{'_id': 0, 'appliance': 1,
      				'rating': 1, 'company': 1})
      for data in x:
      	print(data)
      ```

2. query

   ```python
   import pymongo
   myclient = pymongo.MongoClient("mongodb://localhost:27017/")
   mydb = myclient["mydatabase"]
   mycol = mydb["customers"]
   myquery = { "address": { "$gt": "S" } }
   mydoc = mycol.find(myquery)
   for x in mydoc:
     print(x)
   ```

3. delete

   1. delete_one()

      ```python
      import pymongo
      myclient = pymongo.MongoClient("mongodb://localhost:27017/")
      mydb = myclient["mydatabase"]
      mycol = mydb["customers"]
      myquery = { "address": "Mountain 21" }
      mycol.delete_one(myquery)
      ```

   2. delete_many()

      ```python
      import pymongo
      myclient = pymongo.MongoClient("mongodb://localhost:27017/")
      mydb = myclient["mydatabase"]
      mycol = mydb["customers"]
      myquery = { "address": {"$regex": "^S"} }
      x = mycol.delete_many(myquery)
      print(x.deleted_count, " documents deleted.")
      ```

4. update

   使用db.collection.update()对数据进行更新。

   三种方式：updateOne、updateMany、replaceOne

   replaceOne会用新文档完全替换匹配的文档。使用主键“_id”查询更加高效。

   具体实践：

   - [官方文档:  db.collection.replaceOne()](https://www.mongodb.com/docs/manual/reference/method/db.collection.replaceOne/)

     db.collection.replaceOne(filter, replacement, options)

     <span style='color:red'>**注意认真读官方文档**</span>

   - [参考链接](https://mongoing.com/docs/reference/method/db.collection.replaceOne.html)

   **Using MongoDB's Update API methods in PyMongo**

   update() 方法已弃用，取而代之的是 update_one() 和 update_many().如果您熟悉旧版本 MongoDB 中使用的 update() 方法，请注意此方法已在 MongoDB 版本 3.x 及更高版本中被弃用。因此，您需要在编写代码之前提前计划，以确定 API 调用是只更新一个文档还是更新多个文档。

   **MongoDB对文档字段的更新操作方法**

   MongoDB有许多更新操作符，可以用来修改文档的字段。让我们仔细看一下这些操作符：

   - 更新设置值的运算符：
     - $set --顾名思义，此运算符设置字段的值；
     - $setOnInsert--这仅在插入新文档时更新字段的值；
     - $unset -- 这将从文档中删除该字段及其值；
   - 基于数值的运算符：
     - $inc -- 这将通过参数中指定的增量更改字段的数值；
     - $min and \$max -- 只有当该字段的数值落在指定的最小或最大范围内时，它们才会更新文档的字段；
     - $mul--这会将文档的数值乘以指定的数量。
   - 其他更新操作符：
     - $currentData--这会更改字段的值以匹配当前日期；
     - $rename--这只是更改了字段的名称。

   

   ```shell
   # mongosh的喜欢更新操作
   db.geo_location.replaceOne({'_id':1}, {"_id":1, "text":"hello world", "qa_pair":[{'q':"xxx", "a":"yyy"}]}, {upsert:true})
   
   # output：成功完成替换操作
   {
     acknowledged: true,
     insertedId: null,
     matchedCount: 1,
     modifiedCount: 1,
     upsertedCount: 0
   }
   ```

   **使用pymongo进行更新操作：**

   官方参考文档：[链接](https://www.mongodb.com/blog/post/getting-started-with-python-and-mongodb)

   实践参考案例：[链接](https://kb.objectrocket.com/mongo-db/how-to-update-a-mongodb-document-in-python-356)

   类似于**`insert_one`**并且**`insert_many`**存在帮助您更新 MongoDB 数据的函数，包括**`update_one`**、**`update_many`**和**`replace_one`**. 该**`update_one`**方法将根据与文档匹配的查询更新单个文档。

   使用replace_one的实践：使用场景--就地对数据进行更新操作，

   ```python
   update_data = db[domain_label].replace_one("_id": id, update_data, upsert=True)
   ```

   对mongoDB的部分数据进行更新：

   ```python
   mycollection.update_one({'_id':mongo_id}, {"$set": post}, upsert=False)
   ```

   <span style='color:red'>**多级嵌套结构下的部分数据更新功能**</span>

   ```python
   @csrf_exempt
   def update_part(request):
       # 1、获取需要更新的数据信息
       body = json.loads(request.body.decode('utf-8'))
       table_name = body.get('table_name', '')
       domain_label = body.get('domain_label', '')
       update_data = body.get('update_data', '')
       field_data = body.get('field_data', None)
       db = client[table_name]
       chosen_col = db[domain_label]
       #2、定位到数据并对部分数据进行实时更新
       if field_data == 'text':
           chosen_data = chosen_col.update_one(
           {"_id": update_data["_id"]},
           {"$set": {"text": update_data["text"]}},
           upsert=False)
       elif field_data == 'q':
           chosen_data = chosen_col.update_one(
           {"_id": update_data['_id'], "qa_pair.c_id": update_data['c_id']},
           {"$set": {"qa_pair.$.q": update_data}},
           upsert=False)
       elif field_data == 'a':
           chosen_data = chosen_col.update_one(
           {"_id": update_data['_id'], "qa_pair.c_id": update_data['c_id']},
           {},
           upsert=False)
       ret = {
           "update_part_data": "Successful update."
       }
       return JsonResponse(ret)
   ```

   

   <span style='color:brown'>**使用update_one的实践**</span>

   使用过滤器查询查找 MongoDB 文档，然后使用 PyMongo 更新它：

   1、使用 find_one_and_update() 更新 MongoDB 文档：

   您可以使用 find_one_and_update() 方法查询 MongoDB 文档并同时更新文档的内容，然后让它返回文档作为结果。

   使用find_one_and_update()方法通过其BSON ObjectId找到MongoDB文档，然后更新其内容。尽管你可以通过BSON ObjectId查询一个文档，正如我们在前面的例子中看到的find_one()方法调用，find_one_and_update()方法要求在查询一个文档时传递一个完整的字典对象，包括一个键值对。因此，你需要将键设置为"_id"，并传递一个BSON对象作为查询的值。如果这听起来有点混乱，不要担心，下面的例子表明，语法实际上是很简单的。

   ```json
   # BSON dict for the tuple object's first element
   {"_id" : ObjectId("5cfbb46d6fb0f3245fd8fd34")}
   ```

   正如我们上面提到的，元组中的第二个元素需要你传递上述的 "更新操作符 "之一作为它的字典键。让我们看看一个使用"$set "操作符的 find_one_and_update() 方法调用的例子。

   ```python
   doc = col.find_one_and_update(
       {"_id" : ObjectId("5cfbb46d6fb0f3245fd8fd34")},
       {"$set":
           {"some field": "OBJECTROCKET ROCKS!!"}
       },upsert=True
   )
   ```

   find_one_and_update() 方法返回一个包含 MongoDB 文档的所有数据的字典对象。该字典将包含该文档的"_id"，以及该文档的所有其他字段，作为其键：

   ```json
   {'_id': ObjectId('5cfbb46d6fb0f3245fd8fd34'), 'some field': 'OBJECTROCKET ROCKS!!'}
   ```

   2、使用 update_one() 方法修改 MongoDB 文档：

   update_one() API 方法的工作方式与 find_one_and_update() 非常相似，只是它不返回文档，而是返回结果对象。

   update_one() 方法返回的 PyMongo 结果对象的属性。如果 API 调用成功，它应该在使用 update_one() 或 update_many() 方法更新 MongoDB 文档时返回一个 results.UpdateResult 对象。

   使用 update_one() 方法递增 MongoDB 文档的整数：

   在我们的下一个示例中，我们将在将参数传递给 update_one() 方法时使用“\$inc”键。这将使特定字段的值增加“$inc”键值中指定的数字量：

   ```python
   result = db["Some Collection"].update_one(
       {"num": 41}, {"$inc":
           {'num': 1} # new value will be 42
       }
   )
   ```

   3、如果不存在，则将 Upsert 标志设置为“True”以创建新文档：

   正如我们在上一节中提到的，如果没有找到与更新语句的指定查询匹配的文档，则不会更新任何文档。但是，您可以使用 upsert 布尔选项作为方法调用的元组对象中的最后一个元素来更改此行为——如果调用的查询未找到匹配的文档，此标志将指示 MongoDB 插入新文档。让我们在下一个示例中使用 upsert 选项：

   ```python
   # the document's content will change to this:
   new_val = {"some text": "ObjectRocket: Database Management and Hosting"}
   
   # pass the 'new_val' obj to the method call
   result = col.update_one(
       {"_id" : ObjectId("5cfbb46d6fb0f3245fd8fd34")},
       {"$set": new_val},
       upsert=True
   )
   ```

   您可以通过访问 results.UpdateResult 对象的 upserted_id 属性来验证是否插入了新文档 _id：

   ```python
   # upserted_id is 'None' if ID already existed at time of call
   print ("Upserted ID:", result.upserted_id)
   ```

   如果插入了一个新文档（其 _id 使用 BSON ObjectId），则结果对象的 upserted_id 属性应该是 bson.objectid.ObjectId 对象；否则，此属性的值将只是一个 Python NoneType 对象。

   4、使用 $currentDate 运算符更新或插入具有当前日期的 MongoDB 字段

   让我们看一个例子，其中 find_one_and_update() 方法调用使用 $currentDate 运算符。此调用将完成两件事：插入或创建具有日期类型字段的 MongoDB 文档（使用名称“某个日期”），并为该新字段提供具有当前日期和时间的日期字符串值：

   ```python
   col.find_one_and_update(
       {"_id" : ObjectId("5cfa4c0ad9edff487939dda0")},
       {"$currentDate": {"some date": True}},
       upsert = True
   )
   ```

   请注意，此示例使用 upsert = True 选项来确保如果该特定时间戳 _id 不存在，将插入文档。

   $currentDate 运算符返回 yyyy-MM-dd HHTmm:ss.SSS 格式的字符串（例如 2019-06-08T14:09:07.247+00:00）。

5. Limit

   为了限制 MongoDB 中的结果，我们使用 limit() 方法。limit() 方法采用一个参数，一个定义要返回多少文档的数字。

   ```json
   {'_id': 1, 'name': 'John', 'address': 'Highway37'}
   {'_id': 2, 'name': 'Peter', 'address': 'Lowstreet 27'}
   {'_id': 3, 'name': 'Amy', 'address': 'Apple st 652'}
   {'_id': 4, 'name': 'Hannah', 'address': 'Mountain 21'}
   {'_id': 5, 'name': 'Michael', 'address': 'Valley 345'}
   {'_id': 6, 'name': 'Sandy', 'address': 'Ocean blvd 2'}
   {'_id': 7, 'name': 'Betty', 'address': 'Green Grass 1'}
   {'_id': 8, 'name': 'Richard', 'address': 'Sky st 331'}
   {'_id': 9, 'name': 'Susan', 'address': 'One way 98'}
   {'_id': 10, 'name': 'Vicky', 'address': 'Yellow Garden 2'}
   {'_id': 11, 'name': 'Ben', 'address': 'Park Lane 38'}
   {'_id': 12, 'name': 'William', 'address': 'Central st 954'}
   {'_id': 13, 'name': 'Chuck', 'address': 'Main Road 989'}
   {'_id': 14, 'name': 'Viola', 'address': 'Sideway 1633'}
   ```

   ```python
   import pymongo
   myclient = pymongo.MongoClient("mongodb://localhost:27017/")
   mydb = myclient["mydatabase"]
   mycol = mydb["customers"]
   myresult = mycol.find().limit(5)
   #print the result:
   for x in myresult:
     print(x)
   ```

6. 查看names of all collections using PyMongo

   ```python
   import pymongo
   import json
   if __name__ == '__main__':
       client = pymongo.MongoClient("localhost", 27017, maxPoolSize=50)
       d = dict((db, [collection for collection in client[db].collection_names()])
                for db in client.database_names())
       print json.dumps(d)
   ```

7. insert--插入数据

   参考：[官网文档](https://www.mongodb.com/blog/post/getting-started-with-python-and-mongodb)

   insert_one()、insert_many()

   ```python
   # insert one
   chosen_data = chosen_col.insert_one(item)
   # insert many
   chosen_data = chosen_col.insert_one(data)
   ```

8. exist--判断_id数据是否已经存在

   参考链接：[stack overflow](https://stackoverflow.com/questions/67651034/how-can-i-check-if-something-already-exists-in-a-mongodb-database)

   ```python
   import pymongo
   conn = pymongo.MongoClient("mongodb://localhost:27017")
   dataID = 12465465463
   if conn.mydb.mycol.count_documents({'dataID':dataID}):
       print('**Error: The data is already in the database.')
   else:
       print("Adding new data into the mongoDB database.")
       mydict = { "userID": userID, "coin": "0", "inv": {"inv1": "", "inv2": "", "inv3": "", "inv4": "", "inv5": ""} }
       y = conn.mydb.mycol.insert_one(mydict)
   ```



## 4、<span style="color:brown">**MongoDB的游标使用**</span>--实现分页功能

数据库会使用游标返回find()执行的结果。最常见的查询选项是限制返回结果的数量、略过一定数量的结果以及排序。

<span style='color:blue'>**limit、skip、sort**</span>

不使用skip对结果进行分页：对于结果非常多的情况，skip会非常慢，因为需要先找到被略过的结果，然后再丢弃这些数据。

例如：按照“date”降序显示文档(题库中可以使用_id实现相同的功能)

```c++
var page1 = db=foo.find().sort({"date":-1,}).limit(100)
```

假设日期是唯一的，可以使用最后一个文档的“data”值作为获取下一页的查询条件：

```c++
var latest = null;

//显示第一页
while (page1.hasNext()){
    latest = page1.next();
    display(latest);
}

//获取下一页
var page2 = db.foo.find({"date":{"$lt": latest.date}});
page2.sort({"date": -1}).limit(100)
```

这样查询中就没有了skip。



## 5、对关键词的模糊搜索功能探索

### <span style='color:brown'>**find()**</span>

**简单实现代码示例：**

```python
import re
data_re = re.compile(r'20220425')
db.collection.find({'data':data_re})
```

```python
db.collection.find({'data':{'$regex':'20220425'}})
```



**使用n_grams对查询词进行扩展：**

- 参考资料：
  - [Fuzzy Text Search with MongoDB and Python](https://mschmitt.org/blog/fuzzy-search-mongodb-python/)

```python
import fuzzy

# Building the search term collection
for word in terms:
    if len(word) <= 2 or word in stop_words:
        # skip short words or ignorable words
        continue
    fuzzy_terms = []
    fuzzy_terms.append(dmeta(word)[0])
    fuzzy_terms.append(fuzzy.nysiis(word))
    for term in fuzzy_terms:
        search_terms_collection.insert({
            "keyword":term,
            "original": word,
            "item": item["_id"]
        })
        
        
# Searching our fuzzy collection
search_words = search_query.split(' ')
fuzzy_terms = []
for word in search_words:
    if word in stop_words:
        continue
    fuzzy_terms.append(dmeta(word)[0])
    fuzzy_terms.append(fuzzy.nysiis(word))

results = search_terms_collection.find({
    "$or": [
        {"keyword": {"$in": fuzzy_terms}},
        {"original":{
            "$regex": search_query, "$options": "i"}}
    ]
})

# Sorting our results
result_map = {}
for result in results:
    result_item = str(result["item"])
    keyword_distance = float(distance(search_phrase, result['original']))
    if not result_map.has_key(result_item):
        result_map[result_item] = keyword_distance
    else:
        result_map[result_item] = min(keyword_distance, result_map[result_item])
```



### <span style='color:brown'>**aggregate()**</span>

官方文档参考：

- [Getting Started with Aggregation Pipelines in Python](https://www.mongodb.com/developer/quickstart/python-quickstart-aggregation/)
- [Document Aggregation Example](https://pymongo.readthedocs.io/en/stable/examples/aggregation.html)
- [Medium example](https://blog.manash.io/how-to-use-regex-on-aggregation-using-pymongo-b9377935c721)

PyMongo使用Collection的aggregate()方法执行聚合管道，聚合管道是对集合中的所有数据进行操作，管道中的每个阶段都应用于通过的文档，并且从一个阶段发出的任何文档都作为输入传递到下一个阶段，知道没有更多的阶段。此时，管道最后阶段发出的文档将返回到客户端程序。最后某些阶段，比如$group，将根据传入的文档作为一个整体创建一组全新的文档，这些阶段都不会改变储存在MongoDB本身中的数据，他们只是在将数据返回到你的程序之前更改了数据。

#### <span style='color:brown'>Using MongoDB Supported Regular Expression</span>

```python
from bson import regex
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['your_db_name']
collection = db['your_db_collection']

result = collection.aggregate([
    {"$match": {"$key_to_apply_regex": {"$regex": regex.Regex("regex|expression")}}}
])

print(list(result))
```

#### <span style='color:brown'>Using Native Regular Expressions</span>

```python
import re
from bson import regex
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['your_db_name']
collection = db['your_db_collection']
result = collection.aggregate([
    {"$match":{"$key_to_apply_regex":{"$regex":regex.Regex.from_native(re.compile(".*"))}}}
])
print(list(result))
```



**在本项目中的代码实践：**

```python
# aggregate--用户进行模糊搜索，从词条中找到相应的数据并返回
def search_aggregate(request):
    body = json.loads(request.body.decode('utf-8'))
    query_word = body.get('query_word')
    table_name = body.get('table_name')
    domain_label = body.get('domain_label')
    db = client[table_name]
    chosen_col = db[domain_label]
    results = chosen_col.aggregate([
        {"$match": {"text": {"$regex": query_word}}}
    ])
    ret = {
        'search_resluts': results
    }
    return JsonResponse(ret)
```

