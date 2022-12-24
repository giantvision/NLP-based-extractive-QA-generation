# NLP数据集格式处理

### 文件转换

**注意点：**

原txt文件需要更改编码格式为utf-8，然后再用python进行转换处理，可避免解码转换报错。

转换脚本：

```python 
import csv
with open('./wiki_data/ebook.txt', 'r', encoding='utf-8') as in_file:
    in_file = in_file.readlines()
    with open('./wiki_data/Encyclopedia.csv', 'w', encoding='utf-8-sig') as out_file:
        for line in in_file:
            out_file.write(line)
```



## 中药相关的数据集格式

中药训练集格式范例：

```json
[
  {
    "id": 828,
    "text": "反佐配伍的典范，始见于张仲景《伤寒杂病论》，其中记载“干呕、吐涎沫、头痛者吴茱萸汤主之”。患者病机为肝寒犯胃，浊气上逆所致头痛。胃阳不布产生涎沫随浊气上逆而吐出，肝脉与督脉交会于巅顶，肝经寒邪，循经上冲则头痛，以吴茱萸汤主治。可在吴茱萸汤中加入少许黄连反佐，用以防止方中吴茱萸、人参、干姜等品辛热太过，从而达到温降肝胃、泄浊通阳而止头痛的功效。后代医者多在清热剂和温里剂中运用此法。",
    "annotations": [
      {
        "Q": "“干呕、吐涎沫、头痛者吴茱萸汤主之”这句话曾出现在哪本医学巨著中？",
        "A": "《伤寒杂病论》"
      },
      {
        "Q": "《伤寒杂病论》的作者是谁？",
        "A": "张仲景"
      },
      {
        "Q": "关于反佐配伍，在吴茱萸汤中加入少许黄连反佐，能起到什么作用？",
        "A": "用以防止方中吴茱萸、人参、干姜等品辛热太过，从而达到温降肝胃、泄浊通阳而止头痛的功效。"
      }
    ]
  }
]
```

目前待处理的数据集格式：

| index | title |
| ----- | ----- |
| 1     | xxx   |
| 2     | xxx   |


想要的json格式：

```json
[
    {
        "id": xxx,
        "text": "xxx",
        "annotations": [
            {
                "Q": "XXX",
                "A": "XXX"
            },
            {
                "Q": "XXX",
                "A": "XXX"
            },
        ]
    }
]
```

对应关系为：

- index  -->  id

- title    -->  text



## CMRC2018数据集格式

训练集的格式为：



```json
[
    {
        "context_id": xxx,
        "context_text": xxx,
        "qas":[
            {
                "query_text": xxx,
            	"query_id": xxx,
            	"answers": xxx
            },
            {
               xxx 
            }
        ]
        "title": xxx
    }
]
```











