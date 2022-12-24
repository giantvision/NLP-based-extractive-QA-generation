# KMcha网站的近义词查询实践

**目标**

- 实践在内网构建的题库生成工具上进行错误答案的生成工作，

**实现步骤**

1. 在外网验证爬取数据的可行性，
2. 将代码放入外网的Django进行测试，
3. 验证通过后，将代码进行调试：
   - 辅助条件：需要前端协作进行完成，包括展现形式及选择



## 外网的代码验证

### 1、使用Scrapy工具进行实践

```python
# basic.py

import scrapy
from scrapy.loader import ItemLoader
from WordSearch.items import WordsearchItem

class BasicSpider(scrapy.Spider):
    name = 'basic'
    allowed_domains = ['kmcha.com']
    start_urls = ['https://kmcha.com/similar/李白']

    def parse(self, response):
        l = ItemLoader(item=WordsearchItem(), response=response)
        l.add_xpath('title', '//*[@id="similar-words"]/strong/text()')
        l.add_xpath('p1', '/html/body/div[3]/div[2]/div[3]/p[6]//span/text()')
        return l.load_item()
```



### 2、基于BeautifulSoup进行爬取实践

```python
# bs_test.py

from bs4 import BeautifulSoup
import urllib.request as urllib2
from lxml import etree, html
import requests

page = requests.get('https://kmcha.com/similar/杜甫')
tree = html.fromstring(page.content)
#This will create a list of buyers:
title = tree.xpath( '//*[@id="similar-words"]/strong/text()')
#This will create a list of prices
text = tree.xpath('/html/body/div[3]/div[2]/div[3]/p[6]//span/text()')

print('Title: ', title)
print('Choice: ', text)
```

