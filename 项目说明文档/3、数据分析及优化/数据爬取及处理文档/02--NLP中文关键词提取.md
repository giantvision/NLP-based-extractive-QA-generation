# 中文关键词提取

目录：

1. 参考链接
2. JioNLP功能介绍
3. jieba关键词抽取
4. 基于DGCNN和概率图的轻量级信息抽取模型
5. HanLP：面向生成环境的自然语言处理工具包
6. BERT-KPE关键词提取
7. T5_in_bert4keras





## 1、参考链接：

- http://182.92.160.94:16666/#/extract_keyphrase
- https://github.com/Geekzhangwei/TimeNLP
- https://github.com/fxsjy/jieba
- https://github.com/hankcs/pyhanlp
- https://github.com/thunlp/BERT-KPE
- https://github.com/bojone/t5_in_bert4keras



## 2、JioNLP功能介绍

参考链接：

- https://github.com/dongrixinyu/JioNLP

#### 简介：

- JioNLP 提供 NLP 任务预处理功能，准确、高效、零使用门槛，并提供一步到位的查阅入口。

目前具备的功能：

1. 关键短语抽取
2. 时间语义解析
3. 时间实体抽取
4. 地址解析

JioNLP使用说明：

```markdown
Args:
text: 中文文本
top_k: (int) 选取多少个关键短语返回，默认为 5，若为 -1 返回所有短语
with_weight: 指定返回关键短语是否需要短语权重
func_word_num: 允许短语中出现的虚词个数，strict_pos 为 True 时无效
stop_word_num: 允许短语中出现的停用词个数，strict_pos 为 True 时无效
max_phrase_len: 允许短语的最长长度，默认为 25 个字符
topic_theta: 主题权重的权重调节因子，默认0.5，范围（0~无穷）
strict_pos: (bool) 为 True 时仅允许名词短语出现，默认为 True
allow_pos_weight: (bool) 考虑词性权重，即某些词性组合的短语首尾更倾向成为关键短语，默认为 True
allow_length_weight: (bool) 考虑词性权重，即 token 长度为 2~5 的短语倾向成为关键短语，默认为 True
allow_topic_weight: (bool) 考虑主题突出度，它有助于过滤与主题无关的短语（如日期等），默认为 True
without_person_name: (bool) 决定是否剔除短语中的人名，默认为 False
without_location_name: (bool) 决定是否剔除短语中的地名，默认为 False
remove_phrases_list: (list) 将某些不想要的短语剔除，使其不出现在最终结果中
remove_words_list: (list) 将某些不想要的词剔除，使包含该词的短语不出现在最终结果中
specified_words: (dict) 行业名词:词频，若不为空，则仅返回包含该词的短语
bias: (int|float) 若指定 specified_words，则可选择定义权重增加值
```



### 关键短语抽取-- extract_keyphrase

给定一段文本，返回其中的关键短语，默认为5个。

```python
>>> import jionlp as jio
>>> text = '朝鲜确认金正恩出访俄罗斯 将与普京举行会谈...'
>>> key_phrases = jio.keyphrase.extract_keyphrase(text)
>>> print(key_phrases)
>>> print(jio.keyphrase.extract_keyphrase.__doc__)

# ['俄罗斯克里姆林宫', '邀请金正恩访俄', '举行会谈',
#  '朝方转交普京', '最高司令官金正恩']
```

- 原理简述：在 tfidf 方法提取的碎片化的关键词（默认使用 pkuseg 的分词工具）基础上，将在文本中相邻的关键词合并，并根据权重进行调整，同时合并较为相似的短语，并结合 LDA 模型，寻找突出主题的词汇，增加权重，组合成结果进行返回。



## 3、jieba关键词抽取

Github地址：

- https://github.com/fxsjy/jieba

两种关键词提取方法

1. 基于TF-IDF算法的关键词提取
2. 基于TextRank算法的关键词抽取

### 基于TF-IDF算法的关键词提取

```python
import jieba.analyse
```

- jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
  - sentence 为待提取的文本
  - topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
  - withWeight 为是否一并返回关键词权重值，默认值为 False
  - allowPOS 仅包括指定词性的词，默认值为空，即不筛选
- jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件

##### **代码示例**

>  测试样例文本：
>
> 中国传统绘画，又称丹青，在中国又称国画，泛指中华文化的传统绘画艺术，是琴棋书画四艺之一。狭义的国画指青绿设色画水墨画，而广义的国画则是中国传统风格的壁画、锦画、刺绣、重彩、水墨画、石刻乃至年画和陶瓷上的绘画等的艺术，也包括近代的中国油画和水彩画等。  国画历史悠久，在东周墓葬中出土过最早的帛画作《龙凤仕女图》，传世最早最完整的作品是顾恺之《女史箴图》的唐朝摹本。在五代十国以后中国文人艺术家得到了很高的社会地位。宋代以前绘图在绢帛上，材料昂贵，因此国画题材多以王宫贵族肖像或生活记录等，直至宋元两代后，纸材改良，推广与士大夫文人画兴起等，让国画题材技法多元。明代绘画推广到大众，成为市民生活的一部分，风俗画因此产生。清末，绘画材料多元，朝多方面发展。

```python 
# jieba关键词提取--extract_tags_idfpath
import sys
import jieba
import jieba.analyse

file_name = 'test.txt'
topK = 10
content = open(file_name, 'rb').read()
jieba.analyse.set_idf_path("./jieba-extra_dict/idf.txt.big")
tags = jieba.analyse.extract_tags(content, topK=topK)

print(",".join(tags))
```

> 输出结果:
>
> 顾恺之,理论,箴图,现剩,唐绢,临本藏于,大英博物馆,清宫,英法联军,圆明园

```python
# jieba关键词提取--extract_tags_with_weight
import jieba
import jieba.analyse

file_name = 'test.txt'
topK = 10
withWeight = True

content = open(file_name, 'rb').read()
tags = jieba.analyse.extract_tags(content, topK=topK, withWeight=withWeight)

if withWeight is True:
  for tag in tags:
    print("tag: %s\t\t weight: %f" % (tag[0],tag[1]))
else:
  print(",".join(tags))
```

 输出结果：

tag: 顾恺之		        weight: 0.391960 
tag: 理论		            weight: 0.391960 
tag: 箴图		            weight: 0.195980 
tag: 现剩		            weight: 0.195980 
tag: 唐绢		            weight: 0.195980 
tag: 临本藏于	        weight: 0.195980 
tag: 大英博物馆	    weight: 0.195980 
tag: 清宫		            weight: 0.195980 
tag: 英法联军	        weight: 0.195980 
tag: 圆明园	            weight: 0.195980



### 基TextRank算法的关键词提取

- jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 直接使用，接口相同，注意默认过滤词性。
- jieba.analyse.TextRank() 新建自定义 TextRank 实例

算法实现思想：

1. 将待抽取关键词的文本进行分词
2. 以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系构建图
3. 计算图中节点的PageRank，注意是无向带权图





## 4、基于DGCNN和概率图的轻量级信息抽取模型

参考链接：

- Github：https://github.com/bojone/dgcnn_for_reading_comprehension
- [基于DGCNN和概率图的轻量级信息抽取模型](https://kexue.fm/archives/6671)



## 5、HanLP：面向生产环境的自然语言处理工具包

参考链接：

- https://github.com/hankcs/pyhanlp
- https://hanlp.hankcs.com/



## 6、**BERT for Keyphrase Extraction** (PyTorch)

简介：

该存储库提供论文 Capturing Global Informativeness in Open Domain Keyphrase Extraction 的代码。在本文中，我们对5个关键词提取模型与3个BERT变体进行了实证研究，然后提出一个多任务模型BERT-JointKPE。在两个KPE基准上的实验，OpenKP与Bing网页和KP20K证明了JointKPE的先进性和强大的有效性。我们的进一步分析还表明，JointKPE在预测长关键词和非实体关键词方面具有优势，这对以前的KPE技术来说是一个挑战。





## 7、T5 in bert4keras

参考：

- 博客链接：https://kexue.fm/archives/7867

- 本项目实验环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.9.1



















