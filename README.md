# 基于NLP的常识性问答题库生成的需求调研

**背景介绍：**

了解目前游戏场景中的答题模块的现状，分析是否需要使用NLP或者其他方式进行补充，用来生成更多样性的题目，并满足游戏的使用条件。

#### 题例：

1. 传统文化

   | 题目                                | 选项                         |
   | :---------------------------------- | :--------------------------- |
   | “民为贵，社稷次之...”出自哪部作品？ | 《孟子》、《孔子》、《庄子》 |
   | 佛光寺是现存的唐代哪种结构的建筑？  | 木结构、砖结构、石结构       |
   | 三国演义中的“卧龙”是？              | 诸葛亮、赵云、华佗           |
   | 明朝戏剧《牡丹亭》又称？            | 还魂记、西厢记、长生殿       |
   | 古人用“廿”表示多少？                | 二十、四十、三十             |

   

2. 自然科学

   | 题目                                         | 选项                                                 |
   | -------------------------------------------- | ---------------------------------------------------- |
   | 不移动圆饼切三刀，最多可以切出多少块？       | 8、10、14                                            |
   | 蜂巢的内部结构是什么形状？                   | 六边形、方形、圆形                                   |
   | 雷雨天，人们先看到闪点后听到雷声是什么原因？ | 光速大于声速、先产生闪电后产生雷声、耳朵反应比眼睛慢 |
   | 人脑中控制平衡性的部位是？                   | 小脑、大脑、脑干                                     |
   | 光年是天文学上的什么单位？                   | 长度单位、光速单位、时间单位                         |

   

3. 生活百科

   | 题目                                     | 选项                   |
   | ---------------------------------------- | ---------------------- |
   | 飞行最快的鸟是什么？                     | 雨燕、苍鹰、海鸥       |
   | 俗称的长生果是什么？                     | 花生、腰果、核桃       |
   | “天无三日晴、地无三尺平”值我国哪个省份？ | 贵州、广西、宁夏       |
   | “狼毫”原料取自哪里？                     | 黄鼠狼尾、野狼尾、马尾 |
   | 诺贝尔是哪国人？                         | 瑞典、瑞士、奥地利     |

   

   





## 目前的问答题库推进方案



### 1、确定评估模型性能的测试数据集，具体方案为：

- 首先根据语料的类别进行分类处理

  - 生活常识、自然科学、传统文化
  - 注意问题的分布要均衡

- 研究其他的可量化评估工具

  1. 指标-1：精准匹配率（Exact Match, EM）
     - 计算预测结果与标准答案是否完全一致
     - 一致=1分，不一致=0分
  2. 指标-2：模糊匹配率（F1-score）
     - 计算预测结果与标准答案之间的字级别（character-level）匹配程度

  目前该方法如何使用，还需要进一步研究。





### 2、目前模型算法的推进方案

- 简介目前使用的算法

  目前使用的算法基于bert-base的预训练模型，在CNMed数据集(中药数据集)上进行训练，然后再在CMRC2018数据集上进行fine-tune，目前这样的模型的性能最佳。之前使用A/B测试的方法对基础模型进行训练，实验的设计为：基于bert-base + CNMed + CMRC2018 训练出来的模型比bert-base + CMRC2018 的模型具有更好的泛化性能。可以得出的结论为：更多的高质量数据集对于提升模型的性能具有决定性作用。

- 后续可以采集的研究方向

  1. 更换底层的base-model，进行更多预训练模型的尝试：

     - 高规模调参的尝试
     - 中文预训练模型的尝试

  2. 收集或者制作更多的目前项目所需的训练数据集

     - 高质量数据集的作用效果显著

     - 针对后续的问题生成的定向数据集制作，比如先尝试制作一个小型数据集(自然科学方向)

       然后确定多大规模的数据集，可以显著继续提升模型性能，即：提高问题生成的质量

  3. 使用HuggingFace的Transformer库

     - 使用强大的可用工具进行探索
