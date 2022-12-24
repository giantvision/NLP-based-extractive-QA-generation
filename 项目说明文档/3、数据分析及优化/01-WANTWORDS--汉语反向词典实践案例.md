# WANTWORDS--汉语反向词典实践案例

**目标：**

- 作为KMcha的替代方案，主要解决目前KMcha方法在内网无法使用的问题，作为一种离线的近义词解决方案；

**实现步骤：**

1. - [x] 先在外网验证该方法的有效性，即：可用
2. - [ ] 迁移到内网进行部署，然后验证其有效性；
3. - [ ] 实现错误答案在题库工具上的正式功能，需要豆豉的前端技术支持；

**官网代码及Demo地址：**

- [Github--WANTWORDS](https://github.com/thunlp/WantWords)
- [Demo](https://wantwords.net/)
- [model-download](https://cloud.tsinghua.edu.cn/d/811dcb428ed24480bc60/)

<span style='color:brown'>**测试功能的实践案例：**</span>

- [Colab-notebook](https://colab.research.google.com/drive/1Lqovet7ZaLUop2WnID9vcTddIe7oj9jm?usp=sharing)









## 遇到的问题汇总：

### 1、torch.load(...),   no module  named 'models'

代码分析：

```python
# WantWords/website_RD/views.py
MODEL_FILE = BASE_DIR + 'Zh.model'
model = torch.load(MODEL_FILE, map_location=lambda storage, loc:storage)
model.eval()
```

解决方案：

- 参考链接：[Web-Links](https://github.com/pytorch/pytorch/issues/18325#issuecomment-661759359)

  torch.load()  requires model module in the same folder.



### 2、[AttributeError: module 'time' has no attribute 'clock' in Python 3.8](https://stackoverflow.com/questions/58569361/attributeerror-module-time-has-no-attribute-clock-in-python-3-8)

代码分析：

```python
# WantWords/website_RD/views.py
with torch.no_grad():
    def_words = [w for w, p in lac.cut(description)]
```

**原因分析：**

- From the [Python 3.8 doc](https://docs.python.org/3/whatsnew/3.8.html):

> The function `time.clock()` has been removed, after having been deprecated since Python 3.3: use `time.perf_counter()` or `time.process_time()` instead, depending on your requirements, to have well-defined behavior.

**解决方案：**

- 发布这个丑陋的猴子补丁

  ```python
  import time
  time.clock = time.time
  ```

  

