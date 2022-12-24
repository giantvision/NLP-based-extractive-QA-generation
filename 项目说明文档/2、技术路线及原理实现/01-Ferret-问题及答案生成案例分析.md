# Ferret--Chrome插件：问题及答案生成案例分析

网站链接：

- [Ferret](https://samgorman.notion.site/samgorman/Ferret-c7508ec65df841859d1f84e518fcf21d)  --By Sam Gorman and Kanyes Thaker
- [Github--kanyesthaker/qgqa-flashcards](https://github.com/kanyesthaker/qgqa-flashcards)



Ferret 是一个 Chrome 扩展程序，它使用 NLP 生成有用的基于回忆的问题，以强化您在互联网上阅读的几乎所有内容的关键概念。

<img src="../imgs/Ferret_1.png" alt="Ferret_1" style="zoom:80%;" />



###  过去的解决方法

我们每天用来学习的产品通常并没有帮助我们尝试保留我们的知识。 Quizlet 等经典服务因帮助学生记忆考试词汇而出名。最专门的使用 Anki 等工具配置个人系统，以增加他们的记忆力。但这些方法忽略了一个基本事实：

> 增强记忆的方法如果是高摩擦性的，就不能获得广泛的采用。

最近在学习中增强记忆的方法已经显示出比 Quizlet 和 Anki 等更经典方法的进步。研究人员 Andy Matuschak 和 Michael Nielsen 在 2019 年介绍了 Mmnemonic 媒介，并在 Quantum Country 中展示了它的潜力，这是一篇融合了技术内容和内联间隔重复抽认卡的实验性文章。 Orbit 等开源方法为作者提供了工具来实现他们自己的内联间隔重复抽认卡，例如在 Quantum Country 中看到的那些。



### <span style='color:brown'>**本方案功能设计**</span>

我们在设计过程中采用了几个核心原则来执行我们的信念，即只有极低摩擦的工具才能有效地增强记忆力。

1. 产品需要满足用户所处的位置

   只有浏览器扩展才能满足用户的需求，以减少使用摩擦。我们知道，要求用户将他们正在阅读的内容复制粘贴到 Web 或桌面应用程序中是注定要失败的。

2. 一键获取价值

   无需配置、手动输入或入职。只需打开扩展程序，抽认卡就会神奇地生成，就您正在阅读的内容提出深思熟虑的问题。

3. 不显眼

   从用户反馈中，我们很快学会了减小扩展的轮廓，尤其是高度。这可以防止扩展程序妨碍用户阅读的文本。

4. 建立友好而非自命不凡

   思维空间工具中的许多工具都受到严重的品牌影响，从而缩小了它们的可用性。所以，我们给自己取了一个滑稽的动物名字（雪貂），并设计了一个与之相配的俏皮标志。虽然 Sans Serif 字体肯定有严肃的表情，但我们决定做出一个颇有争议的决定，将其与更有趣的视觉方向配对。

5. 显然是人工智能产品

   这需要明确是一个人工智能产品，以便用户在遇到不好的问题或等待几秒钟以加载模型时相应地设定期望。我们通过“模型加载”屏幕和用户报告错误问题并帮助训练 Ferret 提出更好问题的能力来实现这一点。 Google PAIR 设计指南在这里提供了帮助。



### **模型架构**

我们在 Ferret 中执行两项 NLP 任务：1. 问题生成、2. 问答。

#### 1、问题生成

问题生成是NLP中一个未得到服务的研究领域，具有开放性问题。 Du等人（2017年）和Chan等人（2019年）过去的工作证明了手工制作的Seq2Seq模型的功效，但这两个实现都因为是答案感知的而受到影响。 换句话说，他们训练他们的模型，通过向他们的模型输入上下文、答案、问题图元来生成问题。 然而，这在像Ferret这样的实际应用中并不实用，该应用在不知道答案是什么的情况下对上下文块进行推理以生成问题。 Lopez等人（2020年）证明了在单一的大型语言模型上基于变换器的微调可能比过去的方法的手工架构更出色。此外，作者发现，当这些方法是答案盲目的，而不是答案意识的时候，它们实际上达到了更高的水平。 对文献的回顾有助于指导我们跟随Lopez等人，寻找基于转化器的方法。

我们根据我们的需要调整了开源的question_generation库的端到端QG管道，并将该库重构为<100行的相关代码。 在这里，我们使用了在Squad 2.0上训练的预训练T5-Base模型，并在HuggingFace上托管。

#### 2、问答

问答是 NLP 中的一项常见任务，我们探索了许多现成的基线。在评估了 DistillBERT 和 BERT-tiny 之后，我们选择了托管在 HuggingFace 上的预训练 RoBERTa-Base-SQuAD2 模型。



### **减少 CPU 推理的延迟**

我们在开发早期决定仅在 CPU 上执行推理。与 CPU 相比，GPU 提供了明显而显着的加速，但也有几个缺点。这些包括不友好的生产成本和处理单一投入而不是批次的低效率。在实践中，这意味着我们需要积极优化 CPU 推理的延迟。该项目的其余部分将是在减少延迟和提高模型准确性之间进行一系列权衡。



### **提高问题生成/问答的准确性**

在 QG 中确定合适的评估指标比其他 NLP 子领域更抽象。是的，我们可以使用 BLEU 或 ROUGE 分数来评估自然生成的句子的效果，但与人类定性评估相比，这些指标在实践中对我们来说还不够。我们在 Colab 上编写了实用程序函数，后来又编写了一个 shell 脚本，用于解析来自 URL 的文本并显示生成的问题供我们查看。我们花了几个小时一起观察几百个问题，并从两个方向迭代提高性能：

1. 实施更好的数据预处理；
2. 在生成问题后引入过滤；



### **部署到生产**

我们将模型代码部署到 AWS Sagemaker，端点位于 API Gateway 上。如果时间允许，我们将添加更多关于部署过程的描述。



### **Building the Chrome Extension**

我们很早就决定在 React.JS 中构建 Chrome 扩展。 Chrome 扩展的 UI 通常要简单得多，并且需要一个 popup.html 文件才能运行。但是，我们的应用程序需要进行普通 JS/HTML/CSS 应用程序无法提供的广泛的状态管理和条件渲染。因此，通过修改 create-react-app 的 manifest.json 并修改顶层 package.json 中的构建脚本，我们能够从 popup.html 入口点提供 React 应用程序。

我们使用以下架构构建了扩展，其中 popup.html 是完整的 React 应用程序，服务工作者是包含纯 JS 中的事件侦听器的后台脚本，内容脚本是服务工作者调用的自包含函数，可以在目标网站及其 DOM。

<img src="../imgs/Ferret_2.png" alt="Ferret_2" style="zoom: 33%;" />

在高层次上，background.js中的服务工作者会监听目标事件，如用户选择一个新的标签，在同一页面中访问一个新的URL，或Chrome本地存储的某些变量的变化。 这些事件监听器会重置存储在Chrome本地存储中的值，然后调用在目标页面的DOM树中运行的内容脚本。 这些内容脚本执行有用的功能，如解析和预处理文本，以便输入到我们的推理端点，并突出显示与所提问题有关的文本块。该扩展的大部分内容是用ES7风格的异步Javascript编写的，以利用异步Chrome的API并处理各种请求逻辑。



### **减少和屏蔽客户端延迟**

客户端面临的一个关键挑战是减少和掩盖每次推理的平均延迟。我们开始时每次推理的平均延迟为 10 秒，基本上使该扩展对除了最专用的以外的所有人都无法使用。

首先，我们通过在客户端而不是在 AWS 端点上解析和预处理文本块来消除 2 秒的延迟。这是在内容脚本 getAllChunks() 中编写的一些帮助函数的示例，用于访问和预处理目标网站的 <p> 标记。现在，我们平均在 40 毫秒内过滤和预处理我们的文本，而不是 2 秒。好的！

```java
function getAllChunks() {
const divs = [...document.querySelectorAll("p")];
//Helper function to process a selected chunk
  function preprocess_chunk(text_in_div) {
    //Replace all with {} [] <>
    var pat = /(\{.*?\})|(\[.*?\])|(<.*?>)/g;
    var ret = text_in_div.replaceAll(pat, "");
    return ret;
  }

  //Helper function to determine if should discard a chunk
  function shouldBeIncluded(text_in_div) {
    //If contains any strings w more than one word that start w/ a capital letter and end with punctuation, keep it
    var pat = /^[A-Z].+ .+[\.|!|?]/;
    var bool = pat.test(text_in_div);
    return bool;
  }

var arr_of_divs = [];
  for (var i = 0; i < divs.length - 1; ++i) {
    var text_in_div = divs[i].innerText;
    if (shouldBeIncluded(text_in_div)) {
      var processed_chunk = preprocess_chunk(text_in_div);
      arr_of_divs.push(processed_chunk);
    }
  }
```



### **决策：屏蔽客户端延迟而不是减少服务器延迟**

我们走到了一个岔路口，在AWS上大幅减少延迟需要大量的技术投资或损失问题的质量。 将模型转换为量化的ONNX图将提供1.7倍的CPU速度，但在AWS Sagemaker上部署定制的ONNX图用于生产推理的文档很少。  我们试验了几个较小的预训练的T5模型，但在问题的质量上有很大损失。 我们计划在不久的将来再回来实施这些方法，但在当时，这些方法并不值得为发布一个可用的alpha版本而花费几天时间。

因此，我们决定在客户端**掩盖延迟**，在这里，我们将批量请求并行化，预装问题-答案对象，并使用用户体验技巧，将感知到的延迟从每次推理的8秒减少到 "加载模型 "时启动的单个8秒延迟。

**并行化批处理请求**

在向推理端点发出任何POST请求时，我们都会对接下来的4个问题-答案对象进行并发请求。 这是用Axios的Promise.all()实现的。

**预加载问答对象**

在每个 onClick 事件（代表用户单击下一张抽认卡）上，我们对接下来的 4 个问答对象进行另一轮并发 POST 请求。在实践中，这意味着我们维护的可渲染当前对象队列应该至少有 4-8 个问题供用户点击，直到没有更多的文本块要 POST 到我们的推理端点。

**首次加载时的用户体验技巧**

我们认为在用户打开应用程序之前预加载第一批问题会浪费资源。这意味着在用户访问的每个新选项卡或 URL 上请求我们的推理端点。相反，我们决定设计和构建一个模型加载屏幕来掩盖第一个请求的延迟。这有一个额外的好处，即为用户提供额外的上下文，即这是一个人工智能驱动的产品，并相应地管理期望。这是一个将技术限制转化为优势的机会：研究表明，短暂的等待通常会让用户认为内容更加个性化和先进。事实上，旅游公司和银行过去曾利用这一原则引入人工等待时间，以便让用户了解所显示内容的价值。无论如何，用户对新的加载屏幕反应良好，并称赞它如何“加速”应用程序。

实现一个进度条，让用户清楚这是一个人工智能产品，第一次使用时需要多花几秒钟来加载模型。



### <span style='color:brown'>**Future Directions**</span>

**产品设计：**

- 空间重复
- 在上下文中突出显示答案

**NLP：**

- 通过向AWS Sagemaker部署量化的ONNX模型图来减少延迟；
- 通过预训练我们自己的模型并提炼这个模型来减少延迟；
- 一旦我们收到至少 10,000 个不良问题示例，训练一个分类器，将生成的问题分类为有用或无用；

**Engineering:** 

通过重写代码以在使用 ONNX.JS 的浏览器中运行推理并利用浏览器 WebGL GPU 来减少延迟。



### <span style='color:brown'>**Contributions:**</span>

我们都为 Python 中的问题生成构建了自定义推理管道，并设计了 T5 + RoBERTa 模型架构。

- Kanyes 将所有模型代码部署到 AWS Sagemaker 并构建了完整的生产环境。
- Sam 构思和设计了产品，并在 React.js + vanilla JS 中构建了 Chrome 扩展。



### **References**

1. [为非常好奇的人准备的量子计算](https://quantum.country/qcvc) 
2. [T5-Base-E2E-QG](https://huggingface.co/valhalla/t5-base-e2e-qg)
3. [RoBERTa-Base-SQuAD2](https://huggingface.co/deepset/roberta-base-squad2)



