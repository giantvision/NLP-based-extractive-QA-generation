# NLP方向的资料收集

数据来源：爱可可爱生活

作者：俊斌--93845

## 大公司的研究项目

1. Textless NLP：从原始音频生成表达性语音

   - https://ai.facebook.com/blog/textless-nlp-generating-expressive-speech-from-raw-audio/

   - FACEBOOK AI

     https://ai.facebook.com/blog/textless-nlp-generating-expressive-speech-from-raw-audio/
   
2. DeepMind--Improving Language Models by Retrieving from Trillions of Tokens

   - 论文：https://arxiv.org/abs/2112.04426

   - 推特简介：https://twitter.com/JayAlammar/status/1474838408656732160?s=20
   - https://deepmind.com/research/publications/2021/improving-language-models-by-retrieving-from-trillions-of-tokens
   - Github: https://github.com/google-research/google-research/tree/master/scann



## Github资源汇总

1. 基于transformers的自然语言处理(NLP)入门
   
   - https://github.com/datawhalechina/Learn-NLP-with-Transformers
   
2. Questgen.ai

   - https://github.com/ramsrigouthamg/Questgen.ai

     Questgen AI是一个开源的NLP库，致力于开发易于使用的问题生成算法。它正在寻求建立世界上最先进的问题生成人工智能，利用最先进的转化器模型，如T5、BERT和OpenAI GPT-2等。

3. indrajithi/genquest

   - https://github.com/indrajithi/genquest

     这个程序将一个文本文件作为输入，通过分析每个句子来生成问题。





## Reference

[Alphabetical list of part-of-speech tags used in the Penn Treebank Project](http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

[Automatic Factual Question Generation from Text](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.208.5602&rep=rep1&type=pdf)

[TextBlob: Simplified Text Processing](http://textblob.readthedocs.io/en/dev/index.html)

[Automatic Question Generation from Paragraph](http://www.ijaerd.com/papers/finished_papers/Automatic Question Generation from Paragraph-IJAERDV03I1213514.pdf)

[K2Q: Generating Natural Language Questions from Keywords with User Refinements](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37566.pdf)

[Infusing NLU into Automatic Question Generation](http://www.aclweb.org/anthology/W16-6609)

[Literature Review of Automatic Question Generation Systems](https://pdfs.semanticscholar.org/fee0/1067ea9ce9ac1d85d3fd84c3b7f363a3826b.pdf)

[Neural Question Generation from Text: A Preliminary Study](https://arxiv.org/pdf/1704.01792.pdf)

[Learning to Ask: Neural Question Generation for Reading Comprehension Apr 2017](https://arxiv.org/pdf/1705.00106.pdf)

[SQuAD: The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/)





## 相关论文及开源代码

资料来源：微信公众号--AINLPer

NLP From Scratch Without Large-Scale Pretraining: A Simple and Efficient Framework

- https://arxiv.org/abs/2111.04130

  由于强大的性能，预训练的语言模型已经成为许多NLP任务的标准方法，但它们的训练成本非常高。我们提出了一个简单而高效的学习框架，即TLM，它不依赖于大规模的预训练。给定一些标记的任务数据和一个大型的通用语料库，TLM使用任务数据作为查询，以检索通用语料库的一个极小的子集，并联合优化任务目标和语言建模目标，从头开始。在四个领域的八个分类数据集上，TLM取得了优于或类似于预训练语言模型（如RoBERTa-Large）的结果，同时将训练FLOPs减少了两个数量级。凭借高精确度和高效率，我们希望TLM能够为NLP的民主化和加速其发展作出贡献。



### Question Answering(QA)论文整理(二)

1、**TILE: Compositional De-Attention Networks** 

- **Paper:**  http://papers.nips.cc/paper/8845-compositional-de-attention-networks.pdf
- **Code:**    https://github.com/vanzytay/NeurIPS2019_CODA 

- 论文简述：

   注意力模型的显著特征是它们可以学习相对重要性的能力，即为输入值分配不同的权重。本文提出了一种新的具有复合性质的准注意方法，即在学习表示时，学习某一向量是加，是减还是零。这与普通的注意力形成了强烈的对比，传统的注意力方式只是简单地重新调整输入标记的权重。在六个NLP任务上评估CoDA，即开放域问题回答、检索/排序、自然语言推理、[机器翻译](https://cloud.tencent.com/product/tmt?from=10680)、情感分析和text2code生成，得到了比较好的结果。



2、**TILE: Language Models as Knowledge Bases?**

- **Paper:** https://arxiv.org/pdf/1909.01066v2.pdf

  **Code:** https://github.com/facebookresearch/LAMA

- 论文简述：

  本文深入分析了在一系列最先进的预训练语言模型中已经存在（没有微调）的关系知识。我们发现：（1）在没有微调的情况下，BERT包含了与传统NLP方法相竞争的关系知识，后者可以访问oracle知识；（2）BERT在有监督基线的开放域问题回答上也做得非常好，（3）通过标准语言模型的预训练方法，某些类型的事实知识比其他类型的知识更容易学习。这些模型在不进行任何微调的情况下调用事实知识的能力惊人地强，这表明它们作为无监督的开放域QA系统的潜力。

  

3、**TILE: SpanBERT: Improving Pre-training by Representing and Predicting Spans**

- **Paper:** https://arxiv.org/pdf/1907.10529v3.pdf

  **Code:** https://github.com/facebookresearch/SpanBERT

- 论文简述

  本文提出了SpanBERT预训练方法，旨在更好地表示和预测文本的跨度。本文方法通过(1)屏蔽连续的随机跨度，而不是随机Tokens来扩展BERT;(2)训练跨度边界表示来预测屏蔽跨度的全部内容，而不依赖于其中的单个Token表示。实验结果显示SpanBERT始终比BERT和我们更好调优的基线表现更好，在回答问题和共同参考解决等跨度选择任务上取得了巨大的进步。



4、**TILE:  XQA: A Cross-lingual Open-domain Question Answering Dataset**

- **Paper:**https://www.aclweb.org/anthology/P19-1227.pdf

  **Code:** https://github.com/thunlp/XQA

- 论文简述

  本文构建了一个用于跨语言OpenQA研究的新数据集XQA。它包括英语培训集以及其他八种语言的开发和测试集。此外，还为跨语言OpenQA提供了多个基线系统，包括两个基于机器翻译的方法和一个零距离跨语言方法(多语言BERT)。



5、**TILE:  Multi-Hop Paragraph Retrieval for Open-Domain Question Answering**

- **Paper:** https://arxiv.org/pdf/1906.06606v1.pdf

  **Code:** https://github.com/yairf11/MUPPET

- 论文简述

  本文研究的是多跳开放域问答系统。本文提出了一种检索多个支持段落的方法，这些段落嵌套在一个庞大的知识库中，包含了回答给定问题所必需的证据。我们的方法通过形成问题和段落的联合向量表示来迭代检索支持段落。检索是通过考虑知识源中段落的上下文化句子级表示来实现的。本文方法在数据集SQuAD Open和HotpotQA上实现了最好的性能，这两个数据集分别作为我们的单跳和多跳开放域QA基准。



6、**TILE: Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index**

- **Paper:** https://arxiv.org/pdf/1906.05807v2.pdf

  **Code:** https://github.com/uwnlp/denspi

- 论文简述

   现有的开放问答(OpenDomain QA)模型不适合实时使用，因为它们需要为每个输入请求按需处理多个长文档。在本文中，提出了一种与文档短语的查询无关的可索引表示，它可以极大地提高开放QA的速度。除此之外，本文密集-稀疏短语编码有效地捕获短语的语法、语义和词汇信息，消除了上下文文档的管道过滤。

7、**TILE:DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding**

- **Paper:** https://arxiv.org/pdf/2002.12591v1.pdf

  **Code:**  None

- 论文简述

  针对开放域问题回答的研究，使用预先训练的语言模型(如BERT)实现了显著的性能改进。最先进的方法通常遵循“检索和读取”管道，并使用基于BERT的reranker来过滤检索到的文档，然后再将它们提供给阅读器模块。BERT检索器将问题的串联和检索到的每个文档作为输入。尽管这些方法在QA准确性方面取得了成功，但是由于连接的原因，每个问题都包含大量检索到的文档，它们几乎不能处理大量输入的问题。为了解决效率问题，本文提出了一个解耦的上下文编码框架DC-BERT，它具有双重BERT模型:一个在线的BERT只对问题进行一次编码，一个离线的BERT对所有文档进行预编码并缓存它们的编码。



8、**TILE: REALM: Retrieval-Augmented Language Model Pre-Training**

- **Paper:** https://arxiv.org/pdf/2002.08909v1.pdf

  **Code:**  None

- 论文简述

  语言模型的预训练已经被证明能够获取大量的知识，这对于NLP任务(如回答问题)是至关重要的。然而，这些知识隐含在神经网络的参数中，需要更大的网络来覆盖更多的事实。为了以更模块化和可解释性的方式捕获知识，我们在语言模型预训练中增加了一个潜在的知识检索器，该检索器允许模型从一个大型语料库(如Wikipedia)中检索和处理文档，用于预训练、微调和推理。我们展示了如何以一种无监督的方式预先训练这样一个知识检索器，使用掩蔽语言建模作为学习信号，并通过一个考虑数百万文档的检索步骤进行反向传播。通过对具有挑战性的开放式问题回答(Open-domain Question answer, Open-QA)任务进行微调，我们证明了检索增强语言模型预训练(REALM)的有效性。



### Question Answering(QA)论文整理(五)

引言：

 本次整理的关于QA的八篇paper，主要涉及到**增强Ranker-Reader**、**SearchQA的大型数据集**、**PullNet集成框架**、**改进的加权抽样训练策略**、**开放QA中的Bert模型优化**等。（五篇含源码）

1、**TILE: Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering** 

- **Paper:** https://arxiv.org/pdf/1711.05116v2.pdf 

  **Code:** https://github.com/shuohangwang/mprc 

- 论文简述

  在这篇论文中，提出了两个利用多篇文章来产生答案的模型。两者都使用了一种答案重新排序的方法，该方法重新排序由现有的最先进的QA模型生成候选答案。本文提出了两种方法，即基于强度的重新排序和基于覆盖的重新排序，以利用来自不同文章的汇总证据来更好地确定答案。本文模型在三个公开的开放域QA数据集:Quasar-T、SearchQA和TriviaQA的开放域版本上取得了最先进的结果。



2、**Reinforced Reader-Ranker for Open-Domain Question Answering**

- **Paper:** https://arxiv.org/pdf/1709.00023v2.pdf

  **Code:** https://github.com/shuohangwang/mprc

- 论文简述

  本文提出了一种基于两种算法创新的新型开放域质量保证系统——增强Ranker-Reader。文中首先提出了一个带有排名组件的开放域QA新管道，该组件根据生成给定问题的基本真实答案的可能性对检索到的文章进行排名。其次，提出了一种基于强化学习的排序器与答案生成阅读者模型联合训练的新方法。实验结果，本文方法显著地改善了多个开放域QA数据集的现状。

  

3、**TILE: SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine**

- **Paper:** https://arxiv.org/pdf/1704.05179v3.pdf

  **Code:** https://github.com/nyu-dl/SearchQA

- 论文简述

  本文公开发布了一个名为SearchQA的大型数据集，用于机器理解或问答。它由超过140k个问题-答案对组成，每个对平均有49.6个片段。SearchQA的每个问答上下文元组都带有额外的元数据。我们在SearchQA上进行人工评估，并测试了两种基本方法，一种是简单的单词选择，另一种是基于深度学习的。

  

4、**TILE: Reading Wikipedia to Answer Open-Domain Questions**

- **Paper:** https://arxiv.org/pdf/1704.00051v2.pdf

- **Code**: https://github.com/facebookresearch/ParlAI

- 论文简述

  本文提出利用维基百科作为唯一的知识来源来解决开放领域的问题:任何事实性问题的答案都是维基百科文章的一个文本范围。大规模机器阅读的任务结合了文档检索(查找相关文章)和机器理解文本(从这些文章中识别答案)的挑战。我们的方法结合了一个基于二元哈希和TF-IDF匹配的搜索组件和一个多层递归神经网络模型，该模型训练用来检测维基百科段落中的答案。

  

**5、TILE: PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text**

**Author:** Haitian Sun , Tania Bedrax-Weiss , William Cohen

**Paper:** https://www.aclweb.org/anthology/D19-1242.pdf

**Code:**  None

**论文简述：** 本文PullNet是一个集成的框架，用于(1)学习检索以及(2)利用异构信息进行推理以找到最佳答案。PullNet使用一个{迭代}过程来构造一个包含与问题相关信息的特定于问题的子图。在每个迭代中，使用一个graph convolutional network (graph CNN)来识别子图节点，这些子图节点通过对语料库和/或知识库进行检索操作来展开。子图完成后，使用另一个图CNN从子图中提取答案。这个检索和推理过程允许我们使用大型KBs和语料库回答多跳问题。



**6、TILE: Ranking and Sampling in Open-Domain Question Answering**

**Author:** Yanfu Xu , Zheng Lin , Yuanxin Liu , Rui Liu , Weiping Wang , Dan Meng

**Paper:** https://www.aclweb.org/anthology/D19-1245.pdf

**Code:**  None

**论文简述：** 在本文首先介绍了一个利用分段-问题和分段-段落相关性来计算每个段落的置信度的排序模型。在此基础上，我们设计了一种改进的加权抽样训练策略，以减少噪声和干扰段落的影响。在三个公共数据集(Quasar-T、SearchQA和TriviaQA)上进行的实验表明了本文模型的优势。



**7、TILE: Language Models as Knowledge Bases?**

**Author:** Fabio Petroni , Tim Rocktschel , Sebastian Riedel , Patrick Lewis , Anton Bakhtin

**Paper:** https://www.aclweb.org/anthology/D19-1250.pdf

**Code:** https://github.com/facebookresearch/LAMA

**论文简述：** 本文深入分析了在一系列最先进的预训练语言模型中已经存在（没有微调）的关系知识。我们发现：（1）在没有微调的情况下，BERT相比于传统的NLP方法包含了相关知识，但是传统NLP方法可以访问知识库；（2）BERT在基于监督基线的开放域问题回答方面也做得非常好，（iii）通过标准语言模型的预训练方法，某些类型的事实知识比其他类型的知识更容易学习。这些模型在不进行任何微调的情况下调用事实知识的能力表现出惊人地优势，这表明它们作为无监督的开放域QA系统的潜力。



**8、TILE: Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering**

**Author:** Zhiguo Wang , Patrick Ng , Xiaofei Ma , Ramesh Nallapati , Bing Xiang

**Paper:** https://www.aclweb.org/anthology/D19-1599.pdf

**Code:**  None

**论文简述：** BERT模型已成功地应用于开放域QA任务。然而，以往的工作是通过观察与独立训练实例相同的问题对应的段落来训练BERT，这可能会导致不同段落的答案得分存在不可比性。为了解决这个问题，本文提出了一个多通道的BERT模型来对同一问题的所有段落的答案得分进行全局标准化，这种变化使得我们的QA模型能够通过使用更多的段落找到更好的答案。此外，我们还发现，通过滑动窗口将文章拆分成100字的段落，可以将性能提高4%。通过利用一个通道ranker来选择高质量的通道，多通道BERT获得额外的2%提高。



### Question Answering(QA)论文整理(六)

引言：

本次整理的论文还是主要偏向于机器阅读理解的问答（MRC-QA），其中主要涉及到**双向注意流(BIDAF)网络**、**Gated Attention模型**、**AS Reader模型**、**问答句子识别**、**双向注意机制和层次表示学习的关系图神经网络**、**类人问答系统建立**等。

**1、TILE: Bidirectional Attention Flow for Machine Comprehension** **Author:** Minjoon Seo • Aniruddha Kembhavi • Ali Farhadi • Hannaneh Hajishirzi

**Paper:** https://arxiv.org/pdf/1611.01603v6.pdf

**Code:** https://github.com/allenai/bi-att-flow

**论文简述：**机器理解(Machine comprehension, MC)主要用于回答关于给定上下文段落的查询，它需要对上下文和查询之间的复杂交互进行建模。最近，注意力机制已经成功地扩展到MC。通常，这些方法将注意力集中在上下文的一小部分，并以固定大小的向量、时间上的耦合注意力和/或通常形成单向注意力来进行归纳。本文介绍了双向注意流(BIDAF)网络，这是一个多阶段的分层过程，代表了不同粒度级别的上下文，并使用双向注意流机制来获得一个不需要早期摘要的查询感知上下文表示。



**2、TILE: Gated-Attention Readers for Text Comprehension**

**Author:** Bhuwan Dhingra • Hanxiao Liu • Zhilin Yang • William W. Cohen • Ruslan Salakhutdinov

**Paper:** https://arxiv.org/pdf/1606.01549v3.pdf

**Code:** https://github.com/bdhingra/ga-reader

**论文简述：**本文研究了文档的封闭性问答。本文模型Gated Attention (GA)阅读器，集成了一个多跳架构和一个新的注意机制，该机制基于查询嵌入和递归神经网络文档阅读器中间状态之间的乘法交互。这使阅读器能够在文档中构建特定于查询的令牌表示，以便准确地选择答案。



**3、TILE: Text Understanding with the Attention Sum Reader Network**

**Author:** Rudolf Kadlec • Martin Schmid • Ondrej Bajgar • Jan Kleindienst

**Paper:** https://arxiv.org/pdf/1603.01547v2.pdf

**Code:** https://github.com/rkadlec/asreader

**论文简述：** 本文提出了一个新的、简单的模型，它使用注意力直接从上下文中选择答案，而不是使用文档中单词的混合表示来计算答案。这使得该模型特别适合回答问题，其中答案是来自文档的一个单词。



**4、TILE: Deep Learning for Answer Sentence Selection**

**Author:** Lei Yu • Karl Moritz Hermann • Phil Blunsom • Stephen Pulman

**Paper:** https://arxiv.org/pdf/1412.1632v1.pdf

**Code:** https://github.com/brmson/dataset-sts

**论文简述：** 选择答案句的任务是识别包含给定问题答案的句子。本文提出了一种新的分布式表示方法来解决这一问题，并通过语义编码来学习问题与答案的匹配。这与之前在这项任务上的工作形成对比，后者通常依赖于具有大量手工编制的语法和语义特征以及各种外部资源的分类器。本文的方法不需要任何特征工程，也不涉及专业的语言数据，使得这个模型很容易适用于广泛的领域和语言。



**5、TILE: Relational Graph Representation Learning for Open-Domain Question Answering**

**Author:** Salvatore Vivona • Kaveh Hassani

**Paper:** https://arxiv.org/pdf/1910.08249v1.pdf

**Code:**  None

**论文简述：** 本文介绍了一种具有双向注意机制和层次表示学习的关系图神经网络用于开放域问题的求解。本文模型可以通过联合学习和更新查询、知识图和文档表示来学习上下文表示。实验表明，该模型在WebQuestionsSP基准测试中达到了最新的水平。



**6、TILE: Natural Language Generation at Scale: A Case Study for Open Domain Question Answering**

**Author:** Aless Cervone • ra • Ch Khatri • ra • Rahul Goel • Behnam Hedayatnia  

**Paper:** https://www.aclweb.org/anthology/W19-8657.pdf

**Code:**  None

**论文简述：** 本文通过一个编码器-解码器框架，并使用真实用户和开放域QA会话代理之间交互的大型数据集来建模NLG。首先，研究了增加时隙类型数量对生成质量的影响，并使用逐渐增大的本体（多达369个时隙类型）对QA数据的不同分区进行了实验。其次，在开放域QA和面向任务的dialog之间进行了多任务学习实验，并在一个流行的NLG数据集上对本文模型进行了基准测试。此外，我们还尝试使用会话上下文作为附加输入来提高响应生成质量。实验证明了在更大本体的开放领域QA中学习统计NLG模型的可行性。



**7、TILE: Conversational AI : Open Domain Question Answering and Commonsense Reasoning**

**Author:** Kinjal Basu

**Paper:** https://arxiv.org/pdf/1909.08258v1.pdf

**Code:**  None

**论文简述：** 本文研究重点是建立一个能合理回答问题的类人问答系统。本文方法的显著特点是，它将使用自动的常识推理来真正“理解”对话，允许它像人一样交谈。人类在交谈中常常会做出许多假设。我们用常识推断出没有明确说明的事实。在问答系统中加入常识性知识只会使其更加健壮。



**8、TILE: FriendsQA: Open-Domain Question Answering on TV Show Transcripts**

**Author:** Zhengzhe Yang • Jinho D. Choi

**Paper:** https://www.aclweb.org/anthology/W19-5923.pdf

**Code:**  None

**论文简述：** 本文介绍了一个具有挑战性的问题回答数据集FriendsQA，该数据集包含1222个对话和10610个开放域问题，用于解决机器对日常会话的理解。每个对话都有多个说话者参与，每个对话都用关于对话上下文的几种类型的问题进行注释，并且答案在对话中以一定的跨度进行注释。为了保证良好的注释质量，进行了一系列众包任务，使得注释器间的一致性达到了81.82%。



### Question Answering(QA)论文整理(七)

引言：

本次文章主要介绍了**ERNIE-GEN**(语言生成任务)、统一预训练语言模型(**UniLM**)、问答系统数据集(**CoQA**)、端到端神经生成问答(**GENQA**)、生成式问答系统评估方法、自编码自回归语言模型(**PALM**)、答案生成器(**KEAG**)、生成式问答(**gQA**)。（四篇含源码）

**1、TILE: ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation** 

**Author:** Dongling Xiao • Han Zhang • Yukun Li  

**Paper:** https://arxiv.org/pdf/2001.11314v3.pdf 

**Code:** https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen 

**论文简述：** 当前自然语言生成中的预培训工作很少关注下游任务的暴露偏差问题。为了解决这个问题，我们提出了一种增强的多流序列，用于序列预训练和框架微调，名为ERNIE-GEN，它通过增加生成机制和噪声感知生成方法来填补训练和推理之间的差异。为了使生成更接近人类给出的结果，此框架引入了跨接生成了流程，该流程并不是逐字预测，而是连续预测语义上完整的跨距。与现有的预训练方法不同，ERNIE-GEN结合了多粒度目标采样来构造预训练数据，从而增强了编码器和解码器之间的相关性。实验结果表明，ERNIE-GEN在一系列语言生成任务（包括抽象摘要（Gigaword和CNN / DailyMail），问题生成（ SQuAD），对话生成（Persona-Chat）和生成性问答（CoQA）上都得到了较好的结果。



**2、TILE: Unified Language Model Pre-training for Natural Language Understanding and Generation**

**Author:** Li Dong • Nan Yang • Wenhui Wang  

**Paper:** http://papers.nips.cc/paper/9464-unified-language-model-pre-training-for-natural-language-understanding-and-generation.pdf

**Code:** https://github.com/microsoft/unilm

**论文简述：** 本文提出了一种新的统一预训练语言模型（UniLM），它可以针对自然语言理解和生成任务进行微调。使用三种类型的语言建模任务对模型进行预训练：单向，双向和序列到序列的预测。通过使用共享的Transformer网络并利用特定的自注意mask来控制预测条件所处的环境，可以实现统一的建模。UniLM在GLUE基准测试，SQuAD 2.0和CoQA问题解答任务方面与BERT相比具有优势。此外，UniLM在五个自然语言生成数据集上获得了最新的最新结果，包括将CNN / DailyMail抽象摘要ROUGE-L提升到40.51（绝对改进2.04），Gigaword抽象摘要ROUGE-L提升到35.75（0.86）。CoQA生成问题解答F1分数达到82.5（绝对改进37.1），SQuAD问题生成BLEU-4达到22.12（绝对改进3.75）以及DSTC7文档为基础的对话框响应生成NIST-4达到2.67（人类性能为2.65）。



**3、TILE: CoQA: A Conversational Question Answering Challenge**

**Author:** Siva Reddy • Danqi Chen • Christopher D. Manning

**Paper:** https://arxiv.org/pdf/1808.07042v2.pdf

**Code:** https://github.com/stanfordnlp/coqa-baselines

**论文简述：** 人类通过参与一系列的问答对话来收集信息。机器能够回答对话性问题对帮助其信息收集是至关重要的。这里我们介绍CoQA，这是一个用于构建会话问答系统的新型数据集。该数据集包含12.7万个带有答案的问题，这些问题是从关于7个不同领域的文本段落的8k次对话中获得的。问题是对话性的，答案是自由形式的文本，其相应的证据在段落中突出显示。我们深入分析了CoQA，发现会话问题具有挑战性的现象，这些现象在现有的阅读理解数据集中并不存在，例如，引用和语法推理。我们在CoQA上评估了强大的会话和阅读理解模型。最好的系统获得的F1分数为65.4％，比人类的表现低23.4点（88.8％），表明这些模型还有足够的改进空间。



**4、TILE: Neural Generative Question Answering**

**Author:** Jun Yin • Xin Jiang • Zhengdong Lu

**Paper:** https://arxiv.org/pdf/1512.01337v4.pdf

**Code:** https://github.com/jxfeb/Generative_QA

**论文简述：** 本文介绍了一种端到端神经网络模型，称为神经生成问答（GENQA），该模型可以基于知识库中的事实生成简单事实问题的答案。更具体地说，该模型建立在用于序列到序列学习的编码器-解码器框架上，同时具备查询知识库的能力，并在知识库中的问答对及其相关三元组的语料库上进行训练。实证研究表明，该模型能够有效地处理问题和答案的变化，并结合知识库中的事实生成正确、自然的答案。问题回答实验表明，该模型的性能优于基于嵌入的QA模型，也优于基于相同数据训练的神经对话模型。



**5、TILE: KPQA: A Metric for Generative Question Answering Using Word Weights**

**Author:** Hwanhee Lee • Seunghyun Yoon • Franck Dernoncourt  

**Paper:** https://arxiv.org/pdf/2005.00192v1.pdf

**Code:**  None

**论文简述：**  对于生成式问答系统（genQA）的自动评估，必须评估生成的答案的正确性。然而采用广泛用于比较生成的文本和参考的n-gram相似性度量标准进行事实评估容易产生误判，并且缺乏基准数据集来衡量度量标准的正确性。为了研究更好的genQA评价指标，我们在两个标准genQA数据集上收集了人类对正确性的高质量判断，使用我们的人类评估数据集，我们证明了基于n-gram相似性的现有指标与人类判断没有关联。为了缓解这个问题，我们提出了一种新的指标来评估genQA的正确性。具体而言，新的度量标准通过关键短语预测为每个令牌分配不同的权重，从而判断预测的答案句子是否捕获了人类判断者的真实含义。与广泛使用的现有指标相比，我们提出的指标显示出与人为判断的相关性明显更高。



**6、TILE: PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation**

**Author:** Bin Bi • Chenliang Li • Chen Wu

**Paper:** https://arxiv.org/pdf/2004.07159v1.pdf

**Code:**  None

**论文简述：** 自监督的预训练已经成为一种强大的自然语言理解和生成技术，如BERT、MASS和BART。现有的预训练技术将自编码或自回归作为目标，通过从被破坏的文本中恢复原始单词标记来训练基于transformer的模型。在这项工作中，我们提出了PALM，即在一个大的未标记语料库上预训练一个自编码和自回归语言模型，特别是针对基于上下文的下游生成，如问题生成和会话响应生成。PALM最大限度地减少了现有去噪方案在预训练和微调之间的不匹配，因为在微调过程中生成的文本不仅仅是重构原始文本。PALM采用了一种新颖的预训练方案，在各种语言生成基准测试中取得了最新的研究成果，包括生成性问题回答、Gigaword的摘要和康奈尔电影对话的会话反应生成。



**7、TILE: Incorporating External Knowledge into Machine Reading for Generative Question Answering**

**Author:** Bin Bi • Chen Wu • Ming Yan  

**Paper:** https://www.aclweb.org/anthology/D19-1255

**Code:** None

**论文简述：** QA模型需要常识和背景知识来回答许多重要问题。与现有的知识型QA工作不同，我们关注的是一项更具挑战性的任务，即利用外部知识，根据上下文为给定的问题以自然语言生成答案。在本文中，我们提出了一种新的神经模型，即知识丰富的答案生成器(KEAG)，它能够利用聚集来自所有四种信息源的证据来组成一个自然的答案:问题、文章、词汇和知识。在答案生成过程中，KEAG自适应地决定了什么时候使用符号知识以及知识中的哪些事实是有用的。这允许模型利用外部知识，这些知识在给定的文本中没有明确地陈述，但与生成答案相关。对回答生成公共基准的实验研究表明，相比无知识模型和现有的知识感知模型KEAG提高了回答质量，证实了KEAG在利用知识方面的有效性。



**8、TILE: A Generative Approach to Question Answering**

**Author:** Rajarshee Mitra

**Paper:** https://arxiv.org/pdf/1711.06238v2.pdf

**Code:**  None

**论文简述：** 从选择答案、句子、关联问答到阅读和理解，问答已经走过了漫长的道路。我们将注意力转移到生成式问答(gQA)上，通过学习生成答案来帮助机器阅读文章和回答问题。我们将问题构造为一个生成任务，其中编码器是对问题和段落之间的关系进行建模并将其编码为向量的网络，从而有助于解码器直接形成抽象答案。不能保留事实和重复是常见的错误，会影响答案的整体可读性。为了解决这些问题，我们分别在模型中采用复制和覆盖向量的维护机制。我们在MS-MARCO上的结果证明了它比基线的优越性，并且我们也展示了在正确性和可读性方面得到改进的定性示例。

## 2021-12-09-问题生成论文收集

**新网址**：https://arxiv-sanity-lite.com/

- 搜索启发：其关于搜索论文的推荐分数匹配度非常高，值得学习

  <img src="../images/imgs_nlp_research/论文推荐分数匹配.png" alt="论文推荐分数匹配" style="zoom: 50%;" />



1、Answering Open-Domain Questions of Varying Reasoning Steps from Text

- 从文本中回答不同推理步骤的开放领域问题
- 新数据集地址：https://beerqa.github.io/

摘要：

我们开发了一个统一的系统来直接从文本开放域问题中回答，这些问题可能需要不同数量的检索步骤。我们采用一个单一的多任务转换器模型以迭代方式执行所有必要的子任务——检索支持事实，重新排列它们，并从所有检索到的文档中预测答案。我们避免了以前工作中不能很好地转移到现实世界设置的关键假设，包括利用回答每个问题所需的固定数量检索步骤的知识或使用结构化元数据，如知识库或可用性有限的网络链接。相反，我们设计了一个系统，可以在没有推理复杂性的先验知识的情况下回答关于任何文本集合的开放域问题。为了模拟这种设置，我们构建了一个新的基准测试，称为 BeerQA，通过将现有的一步和两步数据集与需要三个维基百科页面回答的 530 个问题的新集合相结合，在此过程中统一维基百科语料库版本。我们展示了我们的模型在现有基准和这个新基准上都展示了具有竞争力的性能。



2、Towards Universal Dense Retrieval for Open-domain Question Answering

- 面向开放域问答的通用密集检索

摘要：

在开放领域的问题回答中，一个模型接收一个文本问题作为输入，并使用一个大型的证据语料库来搜索正确答案。检索步骤特别困难，因为典型的证据语料库有数以百万计的文件，其中每个文件可能有也可能没有问题的正确答案。最近，密集型模型已经取代了稀疏型方法，成为事实上的检索方法。密集型方法不是以词汇重叠来确定相似性，而是建立一个编码函数，通过从一小批问题-答案或问题-背景对中学习来捕捉语义相似性。在本文中，我们研究了在不同输入分布的开放域问题回答背景下的密集检索模型。为此，我们首先介绍了一个由Wikidata事实构建的实体丰富的问题回答数据集，并证明密集模型无法推广到未见过的输入问题分布。其次，我们进行分析，旨在更好地理解问题的来源，并提出新的训练技术，以改善各种数据集的域外性能。我们鼓励该领域进一步研究建立一个单一的、通用的密集检索模型，在所有的输入分布中都能很好地概括。



3、TopiOCQA: Open-domain Conversational Question Answeringwith Topic Switching

- TopiOCQA: 开放域的对话式问题回答与主题转换
- 网站地址：https://mcgill-nlp.github.io/topiocqa/

在对话式问题回答场景中，提问者通过一系列相互依存的问题和回答，寻求提取关于某个主题的信息。随着对话的进行，他们可能会切换到相关的话题，这是在信息搜索会话中经常观察到的现象。然而，目前用于对话式问题回答的数据集在两个方面具有局限性。1）它们不包含话题切换；2）它们假定对话的参考文本是给定的，即设置不是开放域的。我们介绍TobiOCQA（发音为Tapioca），这是一个带有维基百科上的话题转换的开放域对话数据集。TopiOCQA包含3920个带有信息搜索问题和自由格式答案的对话。TopiOCQA为模型提供了一个具有挑战性的测试平台，它需要在同一对话的多个回合中进行有效的检索，同时利用对话历史构建有效的回答。我们通过将最先进的文档检索方法与神经阅读器模型相结合，评估了几个基线。我们最好的模型实现了51.9的F1和42.1的BLEU得分，分别比人类的表现差了18.3分和17.6分，表明我们的数据集的难度。



**4、Asking Questions Like Educational Experts: Automatically Generating Question-Answer Pairs on Real-World Examination Data**

- 像教育专家一样提问：根据真实考试数据自动生成问答对

摘要：

生成高质量的问题-答案对是一项艰难但有意义的任务。虽然以前的工作在答案感知问题的生成方面取得了很大的成果，但很难将其应用到教育领域的实际应用中。本文首次针对现实世界考试数据中的题对生成任务，提出了一个新的统一的RACE框架。为了捕捉输入段落的重要信息，我们首先自动生成（而不是提取）关键词，因此这一任务被简化为关键词-问题-答案三联体的联合生成。因此，我们提出了一个多代理通信模型来生成并反复优化问题和关键词，然后应用生成的问题和关键词来指导答案的生成。为了建立一个坚实的基准，我们将我们的模型建立在强大的生成性预训练模型上。实验结果表明，我们的模型在问题-答案对生成任务中取得了巨大的突破。此外，我们对我们的模型进行了全面的分析，为这项具有挑战性的任务提出了新的方向。

算法介绍

考虑到数据集的偏见和非自然的语言来源，在教育领域采用这些技术是很困难的。此外，以前的工作大多将答案文本视为段落中的一个连续跨度，并通过提取方法直接获得答案，这可能不符合现实世界数据的需求。

<img src="../../论文集/论文笔记/images/imgs-2021-12-09/Figure-1.png" alt="Figure-1" style="zoom:50%;" />

在本文中，我们提出了一个新的架构来处理这个现实世界的QAG任务，如图1所示。它由三部分组成：粗略的关键词生成代理，问题-关键词迭代生成模块和答案生成代理。我们首先根据给定的文档生成关键词，然后用迭代生成模块优化生成的问题和关键词。此外，以生成的关键词和问题为指导，生成相应的答案。为了应对考试文本的复杂表达，我们的模型以生成式预训练模型ProphetNet（Qi等人，2020）为基础。我们在RACE上进行了实验，与基线模型相比，取得了令人满意的改进。

我们的贡献总结如下：

1）我们是第一个在RACE上执行QAG任务的人。所提出的方法可以很容易地应用于现实世界的考试，产生阅读理解数据。

2）我们提出了一个新的架构来做问答对的联合生成，它比基线模型获得了明显的性能增益。

3）我们对新任务和我们的模型进行了全面的分析，为未来的研究建立了一个坚实的基准。

在本文中，我们提出了一个基于生成性预训练模型的Q-A联合生成的新框架，详细的模型结构如图3所示。整个生成过程可以分成三个部分：

步骤1. 粗略的关键词生成：从文件中生成粗略的关键词，并将其反馈给问题生成过程；

第2步。迭代问题和关键词生成：用步骤1的初始输入关键词迭代优化问题和关键词；

第3步。答案生成：用输出的问题和关键词组生成答案。

![Figure-2 模型结构图](../../论文集/论文笔记/images/imgs-2021-12-09/Figure-2 模型结构图.png)

<center>Figure-2   Q-A对联合生成的详细模型结构</center>
值得强调的是，我们的训练框架与底层模型的选择无关，所以我们既可以选择正常的Seq2Seq模型，如LSTM（Hochreiter和Schmidhuber，1997）和Transformer（Vaswani等人，2017），也可以选择BART（Lewis等人，2020）等预训练模型来替代ProphetNet。

我们选择Prophet-Net来生成关键词有两个原因。首先，ProphetNet足够有效，因为它在一些自动生成的工作中被证明是完全胜任的。更重要的是，ProphetNet在我们的统一模型的所有三个阶段都被采用，这确保了我们框架的通用性和简单性。

为了进一步提高生成的关键词的质量，我们采取了一个两阶段的微调策略。首先，我们使用SQuAD作为第一阶段训练的数据增量。关键词生成模型将段落作为输入，并将与段落相对应的所有参考答案用空间分隔符连接起来作为训练目标。然后，我们在RACE数据集上对该模型进行了细致的微调。由于RACE的特点，我们把参考答案中的停顿词去掉，形成几个独立的关键答案短语，作为第二阶段的训练目标。我们将生成的结果表示为k1，这是一个由推理过程中的多个关键短语组成的字符串。

捕捉值得被提问和回答的重要内容，对我们的任务至关重要。针对Q-A的生成，我们可以使用关键短语或关键句子来代表一段话的重要内容。关键句子得益于完整的句法和语义信息，而关键词组则更加灵活，它们不会带来无用的信息来干扰生成过程。

结论：

本文针对QAG任务，提出了一个在教育数据集RACE上训练的统一框架。我们采用了一个三阶段的生成方法，包括一个粗略的关键词生成模型、一个迭代的消息传递模块和一个问题关键词引导的答案生成模型。我们的模型在QG任务上实现了与最先进的答案感知生成模型相近的性能，并且与基本的预训练模型相比，在生成对的可回答性上获得了很大的改善。在我们提出的QAG任务中，有很大的潜力可以进一步改进，以帮助人们在现实世界的应用中产生阅读理解的数据。



5、Discourse Comprehension: A Question Answering Framework to Represent Sentence Connections

- 话语理解：表示句子连接的问答框架

虽然通过简单的事实性问题的回答在文本理解方面已经有了实质性的进展，但对话语的更全面的理解仍然是一个重大挑战。有人在阅读文本时进行批判性反思，会提出好奇心驱动的、通常是开放式的问题，这些问题反映了对内容的深刻理解，需要复杂的推理来回答。为这种类型的话语理解建立和评估模型的一个关键挑战是缺乏注释数据，特别是由于寻找这种问题的答案（可能根本没有答案）需要注释者在长篇文献中承担高认知负荷。本文提出了一种新的范式，能够针对新闻文件的理解进行可扩展的数据收集，通过话语的角度来看待这些问题。由此产生的语料库，DCQA（通过问题回答的话语理解），由607个英语文档中的22430个问题-答案对组成。DCQA以自由形式的开放式问题的形式捕获了句子之间的话语和语义联系。在我们对来自INQUISITIVE数据集的问题进行注释的评估集上，我们表明DCQA为回答开放式问题提供了宝贵的监督。我们还设计了利用现有问题回答资源的预训练方法，并使用合成数据来适应无法回答的问题。



**6、Enhancing Question Generation with Commonsense Knowledge**

- 用常识知识增强问题生成

问题生成（QG）是为了生成自然的、符合语法的问题，这些问题可以在给定的环境下由特定的答案来回答。以前的序列到序列模型存在一个问题，即提出高质量的问题需要常识性知识作为背景，而这些常识性知识在大多数情况下不能直接从训练数据中学习，从而导致被剥夺了知识的问题不尽人意。在本文中，我们提出了一个多任务学习框架，将常识性知识引入问题生成过程。我们首先从成熟的数据库中检索出相关的常识性知识三元组，并选择具有从源语境到问题的转换信息的三元组。基于这些信息丰富的知识三元组，我们设计了两个辅助任务，将常识性知识引入主要的QG模型，其中一个任务是概念关系分类，另一个是尾部概念生成。在SQuAD上的实验结果表明，我们提出的方法能够明显改善QG在自动和人工评估指标上的表现，这表明将外部常识性知识与多任务学习结合起来能够帮助模型生成类似人类的高质量问题。



7、How Well Do You Know Your Audience? Reader-aware Question Generation

- 你有多了解你的听众？读者意识的问题生成

在写作时，一个人可能需要预测来自读者的问题，但不同类型的读者可能会提出非常不同的问题类型。如果有人在写作中征求对一个问题的建议，那么领域专家会问什么问题，这与新手可能的反应是否不同？在本文中，我们解决了读者意识到的问题生成的任务。我们从社交媒体上收集了一个新的问题和帖子的数据集，并对帖子读者的背景信息进行了扩充。基于预测性分析和描述性差异，我们发现不同的读者，如专家和新手，总是提出不同类型的问题。我们接下来开发了几个文本生成模型，其中包含不同类型的读者背景，包括基于读者先前行为的离散和连续的读者表示。我们证明，在某些情况下，读者意识模型的表现与纯文本模型相当或略胜一筹，特别是在一个帖子从不同群体的读者那里吸引到非常不同的问题的情况下。我们的工作有可能帮助作者预测不同读者的信息需求。



8、Improving Stack Overflow question title generation with copying enhanced CodeBERT model and bi-modal information

- 用复制增强的CodeBERT模型和双模式信息改进Stack Overflow问题的标题生成

背景。Stack Overflow对于那些寻求编程问题答案的软件开发者来说非常有帮助。以前的研究表明，越来越多的问题是低质量的，因此获得潜在回答者的关注较少。Gao等人提出了一个基于LSTM的模型（即BiLSTM-CC），从代码片段中自动生成问题标题以提高问题质量。然而，仅使用问题正文中的代码片段不能为标题的生成提供足够的信息，而且LSTM不能捕捉到标记之间的长距离依赖关系。目标。我们提出了CCBERT，一个基于深度学习的新型模型，通过充分利用整个问题体的双模信息来提高问题标题生成的性能。方法。CCBERT遵循编码器-解码器范式，使用CodeBERT将问题主体编码为隐藏表征，使用堆叠的Transformer解码器生成预测的标记，并使用额外的复制注意力层来完善输出分布。编码器和解码器都执行多头自我注意操作，以更好地捕捉长距离的依赖性。我们建立了一个数据集，其中包含了从Stack Overflow正式发布的数据中筛选出来的120,000多个高质量问题，以验证CCBERT模型的有效性。结果。CCBERT在数据集上取得了较好的性能，特别是比BiLSTM-CC和多用途预训练模型（BART）平均分别高出14%和4%。在纯代码和低资源数据集上的实验也显示了CCBERT的优越性，性能下降较少，BiLSTM-CC分别为40%和13.5%，而CCBERT为24%和5%。



**9、Simplifying Paragraph-level Question Generation via Transformer Language Models**

- 通过 Transformer 语言模型简化段落级问题的生成

问题生成（QG）是一项自然语言生成任务，其中一个模型被训练来提出与一些输入文本相对应的问题。最近的大多数方法将QG作为一个序列到序列的问题，并依靠额外的特征和机制来提高性能；然而，这些通常会增加模型的复杂性，并可能依赖于实际使用中不可用的辅助数据。一个基于Transformer的单向语言模型，利用迁移学习，可以用来产生高质量的问题，同时处理额外的特定任务的复杂性。我们的QG模型从GPT-2 Small中进行了微调，在SQuAD数据集上比几个段落级QG基线高出0.95METEOR点。人类评估员对问题的评价是：容易回答，与上下文段落相关，并与人类的自然语音很好地对应。此外，还介绍了RACE数据集上的一组新的基线分数，该数据集以前没有被用于QG任务。我们建议对不同的模型容量和非识别类型的问题数据集进行进一步的实验，以进一步验证基于变形器的预训练LM作为问题生成器的稳健性。



10、Zero-Shot Open Information Extraction using Question Generation and Reading Comprehension

- 使用问题生成和阅读理解的零样本开放信息提取

通常情况下，开放信息提取（OpenIE）侧重于提取三要素，代表一个主题、一个关系和关系的对象。然而，大多数现有的技术都是基于每个领域中预定义的关系集，这限制了它们对较新领域的适用性，因为这些关系可能是未知的，如金融文件。本文提出了一种零散的开放式信息提取技术，利用现成的机器阅读理解（MRC）模型，从一个句子中提取实体（值）和它们的描述（键）。这个模型的输入问题是用一种新颖的名词短语生成方法创建的。这种方法考虑到了句子的上下文，可以创造出各种各样的问题，使我们的技术具有领域独立性。鉴于问题和句子，我们的技术使用MRC模型来提取实体（价值）。与问题相对应的、具有最高置信度的名词短语被作为描述（关键）。

本文还介绍了EDGAR10-Q数据集，该数据集基于在美国证券交易委员会（SEC）上市的公司的公开财务文件。该数据集由段落、标签值（实体）和它们的键（描述）组成，是最大的实体提取数据集之一。这个数据集将是研究界的一个有价值的补充，特别是在金融领域。最后，本文在EDGAR10-Q和Ade语料库药物剂量数据集上展示了所提技术的功效，分别获得了86.84%和97%的准确性。



## 相关论文及开源代码



### 1、TILE:  Multi-Hop Paragraph Retrieval for Open-Domain Question Answering

- 用于开放域问题回答的多跳段落检索

- 2019

- **Paper:** https://arxiv.org/pdf/1906.06606v1.pdf

  **Code:** https://github.com/yairf11/MUPPET

- 论文简述

  本文研究的是多跳开放域问答系统。本文提出了一种检索多个支持段落的方法，这些段落嵌套在一个庞大的知识库中，包含了回答给定问题所必需的证据。我们的方法通过形成问题和段落的联合向量表示来迭代检索支持段落。检索是通过考虑知识源中段落的上下文化句子级表示来实现的。本文方法在数据集SQuAD Open和HotpotQA上实现了最好的性能，这两个数据集分别作为我们的单跳和多跳开放域QA基准。

<img src="../images/imgs_paper/Figure1-paper1.png" alt="Figure1-paper1" style="zoom:50%;" />

<center>图1：HotpotQA数据集中需要多跳推理和检索的问题及其答案语境的例子。第一个推理环节用绿色标示，第二个环节用紫色标示，连接这两个环节的实体用蓝色粗斜体标示。在第一个推理跳中，人们必须推断出有关的经理是Alex Ferguson。如果没有这个知识，就不可能有把握地检索出第二个上下文，因为这个问题可能指的是俱乐部历史上的任何一位经理。因此，需要进行反复检索，以正确检索这对语境。</center>
**数据集：**

- HotpotQA

  Yang等人（2018）介绍了一个基于维基百科的问题数据集，这些问题需要对多个段落进行推理以找到正确答案。该数据集还包括对句子级支持事实的硬监督，这鼓励了模型给出可解释的答案预测。这个数据集有两个基准设置。(1)分散注意力的设置，即给读者一个问题以及一组段落，其中包括支持性事实和不相关的段落；(2)完整的维基设置，这是数据集的一个开放域版本。我们用这个数据集作为我们的多跳检索设置的基准。必须对第3.2节中的读者进行一些扩展，以便使其适用于HotpotQA数据集。

- SQuAD-Open

  Chen等人（2017）将问题与原始SQuAD数据集（Rajpurkar等人，2016）中的相应语境解耦，并通过将整个维基百科转储定义为背景知识源，从中提取问题的答案，形成了一个开放域版本的数据集。我们使用这个数据集来测试我们的方法在经典的单跳检索环境中的有效性。

我们的解决方案，我们称之为MUPPET（多跳段落检索），依赖于以下基本方案，包括两个主要部分。(a) 段落和问题编码器，以及(b) 段落阅读器。编码器被训练为将段落编码为d维向量，并将问题编码为同一向量空间的搜索向量。然后，应用最大内积搜索（MIPS）算法来寻找与给定问题最相似的段落。存在几种快速（可能是近似）MIPS的算法，如Johnson等人（2017）提出的算法。然后，最相似的段落被传递给段落阅读者，反过来，阅读者会提取问题的最可能的答案。



 本次整理主要涉及到**增强Ranker-Reader**、**SearchQA的大型数据集**、**PullNet集成框架**、**改进的加权抽样训练策略**、**开放QA中的Bert模型优化**等。

2、TILE: Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering 

- 开放域问答中答案重新排序的证据聚合

- 2018

- **Paper:** https://arxiv.org/pdf/1711.05116v2.pdf 

  **Code:** https://github.com/shuohangwang/mprc 

- 论文简述

  在这篇论文中，提出了两个利用多篇文章来产生答案的模型。两者都使用了一种答案重新排序的方法，该方法重新排序由现有的最先进的QA模型生成候选答案。本文提出了两种方法，即基于强度的重新排序和基于覆盖的重新排序，以利用来自不同文章的汇总证据来更好地确定答案。本文模型在三个公开的开放域QA数据集:Quasar-T、SearchQA和TriviaQA的开放域版本上取得了最先进的结果。



<img src="../images/imgs_paper/Table-6_paper2.png" alt="Table-6_paper2" style="zoom: 50%;" />

<center>表6：Quasar-T数据集的一个例子。基础真理答案是 "芝麻街"。Q：问题，A：答案，P：包含相应答案的段落。</center>
结论：我们观察到，通过明确地结合来自多个检索段落的证据，可以改善开放域的质量保证。我们试验了两种类型的重排器，一种是针对证据一致的情况，另一种是针对证据互补的情况。这两种重排器都有助于单独显著地改善我们的结果，甚至在一起改善得更多。在三个开放领域的质量保证数据集上，我们的结果大大推进了最先进的水平。尽管我们提出的方法在对多个段落的联合或共同出现进行建模方面取得了一些成功，但在开放领域的质量保证方面仍有更难的问题需要推理和常识推理能力。在未来的工作中，我们将探索上述方向，我们相信我们提出的方法有可能被推广到这些更难的多段落推理场景。



3、TILE: SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine

- SearchQA：使用来自搜索引擎的上下文增强的新问答**数据集**

- 2017

- **Paper:** https://arxiv.org/pdf/1704.05179v3.pdf

  **Code:** https://github.com/nyu-dl/SearchQA

- 论文简述

  本文公开发布了一个名为SearchQA的大型数据集，用于机器理解或问答。它由超过140k个问题-答案对组成，每个对平均有49.6个片段。SearchQA的每个问答上下文元组都带有额外的元数据。我们在SearchQA上进行人工评估，并测试了两种基本方法，一种是简单的单词选择，另一种是基于深度学习的。

结论：我们为问题回答研究构建了一个新的数据集，称为SearchQA。它是用一个正在生产的商业搜索引擎建立的。它密切反映了一个（假设的）一般问题回答系统的全部管道，其中包括信息检索和答案合成。我们进行了人工评估和机器评估。使用最新的技术，即ASR，我们表明人类和机器之间存在着有意义的差距，这表明SearchQA作为问题回答研究的一个基准任务的潜力。我们公开发布了SearchQA，包括我们自己在PyTorch中对ASR和n-gram ASR的实现。

4、TILE: Reading Wikipedia to Answer Open-Domain Questions

- 阅读维基百科以回答开放性领域的问题

- 2017

- **Paper:** https://arxiv.org/pdf/1704.00051v2.pdf

- **Code**: https://github.com/facebookresearch/ParlAI

- 论文简述

  本文提出利用维基百科作为唯一的知识来源来解决开放领域的问题：任何事实性问题的答案都是维基百科文章的一个文本范围。大规模机器阅读的任务结合了文档检索(查找相关文章)和机器理解文本(从这些文章中识别答案)的挑战。

  我们的方法将基于二元哈希和TF-IDF匹配的搜索组件与为检测维基百科段落中的答案而训练的多层递归神经网络模型相结合。我们在现有的多个质量保证数据集上的实验表明：（1）这两个模块与现有的同类模块相比都具有很强的竞争力；（2）在它们的组合上使用远距离监督的多任务学习在这个具有挑战性的任务上是一个有效的完整系统。

<img src="../images/imgs_paper/Figure1-paper4.png" alt="Figure1-paper4" style="zoom:50%;" />



<img src="../images/imgs_paper/Table-1_paper4.png" alt="Table-1_paper4" style="zoom:50%;" />

<center>表1：每个QA数据集的训练数据示例。在每个案例中，我们都显示了一个相关的段落，其中远距离监督（DS）正确地识别了其中的答案，这一点被突出显示。</center>
结论：我们通过使用维基百科作为开放领域质量保证的独特知识源，研究了大规模的机器阅读任务。我们的结果表明，MRS是研究人员需要关注的一个关键的挑战性任务。单纯的机器理解系统不能解决整个任务。我们的方法整合了搜索、远距离监督和多任务学习，以提供一个有效的完整系统。在多个基准上对各个组件以及整个系统进行评估，显示了我们方法的功效。



本次整理的论文还是主要偏向于机器阅读理解的问答（MRC-QA），其中主要涉及到**双向注意流(BIDAF)网络**、**Gated Attention模型**、**AS Reader模型**、**问答句子识别**、**双向注意机制和层次表示学习的关系图神经网络**、**类人问答系统建立**等。

5、TILE: Bidirectional Attention Flow for Machine Comprehension

- 机器理解的双向注意流
- 2018

- **Paper:** https://arxiv.org/pdf/1611.01603v6.pdf

- **Code:** https://github.com/allenai/bi-att-flow

**论文简述：**机器理解(Machine comprehension, MC)主要用于回答关于给定上下文段落的查询，它需要对上下文和查询之间的复杂交互进行建模。最近，注意力机制已经成功地扩展到MC。通常，这些方法将注意力集中在上下文的一小部分，并以固定大小的向量、时间上的耦合注意力和/或通常形成单向注意力来进行归纳。本文介绍了双向注意流(BIDAF)网络，这是一个多阶段的分层过程，代表了不同粒度级别的上下文，并使用双向注意流机制来获得一个不需要早期摘要的查询感知上下文表示。

结论：在本文中，我们介绍了BIDAF，这是一个多阶段的分层过程，在不同的粒度水平上表示上下文，并使用双向注意流机制来实现查询感知的上下文表示，而不需要提前总结。实验评估表明，我们的模型在斯坦福大学问题回答数据集（SQuAD）和CNN/DailyMail的cloze测试中取得了最先进的结果。消减分析表明我们的模型中每个组件的重要性。可视化和讨论表明，我们的模型正在为MC学习一个合适的表述，并且能够通过关注给定段落中的正确位置来回答复杂的问题。未来的工作涉及扩展我们的方法，以纳入注意力层的多个跳动。



6、TILE: Gated-Attention Readers for Text Comprehension

- 用于文本理解的门控式注意力读物
- 2017

**Paper:** https://arxiv.org/pdf/1606.01549v3.pdf

**Code:** https://github.com/bdhingra/ga-reader

**论文简述：**本文研究了文档的封闭性问答。本文模型Gated Attention (GA)阅读器，集成了一个多跳架构和一个新的注意机制，该机制基于查询嵌入和递归神经网络文档阅读器中间状态之间的乘法交互。这使阅读器能够在文档中构建特定于查询的令牌表示，以便准确地选择答案。

在本文中，我们研究了回答文件上的cloze-style问题的问题。我们的模型，即门控注意力（GA）阅读器，整合了一个多跳架构和一个新的注意力机制，该机制基于查询嵌入和循环神经网络文档阅读器的中间状态之间的乘法互动关系。这使阅读器能够在文档中建立特定于查询的标记表示，以进行准确的答案选择。GA阅读器在这项任务的三个基准上获得了最先进的结果--CNN和每日邮报的新闻故事以及Who Did What数据集。乘法互动的有效性通过一项消融研究以及与实现门控注意力的其他组成运算符的比较得到了证明。

结论：

我们展示了 Gated-Attention 阅读器，用于回答文档上的完形填空式问题。 GA 阅读器具有新颖的乘法门控机制，并结合了多跳架构。我们的模型在几个大型基准数据集上实现了最先进的性能，比竞争基准提高了 4% 以上。我们的模型设计得到了一项消融研究的支持，该研究显示使用门控注意力作为信息过滤器的统计显着改进。我们还凭经验表明，在实现门控注意方面，乘法门控优于加法和串联操作，尽管理论依据仍然是未来研究目标的一部分。对读者中间层的文档和查询注意力的分析进一步表明，该模型迭代地关注查询的不同方面以得出最终答案。在本文中，我们专注于文本理解，但我们相信 Gated-Attention 机制可能有益于其他任务以及多个信息源交互的任务。



7、TILE: Text Understanding with the Attention Sum Reader Network

- 用注意力集中的读者网络理解文本
- 2016

**Paper:** https://arxiv.org/pdf/1603.01547v2.pdf

**Code:** https://github.com/rkadlec/asreader

**论文简述：** 本文提出了一个新的、简单的模型，它使用注意力直接从上下文中选择答案，而不是使用文档中单词的混合表示来计算答案。这使得该模型特别适合回答问题，其中答案是来自文档的一个单词。

最近推出了几个大型的Cloze-style上下文问答数据集：CNN和《每日邮报》的新闻数据以及儿童图书测试。由于这些数据集的规模，相关的文本理解任务很适合深度学习技术，目前似乎比所有替代方法都要好。我们提出了一个新的、简单的模型，该模型利用注意力直接从上下文中挑选答案，而不是像类似模型中通常使用文档中的单词混合表示来计算答案。这使得该模型特别适用于回答问题的问题，在这些问题中，答案是文档中的一个词。我们的模型组合在所有被评估的数据集上都达到了新的水平。

结论：在这篇文章中，我们提出了一个用于自然语言文本理解的新的神经网络架构。虽然我们的模型比以前发表的模型更简单，但它在所有评估的数据集上给出了新的最先进的准确性。Chen等人的分析表明，在CNN和《每日邮报》的数据集上，有相当一部分问题是模糊的，或者即使是人类也很难回答（部分原因是实体匿名化），所以我们模型的集合可能非常接近在这些数据集上可以实现的最大准确率。



8、Deep Learning for Answer Sentence Selection

- 深度学习用于选择答案的句子
- 2014

**Paper:** https://arxiv.org/pdf/1412.1632v1.pdf

**Code:** https://github.com/brmson/dataset-sts

**论文简述：** 选择答案句的任务是识别包含给定问题答案的句子。本文提出了一种新的分布式表示方法来解决这一问题，并通过语义编码来学习问题与答案的匹配。这与之前在这项任务上的工作形成对比，后者通常依赖于具有大量手工编制的语法和语义特征以及各种外部资源的分类器。本文的方法不需要任何特征工程，也不涉及专业的语言数据，使得这个模型很容易适用于广泛的领域和语言。

本次文章主要介绍了**ERNIE-GEN**(语言生成任务)、统一预训练语言模型(**UniLM**)、问答系统数据集(**CoQA**)、端到端神经生成问答(**GENQA**)、生成式问答系统评估方法、自编码自回归语言模型(**PALM**)、答案生成器(**KEAG**)、生成式问答(**gQA**)。

问题回答大致可分为两类。一种方法侧重于语义解析，通过将问题转化为数据库查询并随后将该查询应用于现有的知识库来检索答案。另一类是开放领域的问题回答，它与信息检索领域的关系更为密切。开放域问题的回答需要一些中间步骤。例如，要回答诸如 "谁写了《哈利-波特》这本书？"这样的问题，系统首先要识别问题类型并检索相关的文档。随后，在检索到的文档中，选择一个包含答案的句子，最后从相关句子中提取答案（J.K.罗琳）本身。在本文中，我们关注的是答案句子的选择，即从一组候选句子中选择回答事实问题的正确句子的任务。除了在开放领域问题回答中的作用，答案句子选择也是一个独立的任务，在知识库构建和信息提取中都有应用。

结论：本文证明了将分布式句子模型应用于答案句子选择的有效性。我们将问题和答案投射到矢量中，并学习了QA对之间的语义匹配函数。随后，我们将这个函数与一个简单的、加权的QA共同发生率计数器结合起来。

我们证明了这种带有词袋句子模型的方法比原来的基于计数的基线模型明显提高了性能。通过使用基于卷积神经网络的更复杂的句子模型，我们进一步提高了性能，并在答案选择任务上达到了最先进的水平。与以前基于特征工程和外部手工编码语义资源的工作相比，我们的方法要简单得多，也更灵活。

在未来，我们希望为这项任务研究更复杂的句子模型，例如基于卷积网络的句子模型，带有高阶n-grams和多个特征图，以及基于递归神经网络的模型。此外，由于答案句子的选择类似于文本内涵和转述检测，我们希望将这一工作思路扩展到这些任务。



9、ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation

- ERNIE-GEN：一个用于自然语言生成的增强型多流预训练和微调框架
- 2020

**Paper:** https://arxiv.org/pdf/2001.11314v3.pdf 

**Code:** https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen 

**论文简述：** 当前自然语言生成中的预培训工作很少关注下游任务的暴露偏差问题。为了解决这个问题，我们提出了一种增强的多流序列，用于序列预训练和框架微调，名为ERNIE-GEN，它通过增加生成机制和噪声感知生成方法来填补训练和推理之间的差异。为了使生成更接近人类给出的结果，此框架引入了跨接生成了流程，该流程并不是逐字预测，而是连续预测语义上完整的跨距。与现有的预训练方法不同，ERNIE-GEN结合了多粒度目标采样来构造预训练数据，从而增强了编码器和解码器之间的相关性。实验结果表明，ERNIE-GEN在一系列语言生成任务（包括抽象摘要（Gigaword和CNN / DailyMail），问题生成（ SQuAD），对话生成（Persona-Chat）和生成性问答（CoQA）上都得到了较好的结果。



结论：我们提出了一个增强的多流seq2seq预训练和微调框架，名为ERNIE-GEN，用于语言生成，其中包含了一个填充生成机制和一个噪声感知生成方法，以减轻暴露偏差。此外，ERNIE-GEN还集成了一个新的逐个跨度生成任务，以训练模型生成类似人类写作的文本，从而进一步提高下游任务的性能。通过广泛的实验，ERNIEGEN在一系列的NLG任务上取得了最先进的结果。未来的工作包括将强化学习纳入暴露偏差的预训练，并将ERNIE-GEN应用于更多的NLG任务，如机器翻译。

10、TILE: Neural Generative Question Answering

- 神经网络生成性问题回答
- 2016

**Paper:** https://arxiv.org/pdf/1512.01337v4.pdf

**Code:** https://github.com/jxfeb/Generative_QA

**论文简述：** 本文介绍了一种端到端神经网络模型，称为神经生成问答（GENQA），该模型可以基于知识库中的事实生成简单事实问题的答案。更具体地说，该模型建立在用于序列到序列学习的编码器-解码器框架上，同时具备查询知识库的能力，并在知识库中的问答对及其相关三元组的语料库上进行训练。实证研究表明，该模型能够有效地处理问题和答案的变化，并结合知识库中的事实生成正确、自然的答案。问题回答实验表明，该模型的性能优于基于嵌入的QA模型，也优于基于相同数据训练的神经对话模型。



## Demo案例分析

项目地址：

- [“万创杯”中医药天池大数据竞赛——中医文献问题生成挑战][https://github.com/kangyishuai/CHINESE-MEDICINE-QUESTION-GENERATION]

  

### 数据集信息展示：

训练集数据：

```json
{
    "id": 1240,
    "text": "\"胆石症的治疗应区别不同情况分别处理，无症状胆囊结石可不作治疗，但应定期观察并注意良好的饮食习惯。有症状的胆囊结石仍以胆囊切除术为较安全有效的疗法，此外，尚可采用体外震波碎石。胆管结石宜采用以手术为主的综合治疗。胆石症的家庭治疗可采用以下方法：\\n（1）一般治疗    预防和治疗肠道寄生虫病和肠道感染，以降低胆石症的发病率。胆绞痛发作期应禁食脂肪等食物，采用高碳水化合物流质饮食；缓解期应忌食富含胆固醇的食物如脑、肝、肾、蛋黄等。\\n（2）增进胆汁排泄    可选用50%硫酸镁10~15毫升，餐后口服，每日3次；胆盐每次口服0.5~1克，每日3次；去氢胆酸0.25克，每日3次，餐后服用。\\n（3）消除胆绞痛    轻者可卧床休息，右上腹热敷，用硝酸甘油酯0.6毫克，每3~4小时一次，含于舌下；或阿托品0.5毫克，每3~4小时肌肉注射一次。重者应住院治疗。\\n（4）排石疗法以中药治疗为主，若右上腹疼痛有间歇期，无明显发热及黄疸，苔薄白，脉弦，属气滞者，用生大黄6克、木香9克、枳壳9克、金钱草30克、川楝子9克、黄苓9克，水煎服。右上腹痛为持续性，且阵发性加剧，有明显发热及黄疸，舌红苔黄，",
    "annotations": [
      {
        "Q": "什么类型的胆囊结石可不作治疗？",
        "A": "无症状胆囊结"
      },
      {
        "Q": "胆石症的治疗应注意什么？",
        "A": "应区别不同情况分别处理"
      },
      {
        "Q": "胆管结石宜采用什么样的治疗方式？",
        "A": "以手术为主的综合治疗"
      }
    ]
  },

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
  },
{
    "id": 642,
    "text": "《素问·金匮真言论》说：“肾开窍于二阴”，意思就是前后二阴的排泄和生殖功能受肾的调控，如果久病体虚或者泄泻日久，肾阳、肾气均有可能被损伤。如果肾阳受损，则温煦作用减弱，脏腑的气化失常，常见泄泻；如果肾气虚衰，则肾的固摄功能失司，可见久泄滑脱。肾虚则脾胃阳气无法生化，所以会导致脾失健运，无法运化水谷，脾无法发挥升阳举陷的功能，所以出现滑脱的情况。",
    "annotations": [
      {
        "Q": "\"肾开窍于二阴\"出自哪里？",
        "A": "《素问·金匮真言论》"
      },
      {
        "Q": "肾开窍于二阴是什么意思？",
        "A": "意思就是前后二阴的排泄和生殖功能受肾的调控，如果久病体虚或者泄泻日久，肾阳、肾气均有可能被损伤。"
      },
      {
        "Q": "如果肾气虚衰会导致什么？",
        "A": "如果肾气虚衰，则肾的固摄功能失司，可见久泄滑脱。肾虚则脾胃阳气无法生化，所以会导致脾失健运，无法运化水谷，脾无法发挥升阳举陷的功能，所以出现滑脱的情况。"
      }
    ]
  },
```

测试集数据：

```json
{
    "id": 1050,
    "text": "\"橄榄，又名青果、白榄，为橄榄科植物橄榄的果实，产广东、广西、福建等地。宋朝大文学家苏东坡称之为“青子”。早在唐宋之间，橄榄已广泛地被采入药用。现代研究证实，橄榄的果实中含有蛋白质、脂肪、碳水化合物以及钙.磷、铁等。祖国医学认为橄榄味甘酸，性平，能够清肺，利咽，生津：解毒，主治咽喉肿痛，烦渴，咳嗽咯血以及细菌性痢疾、癫痫等，还能解除河豚毒以及酒毒。用新鲜橄榄3枚，白萝卜数片，水煎服，可以用于治疗咽喉肿痛。用橄榄10枚，去核，水煎汤，频频服用，可以治疗饮酒中毒昏闷不适。若肠风下血，可用橄榄烧灰存性，每次6克，用米汤汁调服。将橄榄炒研为末，用猪油调和，外敷，可以治疗口唇干裂生疮。用鲜橄榄20枚，冰糖50克，水炖服，可以用于小儿百日咳的治疗。若妇女妊娠呕吐不止，可将鲜橄榄适量捣烂，用水煎服。\"",
    "annotations": [
      {
        "Q": "",
        "A": "橄榄的果实中含有蛋白质、脂肪、碳水化合物以及钙.磷、铁等。"
      },
      {
        "Q": "",
        "A": "清肺，利咽，生津：解毒"
      },
      {
        "Q": "",
        "A": "青子"
      },
      {
        "Q": "",
        "A": "主治咽喉肿痛，烦渴，咳嗽咯血以及细菌性痢疾、癫痫等，还能解除河豚毒以及酒毒。"
      }
    ]
  },
  {
    "id": 230,
    "text": "黄帝说：我想听听运气学说是怎样创始的。岐伯说：你提这个问题很高明的啊！我曾看到《太始天元册》文记载，赤色的天气，经过牛、女二宿及西北方的戊分；黄色的天气，经过心、尾二宿及东南方的已分；青色的天气，经过危、室二宿与柳、鬼二宿之间；白色的天气，经过亢、氐二宿与昴、毕二宿之间；黑色的天气，经过张、翼二宿与娄、胃二宿之间。所谓戊分，即奎、壁二宿所在处，己分，即角、轸二宿所在处，奎、壁正当秋分时，日渐短，气渐寒，角、轸正当春分时，日渐长，气渐暖，所以是天地阴阳的门户。这是推演气候的开始，自然规律的所在，不可以不通。",
    "annotations": [
      {
        "Q": "",
        "A": "赤色的天气，经过牛、女二宿及西北方的戊分；黄色的天气，经过心、尾二宿及东南方的已分；青色的天气，经过危、室二宿与柳、鬼二宿之间；白色的天气，经过亢、氐二宿与昴、毕二宿之间；黑色的天气，经过张、翼二宿与娄、胃二宿之间"
      },
      {
        "Q": "",
        "A": "所谓戊分，即奎、壁二宿所在处"
      },
      {
        "Q": "",
        "A": "奎、壁正当秋分时，日渐短，气渐寒，角、轸正当春分时，日渐长，气渐暖，所以是天地阴阳的门户。"
      }
    ]
  },
```



### 数据分析

1. 篇章

   篇章通常是根据答案的上下文得来的,所以篇章中答案所在位置的附近与问题的相关性最强，所以取答案所在位置的前64个字符和后128个字符作为截取的篇章。

2. 问题

   问题文本的长度均未超过131个字符，所以相当于未做截断。

3. 答案

   对于答案文本 ，79.38%的长度是小于64的，相对而言答案长度可以取得较短，因为答案基本都是从原文中读取的，对篇章内容的描述，通常根据答案的前面部分，就已经能够推理出对应的提问。所以最终选取答案的长度为64，大于64的进行截断。 优点：保留了绝大部分信息，长度短，可以加速训练和推理。
