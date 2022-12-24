# Hugging Face -- T5模型实践

时间：2022-06-14

网址：[Hugging Face-T5  :  Links](https://huggingface.co/docs/transformers/model_doc/t5)



### 概述--Overview

The T5 model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

The abstract from the paper is the following:

> *Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pretraining objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.*

T5 comes in different sizes:

- t5-small、t5-base、t5-large、t5-3b、t5-11b

在原有 T5 模型的基础上，谷歌发布了一些后续作品：

- T5v1.1：T5v1.1 是 T5 的改进版本，在架构上做了一些调整，并且仅在 C4 上进行了预训练，没有混合监督任务。请参阅可在[此处](https://huggingface.co/docs/transformers/model_doc/t5v1.1)找到的 T5v1.1 文档。
- mT5：mT5 是一种多语言 T5 模型。它在 mC4 语料库上进行了预训练，其中包括 101 种语言。请参阅 mT5 的文档，可在[此处](https://huggingface.co/docs/transformers/model_doc/mt5)找到。
- byT5：byT5 是在字节序列而不是 SentencePiece 子词标记序列上预训练的 T5 模型。请参阅可在[此处](https://huggingface.co/docs/transformers/model_doc/byt5)找到的 byT5 文档。

所有检查点都可以在[hub](https://huggingface.co/models?search=t5)上找到。

### Training 

T5 是一种编码器-解码器模型，将所有 NLP 问题转换为文本到文本格式。它是使用教师强迫训练的。这意味着对于训练，我们总是需要一个输入序列和一个相应的目标序列。输入序列使用 input_ids 馈送到模型。目标序列向右移动，即，在前面加上一个起始序列标记，并使用 decoder_input_ids 馈送到解码器。在teacher-forcing 风格中，目标序列然后附加EOS 令牌并对应于标签。 PAD 令牌在此用作起始序列令牌。 T5 可以以有监督和无监督的方式进行训练/微调。

可以使用[T5ForConditionalGeneration](https://huggingface.co/docs/transformers/v4.19.4/en/model_doc/t5#transformers.T5ForConditionalGeneration)（或 Tensorflow/Flax 变体），它包括解码器顶部的语言建模头。

- 无监督去噪训练

在此设置中，输入序列的跨度被所谓的标记标记（也称为唯一标记标记）屏蔽，输出序列由相同标记标记和实际屏蔽标记的串联形成。每个哨兵标记代表该句子的唯一掩码标记，应以 <extra_id_0>、<extra_id_1>、... 直到 <extra_id_99> 开头。默认情况下，T5Tokenizer 中有 100 个标记令牌可用。

如果您对在新语料库上预训练 T5 感兴趣，请查看示例目录中的  [run_t5_mlm_flax.py](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling) 脚本。

- 有监督训练

在此设置中，输入序列和输出序列是标准的序列到序列输入输出映射。例如，假设我们要微调模型以进行翻译，并且我们有一个训练示例：输入序列“The house is wonderful”。和输出序列“Das Haus ist wunderbar.”，那么它们应该为模型准备如下：

```python
>>> from transformers import T5Tokenizer, T5ForConditionalGeneration

>>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
>>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

>>> input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
>>> labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(input_ids=input_ids, labels=labels).loss
>>> loss.item()
0.2542
```

如您所见，模型只需要 2 个输入来计算损失：input_ids（编码输入序列的 input_ids）和标签（编码目标序列的 input_ids）。该模型将根据标签自动创建decoder_input_ids，方法是将它们向右移动一个位置并预先添加config.decoder_start_token_id，对于T5，它等于0（即填充令牌的id）。还要注意任务前缀：我们在编码之前在输入序列前面加上“将英语翻译成德语：”。这将有助于提高性能，因为在 T5 的预训练期间使用了此任务前缀。

根据这个[论坛帖子](https://discuss.huggingface.co/t/t5-finetuning-tips/684)，任务前缀在 (1) 进行多任务训练 (2) 你的任务与 T5 的预训练混合中使用的监督任务之一相似或相关时很重要（任务见论文附录 D使用的前缀）。

如果在 TPU 上进行训练，建议将数据集的所有示例填充到相同的长度，或者使用 pad_to_multiple_of 具有少量预定义的桶大小以适合所有示例。不建议将批次动态填充到最长的示例TPU 因为它会触发训练期间遇到的每个批次形状的重新编译，从而显着减慢训练速度。仅填充到批处理中最长的示例）会导致 TPU 训练非常缓慢。



### Inference

在推理时，建议使用 generate()。该方法负责对输入进行编码，并通过交叉注意力层将编码的隐藏状态馈送到解码器，并自动回归生成解码器输出。查看这篇博文，了解有关使用 Transformer 生成文本的所有详细信息。还有这篇博客文章解释了生成器在编码器-解码器模型中的一般工作原理。

```python
>>> from transformers import T5Tokenizer, T5ForConditionalGeneration

>>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
>>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

>>> input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
>>> outputs = model.generate(input_ids)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Das Haus ist wunderbar.
```



### Performance

如果您想要更快的训练和推理性能，请安装 apex，然后模型将自动使用 apex.normalization.FusedRMSNorm 而不是 T5LayerNorm。前者使用优化的融合内核，比后者快几倍。



### Example scripts

T5 由几个示例脚本支持，用于预训练和微调：

- 预训练：[run_t5_mlm_flax.py](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py)  脚本允许您根据自己的数据进一步预训练 T5 或从头开始预训练 T5。 [t5_tokenizer_model.py](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/t5_tokenizer_model.py)  脚本允许您进一步训练 T5 标记器或根据您自己的数据从头开始训练 T5 标记器。请注意，Flax（基于 JAX 的神经网络库）对于在 TPU 硬件上进行训练特别有用。
- 微调：官方摘要脚本 ([PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization), [Tensorflow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization), and [Flax](https://github.com/huggingface/transformers/tree/main/examples/flax/summarization)) 和翻译脚本([PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation) and [Tensorflow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/translation))支持 T5。这些脚本使您可以轻松地在自定义数据上微调 T5 以进行摘要/翻译。



## <span style='color:brown'>T5 Finetuning Tips</span>

网站：[Hugging Face discuss:   Links](https://discuss.huggingface.co/t/t5-finetuning-tips/684)

### sshleifer:

- Apparently if you copy AdaFactor from fairseq, as recommended by t5 authors, you can fit batch size = 2 for t5-large lm finetuning
- fp16 rarely works.
- for most tasks, you need to manually add `</s>` to the end of your sequence.
- task specific prefix doesn’t matter much.



### valhalla:

- <span style='color:brown'>task prefixes matter when: </span>
  1. When doing multi-task training
  2. When your task similar or related to one of the supervised tasks used in T5 pre-training mixture.
- <span style='color:brown'>**Needs slightly higher LR**</span> than the default one set in Trainer, in my experiments 1e-4 and 3e-4 worked for almost all problems (classification, QA, que-gen, summ)
- no need to pass `decoder_input_ids` to T5 yourself, just pass `labels` and the `T5Model` will prepare them for you. labels should end with `eos_token`. (important! This is where most of the mistakes are happening).
- T5 uses `pad_token` as the `decoder_start_token_id` so when doing generation without the `generate` function make sure you start it with pad token.
- trimming batches when training on TPU leads to very slower training.
- apparently, because of sentencepiece and some possible leakage of other languages in C4 data, T5 gives somewhat sensible results for french lang. fine-tuned it on FQuAD (french version of SQuAD) for que gen and BLUE-4 against dev set was 15.



### moscow25:

- Training with AdaFactor works quite well for me so far. I use the “constant LR 0.001” recommended in all of Colin Raffel’s finetuning paper and other AdaFactor settings from original Noam Shazeer paper.
- Fair-Seq’s AdaFactor implementation is good – except you need to turn auto-scaling options off – no idea why they are on by default in the init.
- [fairseq/adafactor](https://github.com/pytorch/fairseq/blob/775122950d145382146e9120308432a9faf9a9b8/fairseq/optim/adafactor.py)
- lr=0.001, scale_parameter=False, relative_step=False

但是总而言之 - 我强烈建议使用Afafactor，而不是ADAM进行T5训练和微调：

- this is what the T5 authors use themselves
- AdaFactor was developed specifically with Transformers/T5 in mind (say so in the paper)
- ADAM is a massive waste of memory in general; it’s not surprising that something more efficient would work as well unless you have custom additions to your model



## <span style='color:brown'>**mT5**</span>

NOTE：mT5 仅在 mC4 上进行了预训练，不包括任何监督训练。因此，与原始 T5 模型不同，该模型必须在可用于下游任务之前进行微调。由于 mT5 是在无人监督的情况下进行预训练的，因此在单任务微调期间使用任务前缀并没有真正的优势。如果你在做多任务微调，你应该使用前缀。

Google 发布了以下变体：

- [google/mt5-small](https://huggingface.co/google/mt5-small)
- [google/mt5-base](https://huggingface.co/google/mt5-base)
- [google/mt5-large](https://huggingface.co/google/mt5-large)
- [google/mt5-xl](https://huggingface.co/google/mt5-xl)
- [google/mt5-xxl](https://huggingface.co/google/mt5-xxl).

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The original code can be found [here](https://github.com/google-research/multilingual-t5).







