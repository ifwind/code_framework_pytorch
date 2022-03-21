# Pytorch训练代码框架

## 前言

自己在学习和coding的过程中，感觉每次搞一个模型，需要写一堆的过程代码（大部分是可复用的），有的时候还需要从之前或者各个博客cv一点代码，这样开发起来效率可能比较低，所以整理了一份相对来说比较全面的Pytorch建模&训练框架，一些简单的trick也整理放在了里面，方便取用。

因为个人用NLP比较多，这个框架主要也是在预训练+微调这一范式下写的，但是想去掉预训练模型也很好改，就不多赘述了。

代码框架不是很成熟，主要是从自己开发过程中进行一些总结，欢迎大家提issue，我也会持续更新，希望也可以帮助一些有需要的朋友~

**`完整代码：`[code_frameword_pytorch](https://github.com/ifwind/code_framework_pytorch.git)**

**`Pytorch手册汇总：`[Pytorch手册汇总](https://ifwind.github.io/2022/03/20/Pytorch手册汇总/)**

## 安装包依赖

见`requirement.txt`

```
transformers==4.9.0
datasets==1.11.0
torch==1.7.1
sklearn==0.21.3
```

补充一下，命名实体识别任务中常用到pytorch-crf包，用下面的方式安装：

```
pip install pytorch-crf==0.4.0
```

## 代码介绍

包括了一下几个部分：

1. data_loader
2. data_process
3. main
4. model_utils
5. trainer
6. tricks
7. logging_config
8. config
9. run.sh

### data_process

数据预处理模块，最常用的就是数据预处理+数据集划分（训练、验证、测试）。

另外，我用huggingface的datasets做数据预处理比较多，这里看个人需求进行修改就可以~

### data_loader

数据集加载器，包括了两种数据集加载方法：

1. 全量读取数据`MyDataset、MyDataLoader`
2. 流式读取数据`MyIterableDataset、MyIterableDataLoader`

之前整理的博客：[Pytorch与深度学习自查手册2-数据加载和预处理](https://ifwind.github.io/2021/11/03/Pytorch与深度学习自查手册2-数据加载和预处理/)

### model_utils

模型文件，可以把自己设计的模型放在这里，常用的可以参考[Pytorch与深度学习自查手册3-模型定义](https://ifwind.github.io/2021/11/10/Pytorch与深度学习自查手册3-模型定义/)。

一些时候特别好使的`nn.ModuleDict`、`nn.ModuleList`，使用案例写在里面了。

### main

1. 参数配置args
2. 日志记录器logger
3. 数据集加载
4. 模型初始化：实例化trainer
5. optimizer、scheduler实例化
6. 开始训练
7. 评估
8. 预测

[Pytorch与深度学习自查手册4-训练、可视化、日志输出、保存模型](https://ifwind.github.io/2021/11/10/Pytorch与深度学习自查手册6-训练、验证和预测/)

[Pytorch与深度学习自查手册5-损失函数、优化器](https://ifwind.github.io/2021/11/10/Pytorch与深度学习自查手册5-损失函数、优化器/)

### trainer

`train_one_epoch`：每个epoch中的训练过程

`train`：控制整个训练流程，包括早停、模型保存、评估等步骤

`metric_function`：计算评价指标

`eval`：评估（有标签）

`predict`：预测（无标签）

`save_model`：保存模型（会存下模型、训练参数、optimizer、scheduler等，方便训练中断后，从断点开始继续训练）

### tricks

目前只放了对抗学习的FGM模块，warmup等常见的学习率调整策略、早停等直接写在main.py中了。

### logging_config

日志logger配置。

### config

固定随机种子（为了模型效果可复现）。

可以放常用的固定配置。

### run.sh

使用服务器可以直接用`sh run.sh`运行训练代码。

### huggingface_hands_on

`continue_pretrain_task`：利用huggingface，结合自己的语料继续预训练一个pretrained-model，（然后再微调）。

`train_tokenizer_form_scratch `：利用huggingface从头训练一个tokenizer。

利用huggingface从头开始预训练一个模型可以参考之前写的这篇博客：[Task4-基于深度学习的文本分类3-基于Bert预训练和微调进行文本分类](https://ifwind.github.io/2021/10/16/Task4-基于深度学习的文本分类3-基于Bert的预训练和微调进行文本分类/#设置微调参数)