# CS3307

CS3307 课程大作业：情感分析

## 数据处理

两个csv数据文件保存在data文件夹下

数据处理见get_preprocessd_data.py其调用utils.py中的preprocess进行数据清洗工作，得到的进一步数据也存储在data文件夹下。

## 训练过程介绍

### sklearn简单的词频+svm分类实现（CPU训练）

代码见 frequency_svm.py
训练集和验证集 9：1划分，迭代1000轮
验证集准确率：0.7818
测试集准确率：0.8106

![Alt text](image/image1.png)

### 预处理 + sklearn提供的多种模型实现

代码见 traditional_model.py

### WordAVGModel && RNN

具体模型代码见models.py 准确率也都在80％多
训练的代码为TrochText_train.py

### BERT（GPU训练）

代码见 BERT_train.py
模型参数文件在models文件夹下， 从[distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)处下载
目前取0.1％的数据作为训练集训练得到准确率为 80% 左右。

目前最高准确率为0.855
![Alt text](image/image.png)

### 代码文件介绍

- data文件夹：用于存储数据文件，包括原始数据，经过简单处理的训练集测试及数据，经过复杂处理的训练集测试集数据（文件较大，可以自己生成）
- models文件夹： 用于存储BERT模型相关的原始权重
- results文件夹：用于存储结果和训练好的网络权重
- BERT_train.py: BERT模型的训练代码
- datasets.py: 生成dataloader
- frequency_svm.py: sklearn简单的词频+svm分类实现代码
- get_preprocessed_data.py: 对数据进行预处理的代码
- models.py: WordAVGModel && RNN模型代码
- PCA.py: 用于PCA分析的代码
- T-SNE: 用于对结果进行T-SNE分析的代码
- TorchText_train.py: WordAVGModel && RNN模型训练代码
- traditional_model.py: 预处理 + sklearn提供的多种模型实现
- train.py: 用于pytorch训练的一些工具代码
- utils.py: 用于数据处理等一些的工具代码

