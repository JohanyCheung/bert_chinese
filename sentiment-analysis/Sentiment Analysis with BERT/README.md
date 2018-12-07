# Sentiment Analysis with BERT

## 方法

- 中文：BERT + TextCNN + Bi-LSTM + Self-attention

- 英文：BERT + Bi-LSTM + Self-attention

## 实验

中文34639条，英文13385条

### 预处理：preprocess_bert.py

BERT分词

### 训练：bert_analysis.ipynb

用80%作为train集合，20%作为valid集合。

BertAdam，分类器学习率1e-3，微调学习率5e-5, batch大小24，进行30个epoch，结果取最优。

valid：中文89.68% 英文89.84%

### 测试：evaluate.ipynb

中英文各5000条

中文82+% 英文85+%

## To do

- 用机器翻译方法扩充数据集。

- 尝试Capsule、xgboost等其他分类器结构。

- BERT分层加权


