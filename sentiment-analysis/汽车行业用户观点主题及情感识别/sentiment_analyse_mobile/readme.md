## 基于bert的情感识别

数据集：[训练赛-汽车行业用户观点主题及情感识别](https://www.datafountain.cn/competitions/329/details)

## env
python3


## 训练模型
1. 下载训练数据，并copy到data目录下
2. 生成训练数据：***```python ./baseline/data.py gen_sentiment_files```***
3. 训练模型：***```python sentiment_cls.py --task_name=sent --do_train=true --do_eval=true --data_dir=./data/ --vocab_file=/Users/higgs/beast/code/demo/bert/test/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/Users/higgs/beast/code/demo/bert/test/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/Users/higgs/beast/code/demo/bert/test/chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=64 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./tmp/sent_output/```***

![train](./img/train.png)

## 预测结果
1. 下载测试数据，并copy到data目录下
2. 预测：***```python sentiment_cls.py --task_name=sent --do_train=false --do_predict=true --data_dir=./data/ --vocab_file=/Users/higgs/beast/code/demo/bert/test/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/Users/higgs/beast/code/demo/bert/test/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/Users/higgs/beast/code/demo/bert/test/chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=64 --train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./tmp/sent_output/```***
3. merge成参赛数据：***```python ./baseline/data.py test_merge_test_result```***