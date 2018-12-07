# Baseline

# Competition
[训练赛-汽车行业用户观点主题及情感识别](https://www.datafountain.cn/competitions/329/details/data-evaluation)

## fine-ture
1. 在google-bert的基础上进行微调，代码参见[sentiment_cls.py](../sentiment_cls.py)
2. train: ***``python sentiment_cls.py --task_name=sent --do_train=true --do_eval=true --data_dir=../data/ --vocab_file=/Users/higgs/beast/code/demo/bert/test/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/Users/higgs/beast/code/demo/bert/test/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/Users/higgs/beast/code/demo/bert/test/chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=64 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./tmp/sent_output/``***


## Reference
1. [干货 | 谷歌BERT模型fine-tune终极实践教程](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/84551399)