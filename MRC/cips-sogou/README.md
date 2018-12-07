# **基于bert cips-sogou 2018**

数据集：cips-sogou 2018比赛提供的数据
使用事实类和非事实类数据集，将它们转换为SQuAD数据集的格式，基于bert模型的基础上进行测试。
测试结果：

- 模型：bert

- 训练集：cips-sogou unfactoid training set

- 测试集：cipos-sogou unfactoid eval set

- 实验结果：{

    ​                     "exact_match": 0.657786548265088, 

    ​                     "f1": 0.8098938569053756

    ​                    }

    ​                  {

    ​                      "Total Cnt": 5000
    ​                      "average bleu-4": 0.4400
    ​                      "average rouge-l": 0.5106

    ​                   }

- 实验细节： 计算各个paragraph的最大start\_logits + end\_logits，取最大值对应的text

- 参数设置：
    - doc_stride = 256
    - max_seq_length = 512
    - max_answer_length = 256
    - n_best_size = 10


