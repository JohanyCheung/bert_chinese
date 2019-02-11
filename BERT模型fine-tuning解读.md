BERT模型fine-tuning解读
=======
# 一. 简介

> BERT官方Github地址：https://github.com/google-research/bert ，BERT本质上是一个两段式的NLP模型。第一个阶段叫做：Pre-training，跟WordEmbedding类似，利用现有无标记的语料训练一个语言模型。第二个阶段叫做：Fine-tuning，利用预训练好的语言模型，完成具体的NLP下游任务。其中，只有run_classifier.py和run_squad.py是用来做fine-tuning 的，这里采用run_classifier.py进行句子分类任务。
>

# 二. 代码解析

> 查看run_classifier.py代码，进一步解析代码。

## (一) main函数入口

> 主函数入口指定了必须的参数，如下所示：

```python
if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
```

### 1.data_dir

> data_dir指的是输入数据的文件夹路径，代码中输入数据格式如下：

```python
class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
```

> text_b和label为可选参数，例如要做的是单个句子的分类任务，那么就不需要输入text_b；另外，在test样本中，便不需要输入lable。

### 2.task_name

> task_name为任务名称，代码中定义如下，其主要作用是用来选择processor的。

```python
processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
  }

task_name = FLAGS.task_name.lower()

if task_name not in processors:
  raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
```

> 以mrpc为例，查看MrpcProcessor，代码如下：

```python
class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
```

> 代码主要是用来对data_dir中输入的数据进行预处理，在data_dir中需要将数据处理成.tsv格式，训练集、验证集和测试集分别是train.tsv, dev.tsv, test.tsv，分隔符为"\t"。另外，label在get_labels()设定，如果是二分类，则将label设定为[“0”,”1”]，如果是多分类，可设置为["contradiction", "entailment", "neutral"]。在_create_examples()中，给定了如何获取guid以及如何给text_a, text_b和label赋值。

### 3.其他

> vocab_file和bert_config_file是预训练的配置文件，可到官方github进行下载，output_dir是结果输出文件夹路径。

# 三. fine-tuning修改

## (一) Processor设定

> 仿照上述代码的要求，设计自己的训练任务，如下：

```python
processors = {
      "np": NpProcessor,
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
  }
```

```python
class NpProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["色情", "语言", "饮食", "其他"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
```

> get_labels()中设置4分类的标签["色情", "语言", "饮食", "其他"]；
>
> 为了方便后续test测试效果，所以_create_examples()中没有对test做判断（当文件名是test.tsv时，只赋给text_a，label直接给0）；
>
> guid为自动生成。

## (二) 设置参数

> 设置flags的参数，将train和test整合程shell脚本 run_gpu.sh，如下：

```shell
#!/bin/bash
source ~/.bashrc

if [[ "$1" == "train" ]];then
    /home/work/anaconda3/bin/python3.6 ./run_classifier.py \
        --task_name NP \
        --do_train \
        --do_eval \
        --data_dir ./data \
        --init_checkpoint ./chinese_L-12_H-768_A-12/bert_model.ckpt \
        --vocab_file ./chinese_L-12_H-768_A-12/vocab.txt \
        --bert_config_file ./chinese_L-12_H-768_A-12/bert_config.json \
        --max_seq_length 16 \
        --train_batch_size 64 \
        --learning_rate 2e-5 \
        --num_train_epochs 20.0 \
        --output_dir ./gpu_result/
elif [[ "$1" == "test" ]];then
    /home/work/anaconda3/bin/python3.6 ./run_classifier.py \
        --task_name NP \
        --do_predict \
        --data_dir ./data \
        --bert_config_file ./chinese_L-12_H-768_A-12/bert_config.json \
        --init_checkpoint ./gpu_result \
        --vocab_file ./chinese_L-12_H-768_A-12/vocab.txt \
        --max_seq_length 16 \
        --output_dir ./gpu_result/
fi
```

- fine-tuning训练时，将do_train和do_eval设置为True，do_test设置为False(默认)；

- 当模型训练好了，就可以将do_test设置为True，将会自动调用保存在output_dir中已经训练好的模型，进行测试；init_checkpoint也可以明确设置，例如：--init_checkpoint ./gpu_result/model.ckpt-56000 \

- max_seq_length、train_batch_size、learning_rate、num_train_epochs 可以根据自己的设备情况适当调整；

> 运行方式如下：

```shell
# train
sh run_gpu.sh train

# test
sh run_gpu.sh test
```
