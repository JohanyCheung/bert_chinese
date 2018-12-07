import os
import tensorflow as tf
import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import codecs
import collections
import json
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf_config = tf.ConfigProto(allow_soft_placement = True,log_device_placement=True)
tf_config.gpu_options.allow_growth = True


#路径设置
# BERT_BASE_DIR='/home1/shenxing/uncased_L-12_H-768_A-12/'
BERT_BASE_DIR='/home1/shenxing/chinese_L-12_H-768_A-12/'
GLUE_DIR='/home1/shenxing//bert/glue_data/'
vocab_file=os.path.join(BERT_BASE_DIR,"vocab.txt")
bert_config_file = os.path.join(BERT_BASE_DIR,"bert_config.json")
init_checkpoint = os.path.join(BERT_BASE_DIR,"bert_model.ckpt")
max_seq_length=128
train_batch_size=32
learning_rate=2e-5
num_train_epochs=3.0
output_dir="~/bert_test_model"
use_one_hot_embeddings = False

def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1
  return examples

def read_examples_text(text):
  """Read `InputExample`s from a text,
  return InputExample"""
  unique_id = 0
  line = tokenization.convert_to_unicode(text)
  line = line.strip()
  text_a = None
  text_b = None
  m = re.match(r"^(.*) \|\|\| (.*)$", line)
  if m is None:
    text_a = line
  else:
    text_a = m.group(1)
    text_b = m.group(2)
  example = InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
  unique_id += 1
  return example

def convert_example_to_feature(examples, seq_length, tokenizer):
  """Loads a example into a list of `InputBatch`s."""

  features = []

  tokens_a = tokenizer.tokenize(example.text_a)

  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
      tokens_a = tokens_a[0:(seq_length - 2)]
  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  input_type_ids = []
  tokens.append("[CLS]")
  input_type_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    input_type_ids.append(0)
  tokens.append("[SEP]")
  input_type_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      input_type_ids.append(1)
    tokens.append("[SEP]")
    input_type_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < seq_length:
    input_ids.append(0)
    input_mask.append(0)
    input_type_ids.append(0)

  assert len(input_ids) == seq_length
  assert len(input_mask) == seq_length
  assert len(input_type_ids) == seq_length

  # tf.logging.info("*** Example ***")
  # tf.logging.info("unique_id: %s" % (example.unique_id))
  # tf.logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
  # tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
  # tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
  # tf.logging.info(
  #     "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))


  feature = InputFeatures(
        unique_id=example.unique_id,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids)
  return feature


class InputExample(object):
  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b

def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

# input_file = '/home1/shenxing/bert/tmp/input.txt'
# examples = read_examples(input_file)
# tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
# convert_examples_to_features(examples,seq_length=128,tokenizer=tokenizer)

text = "你好哇！"
example = read_examples_text(text)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
feature = convert_example_to_feature(example,seq_length=128,tokenizer=tokenizer)
print(feature)
