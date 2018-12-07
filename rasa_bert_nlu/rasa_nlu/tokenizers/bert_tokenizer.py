from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
import pdb
from rasa_nlu.bert_source_code import tokenization
import os
import glob
import shutil
import re


from rasa_nlu.bert_source_code import tokenization
from rasa_nlu.bert_source_code.extract_features import InputExample,InputFeatures,_truncate_seq_pair,convert_examples_to_features

class BertTokenizer(Tokenizer, Component):
    name = "tokenizer_bert"

    provides = ["int_features"]
    #返回一个InputFeatures,暂时没有加上多个句子的处理
    #InputFeatures.unique_id是bert源码训练中用到，忽略。
    # InputFeatures.input_ids：type:list,length:max_seq_length,是文本对应的字典的index
    # InputFeatures.input_mask：type:list,length:max_seq_length,input_ids的mask，[input_ids[i]!=0]
    # InputFeatures.input_type_ids：tyep:list,length:max_seq_length,输入有两句话时，[0, 0, 0,...1, 1, 1, 1, 0, 0, 0, 0]把句子分为两句。IsNext和NotNext

    defaults = {"BERT_BASE_DIR":"./bert_pretrain_model/chinese_L-12_H-768_A-12/","max_seq_length":128}

    language_list = ["zh"]

    def __init__(self,
                 component_config=None, # type: Dict[Text, Any]
                 tokenizer=None
                 ):
        # type: (...) -> None

        super(BertTokenizer, self).__init__(component_config)
        self.tokenizer = tokenizer
        if component_config!=None:
            self.max_seq_length = component_config['max_seq_length']

    @classmethod
    def create(cls,cfg):
        component_conf = cfg.for_component(cls.name, cls.defaults)
        vocab_file = os.path.join(component_conf['BERT_BASE_DIR'],"vocab.txt")
        #加载词表
        component_conf["vocab_file"] = vocab_file
        tokenizer = tokenization.FullTokenizer(vocab_file=component_conf['vocab_file'], do_lower_case=True)
        return cls(component_conf, tokenizer)

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None,
             **kwargs):
        """Load this component from file."""
        return cls.create(model_metadata)


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        example = self.read_examples_text(message.text)
        feature = self.convert_example_to_feature(example, seq_length=self.max_seq_length, tokenizer=self.tokenizer)

        message.set("int_features", feature)

    def train(self,training_data,cfg,**kwargs):
        """没有对模型进行训练，只是为了给之后的训练过程提供每句话对应的int_features"""
        Text = []
        for i, example in enumerate(training_data.training_examples):
            Text.append(example.text)
        examples = self.read_list_examples(Text)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.max_seq_length, tokenizer=self.tokenizer)
        for i,example in enumerate(training_data.training_examples):
            example.set("int_feature", features[i])


    def read_examples_text(self,text):
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

    def convert_example_to_feature(self,example, seq_length, tokenizer):
        """Loads a example into a feature"""

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

        feature = InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids)
        return feature

    def read_list_examples(self,Text):
        """Read a list of `InputExample`s from a list of text."""
        examples = []
        unique_id = 0
        for text in Text:
            line = text.strip()
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
