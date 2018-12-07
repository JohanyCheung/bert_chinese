from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter
import time
import pdb
from rasa_nlu import utils, config
import os
import argparse
from rasa_nlu.model import Metadata, Interpreter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_argument_parser():
    parser = argparse.ArgumentParser(
            description='train a custom language parser')

    parser.add_argument('--model',
                        default="lm_spell_checker",
                        help="lm_spell_checker: lm纠错\n"
                             "mask_spell_checker: mask纠错\n"
                             "en_spell_checker: 英文纠错(开发中)"
                             "intent_pooled: pooled intent分类模型\n"
                             "intent_unpooled: unpooled intent分类模型\n"
                             "generator_replace_one: 将原句替换一个字生成新的句子\n"
                             "generator_add_one: 将原句加一个字生成新的句子\n"
                             "generator_delete_one: 将原句减一个字生成新的句子\n"
                             "generator_add_one: 将原句加一个字生成新的句子\n"
                             "generator_replace_two: 将原句替换两个字生成新的句子\n"
                             "NER: 一个NER demo,还未进行调参等"
                             "sentiment: 情感分析demo。")
    parser.add_argument('--train',
                        default="False",
                        help="True:用数据重新训练，False:使用训练好的模型")
    return parser

def lm_spell_checker_model(is_train):
    if is_train:
        training_data = load_data('./data/examples/rasa/demo_rasa.json')
        config_file = './sample_configs/config_bert_spell_checker_default_lm.yml'
        ModelConfig = config.load(config_file)
        trainer = Trainer(ModelConfig)
        interpreter = trainer.train(training_data)
    else:
        model_directory = './models/spell_checker/rasa_bert_spell_checker_lm'
        interpreter = Interpreter.load(model_directory)
    query = "今天阵是个好天气"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('spell_checker close')

def mask_spell_checker_model(is_train):
    if is_train:
        training_data = load_data('./data/examples/rasa/demo_rasa.json')
        config_file = './sample_configs/config_bert_spell_checker_default_mask.yml'
        ModelConfig = config.load(config_file)
        trainer = Trainer(ModelConfig)
        interpreter = trainer.train(training_data)
    else:
        model_directory = './models/spell_checker/rasa_bert_spell_checker_mask'
        interpreter = Interpreter.load(model_directory)
    query = "我想听张雨声的哥曲"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('spell_checker close')

def en_spell_checker_model(is_train):
    if is_train:
        training_data = load_data('./data/examples/rasa/demo-rasa.json')
        config_file = './sample_configs/config_bert_spell_checker_en.yml'
        ModelConfig = config.load(config_file)
        trainer = Trainer(ModelConfig)
        interpreter = trainer.train(training_data)
    else:
        model_directory = './models/spell_checker/rasa_bert_spell_checker_en'
        interpreter = Interpreter.load(model_directory)
    query = "How old aer you?"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('spell_checker close')

def intent_pooled_model(is_train):
    if is_train:
        training_data = load_data('./data/nlpcc_intent/rasa_nlpcc_train.json')
        config_file = './sample_configs/config_bert_intent_classifier_pooled.yml'
        ModelConfig = config.load(config_file)
        trainer = Trainer(ModelConfig)
        interpreter = trainer.train(training_data)
    else:
        model_directory = './models/rasa_bert/nlpcc_pooled'
        interpreter = Interpreter.load(model_directory)
    query = "播放一首歌"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('intent classifier close')

def intent_unpooled_model(is_train):
    if is_train:
        training_data = load_data('./data/nlpcc_intent/rasa_nlpcc_train.json')
        config_file = './sample_configs/config_bert_intent_classifier_unpooled.yml'
        ModelConfig = config.load(config_file)
        trainer = Trainer(ModelConfig)
        interpreter = trainer.train(training_data)
    else:
        model_directory = './models/rasa_bert/nlpcc_unpooled'
        interpreter = Interpreter.load(model_directory)
    query = "播放一首歌"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('intent classifier close')

def generator_replace_one():
    pipeline = [{"name": "tokenizer_bert"},
                {"name": "generator_bert",
                 "spell_checker": "mask",
                 "task":"replace_one",
                 "g_score":0
                 }
                ]
    trainer = Trainer(RasaNLUModelConfig({"pipeline": pipeline, "language": "zh"}))
    training_data = load_data('./data/examples/rasa/demo-rasa_zh.json')
    interpreter = trainer.train(training_data)
    query = "今天真的是个好天气"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('generator close')

def generator_delete_one():
    pipeline = [{"name": "tokenizer_bert"},
                {"name": "generator_bert",
                 "spell_checker": "mask",
                 "task":"delete_one",
                 "g_score":0
                 }
                ]
    trainer = Trainer(RasaNLUModelConfig({"pipeline": pipeline, "language": "zh"}))
    training_data = load_data('./data/examples/rasa/demo-rasa_zh.json')
    interpreter = trainer.train(training_data)
    query = "今天擦真的是个好天气"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('generator close')

def generator_add_one():
    pipeline = [{"name": "tokenizer_bert"},
                {"name": "generator_bert",
                 "spell_checker": "mask",
                 "task":"add_one",
                 "g_score":2
                 }
                ]
    trainer = Trainer(RasaNLUModelConfig({"pipeline": pipeline, "language": "zh"}))
    training_data = load_data('./data/examples/rasa/demo-rasa_zh.json')
    interpreter = trainer.train(training_data)
    query = "我想听一首歌"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('generator close')

def generator_replace_two():
    pipeline = [{"name": "tokenizer_bert"},
                {"name": "generator_bert",
                 "spell_checker": "mask",
                 "task":"replace_two",
                 "g_score":0
                 }
                ]
    trainer = Trainer(RasaNLUModelConfig({"pipeline": pipeline, "language": "zh"}))
    training_data = load_data('./data/examples/rasa/demo-rasa_zh.json')
    interpreter = trainer.train(training_data)
    query = "我想要听儿歌"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('generator close')

def NER(is_train):
    if is_train:
        training_data = load_data('./data/ner/bert_ner_train.json')
        config_file = './sample_configs/config_bert_ner.yml'
        ModelConfig = config.load(config_file)
        trainer = Trainer(ModelConfig)
        interpreter = trainer.train(training_data)
    else:
        model_directory = './models/rasa_bert/ner_demo'
        interpreter = Interpreter.load(model_directory)
    query = "这是中国领导人首次在哈佛大学发表演讲。"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('Ner close')

def sentiment_analyzer(is_train):
    if is_train:
        training_data = load_data('./data/sentiment_analyzer/trainset.json')
        config_file = './sample_configs/config_bert_sentiment.yml'
        ModelConfig = config.load(config_file)
        trainer = Trainer(ModelConfig)
        interpreter = trainer.train(training_data)
    else:
        model_directory = './models/sentiment/sentiment_demo'
        interpreter = Interpreter.load(model_directory)
    query = "今天好开心呀"
    while query != "Stop":
        print(interpreter.parse(query))
        query = input("input query: (insert Stop to close)\n")
    print('sentiment_analyzer close')

if __name__ == '__main__':
    cmdline_args = create_argument_parser().parse_args()
    is_train = eval(cmdline_args.train)
    if cmdline_args.model == "lm_spell_checker":
        lm_spell_checker_model(is_train)
    elif cmdline_args.model == "mask_spell_checker":
        mask_spell_checker_model(is_train)
    elif cmdline_args.model == "en_spell_checker":
        en_spell_checker_model(is_train)
    elif cmdline_args.model == "intent_pooled":
        intent_pooled_model(is_train)
    elif cmdline_args.model == "intent_unpooled":
        intent_unpooled_model(is_train)
    elif cmdline_args.model == "generator_replace_one":
        generator_replace_one()
    elif cmdline_args.model == "generator_delete_one":
        generator_delete_one()
    elif cmdline_args.model == "generator_add_one":
        generator_add_one()
    elif cmdline_args.model == "generator_replace_two":
        generator_replace_two()
    elif cmdline_args.model == "NER":
        NER(is_train)
    elif cmdline_args.model == "sentiment":
        sentiment_analyzer(is_train)
    else:
        print("wrong input!")