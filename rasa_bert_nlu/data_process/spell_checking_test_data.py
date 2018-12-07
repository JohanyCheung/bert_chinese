import pandas as pd
import json
import random
import re
spell_checking_test_data = open('../data/bert_data/custom_confusion.txt','r').readlines()
spell_checking_test_query = [line.split()[0] for line in spell_checking_test_data]

examples = []
for line in spell_checking_test_data:
    init_text = line.split()[0]
    init_text = init_text + "。"
    correct_text = line.split()[1]
    correct_text = correct_text + "。"
    example = {}
    example['text'] = init_text
    example['correct_text'] = correct_text
    examples.append(example)

data_json = {"rasa_nlu_data": {"common_examples": examples}}
data_save = open("../data/bert_data/spell_checking_testset.json", 'w')
json.dump(data_json, data_save,indent=4,ensure_ascii=False)

# from rasa_nlu.training_data import load_data
# training_data = load_data("/home1/shenxing/rasa_bert/data/bert_data/spell_checking_testset.json")