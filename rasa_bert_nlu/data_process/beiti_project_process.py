import pandas as pd
import json
import random
data = pd.read_excel('../data/beiti_project/在线客服问题清单.xlsx')

col = list(data.columns)
col[0] = 'intent'
data.columns = col

train_examples = []
test_examples = []
for i in range(len(data)):
    data_i = data.loc[i]
    intent_i = data_i.intent
    test_random_number = random.sample(range(2, len(data_i.keys())), 2)
    # 每个intent随机取两个句子作为测试集，第一个不取.
    for j in range(1, len(data_i.keys())):
        example = {}
        example['intent'] = intent_i
        test_example = {}
        test_example['intent'] = intent_i

        key = data_i.keys()[j]
        text = data_i[key]

        if type(text) != str:
            # 忽略nan
            continue
        if j in test_random_number:
            # 取随机值的句子作为测试集
            test_example['text'] = text
            test_example['entities'] = []
            test_examples.append(test_example)
        else:
            example['text'] = text
            example['entities'] = []
            train_examples.append(example)

data_train_json = {"rasa_nlu_data": {"common_examples": train_examples}}
data_train_save = open("/home1/shenxing/rasa_bert/data/beiti_project/trainset.json", 'w')
json.dump(data_train_json, data_train_save,indent=4,ensure_ascii=False)

data_test_json = {"rasa_nlu_data": {"common_examples": test_examples}}
data_test_save = open("/home1/shenxing/rasa_bert/data/beiti_project/testset.json", 'w')
json.dump(data_test_json, data_test_save,indent=4,ensure_ascii=False)

from rasa_nlu.training_data import load_data
training_data = load_data('/home1/shenxing/rasa_bert/data/beiti_project/trainset.json')