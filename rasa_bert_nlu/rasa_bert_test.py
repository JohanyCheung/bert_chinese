from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter
import time
import pdb
from rasa_nlu import utils, config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def cal_time():
    tic = time.time()
    i = 0
    for examples in training_data.training_examples:
        text = examples.text
        result = interpreter.parse(text)
        print(result)
        # print(result["max_n_lm_score"])
        if i%10==0 and i>0:
            toc = time.time()
            print("time:",(toc-tic)/i)
            # break
        i+=1
    toc = time.time()
    time_per_text = (toc-tic)/len(training_data.training_examples)
    print("mean_time:",time_per_text)

# training_data = load_data('./data/examples/rasa/demo-rasa_zh.json')
training_data = load_data('./data/ner/bert_ner_train.json')
config_file = './sample_configs/config_bert_ner.yml'
ModelConfig = config.load(config_file)
trainer = Trainer(ModelConfig)
interpreter = trainer.train(training_data)
query = "这是中国领导人首次在哈佛大学发表演讲。"
while query != "Stop":
    print(interpreter.parse(query))
    query = input("input query: (insert Stop to close)\n")
print('Ner close')
