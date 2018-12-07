from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter
import time
import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def cal_time():
    T1 = []
    T2 = []
    T3 = []
    file = open('./data/bert_data/spell_checker_result.txt','w')
    file.write('text,init_word,init_score,max_word,max_score,second_word,second_score,score_diff,correct\n')
    tic = time.time()
    i = 0
    for examples in training_data.training_examples:
        text = examples.text
        result = interpreter.parse(text)
        # T1.append(result['time_bert_features'])
        # T2.append(result['lm_time'])
        # T3.append(result['similar_time'])
        max_n_lm_score = result["max_n_lm_score"]
        for item in max_n_lm_score:
            init_word,init_score = item[0]
            max_word,max_score = item[1]
            if len(item)<3:
                #有些情况只有一个结果
                second_word, second_score = "None","0"
                correct = init_word == max_word
                score_diff = "0"
            else:
                second_word,second_score = item[2]
                correct = init_word == max_word
                score_diff = max_score - second_score
            init_word,init_score,max_word,max_score,second_word,second_score,score_diff,correct = str(init_word),str(init_score),str(max_word),str(max_score),str(second_word),str(second_score),str(score_diff),str(correct)
            file.write(text+","+init_word+","+init_score+","+max_word+","+max_score+","+second_word+","+second_score+","+score_diff+","+correct+"\n")
            if correct != "True":
                print(text+","+init_word+","+init_score+","+max_word+","+max_score+","+second_word+","+second_score+","+correct)
        if i%10==0 and i>0:
            toc = time.time()
            print("time:",(toc-tic)/i)
            # break
        i+=1
    file.close()
    toc = time.time()
    time_per_text = (toc-tic)/len(training_data.training_examples)
    print("mean_time:",time_per_text)
    # print(T1,T2,T3)


# training_data = load_data('data/sentiment_analyzer/trainset.json')
# training_data = load_data('./data/examples/rasa/demo-rasa_zh.json')
training_data = load_data('data/data2/rasa_dataset_train.json')
# training_data = load_data('data/bert_data/spell_checking_testset.json')

pipeline = [{"name": "tokenizer_bert"},
            {"name": "intent_featurizer_bert","do_spell_checking": False,"LM_output": True,'mul_similar_matrix': True}
            ]

##train:
# trainer = Trainer(RasaNLUModelConfig({"pipeline": pipeline,"language":"zh"}))
# interpreter = trainer.train(training_data)
# model_directory = trainer.persist('projects/spell_checker/')
# cal_time()


#test:
# training_data = load_data('data/sentiment_analyzer/testset.json')
model_directory = "/home1/shenxing/rasa_bert_nlu/projects/spell_checker/default/model_20190107-141143"
from rasa_nlu.model import Metadata, Interpreter
interpreter = Interpreter.load(model_directory)
text = "怎样良好的生活习惯才能预防生病呢"
result = interpreter.parse(text)
print(result)
# cal_time()


##evaluate
# from rasa_nlu.evaluate import *
# run_evaluation("data/data2/out_test.json", "projects/default//model_20181219-200508")