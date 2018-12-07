import pandas as pd
file = '/home1/shenxing/rasa_bert_nlu/data/bert_data/spell_checker_result.txt'
Data = pd.read_csv(file)
print(Data[Data['correct']==False])
a=1