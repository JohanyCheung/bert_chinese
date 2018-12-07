import numpy as np
import pandas as pd
import json
import numpy as np
import pandas as pd
import json
trainset_url = '/home1/shenxing/NLPCC_sentiment/data/dev.txt'
trainset = open(trainset_url,'r')
trainset = trainset.read()
from lxml import etree
parser = etree.XMLParser(recover=True)
tree = etree.fromstring(trainset, parser=parser)
df = pd.DataFrame(columns=['Happiness','Sadness','Anger','Fear','Surprise','Content'])
i=0
for tweet in tree.iterchildren():
    sample = []
    for label in tweet.iterchildren():
        text = label.text
        text = ''.join(text.split())
        sample.append(text)
    df.loc[i] = sample[0:6]
    i+=1
    if i%100 ==0:
        print(i)
df.to_csv('/home1/shenxing/NLPCC_sentiment/data/dev.csv')

