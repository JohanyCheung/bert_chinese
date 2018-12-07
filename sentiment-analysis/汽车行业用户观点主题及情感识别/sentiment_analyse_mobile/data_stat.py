# !/usr/bin/env python3

import pandas as pd

'''
1. 统计由多个情感分类的文本
'''

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

train_file = './data/train.csv'
train_df = pd.read_csv(train_file)

train_df_grouped = train_df.groupby('content_id')

for name, group in train_df_grouped:
    print(name)
    print(group)