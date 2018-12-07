# !/usr/bin/env python3

import os
import pandas as pd
import fire

subjects_list = ['动力','价格','内饰','配置','安全性','外观','操控','油耗','空间','舒适性']
subjects_dict = {}
subjects_dict_id2label = {}
cnt = 0
for subject in subjects_list:
    subjects_dict[subject] = cnt
    subjects_dict_id2label[cnt] = subject
    cnt += 1

def get_sentiment_label(infile):
    df = pd.read_csv(infile)
    df_grouped = df.groupby(by='content_id')
    sents_list = []
    labels_list = []
    for name, group in df_grouped:
        len_group = len(group)
        if len_group >= 1:
            sents_list.append(list(group['content'])[0])
        labels = [0]*10
        for index, row in group.iterrows():
            subject_id = subjects_dict[row['subject']]
            labels[subject_id] += int(row['sentiment_value']) + 2
        labels = list(map(str, labels))
        labels = ' '.join(labels)
        labels_list.append(labels)
    return sents_list, labels_list

def gen_sentiment_files():
    infile = '../data/train.csv'
    outdir = '../data'
    sents_list, labels_list = get_sentiment_label(infile)
    assert len(sents_list) == len(labels_list)
    total_num = len(sents_list)
    train_num = int(0.8*total_num)
    val_num = int(0.9*total_num)
    with open(os.path.join(outdir, 'sents_train.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(sents_list[:val_num]))
    with open(os.path.join(outdir, 'labels_train.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(labels_list[:val_num]))
    with open(os.path.join(outdir, 'sents_val.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(sents_list[train_num:val_num]))
    with open(os.path.join(outdir, 'labels_val.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(labels_list[train_num:val_num]))
    with open(os.path.join(outdir, 'sents_test.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(sents_list[val_num:]))
    with open(os.path.join(outdir, 'labels_test.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(labels_list[val_num:]))

def merge_test_result(in_csv_file, in_result_file, out_csv_file):
    df_in = pd.read_csv(in_csv_file)
    df_out1 = pd.DataFrame(columns=['content_id', 'content', 'subject', 'sentiment_value', 'sentiment_word'])
    df_out = pd.DataFrame(columns=['content_id', 'subject', 'sentiment_value', 'sentiment_word'])
    res_list = []
    with open(in_result_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            res_list.append(line)
    jj = 0
    for index, row in df_in.iterrows():
        res = res_list[index]
        ss = res.split('\t')
        cnt_jj = 0
        for i in range(len(ss)):
            if int(ss[i]) > 0:
                df_out1.loc[jj] = [row['content_id'], row['content'], subjects_dict_id2label[i], int(ss[i])-2, '']
                df_out.loc[jj] = [row['content_id'], subjects_dict_id2label[i], int(ss[i]) - 2, '']
                jj += 1
                cnt_jj += 1
        if cnt_jj == 0:
            df_out.loc[jj] = [row['content_id'], '', 0, '']
            jj += 1

    # df_out.reset_index()
    df_out1.to_csv('./test_public_all.csv', index=False, encoding='utf-8-sig')
    df_out.to_csv(out_csv_file, index=False, encoding='utf-8-sig')
    print('hello world!')

def test_merge_test_result():
    in_csv_file = '../data/test_public.csv'
    in_res_file = '../bert/tmp/sent_output/test_results.tsv'
    out_csv_file = './test_public.csv'
    merge_test_result(in_csv_file, in_res_file, out_csv_file)

def stat_senti_word(in_csv_file):
    df = pd.read_csv(in_csv_file)
    df = df[df['sentiment_word'].notnull()]
    df_group1 = df.groupby(['sentiment_word'])
    df_group2 = df.groupby(['subject'])
    df['subject']
    print('hello world!')


if __name__ == '__main__':
    # get_sentiment_label('../data/train.csv', '内饰')
    # gen_sentiment_files()
    # test_merge_test_result()
    stat_senti_word('../data/train.csv')
    # fire.Fire()
