# -*- coding:utf-8 -*-

import sys
sys.path.append('/home/lxp/sentiment-analysis')
import os
import xml.dom.minidom
import re
import h5py
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

root = 'raw_corpus'
direp = {'CN':"cn_sample_data", 'EN':"en_sample_data"}
filep = {'raw_POS':"positive.xml", 'raw_NEG':"negative.xml", 'POS':"pure_positive.xml", 'NEG':"pure_nagative.xml"}
en_corpus_path = os.path.join(root, 'en_corpus.txt')
cn_corpus_path = os.path.join(root, 'cn_corpus.txt')
embedding_dir = 'word_embedding'
en_word2vec_path = os.path.join(embedding_dir, 'en_word2vec.dat')
cn_word2vec_path = os.path.join(embedding_dir, 'cn_word2vec.dat')
en_dict_path = os.path.join(embedding_dir, 'en_dict.h5')
cn_dict_path = os.path.join(embedding_dir, 'cn_dict.h5')

def preprocess_string(s, tag):
    """
        s: string to be processed
        tag: 'CN' or 'EN'
        Delete invalid characters.
    """
    if s[-1] == '\n':
        s = s[:-1]
    s = s.replace('\t', '')
    if "<review" not in s and "<reviews>" not in s and "</review>" not in s and "</reviews>" not in s:
        s = s.replace('&lt;', '《')
        s = s.replace('&gt;', '》')
    if tag == "CN":
        s.replace('<<', '《')
        s.replace('>>', '》')
    s = ' '.join(s.split())
    return re.sub(r'&(?!(lt;)|(gt;)|(amp;)|(apos;)|(quot;))', '&amp;', s)

def preprocess_file(lang, tag):
    """
        lang: 'CN' or 'EN'
        tag: 'pos' or 'neg'
        Preprocess all the sentences in file.
    """
    fname = os.path.join(direp[lang], filep['raw_' + tag])
    with open(fname, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        lines = list(filter(lambda x:x != '\n', lines))
        for i in range(len(lines)):
            lines[i] = preprocess_string(lines[i], lang)
    
    fname = os.path.join(direp[lang], filep[tag])
    with open(fname, 'w', encoding='utf-8') as f_out:
        for line in lines:
            f_out.write(line + '\n')

# 英文预处理
en_file = open(en_corpus_path, 'w', encoding='UTF-8')
preprocess_file('EN', 'POS')
fname = os.path.join(direp['EN'], filep['POS'])
dom = xml.dom.minidom.parse(fname)
collection = dom.documentElement
reviews = collection.getElementsByTagName("review")
en_sen_1 = np.zeros(len(reviews))
for i in range(len(reviews)):
    s = reviews[i].firstChild.data.strip()
    s = ' '.join(s.split())
    en_file.write(s + '\n')
    en_sen_1[i] = 1
preprocess_file('EN', 'NEG')
fname = os.path.join(direp['EN'], filep['NEG'])
dom = xml.dom.minidom.parse(fname)
collection = dom.documentElement
reviews = collection.getElementsByTagName("review")
en_sen_2 = np.zeros(len(reviews))
for i in range(len(reviews)):
    s = reviews[i].firstChild.data.strip()
    s = ' '.join(s.split())
    en_file.write(s + '\n')
    en_sen_2[i] = -1
en_sen=np.concatenate((en_sen_1, en_sen_2),axis=0)
en_file.close()

# 中文预处理
cn_file = open(cn_corpus_path, 'w', encoding='UTF-8')
preprocess_file('CN', 'POS')
fname = os.path.join(direp['CN'], filep['POS'])
dom = xml.dom.minidom.parse(fname)
collection = dom.documentElement
reviews = collection.getElementsByTagName("review")
cn_sen_1 = np.zeros(len(reviews))
for i in range(len(reviews)):
    s = reviews[i].firstChild.data.strip()
    s = ' '.join(s.split())
    cn_file.write(s + '\n')
    cn_sen_1[i] = 1
preprocess_file('CN', 'NEG')
fname = os.path.join(direp['CN'], filep['NEG'])
dom = xml.dom.minidom.parse(fname)
collection = dom.documentElement
reviews = collection.getElementsByTagName("review")
cn_sen_2 = np.zeros(len(reviews))
for i in range(len(reviews)):
    s = reviews[i].firstChild.data.strip()
    s = ' '.join(s.split())
    cn_file.write(s + '\n')
    cn_sen_2[i] = -1
cn_sen=np.concatenate((cn_sen_1, cn_sen_2),axis=0)
cn_file.close()

# 英文bert_tokenize
en_corpus = open(en_corpus_path, 'r', encoding='UTF-8')
lines = en_corpus.readlines()
en_corpus.close()
tokenizer = BertTokenizer.from_pretrained('bert_cache/bert-base-uncased-vocab.txt')
print('successfully loaded English')
en_tot = len(lines)
assert en_tot == en_sen.shape[0]
print("Total sentences:", en_tot)
en_mask = np.ones((en_tot, 128))
en_index = np.zeros((en_tot, 128))
for i in range(len(lines)):
    s = lines[i]
    tokenized_text = tokenizer.tokenize(s)[:128]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    k = len(indexed_tokens)
    while k < 128:
        en_mask[i, k] = 0
        indexed_tokens.append(0)
        k += 1
    en_index[i, :] = np.array(indexed_tokens)

ll = list(np.arange(en_tot))
parser_train = sorted(list(np.random.choice(ll, int(en_tot * 0.8), replace=False)))
parser_valid = sorted(list(set(ll).difference(set(parser_train))))
print("Total train:", len(parser_train))
print("Total valid:", len(parser_valid))

with h5py.File('en_train_data.h5', 'w') as f:
    f['data'] = en_index[parser_train, :]
    f['mask'] = en_mask[parser_train, :]
    f['annot'] = en_sen[parser_train]

with h5py.File('en_valid_data.h5', 'w') as f:
    f['data'] = en_index[parser_valid, :]
    f['mask'] = en_mask[parser_valid, :]
    f['annot'] = en_sen[parser_valid]

# 中文bert_tokenize
cn_corpus = open(cn_corpus_path, 'r', encoding='UTF-8')
lines = cn_corpus.readlines()
cn_corpus.close()
tokenizer = BertTokenizer.from_pretrained('bert_cache/bert-base-chinese-vocab.txt')
print('successfully loaded Chinese')
cn_tot = len(lines)
assert cn_tot == cn_sen.shape[0]
print("Total sentences:", cn_tot)
cn_mask = np.ones((cn_tot, 128))
cn_index = np.zeros((cn_tot, 128))
for i in range(len(lines)):
    s = lines[i]
    tokenized_text = tokenizer.tokenize(s)[:128]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    k = len(indexed_tokens)
    while k < 128:
        cn_mask[i, k] = 0
        indexed_tokens.append(0)
        k += 1
    cn_index[i, :] = np.array(indexed_tokens)

ll = list(np.arange(cn_tot))
parser_train = sorted(list(np.random.choice(ll, int(cn_tot * 0.8), replace=False)))
parser_valid = sorted(list(set(ll).difference(set(parser_train))))
print("Total train:", len(parser_train))
print("Total valid:", len(parser_valid))

with h5py.File('cn_train_data.h5', 'w') as f:
    f['data'] = cn_index[parser_train, :]
    f['mask'] = cn_mask[parser_train, :]
    f['annot'] = cn_sen[parser_train]

with h5py.File('cn_valid_data.h5', 'w') as f:
    f['data'] = cn_index[parser_valid, :]
    f['mask'] = cn_mask[parser_valid, :]
    f['annot'] = cn_sen[parser_valid]
