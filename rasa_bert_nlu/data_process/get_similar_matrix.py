import numpy as np
from rasa_nlu.bert_source_code import tokenization
import os
import pdb
import tensorflow as tf
import time

tf_config = tf.ConfigProto(allow_soft_placement = True,log_device_placement=True)
tf_config.gpu_options.allow_growth = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BERT_BASE_DIR = "../bert_pretrain_model/chinese_L-12_H-768_A-12/"
vocab_file = os.path.join(BERT_BASE_DIR, "vocab.txt")
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
token_dict = tokenizer.vocab
id_dict = {v: k for k, v in token_dict.items()}
vocab_size = len(token_dict)

def get_similar_numpy_matrix():
    similar_pinyin_dict = open('../data/bert_data/same_pinyin.txt')
    #拼音相似的词表
    similar_pinyin_dict = similar_pinyin_dict.readlines()[1:]
    similar_matrix = np.eye(vocab_size)
    #去掉第一行

    for line in similar_pinyin_dict:
        all_word = ''.join(line.split())
        all_ids = []
        for word in all_word:
            if word in token_dict:
                all_ids.append(token_dict[word])
        for id in all_ids:
            other_ids = all_ids.copy()
            other_ids.remove(id)
            for other_id in other_ids:
                similar_matrix[id,other_id] = 1
                similar_matrix[other_id,id] = 1
    np.save('../data/bert_data/similar_matrix.npy',similar_matrix)
    all_similar_num = (np.sum(similar_matrix)-vocab_size)/2
    #共有8万对相似字

similar_matrix = np.load("../data/bert_data/similar_matrix.npy")

def test_similar_matrix(test_str):
    test_id = token_dict[test_str]
    vector_i = similar_matrix[test_id]
    for i in range(len(vector_i)):
        if vector_i[i]==1:
            print(id_dict[i])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

logits = np.random.random([3,21128])

tic =time.time()
slice_similar_matrix = similar_matrix[[1,3,4],:]
logits_2 = np.multiply(slice_similar_matrix,logits)
log_prob = softmax(logits_2)
toc = time.time()
#这一步要0.001s
print(toc -tic)

#
# tf.reset_default_graph()
# # # 重置所有图
# sess = tf.Session(config=tf_config)
# #
# # similar_tensor = tf.convert_to_tensor(similar_matrix,dtype=tf.float32)
#
# a = tf.constant([[1], [3]])
# lm_input_ids = tf.placeholder(shape=[None,1], dtype=tf.int32)
# similar_tensor = tf.convert_to_tensor(similar_matrix,dtype=tf.float32)
# result = tf.gather_nd(similar_tensor,lm_input_ids)
# R = sess.run(result,feed_dict={lm_input_ids:[[1], [3]]})
# print(R.shape())

# similar_matrix = self.get_similar_matrix()
# slice_similar_matrix = similar_matrix[int_features.input_ids[1:-1],:]
# masked_lm_logits = np.multiply(masked_lm_logits,slice_similar_matrix)
