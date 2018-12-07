#测试计算语言模型的时间
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import typing
from typing import Any
from typing import List
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.tokenizers import Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from rasa_nlu.bert_source_code import modeling,optimization,tokenization
from rasa_nlu.featurizers.bert_featurizer import BertFeaturizer
import re
import tensorflow as tf
import os
import time
import pdb

tf_config = tf.ConfigProto(allow_soft_placement = True,log_device_placement=True)
tf_config.gpu_options.allow_growth = True

class BertGenerator(BertFeaturizer):
    #直接从BertFeaturizer继承函数
    name = "generator_bert"

    provides = ["generator_text"]
    # 返回list[numpy],len(list) = len(self.layer_indexes),numpy shape: [1,max_seq_length,layer_size]
    requires = ["int_features"]

    defaults = {"BERT_BASE_DIR": "./bert_pretrain_model/chinese_L-12_H-768_A-12/",
                "max_seq_length":128,"layer_indexes": [-1],
                "pooled_output": False,"spell_checker": "mask",
                "mul_similar_matrix": False,"sentence_embedding_type": "pooled","spell_checker_score": 1,
                "g_score": 0,"g_num": 5,"task":"add_one"
                }
    #g_num: 每个mask生成的句子数量
    #when pooled_output = True,output:[num_example,1,hidden_size].取第一个token的hidden state的输出作为训练集。
    #pooled_output不能和语言模型同用
    #spell_checker: lm不对输入进行mask,mask:对输入进行mask，其他不进行纠错
    #spell_checker_score：当得分最大的字比当前字的得分大一个得分时进行替换
    #sentence_embedding_type: mean 每个输入的字的位置平均值，pooled 第一个字的结果作为sentence_embedding，其他不进行sentence_embedding
    #mul_similar_matrix: 语言模型结果乘以相似矩阵。
    #task: add_one:添加一个字,delete_one:删除一个字，replace_one:替换一个字 ,replace_two:替换两个字(效果调整中)，
    def __init__(self,
                 component_config=None, # type: Dict[Text, Any],
                 model = None,
                 sess = None
                 ):

        super(BertGenerator, self).__init__(component_config)
        if component_config!= None:
            #after creat
            self.sess = sess
            self.model = model
            self.max_seq_length = component_config['max_seq_length']
            self.pooled_output = False
            self.spell_checker = component_config['spell_checker']
            self.mul_similar_matrix = False
            self.similar_matrix = self.get_similar_matrix()
            self.g_num = component_config['g_num']
            self.g_score = component_config['g_score']
            self.task = component_config['task']

    def process(self, message, **kwargs):
        #bert特征提取
        int_features = message.get("int_features")
        if self.task == "add_one":
            max_n_lm_score, lm_mean_score = self.cal_add_one_mask(int_features)
            generator_texts = self.generate_add_one_text(int_features, max_n_lm_score)
        elif self.task == "delete_one":
            max_n_lm_score, lm_mean_score = self.cal_replace_one_mask(int_features)
            generator_texts = self.generate_delete_one_text(int_features,max_n_lm_score)
        elif self.task == "replace_one":
            max_n_lm_score, lm_mean_score = self.cal_replace_one_mask(int_features)
            generator_texts = self.generate_replace_one_text(int_features,max_n_lm_score)
        elif self.task == "replace_two" and len(int_features.tokens)>3:
            max_n_lm_score, lm_mean_score = self.cal_replace_n_mask(int_features)
            generator_texts = self.generate_replace_n_text(int_features,max_n_lm_score)
        else:
            generator_texts = []
        message.set("generator_texts",generator_texts,add_to_output=True)

    def add_one_masked_lm_input(self, int_features):
        """
        把当前句子每个位置中间插入一个mask
        """
        masked_token = "[MASK]"
        masked_id = self.model.token_dict[masked_token]
        #mask对应的vocab中的id

        tokens = int_features.tokens
        init_input_ids = int_features.input_ids
        init_input_mask = np.asarray(int_features.input_mask)
        init_input_type_ids = np.asarray(int_features.input_type_ids)

        tokens_len = len(tokens)
        assert tokens_len < self.max_seq_length -1
        #若果句子长度等于max_seq_length 就无法加字
        masked_input_ids = []
        for i in range(1,tokens_len):
            tmp_input_ids = init_input_ids.copy()
            tmp_input_ids[i] = masked_id
            tmp_input_ids[i+1:] = init_input_ids[i:-1]
            masked_input_ids.append(tmp_input_ids)
        masked_input_ids = np.asarray(masked_input_ids)
        N = masked_input_ids.shape[0]
        #shape:[N,max_seq_len] (输入到model中先获取sequence_output,所以长度必须max_seq_len)
        masked_input_mask = np.repeat(init_input_mask[:,np.newaxis],N,axis=1).transpose()
        masked_input_type_ids = np.repeat(init_input_type_ids[:,np.newaxis],N,axis=1).transpose()
        return masked_input_ids,masked_input_mask,masked_input_type_ids

    def cal_add_one_mask(self, int_features):
        N = len(int_features.tokens) - 1
        #加了一个字
        masked_input_ids, masked_input_mask, masked_input_type_ids = self.add_one_masked_lm_input(int_features)
        input_feed = {self.model.input_ids: masked_input_ids, self.model.input_mask: masked_input_mask,
                      self.model.input_type_ids: masked_input_type_ids}
        masked_bert_features = self.sess.run(self.model.get_sequence_output(), input_feed)
        # shape [N,max_seq_len,hidden_size]
        # 返回每个masked位置的概率值
        masked_position_features = [masked_bert_features[i, i + 1, :] for i in range(N)]
        masked_position_features = np.asarray(masked_position_features)
        # shape [N,hidden_size]
        input_feed = {self.model.lm_input_tensor: masked_position_features}
        masked_lm_log_probs = self.sess.run(self.model.lm_log_probs, input_feed)
        # shape:[N,|V|]
        mask_tokens = ["[MASK]" for i in range(N)]
        max_n_lm_score, lm_mean_score = self.get_max_n_lm_score(mask_tokens, masked_lm_log_probs,n=self.g_num)
        return max_n_lm_score, lm_mean_score

    def generate_add_one_text(self,int_features,max_n_lm_score):
        """
        如果返回的原始的字的得分小于最大得分一定值(score)，就进行替换
        :return:
        """
        generator_texts = []
        tokens = int_features.tokens[1:-1]
        tokens_list = []
        for k in range(len(max_n_lm_score)):
            mask_tokens = tokens.copy()
            mask_tokens.insert(k,"[MASK]")
            tokens_list.append(mask_tokens)
        #tokens_list输入的句子:[['[MASK]', '今', '天', '天', '气', '真', '好'],['今', '[MASK]', '天', '天', '气', '真', '好']]

        for i in range(len(max_n_lm_score)):
            new_tokens = tokens_list[i].copy()
            item = max_n_lm_score[i]
            init_word, init_score = item[0]
            max_word, max_score = item[1]
            for j in range(1,len(item)):
                new_word = item[j][0]
                new_score = item[j][1]
                re_word = re.sub(u'[^\u4e00-\u9fa5]+',"",new_word)
                if new_word != init_word \
                        and new_score - init_score > self.g_score \
                        and new_score >-8 and len(re_word) == len(new_word):
                    #如果备选词得分不比原始得分少一定值就生成这句话，并且是中文
                    new_tokens[i] = new_word
                    new_sentence = "".join(new_tokens)
                    generator_texts.append(new_sentence)
        return generator_texts

    def replace_n_masked_lm_input(self, int_features, mask_sizes = [2]):
        """
        从int_features中获取masked的信息，mask 每一个字，组成一个batch
        第一个和最后一个是CLS,SEP。不进行处理
        masked_lm_position:该句子mask的位置,比如:[1]
        masked_lm_ids:该句子mask的位置对应的token的ids，比如:[791]
        masked_lm_labels:该句子mask的位置对应的token的字符,比如:['今']
        masked_lm_weights:这里全都使用1.
        这里的mask 指的是把某个字用[MASK]代替，跟前面的input_mask不同
        """
        masked_token = "[MASK]"
        masked_id = self.model.token_dict[masked_token]
        #mask对应的vocab中的id

        tokens = int_features.tokens
        init_input_ids = int_features.input_ids
        init_input_ids = np.asarray(init_input_ids)
        init_input_mask = np.asarray(int_features.input_mask)
        init_input_type_ids = np.asarray(int_features.input_type_ids)

        tokens_len = len(tokens)
        masked_input_ids = []
        for mask_size in mask_sizes:
            #mask连续的N个字
            for i in range(1,tokens_len-mask_size):
                tmp_input_ids = init_input_ids.copy()
                tmp_input_ids[i:i+mask_size] = masked_id
                masked_input_ids.append(tmp_input_ids)
        masked_input_ids = np.asarray(masked_input_ids)
        N =masked_input_ids.shape[0]
        #shape:[N,max_seq_len] (输入到model中先获取sequence_output,所以长度必须max_seq_len)
        masked_input_mask = np.repeat(init_input_mask[:,np.newaxis],N,axis=1).transpose()
        masked_input_type_ids = np.repeat(init_input_type_ids[:,np.newaxis],N,axis=1).transpose()
        return masked_input_ids,masked_input_mask,masked_input_type_ids

    def cal_replace_n_mask(self, int_features):
        """
        一句话连续mask n个位置，在mask的位置生成所以可能的话
        :param int_features:
        :param masked_bert_features:
        :return:
        """
        # 计算出seq_output
        tic = time.time()
        # N = len(int_features.tokens) - 2
        masked_input_ids, masked_input_mask, masked_input_type_ids = self.replace_n_masked_lm_input(int_features)
        input_feed = {self.model.input_ids: masked_input_ids, self.model.input_mask: masked_input_mask,
                      self.model.input_type_ids: masked_input_type_ids}
        masked_bert_features = self.sess.run(self.model.get_sequence_output(), input_feed)
        # shape [N,max_seq_len,hidden_size]

        # 返回每个masked位置的概率值
        N = masked_bert_features.shape[0]
        masked_position_features = [masked_bert_features[i, i + 1 :i+3, :] for i in range(N)]
        #先替换两个字
        masked_position_features = np.reshape(np.asarray(masked_position_features),(-1,self.model.hidden_size))
        # shape [N*n_replace,hidden_size]
        input_feed = {self.model.lm_input_tensor: masked_position_features}
        masked_lm_log_probs = self.sess.run(self.model.lm_log_probs, input_feed)
        # shape:[N,|V|]
        #mask位置的原来的字符
        tokens = int_features.tokens[1:-1]
        init_tokens = []
        for i in range(len(tokens)-1):
            init_tokens.extend(tokens[i:i+2])

        max_n_lm_score, lm_mean_score = self.get_max_n_lm_score(init_tokens, masked_lm_log_probs,
                                                                n=self.g_num)
        return max_n_lm_score, lm_mean_score

    def generate_replace_n_text(self,int_features,max_n_lm_score):
        tokens = int_features.tokens[1:-1]
        generator_texts = []
        for i in range(len(max_n_lm_score)//2):
            new_tokens = tokens.copy()
            item1 = max_n_lm_score[2*i]
            item2 = max_n_lm_score[2*i+1]
            init_word1, init_score1 = item1[0]
            init_word2, init_score2 = item2[0]
            for (word,score) in item1:
                if word != init_word1:
                    max_word1,max_score1 = word,score
                    break
            for (word,score) in item2:
                if word != init_word2:
                    max_word2,max_score2 = word,score
                    break
            if max_score1 - init_score1 >self.g_score and max_score2 - init_score2 >self.g_score:
                new_tokens[i] = max_word1
                new_tokens[i+1] = max_word2
                new_sentence = "".join(new_tokens)
                generator_texts.append(new_sentence)
            # for j in range(1,len(item1)):
            #     for k in range(1, len(item2)):
            #         new_word1 = item1[j][0]
            #         new_score1 = item1[j][1]
            #         re_word1 = re.sub(u'[^\u4e00-\u9fa5]+', "", new_word1)
            #         new_word2 = item2[k][0]
            #         new_score2 = item2[k][1]
            #         re_word2 = re.sub(u'[^\u4e00-\u9fa5]+', "", new_word2)
            #         if new_word1 != init_word1 \
            #             and new_score1 - init_score1 > self.g_score \
            #             and new_score1 >-10 and len(re_word1) == len(new_word1) \
            #             and new_word2 != init_word2 \
            #             and new_score2 - init_score2 > self.g_score \
            #             and new_score2 > -10 and len(re_word2) == len(new_word2):
            #             new_tokens[i] = new_word1
            #             new_tokens[i+1] = new_word2
            #             new_sentence = "".join(new_tokens)
            #             generator_texts.append(new_sentence)
        return generator_texts

    def cal_replace_one_mask(self,int_features):
        """
        先将每个字都mask，组成一个batch
        跑bert_model,把sequence_output里的masked位置的向量取出，
        输入lm层，计算出masked位置的字的概率
        获取masked位置的
        :param int_features:
        :param masked_bert_features:
        :return:
        """
        #计算出seq_output
        tic = time.time()
        N = len(int_features.tokens)-2
        masked_input_ids, masked_input_mask, masked_input_type_ids = self.get_masked_lm_input(int_features)
        input_feed = {self.model.input_ids: masked_input_ids, self.model.input_mask: masked_input_mask,
                      self.model.input_type_ids: masked_input_type_ids}
        masked_bert_features = self.sess.run(self.model.get_sequence_output(), input_feed)
        # shape [N,max_seq_len,hidden_size]

        # 返回每个masked位置的概率值
        masked_position_features = [masked_bert_features[i, i + 1, :] for i in range(N)]
        masked_position_features = np.asarray(masked_position_features)
        #shape [N,hidden_size]
        lm_input_ids = int_features.input_ids[1:N+1]
        input_feed = {self.model.lm_input_tensor: masked_position_features}
        masked_lm_log_probs = self.sess.run(self.model.lm_log_probs, input_feed)
        # shape:[N,|V|]
        max_n_lm_score, lm_mean_score = self.get_max_n_lm_score(int_features.tokens[1:-1],masked_lm_log_probs,n=self.g_num)
        return max_n_lm_score,lm_mean_score

    def generate_replace_one_text(self,int_features,max_n_lm_score):
        """
        如果返回的原始的字的得分小于最大得分一定值(score)，就进行替换
        :return:
        """
        tokens = int_features.tokens[1:-1]
        generator_texts = []
        assert len(max_n_lm_score) == len(tokens)
        for i in range(len(max_n_lm_score)):
            new_tokens = tokens.copy()
            item = max_n_lm_score[i]
            init_word, init_score = item[0]
            max_word, max_score = item[1]
            for j in range(1,len(item)):
                new_word = item[j][0]
                new_score = item[j][1]
                re_word = re.sub(u'[^\u4e00-\u9fa5]+', "", new_word)
                if new_word != init_word \
                        and new_score - init_score > self.g_score \
                        and new_score >-10 \
                        and len(re_word) == len(new_word):
                #如果备选词得分不比原始得分少一定值就生成这句话
                    new_tokens[i] = new_word
                    new_sentence = "".join(new_tokens)
                    generator_texts.append(new_sentence)
        return generator_texts

    def generate_delete_one_text(self,int_features,max_n_lm_score):
        """
        去掉得分最低的一个字。
        :return:
        """
        tokens = int_features.tokens[1:-1]
        generator_texts = []
        scores = [item[0][1] for item in max_n_lm_score]
        index = np.argmin(scores)
        del tokens[index]
        new_sentence = "".join(tokens)
        generator_texts.append(new_sentence)
        return generator_texts


    def get_max_n_lm_score(self,lm_input_tokens,lm_log_probs,n=2):
        """
        lm_log_probs:[N,|V|]
        每个token取概率值最大的n个值和对应的字符,以及原本输入的token的得分
        返回语言模型总得分，每个token的最大得分的n个值。
        因为这里的结果是log之后的，直接加起来
        """
        num_tokens = lm_log_probs.shape[0]
        max_n_lm_score = []
        lm_sum_score = 0
        #语言模型总得分
        for j in range(num_tokens):
            lm_input_token = lm_input_tokens[j]
            lm_input_id = self.model.token_dict[lm_input_token]
            lm_log_prob = lm_log_probs[j]
            #第j个token的概率值
            lm_sum_score += lm_log_prob[lm_input_id]
            if self.mul_similar_matrix == True:
                k = np.sum(lm_log_prob<0)
                #乘以similar_matrix之后是[0,0,-0.7,0,-11..]这样的结果。所以先去掉零的index
                max_n_index = lm_log_prob.argsort()[:k][-n:][::-1]
                max_n_value = lm_log_prob[max_n_index]
                max_n_result = [(lm_input_token, lm_log_prob[lm_input_id])]
                if k<n:
                    max_n_result.extend([(self.model.id_dict[max_n_index[i]], max_n_value[i]) for i in range(k)])
                else:
                    max_n_result.extend([(self.model.id_dict[max_n_index[i]], max_n_value[i]) for i in range(n)])
            else:
                max_n_index = lm_log_prob.argsort()[-n:][::-1]
                max_n_value = lm_log_prob[max_n_index]
                max_n_result = [(lm_input_token, lm_log_prob[lm_input_id])]
                max_n_result.extend([(self.model.id_dict[max_n_index[i]], max_n_value[i]) for i in range(n)])
            max_n_lm_score.append(max_n_result)
            #[('今',0.5),('some_token',0.1),...]
        lm_mean_score = lm_sum_score/(num_tokens + 1e-5)
        return max_n_lm_score,lm_mean_score
