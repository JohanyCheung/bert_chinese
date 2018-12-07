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

import tensorflow as tf
import os
import time
import pdb
from scipy import sparse

tf_config = tf.ConfigProto(allow_soft_placement = True,log_device_placement=True)
tf_config.gpu_options.allow_growth = True

class BertFeaturizer(Featurizer):
    #todo:如何gpu并行?
    name = "intent_featurizer_bert"

    provides = ["bert_features"]
    # 返回list[numpy],len(list) = len(self.layer_indexes),numpy shape: [1,max_seq_length,layer_size]
    requires = ["int_features"]

    defaults = {"BERT_BASE_DIR": "./bert_pretrain_model/chinese_L-12_H-768_A-12/",
                "max_seq_length":128,"layer_indexes": [-1],
                "pooled_output": False,"spell_checker": "lm",
                "mul_similar_matrix": True,"spell_checker_score": 1,
                "sentence_embedding_type": "pooled"}
    #when pooled_output = True,output:[num_example,1,hidden_size].取第一个token的hidden state的输出作为训练集。
    #pooled_output不能和语言模型同用
    #spell_checker: lm不对输入进行mask,mask:对输入进行mask，其他不进行纠错
    #spell_checker_score：当得分最大的字比当前字的得分大一个得分时进行替换
    #sentence_embedding_type: mean 每个输入的字的位置平均值，pooled 第一个字的结果作为sentence_embedding，其他不进行sentence_embedding
    #mul_similar_matrix: 语言模型结果乘以相似矩阵。
    def __init__(self,
                 component_config=None, # type: Dict[Text, Any],
                 model = None,
                 sess = None
                 ):

        super(BertFeaturizer, self).__init__(component_config)
        if component_config!= None:
            #after creat
            self.sess = sess
            self.model = model
            self.layer_indexes = component_config['layer_indexes']
            self.max_seq_length = component_config['max_seq_length']
            self.pooled_output = component_config['pooled_output']
            self.spell_checker = component_config['spell_checker']
            self.mul_similar_matrix = component_config['mul_similar_matrix']
            self.spell_checker_score = component_config['spell_checker_score']
            self.sentence_embedding_type = component_config['sentence_embedding_type']
            self.similar_matrix = self.get_similar_matrix()
    @classmethod
    def create(cls,cfg):
        tic = time.time()

        component_conf = cfg.for_component(cls.name, cls.defaults)

        #文件路径
        vocab_file = os.path.join(component_conf['BERT_BASE_DIR'], "vocab.txt")
        bert_config_file = os.path.join(component_conf['BERT_BASE_DIR'], "bert_config.json")
        init_checkpoint = os.path.join(component_conf['BERT_BASE_DIR'], "bert_model.ckpt")

        #参数
        is_training = False
        max_seq_length = component_conf['max_seq_length']
        use_one_hot_embeddings = False

        #加载词表
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        token_dict = tokenizer.vocab
        #key:token,value:id
        id_dict = {v: k for k, v in token_dict.items()}
        #key:id,value:token


        tf.reset_default_graph()
        #重置所有图
        sess = tf.Session(config=tf_config)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        #输入
        input_ids = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32)
        #None: num_examples
        input_mask = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32)
        input_type_ids = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32)

        #模型
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # model.similar_matrix = cls.get_similar_matrix(cls)

        #to do :这里不用place_holder，直接用slice获取输入(怀疑这一步速度慢)
        #语言模型部分代码
        lm_input_tensor = tf.placeholder(shape=[None,bert_config.hidden_size],dtype=tf.float32)
        #输入是model.sequence_output。把mask的所有位置的结果合为一个batch。output_tensor shape[N, width](width: hidden_size)
        #因为之后也要用到model.sequence_output的结果，所以先后面运行的时候先sess.run(model.sequence_output)，再把seq_output作为input_feed加入，再sess.run(lm_logits)
        #可以把lm_input_tensor换成model.sequence_output速度的差别未测试
        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                lm_tensor = tf.layers.dense(
                    lm_input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                lm_tensor = modeling.layer_norm(lm_tensor)
            #lm_tensor shape:[None,hidden_size]
            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            #output_bias shape [|V|]
            output_weights = model.get_embedding_table()
            #shape [|V|,hidden_size]
            logits = tf.matmul(lm_tensor, output_weights, transpose_b=True)
            #shape [N,|V|]
            lm_logits = tf.nn.bias_add(logits, output_bias)
            # if component_conf['mul_similar_matrix'] == True:
            #     model.lm_input_ids = tf.placeholder(shape=[None, 1], dtype=tf.int32)
            #     #[[1],[3],...]
            #     sliced_similar_matrix = tf.gather_nd(model.similar_matrix,model.lm_input_ids)
            #     #取输入对应的ids的行 shape:[N,|V|]
            #     lm_logits = tf.multiply(lm_logits,sliced_similar_matrix)
            log_probs = tf.nn.log_softmax(lm_logits, axis=-1)
            log_probs = log_probs - 1e-5
            #有些结果会等于0.0，之后会出错,保证结果是负值

        saver = tf.train.Saver()
        saver.restore(sess, init_checkpoint)

        #把所有需要传到下一步的变量都用model传下去
        model.lm_logits = lm_logits
        model.lm_log_probs = log_probs
        model.hidden_size = bert_config.hidden_size
        model.input_ids = input_ids
        model.input_mask = input_mask
        model.input_type_ids = input_type_ids
        model.lm_input_tensor = lm_input_tensor
        model.token_dict = token_dict
        model.id_dict = id_dict

        toc = time.time()
        print("creat bert featurizer using time: ",toc-tic,"s")
        return cls(component_conf, model, sess)

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None,
             **kwargs):
        """Load this component from file."""
        return cls.create(model_metadata)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        # tic = time.time()

        #bert特征提取
        int_features = message.get("int_features")
        #InputFeature类
        input_ids = np.asarray(int_features.input_ids)
        input_ids = np.reshape(input_ids,newshape=(-1,self.max_seq_length))
        input_mask = np.asarray(int_features.input_mask)
        input_mask = np.reshape(input_mask,newshape=(-1,self.max_seq_length))
        input_type_ids = np.asarray(int_features.input_type_ids)
        input_type_ids = np.reshape(input_type_ids,newshape=(-1,self.max_seq_length))

        input_feed = {self.model.input_ids:input_ids,self.model.input_mask:input_mask,self.model.input_type_ids:input_type_ids}

        output_features = self.get_output_features()
        numpy_features = self.sess.run(output_features,input_feed)
        bert_features = numpy_features[0]
        message.set("bert_features", numpy_features)

        #sentence_embedding
        if self.sentence_embedding_type == 'mean':
            #使用每个输入的词的特征的平均
            valid_len = len(int_features.tokens)
            if self.pooled_output == False:
                sentence_embed = np.mean(bert_features[0,:valid_len],0)
            else:
                seq_features =  self.sess.run(self.model.get_sequence_output(), input_feed)
                sentence_embed = np.mean(seq_features[0, :valid_len], 0)
            sentence_embed = np.reshape(sentence_embed,(-1,1))
            #shape [768,1]
        elif self.sentence_embedding_type == 'pooled':
            if self.pooled_output == True:
                sentence_embed = bert_features
            else:
                sentence_embed = self.sess.run(self.model.get_pooled_output(), input_feed)
            sentence_embed = np.reshape(sentence_embed, (-1, 1))
        else:
            sentence_embed = None

        message.set("sentence_embedding", sentence_embed)
        message.set("sentence_embedding_shape", sentence_embed.shape, add_to_output=True)
        # message.set("bert_features_shape", bert_features.shape, add_to_output=True)
        # toc = time.time()
        # message.set("time_bert_features", toc-tic, add_to_output=True)

        #纠错模块
        if self.spell_checker == 'lm':
            #没有mask，速度快,短文本效果差
            assert self.pooled_output == False
            max_n_lm_score, lm_mean_score = self.get_lm_score(int_features,bert_features)
            checked_text,replace_words = self.get_checked_text(int_features, max_n_lm_score, self.spell_checker_score)
        elif self.spell_checker == 'mask':
            assert self.pooled_output == False
            #sess 和 model 不知道怎么传到后面去，纠错模型暂时和featurizer放在一起
            #返回分别mask每一个字后输出的语言模型得分，(等于输入一个batch的样本)，batch_size = N
            # [[mask,token2,token3,...tokenN],[token1,mask,...tokenN],...[token1,token2,...mask]]
            max_n_lm_score, lm_mean_score = self.mask_spell_checking(int_features)
            checked_text,replace_words = self.get_checked_text(int_features, max_n_lm_score, self.spell_checker_score)
        else:
            max_n_lm_score,lm_mean_score,replace_words,checked_text = None,None,None,None
        message.set("lm_mean_score", lm_mean_score, add_to_output=True)
        message.set("max_n_lm_score", max_n_lm_score, add_to_output=True)
        message.set("checked_text", checked_text, add_to_output=True)
        message.set("replace_words", replace_words, add_to_output=True)

    def get_similar_matrix(self):
        """
        获取相似矩阵,shape[|V|,|V|],
        :return:
        """
        similar_matrix = sparse.load_npz("./data/bert_data/similar_matrix.npz")
        # similar_matrix = np.load("./data/bert_data/similar_matrix.npy")
        # similar_matrix = tf.convert_to_tensor(similar_matrix,dtype=tf.float32)
        return similar_matrix

    def get_lm_score(self,int_features,bert_features):
        """
        :param lm_input_tokens:
        :param bert_features:
        :return: 获取当前词的概率值，以及对应的概率值最高的词的得分。还有语言模型总得分
        """
        tic = time.time()

        tokens = int_features.tokens
        tokens_len = len(tokens)
        lm_input_tokens = tokens[1:-1]
        lm_input_ids = int_features.input_ids[1:tokens_len-1]
        # 比如：今天天气真好

        # 测试发现有没有加入cls和seq结果是一样的，所以去掉cls和seq加快速度
        lm_input_tensor = np.reshape(bert_features, (-1, self.model.hidden_size))
        lm_input_tensor = lm_input_tensor[1:tokens_len - 1, :]
        input_feed = {self.model.lm_input_tensor: lm_input_tensor}
        # 输入只有一句话，所以shape:[1,max_seq_len,hidden_size],取只有字符输入的位置的结果,去掉CLS和SEP.
        # self.lm_input_tensor shape:[tokens_len-2,hidden_size]
        lm_log_probs = self.sess.run(self.model.lm_log_probs, input_feed)

        toc = time.time()
        lm_time = toc-tic
        # softmax之后的结果
        tic = time.time()

        if self.mul_similar_matrix == True:
            sliced_similar_matrix = self.similar_matrix[lm_input_ids]
            lm_log_probs = sliced_similar_matrix.multiply(lm_log_probs)
            lm_log_probs = lm_log_probs.toarray()
            # lm_log_probs = lm_log_probs * sliced_similar_matrix
        max_n_lm_score,lm_mean_score = self.get_max_n_lm_score(lm_input_tokens, lm_log_probs)

        toc = time.time()
        similar_time = toc-tic

        return max_n_lm_score,lm_mean_score

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

    def pass_word(self):
        """
        纠错过程中部分词不进行纠错
        :return:
        """
        self.passed_words = ['哟','唉','呀','哈','暂','亿','贷','抵','滴','嘿','咱','诈']

    def get_checked_text(self,int_features,max_n_lm_score,score=1):
        """
        如果返回的原始的字的得分小于最大得分一定值(score)，就进行替换
        :return:
        """
        self.pass_word()
        tokens = int_features.tokens[1:-1]
        checked_text = ""
        if len(tokens)<2:
            #[ClS,WORD,SEP],只有单个字
            checked_text = "".join(tokens)
            return checked_text
        assert len(max_n_lm_score) == len(tokens)
        replace_words = []
        for i in range(len(max_n_lm_score)):
            item = max_n_lm_score[i]
            init_word, init_score = item[0]
            max_word, max_score = item[1]
            if init_word in self.passed_words:
                #忽略部分词
                checked_text += tokens[i]
                continue
            # if init_word != max_word and max_score - init_score > score and max_score >-10:
            elif init_word != max_word and max_score - init_score > score and max_score >-5:
                #语言模型结果与输入不同，而且最大结果得分比输入的得分大score
                checked_text += max_word
                replace_word = {}
                replace_word['init_word'] = init_word
                replace_word['replace_word'] = max_word
                replace_word['index'] = i
                replace_words.append(replace_word)
            else:
                checked_text += tokens[i]
        return checked_text,replace_words

    def get_masked_lm_input(self, int_features):
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
        init_input_mask = np.asarray(int_features.input_mask)
        init_input_type_ids = np.asarray(int_features.input_type_ids)

        tokens_len = len(tokens)
        N = tokens_len -2
        masked_input_ids = []
        for i in range(1,tokens_len-1):
            tmp_input_ids = init_input_ids.copy()
            tmp_input_ids[i] = masked_id
            masked_input_ids.append(tmp_input_ids)
        masked_input_ids = np.asarray(masked_input_ids)
        #shape:[N,max_seq_len] (输入到model中先获取sequence_output,所以长度必须max_seq_len)
        masked_input_mask = np.repeat(init_input_mask[:,np.newaxis],N,axis=1).transpose()
        masked_input_type_ids = np.repeat(init_input_type_ids[:,np.newaxis],N,axis=1).transpose()
        return masked_input_ids,masked_input_mask,masked_input_type_ids

    def mask_spell_checking(self,int_features):
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
        if self.mul_similar_matrix == True:
            sliced_similar_matrix = self.similar_matrix[lm_input_ids]
            masked_lm_log_probs = sliced_similar_matrix.multiply(masked_lm_log_probs)
            masked_lm_log_probs = masked_lm_log_probs.toarray()
            # masked_lm_log_probs = masked_lm_log_probs * sliced_similar_matrix
        max_n_lm_score, lm_mean_score = self.get_max_n_lm_score(int_features.tokens[1:-1],masked_lm_log_probs)
        return max_n_lm_score,lm_mean_score


    def get_output_features(self):
        """定义输出"""
        output_features = []
        assert len(self.layer_indexes)>0
        if len(self.layer_indexes) ==1:
            assert self.layer_indexes[0] == -1
            #只需要模型的最后一层的输出。
            if self.pooled_output == True:
                #取第一个token作为输入，用于分类问题。
                output_features.append(self.model.get_pooled_output())

                # 返回list[numpy],len(list) = len(self.layer_indexes),numpy shape: [num_example,layer_size]
            else:
                output_features.append(self.model.get_sequence_output())
                # 返回list[numpy],len(list) = len(self.layer_indexes),numpy shape: [num_example,max_seq_length,layer_size]
        else:
            #多层的输出未测试
            all_layers = self.model.get_all_encoder_layers()
            for (_, layer_index) in enumerate(layer_indexes):
                output_features.append(all_layers[layer_index])
        return output_features

    def train(self,training_data,cfg,**kwargs):
        """没有对模型进行训练，只是为了给之后的训练过程提供每句话对应的bert_features"""
        tic = time.time()

        int_features = []
        input_ids = []
        input_mask = []
        input_type_ids = []

        #get list of all input
        for i, example in enumerate(training_data.training_examples):
            int_feature = example.get("int_feature")
            input_ids.append(int_feature.input_ids)
            input_mask.append(int_feature.input_mask)
            input_type_ids.append(int_feature.input_type_ids)
        num_samples = len(input_ids)

        #reshape
        input_ids = np.asarray(input_ids)
        input_ids = np.reshape(input_ids, newshape=(num_samples, self.max_seq_length))
        input_mask = np.asarray(input_mask)
        input_mask = np.reshape(input_mask, newshape=(num_samples, self.max_seq_length))
        input_type_ids = np.asarray(input_type_ids)
        input_type_ids = np.reshape(input_type_ids, newshape=(num_samples, self.max_seq_length))


        batch_size = 32
        numpy_features = []
        last_batch = 0
        for i in range(num_samples//batch_size-1):
            print('calculated ',i*batch_size, ' samples')
            #按batch 进行计算，加快train的速度,batch_size不能太大，否则会OOM，(batch_size=1,memory 700M)
            input_ids_batch = input_ids[i*batch_size:(i+1)*batch_size,:]
            input_mask_batch = input_mask[i*batch_size:(i+1)*batch_size,:]
            input_type_ids_batch = input_type_ids[i*batch_size:(i+1)*batch_size,:]
            input_feed = {self.model.input_ids: input_ids_batch, self.model.input_mask: input_mask_batch, self.model.input_type_ids: input_type_ids_batch}
            output_features = self.get_output_features()
            numpy_features_batch = self.sess.run(output_features, input_feed)[0]
            numpy_features.extend(list(numpy_features_batch))
            last_batch = (i+1)*batch_size
        input_ids_batch = input_ids[last_batch:,:]
        input_mask_batch = input_mask[last_batch:,:]
        input_type_ids_batch = input_type_ids[last_batch:,:]
        input_feed = {self.model.input_ids: input_ids_batch, self.model.input_mask: input_mask_batch, self.model.input_type_ids: input_type_ids_batch}
        output_features = self.get_output_features()
        numpy_features_batch = self.sess.run(output_features, input_feed)[0]
        numpy_features.extend(list(numpy_features_batch))
        numpy_features = np.asarray(numpy_features)

        for i,example in enumerate(training_data.training_examples):
            example.set("bert_feature", numpy_features[i])
        toc = time.time()
        print("extract feature from ",num_samples," training_data using all :",toc-tic,",")
