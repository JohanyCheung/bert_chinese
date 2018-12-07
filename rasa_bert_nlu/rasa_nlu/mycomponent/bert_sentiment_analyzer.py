from rasa_nlu import utils
from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Input
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.models import load_model,Model
from keras import optimizers
from keras.layers.core import Flatten
from keras import regularizers
import pdb
import os
import pickle
import io
import numpy as np
import time
import logging
logger = logging.getLogger(__name__)

class BertSentimentAnalyzer(Component):
    """
    基于bert结果的情感分析模块，
    训练数据https://github.com/z17176/Chinese_conversation_sentiment
    输出[0,1]的情感得分，0为负面情绪，1为正面情绪
    """
    name = "sentiment_analyzer_bert_keras"
    provides = ["sentiment_score"]
    requires = ["bert_features"]

    defaults = {
        "is_training":True,
        "fine_turing":False,
        "pooled_output":False,
        "num_hidden_layers":1,
        "hidden_size":768,
        "max_seq_length":128,
        "batch_size": 128,
        "epoch":30,
        "learning_rate": 1e-3,
        "lr_decay": 0,
        "droprate":0.2,
        "loss":"binary_crossentropy",
        "optimizer":"Adam",
        'activation':"relu",
        "valid_rate":0.1,
        'regularize_rate':0.01,
        'early_stop_patience':5
                }

    def __init__(self, component_config = None, model = None):
        super(BertSentimentAnalyzer, self).__init__(component_config)
        self.component_config = component_config
        self.model = model
        if self.component_config!=None:
            self._creat_hyperparameters(component_config)

    @classmethod
    def create(cls,cfg):
        tic = time.time()
        component_conf = cfg.for_component(cls.name, cls.defaults)
        is_training = component_conf['is_training']
        if is_training:
            model = None
        else:
            load_path = component_conf['load_path']
            assert load_path==None
            # must specify load_path when load
            model = load_model(load_path)
        return cls(component_conf,model)

    def _creat_hyperparameters(self,component_conf):
        self.pooled_output = component_conf['pooled_output']
        self.hidden_size = component_conf['hidden_size']
        self.batch_size = component_conf['batch_size']
        self.num_hidden_layers = component_conf['num_hidden_layers']
        self.epoch = component_conf['epoch']
        self.learning_rate = component_conf['learning_rate']
        self.droprate = component_conf['droprate']
        self.loss = component_conf['loss']
        self.valid_rate = component_conf['valid_rate']
        self.activation = component_conf['activation']
        self.fine_turing = component_conf['fine_turing']
        self.lr_decay = component_conf['lr_decay']
        self.max_seq_length = component_conf['max_seq_length']
        self.regularize_rate = component_conf['regularize_rate']
        self.early_stop_patience = component_conf['early_stop_patience']
        self.is_training = component_conf['is_training']
        optimizer = component_conf['optimizer']
        if optimizer == "Adam":
            self.optimizer = optimizers.Adam(lr=self.learning_rate)
        elif optimizer == "SGD":
            self.optimizer = optimizers.SGD(lr=self.learning_rate, momentum=0.0, decay=self.lr_decay, nesterov=False)
        elif optimizer == "RMSprop":
            self.optimizer = optimizers.RMSprop(lr=self.learning_rate)
        else:
            raise IOError("not support this optimizer")

    def build_model_fn(self):
        """creat keras model function according to component conf"""
        #输入的时候就包含了mask，所以这里不需要加上mask层
        x_input = Input(shape = (self.max_seq_length, 768))
        x_tmp = x_input
        for i in range(self.num_hidden_layers):
            x_tmp = Dense(self.hidden_size, activation=self.activation,kernel_regularizer=regularizers.l2(self.regularize_rate))(x_tmp)
            x_tmp = Dropout(self.droprate)(x_tmp)
        #如果layer size=0:则认为直接把结果连接上输出
        x_tmp = Flatten()(x_tmp)
        y_output = Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(self.regularize_rate))(x_tmp)
        model = Model(inputs = x_input,outputs = y_output)
        return model

    def build_pooled_model_fn(self):
        """creat keras model function according to component conf,feature 为pooled_output时使用"""
        assert self.pooled_output == True
        x_input = Input(shape=(768,))
        x_tmp = x_input
        for i in range(self.num_hidden_layers):
            x_tmp = Dense(self.hidden_size, activation=self.activation,kernel_regularizer=regularizers.l2(self.regularize_rate))(x_tmp)
            x_tmp = Dropout(self.droprate)(x_tmp)
        # 如果layer size=0:则认为直接把结果连接上输出
        y_output = Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(self.regularize_rate))(x_tmp)
        model = Model(inputs=x_input, outputs=y_output)
        return model

    @staticmethod
    def _create_intent_dict(training_data):
        """Create intent dictionary"""

        distinct_intents = set([example.get("intent")
                                for example in training_data.intent_examples])
        return {intent: idx
                for idx, intent in enumerate(sorted(distinct_intents))}

    def train(self, training_data, cfg, **kwargs):
        if self.is_training == False:
            return 0;
        print("sentiment_training!")
        x_train = np.array([intent_example.data["bert_feature"] for intent_example in training_data.training_examples])
        x_train = np.squeeze(x_train)
        #shape should be [num_example,max_seq_length,layer_size(768)]

        y_train = [int(intent_example.data["intent"]) for intent_example in training_data.intent_examples]
        #返回的是0,1

        if self.model == None:
            if self.pooled_output==True:
                self.model = self.build_pooled_model_fn()
            else:
                self.model = self.build_model_fn()
        elif self.fine_turing == True:
            #fine turing:保存前n层,更改最后一层的预测的intent的数量。重新训练。效果待测试
            self.model.layers = self.model.layers[:-1]
            self.model.add(Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(self.regularize_rate)))

        self.model.summary()
        esCallBack=keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stop_patience, verbose=0, mode='auto')

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        self.model.fit(x_train, y_train,
                       validation_split=self.valid_rate,
                       epochs=self.epoch,
                       batch_size=self.batch_size,
                       callbacks=[esCallBack]
                       )
        # callbacks = [esCallBack]
                       # callbacks = [tbCallBack, mcpCallBack, esCallBack]

        # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10)

    def process(self, message, **kwargs):
        """
        Return intent as a dict.

        Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""

        sentiment_score = self.model.predict(message.get("bert_features"))
        message.set("sentiment", float(sentiment_score),
                    add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this component to disk for future loading."""

        classifier_file = os.path.join(model_dir, self.name+'.h5')
        self.model.save(classifier_file)

        return {"classifier_file": classifier_file}

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None,
             **kwargs):
        """Load this component from file."""
        print("loading intent classifier...")
        meta = model_metadata.for_component(cls.name)
        classifier_file = os.path.join(model_dir, cls.name + '.h5')
        model = keras.models.load_model(classifier_file)
        return BertSentimentAnalyzer(model=model)
