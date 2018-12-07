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

class BertKerasIntentClassfier(Component):
    """keras分类器base模型"""
    name = "intent_classifier_bert_keras"
    provides = ["intent", "intent_ranking"]
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
        "loss":"categorical_crossentropy",
        "optimizer":"Adam",
        'activation':"relu",
        "valid_rate":0.1,
        'regularize_rate':0.01,
        'early_stop_patience': 5
                }

    def __init__(self, component_config = None, model = None, inv_intent_dict = None):
        super(BertKerasIntentClassfier, self).__init__(component_config)
        self.component_config = component_config
        self.model = model
        self.inv_intent_dict = inv_intent_dict
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
            x_tmp = Dense(self.hidden_size, activation=self.activation)(x_tmp)
            x_tmp = Dropout(self.droprate)(x_tmp)
        #如果layer size=0:则认为直接把结果连接上输出
        x_tmp = Flatten()(x_tmp)
        y_output = Dense(self.intent_len,activation='softmax')(x_tmp)
        model = Model(inputs = x_input,outputs = y_output)
        return model

    def build_pooled_model_fn(self):
        """creat keras model function according to component conf,feature 为pooled_output时使用"""
        assert self.pooled_output == True
        x_input = Input(shape=(768,))
        x_tmp = x_input
        for i in range(self.num_hidden_layers):
            x_tmp = Dense(self.hidden_size, activation=self.activation)(x_tmp)
            x_tmp = Dropout(self.droprate)(x_tmp)
        # 如果layer size=0:则认为直接把结果连接上输出
        y_output = Dense(self.intent_len, activation='softmax')(x_tmp)
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
        print("intent_classifier_training!")
        x_train = np.array([intent_example.data["bert_feature"] for intent_example in training_data.training_examples])
        x_train = np.squeeze(x_train)
        #shape should be [num_example,max_seq_length,layer_size(768)]
        intent_dict = self._create_intent_dict(training_data)
        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}

        self.intent_len = len(intent_dict)

        y_train_str = [intent_example.data["intent"] for intent_example in training_data.intent_examples]
        y_train_num = [intent_dict[i] for i in y_train_str]
        y_train_num = keras.utils.to_categorical((y_train_num), num_classes=self.intent_len)

        if self.model == None:
            if self.pooled_output==True:
                self.model = self.build_pooled_model_fn()
            else:
                self.model = self.build_model_fn()
        elif self.fine_turing == True:
            #fine turing:保存前n层,更改最后一层的预测的intent的数量。重新训练。效果待测试
            self.model.layers = self.model.layers[:-1]
            self.model.add(Dense(intent_len, activation='softmax',kernel_regularizer=regularizers.l2(self.regularize_rate)))

        self.model.summary()
        tbCallBack = TensorBoard(log_dir='./models/' + self.name + 'tensorboard_dir',  # log 目录
                                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                 write_graph=True,  # 是否存储网络结构图
                                 write_grads=True,  # 是否可视化梯度直方图
                                 write_images=True,  # 是否可视化参数
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None)
        esCallBack=keras.callbacks.EarlyStopping(monitor='acc', patience=self.early_stop_patience, verbose=0, mode='auto')

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        self.model.fit(x_train, y_train_num,
                       validation_split=self.valid_rate,
                       epochs=self.epoch,
                       batch_size=self.batch_size,
                       callbacks=[esCallBack,tbCallBack]
                       )
        # callbacks = [esCallBack]
                       # callbacks = [tbCallBack, mcpCallBack, esCallBack]
        score = self.model.evaluate(x_train, y_train_num, batch_size=128)
        print("score:",score)

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

        x = self.model.predict(message.get("bert_features"))
        x = np.array(x[0])
        index = x.argmax()

        intent = {"name": str(self.inv_intent_dict[index]), "confidence": float(np.max(x))}
        if intent['confidence']<0.5:
            intent['name'] = 'UNK'
        message.set("intent", intent,
                    add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this component to disk for future loading."""

        classifier_file = os.path.join(model_dir, self.name+'.h5')
        self.model.save(classifier_file)

        with io.open(os.path.join(
                model_dir,
                self.name + "_inv_intent_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)

        return {"classifier_file": classifier_file}

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None,
             **kwargs):
        """Load this component from file."""
        print("loading intent classifier...")
        meta = model_metadata.for_component(cls.name)
        classifier_file = os.path.join(model_dir, cls.name + '.h5')
        model = keras.models.load_model(classifier_file)

        with io.open(os.path.join(
                model_dir,
                cls.name + "_inv_intent_dict.pkl"), 'rb') as f:
            inv_intent_dict = pickle.load(f)
        print("loading finish.")
        return BertKerasIntentClassfier(model=model,inv_intent_dict = inv_intent_dict)
