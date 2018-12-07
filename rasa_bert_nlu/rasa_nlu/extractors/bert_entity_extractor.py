from rasa_nlu import utils
from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Input,TimeDistributed
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

class BertEntityExtractor(Component):
    """
    bert_ner模块
    """
    name = "ner_bert_keras"
    provides = ["entities"]
    requires = ["bert_features"]

    defaults = {
        "is_training": True,
        "num_hidden_layers": 0,
        "hidden_size": 768,
        "max_seq_length": 128,
        "batch_size": 128,
        "epoch": 30,
        "learning_rate": 1e-3,
        "lr_decay": 0,
        "droprate": 0.2,
        "loss": "categorical_crossentropy",
        "optimizer": "Adam",
        'activation': "relu",
        "valid_rate": 0.1,
        'regularize_rate': 0.01,
        'early_stop_patience': 3
    }

    def __init__(self, component_config=None, model=None, inv_entity_dict=None):
        super(BertEntityExtractor, self).__init__(component_config)
        self.component_config = component_config
        self.model = model
        self.inv_entity_dict = inv_entity_dict
        if self.component_config != None:
            self._creat_hyperparameters(component_config)

    @classmethod
    def create(cls, cfg):
        tic = time.time()
        component_conf = cfg.for_component(cls.name, cls.defaults)
        is_training = component_conf['is_training']
        if is_training:
            model = None
        else:
            load_path = component_conf['load_path']
            assert load_path == None
            # must specify load_path when load
            model = load_model(load_path)
        return cls(component_conf, model)

    def _creat_hyperparameters(self, component_conf):
        self.hidden_size = component_conf['hidden_size']
        self.batch_size = component_conf['batch_size']
        self.num_hidden_layers = component_conf['num_hidden_layers']
        self.epoch = component_conf['epoch']
        self.learning_rate = component_conf['learning_rate']
        self.droprate = component_conf['droprate']
        self.loss = component_conf['loss']
        self.valid_rate = component_conf['valid_rate']
        self.activation = component_conf['activation']
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
        model = Sequential()
        for i in range(self.num_hidden_layers):
            if i == 0:
                #第一层
                model.add(TimeDistributed(Dense(self.hidden_size), input_shape=(self.max_seq_length, 768)))
            else:
                model.add(TimeDistributed(Dense(self.hidden_size)))
        if self.num_hidden_layers == 0:
            #直接接上一层输入
            model.add(TimeDistributed(Dense(self.entity_len), input_shape=(self.max_seq_length, 768)))
        else:
            model.add(TimeDistributed(Dense(self.entity_len)))
        model.add(Activation('softmax'))
        return model

    @staticmethod
    def _create_entity_dict(training_data):
        """
        Create entity dictionary.BIO格式
        """
        entities = training_data.entities
        entity_dict = {}
        entity_dict["P"] = 0
        #PAD,CLS,SEP等字符
        entity_dict["O"] = 1
        # 没有命中ner的字
        idx = 2
        for entity in entities:
            name_B = "B-" + entity
            entity_dict[name_B] = idx
            idx += 1
            name_I = "I-" + entity
            entity_dict[name_I] = idx
            idx += 1
        return entity_dict

    def get_label_from_data(self,training_data):
        """
        构建BIO格式的y_train
        :param training_data:
        :return:
        """
        y_train = []
        for entity_example in training_data.training_examples:
            y_train_i = np.zeros(self.max_seq_length)
            int_feature_len = len(entity_example.data['int_feature'].tokens)
            #输入句子长度(带有CLS和SEP)
            y_train_i[1:int_feature_len-1] = 1
            #有效输入的字
            if 'entities' not in entity_example.data:
                #输入句子没有entity
                y_train.append(y_train_i)
            else:
                entities = entity_example.data['entities']
                for entity in entities:
                    start = entity['start']
                    end = entity['end']
                    entity_name = entity['entity']
                    entity_name_B = "B-" + entity_name
                    entity_name_I = "I-" + entity_name
                    label_B = self.entity_dict[entity_name_B]
                    label_I = self.entity_dict[entity_name_I]
                    y_train_i[start+1] = label_B
                    #有CLS，所以+1
                    y_train_i[start+2:end+1] = label_I
                y_train.append(y_train_i)
        y_train = np.asarray(y_train)
        # y_train = np.reshape(y_train,(-1,self.max_seq_length,1))
        return y_train

    def train(self, training_data, cfg, **kwargs):
        print("bert_ner_training!")
        x_train = np.array([intent_example.data["bert_feature"] for intent_example in training_data.training_examples])
        x_train = np.squeeze(x_train)
        # shape should be [num_example,max_seq_length,layer_size(768)]
        self.entity_dict = self._create_entity_dict(training_data)
        self.inv_entity_dict = {v: k for k, v in self.entity_dict.items()}

        self.entity_len = len(self.entity_dict)

        y_train_num = self.get_label_from_data(training_data)
        y_train_num = keras.utils.to_categorical((y_train_num), num_classes=self.entity_len)

        # x_train = self.get_dataset_from_data(training_data)

        self.model = self.build_model_fn()

        self.model.summary()
        tbCallBack = TensorBoard(log_dir='./models/' + self.name + 'tensorboard_dir',  # log 目录
                                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                 write_graph=True,  # 是否存储网络结构图
                                 write_grads=True,  # 是否可视化梯度直方图
                                 write_images=True,  # 是否可视化参数
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None)
        esCallBack = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stop_patience, verbose=0,
                                                   mode='auto')

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        #TODO:设置my_metric
        self.model.fit(x_train, y_train_num,
                       validation_split=self.valid_rate,
                       epochs=self.epoch,
                       batch_size=self.batch_size,
                       callbacks=[esCallBack,tbCallBack]
                       )
        # callbacks = [tbCallBack, mcpCallBack, esCallBack]

        score = self.model.evaluate(x_train, y_train_num, batch_size=128)
        print("score:", score)

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
        int_feature = message.get("int_features")
        tokens = int_feature.tokens
        tokens_len = len(tokens)
        y = self.model.predict(message.get("bert_features"))
        index = np.argmax(y,2)[0]
        #shape [128,]
        index = index[1:tokens_len-1]
        #大小与输入长度相同
        entities = []
        entity_tokens = [self.inv_entity_dict[i] for i in index]
        for i in range(len(entity_tokens)):
            entity_token = entity_tokens[i]
            if entity_token.startswith("B-"):
                entity_dict = {}
                entity_dict['start'] = i
                entity = entity_token.split("-")[1]
                entity_I = "I-" + entity
                entity_dict['entity'] = entity
                value = tokens[i+1]
                i+=1
                while i < len(entity_tokens) and entity_tokens[i] == entity_I:
                    value += tokens[i+1]
                    i+=1
                entity_dict['end'] = i
                entity_dict['value'] = value
                entities.append(entity_dict)
        message.set("entities",entities,add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this component to disk for future loading."""

        model_file = os.path.join(model_dir, self.name + '.h5')
        self.model.save(model_file)

        with io.open(os.path.join(
                model_dir,
                self.name + "_inv_entity_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_entity_dict, f)

        return {"model_file": model_file}

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None,
             **kwargs):
        """Load this component from file."""
        print("loading entity extractor...")
        meta = model_metadata.for_component(cls.name)
        classifier_file = os.path.join(model_dir, cls.name + '.h5')
        model = keras.models.load_model(classifier_file)

        with io.open(os.path.join(
                model_dir,
                cls.name + "_inv_entity.pkl"), 'rb') as f:
            inv_entity_dict = pickle.load(f)
        print("loading finish.")
        return BertKerasIntentClassfier(model=model, inv_entity_dict=inv_entity_dict)
