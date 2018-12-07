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

class BertAutoKerasClassfier(Component):
    """bert提取特征后用auto-keras训练模型"""
    name = "classifier_bert_auto_keras"
    provides = ["intent", "intent_ranking"]
    requires = ["bert_features"]

    defaults = {
        "is_training":True,
        "pooled_output":False,
        "max_seq_length":128,
                }

    def __init__(self, component_config = None, model = None, inv_intent_dict = None):
        super(BertAutoKerasClassfier, self).__init__(component_config)
        self.component_config = component_config
        self.model = model
        self.inv_intent_dict = inv_intent_dict
        if self.component_config!=None:
            self._creat_hyperparameters(component_config)

    @classmethod
    def create(cls,cfg):
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

        np.save('/home1/shenxing/rasa_bert_nlu/data/auto_keras_test_data/x_train.npy', x_train)
        np.save('/home1/shenxing/rasa_bert_nlu/data/auto_keras_test_data/y_train.npy', y_train_num)
        print("saved!!!!")
        # score = self.model.evaluate(x_train, y_train_num, batch_size=128)
        # print("score:",score)

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

        # intent = {"name": str(self.inv_intent_dict[index]), "confidence": str(np.max(x))}
        intent = {"name": str(self.inv_intent_dict[index]), "confidence": float(np.max(x))}
        # response = self.get_response(intent)
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
        return BertAutoKerasClassfier(model=model,inv_intent_dict = inv_intent_dict)
