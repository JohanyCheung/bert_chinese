#!/usr/bin/env python
# coding=utf-8
import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.options
import os.path
import json
# from models.model_demo import *
from tornado.options import define, options
from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer

define("port", default=8886, help="run on the given port", type=int)
class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/parse", ParserHandler),
        ]
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            debug=True,
        )
        tornado.web.Application.__init__(self, handlers, **settings)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html', result='')

class ParserHandler(tornado.web.RequestHandler):
    def post(self):
        args = {k:self.get_argument(k) for k in self.request.arguments}
        print('args:', args)
        input_sentence = args.get('input_sentence', '')
        try:
             r = model.parse(input_sentence)
             # print(r)
             # result = r['intent'] + " " +str(r['intent_ranking'])
             # result = r['checked_text']
             result = str(r)
        except:
            result = "ERROR"
        # result = input_sentence
        self.render('index.html', result=result)

if __name__ == "__main__":
    print('loading')
    pipeline = [{"name": "tokenizer_bert"},
                {"name": "intent_featurizer_bert",
                 "lm_spell_checker": True,
                 "mask_spell_checker": False,
                 "mul_similar_matrix": True,
                 "spell_checker_score": 1}
                ]

    # pipeline = [{"name": "tokenizer_bert"},
    #             {"name": "intent_featurizer_bert",
    #              "lm_spell_checker": False,
    #              "mask_spell_checker": True,
    #              "mul_similar_matrix": True,
    #              "spell_checker_score": 1}
    #             ]
    training_data = load_data('./data/examples/rasa/demo-rasa_zh.json')
    trainer = Trainer(RasaNLUModelConfig({"pipeline": pipeline, "language": "zh"}))
    interpreter = trainer.train(training_data)
    model = interpreter
    # model = Interpreter.load('./projects/spell_checker/default/model_20190115-163425')
    print('loaded')
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
