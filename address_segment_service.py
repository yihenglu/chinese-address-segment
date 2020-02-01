#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
使用Tornado框架部署服务
参考：https://www.jianshu.com/p/d1085dbf321f?tdsourcetag=s_pcqq_aiomsg
'''
import random
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tornado.ioloop
import tornado.web
import tensorflow as tf
from collections import OrderedDict
from gevent import monkey
monkey.patch_all()

from model import Model
from utils import get_logger,load_config,create_model,save_config
from utils import make_path
from data_utils import load_word2vec, create_input, input_from_line, BatchManager


def seed_everything(seed=1234):
    '''固定随机种子'''
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_model():
    '''获取地址分词模型'''

    '''模型参数'''
    flags = tf.app.flags
    # configurations for training
    flags.DEFINE_string("ckpt_path", "ckpt", "Path to save model")
    flags.DEFINE_string("log_file", "train.log", "File for log")
    flags.DEFINE_string("map_file", "maps.pkl", "file for maps")  # （词与词频，标签与整数）文件
    flags.DEFINE_string("config_file", "config_file", "File for config")
    FLAGS = tf.app.flags.FLAGS

    '''加载配置文件与日志文件'''
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    log_path = os.path.join("log", FLAGS.log_file)
    if not os.path.isdir("log"):
        os.makedirs("log")
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    logger = get_logger(log_path)

    '''加载模型'''
    tf_config = tf.ConfigProto()
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)

    return model, sess, char_to_id, id_to_tag


class NERService(object):
    '''定义NER服务对象'''
    def __init__(self):
        pass
    def address_segment(self, address):
        # print(address, type(address))  # <class 'str'>
        model, sess, char_to_id, id_to_tag = get_model()
        address_segment_result = model.evaluate_line(sess, input_from_line(address, char_to_id), id_to_tag)
        # print(address_segment_result, type(address_segment_result))  # <class 'dict'>
        return address_segment_result


class NERHandler(tornado.web.RequestHandler):
    '''提供地址分词服务'''
    service = NERService()

    def get(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Max-Age', 1000)
        # self.set_header('Content-type', 'application/json')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Access-Control-Allow-Headers',  # '*')
                        'authorization, Authorization, Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')

        address = str(self.get_argument("inputStr"))  # 获取url的参数值
        self.write(str(self.service.address_segment(address)))  # 使用NER服务


def make_app():
    return tornado.web.Application([
        (r"/fact", NERHandler),  # 注册路由
    ])


if __name__ == "__main__":
    seed_everything()
    app = make_app()
    app.listen(5002)
    tornado.ioloop.IOLoop.current().start()

    '''多线程'''
    # app = make_app()
    # server = tornado.httpserver.HTTPServer(app)
    # server.bind(5009)
    # server.start(0)  # forks one process per cpu
    # tornado.ioloop.IOLoop.current().start()

    '''访问示例'''
   # http://localhost:5002/fact?inputStr=长沙市芙蓉中路3段398号
   # http://192.168.3.149:5002/fact?inputStr=湖南省长沙市天心区大托铺街道中建芙蓉工社3栋

