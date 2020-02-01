'''
训练并评估模型，同时可以逐行测试
'''
import random
import os
import pickle
import itertools
import time
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from collections import OrderedDict
from model import Model  # 引用其他文件中的函数没有报错
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def seed_everything(seed=1234):
    '''固定随机种子'''
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def config_model(FLAGS, char_to_id, tag_to_id):
    '''config for the model'''
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(FLAGS, sess, model, name, data, id_to_tag, logger):
    '''评估模型'''
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train_test(FLAGS, is_train, is_dev, is_test):
    '''训练模型'''
    # load data sets  sentences = [sentence,sentence],sentence=[word,word],word=['hanzi','tag']
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)  # 训练集被转换成[[句子1],...,[word1 tag1,...,wordn tagn]]，
    update_tag_scheme(train_sentences, FLAGS.tag_schema)  # Use selected tagging scheme (IOB / IOBES) # 检查一遍标签是否合法
    if is_dev:
        dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
        update_tag_scheme(dev_sentences, FLAGS.tag_schema)
    if is_test:
        test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
        update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):  # 若不存字典（词与词频，标签与整数）文件，则重新创造
        if FLAGS.pre_emb:  # Whether use pre-trained embedding
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]  # 包含三个字典：字与字频的字典，另外两个是对应的字典
            # 根据测试集等扩大字典规模，比如：出现在测试集中但没有出现在训练集中的单词，初始频次设置为0
            # itertools.chain.from_iterable()参考https: // blog.csdn.net / pipisorry / article / details / 45171451
            # 使用copy防止共享内存引起的问题
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(  # 见函数定义时的参数说明
                dico_chars_train.copy(),
                FLAGS.emb_file,   # 预训练词嵌入的向量
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences]))  # 生成迭代序列 参考https://blog.csdn.net/pipisorry/article/details/45171451
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:    # 写到.pkl文件中
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)  # 将对象obj保存到文件file中去。序列化
            # print("tag_to_id",tag_to_id)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)  # 从file中读取一个字符串，并将它重构为原来的python对象。

    # prepare data, get a collection of list containing index
    # prepare_dataset()返回 字符、字符id、根据词的长度，标记为0、1 2、1 2 3、1 2 2 3等、tag的id
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, FLAGS.lower)
    train_manager = BatchManager(train_data, FLAGS.batch_size)  # 见该类的定义说明，将训练数据包装成batch
    print("%i sentences in train" %(len(train_data)))
    print("*** train data example ****")
    print(train_data[0])
    if is_dev:
        dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id, FLAGS.lower)
        dev_manager = BatchManager(dev_data, 100)
        print("%i sentences in dev" % (len(dev_data)))
    if is_test:
        test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower)
        test_manager = BatchManager(test_data, 100)
        print("%i sentences in test" % (len(test_data)))

    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)  # 通过json解析
    else:
        config = config_model(FLAGS, char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)  # 保存配置文件
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)  # 设置日志类的基本参数
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True  # 刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
    steps_per_epoch = train_manager.len_data  # 成员变量

    # 训练模型
    if is_train:
        start = time.time()
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
            logger.info("start training")
            loss = []
            for i in range(FLAGS.max_epoch):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, ""NER loss:{:>9.6f}".format(
                            iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []

                if (i % 7 == 0 and i != 0) or i == FLAGS.max_epoch-1:  # 每7个epoch检查一次模型
                    if is_dev:
                        print("********************dev**************************")
                        best = evaluate(FLAGS, sess, model, "dev", dev_manager, id_to_tag, logger)
                        if best:
                            print("*****************save model**********************")
                            save_model(sess, model, FLAGS.ckpt_path, logger)
            # if is_test:
            #     print("********************test**************************")
            #     logger.info("start testing")
            #     evaluate(FLAGS, sess, model, "test", test_manager, id_to_tag, logger)

            elapsed = (time.time() - start)  # 记录训练时间
            print("Training time used {:.1f} s,samples :{} ".format(elapsed, len(train_data)))

    # 测试
    if is_test:
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
            print("start testing")
            evaluate(FLAGS, sess, model, "test", test_manager, id_to_tag, logger)


def evaluate_line(FLAGS):
    '''读取控制台的句子并进行标注'''
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)  # 从file中读取一个字符串，并将它重构为原来的python对象。
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            line = input("请输入测试句子:")
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            print(result)


def main(_):
    seed_everything()

    flags = tf.app.flags
    flags.DEFINE_boolean("clean", False, "clean train folder")
    flags.DEFINE_boolean("train", True, "Whether train the model")
    flags.DEFINE_boolean("test", True, "Whether test the model")
    flags.DEFINE_boolean("dev", True, "Whether dev the model")
    flags.DEFINE_boolean("test_sample", False, "Whether test sample")

    # configurations for the model
    flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")  # 分词向量，如1 2 3分别表示字位于词的开始中间与结尾
    flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
    flags.DEFINE_integer("lstm_dim", 100, "Num of hidden units in LSTM, or num of filters in IDCNN")  # IDCNN和LSTM可以随时替换
    flags.DEFINE_string("tag_schema", "iobes",
                        "tagging schema iobes or iob")  # 选择使用iobes还是iob标注体系,I表示internal，B表示begin，O表示other

    # configurations for training
    flags.DEFINE_float("clip", 5, "Gradient clip")  # 梯度截斷，避免梯度爆炸
    flags.DEFINE_float("dropout", 0.5, "Dropout rate")
    flags.DEFINE_float("batch_size", 60, "batch size")  # 最初为60
    flags.DEFINE_float("lr", 0.001, "Initial learning rate")
    flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
    flags.DEFINE_boolean("pre_emb", False, "Wither use pre-trained embedding")
    flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")  # 把字符串中的数字都替换成0
    flags.DEFINE_boolean("lower", False, "Wither lower case")

    flags.DEFINE_integer("max_epoch", 1, "maximum training epochs")
    flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")
    flags.DEFINE_string("ckpt_path", "ckpt", "Path to save model")
    flags.DEFINE_string("summary_path", "summary", "Path to store summaries")
    flags.DEFINE_string("log_file", "train.log", "File for log")
    flags.DEFINE_string("map_file", "maps.pkl", "file for maps")  # （词与词频，标签与整数）文件
    flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")
    flags.DEFINE_string("config_file", "config_file", "File for config")
    flags.DEFINE_string("script", "conlleval", "evaluation script")
    flags.DEFINE_string("result_path", "result", "Path for results")
    flags.DEFINE_string("emb_file", r"../data/vec.txt", "Path for pre_trained embedding")
    flags.DEFINE_string("train_file", "../data/cs_nj_184_train4.txt", "Path for train data")
    flags.DEFINE_string("dev_file", "../data/cs_nj_184_dev4.txt", "Path for dev data")
    flags.DEFINE_string("test_file", "../data/cs_nj_184_test4.txt", "Path for test data")

    flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")

    FLAGS = tf.app.flags.FLAGS

    '''必要的限制'''
    assert FLAGS.clip < 5.1, "gradient clip should't be too much"
    assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
    assert FLAGS.lr > 0, "learning rate must larger than zero"
    assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]

    if FLAGS.clean:
        clean(FLAGS)  # 清空数据
    if FLAGS.train or FLAGS.test:
        train_test(FLAGS, FLAGS.train, FLAGS.dev, FLAGS.test)
    if FLAGS.test_sample:
        evaluate_line(FLAGS)


if __name__ == "__main__":
    tf.app.run(main)