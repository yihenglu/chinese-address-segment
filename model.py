# encoding = utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from utils import result_to_json
from data_utils import create_input, iobes_iob,iob_iobes


class Model(object):
    #初始化模型参数
    def __init__(self, config):

        self.config = config
        
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]  # 字符潜入维度
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]  # 样本中总字数
        self.num_segs = 4  # 表示分词信息 0 1 2 3

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()  # Xavier初始化器得作用，就是在初始化深度学习网络得时候让权重不大不小。


        # add placeholders for the model
        # tf.placeholder好像可以看成普通函数中的虚参
        # char_inputs: one-hot encoding of sentence
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))  # 为什么要用abs判断？？？输入的char不都是大于0吗？shape=(?,?)
        length = tf.reduce_sum(used, reduction_indices=1)  # 降维求和 shape=(?,)
        self.lengths = tf.cast(length, tf.int32)  # 为什么要转化为整数？？？
        self.batch_size = tf.shape(self.char_inputs)[0]  # 每一个batch含有的样本数量
        self.num_steps = tf.shape(self.char_inputs)[-1]  # 每一个样本的维数
        
        
        #Add model type by crownpku bilstm or idcnn
        self.model_type = config['model_type']
        #parameters for idcnn
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]  # 列表里面存字典
        self.filter_width = 3
        self.num_filter = self.lstm_dim  # ？？？表示LSTM神经元的个数还是神经元的维度，应该是神经元的个数，即时间步的数量
        self.embedding_dim = self.char_dim + self.seg_dim  # 字维度加上分词相关信息的维度（seg有1/2/3），和LSTM神经元的维度成正比
        self.repeat_times = 4  # 为什么重复4次？？？难道LSTM有4层（深层神经网络）
        self.cnn_output_width = 0  # 输出0维？？？
        
        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        if self.model_type == 'bilstm':
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)  # biLSTM_layer()还没有实现，后期可以自己尝试实现

            # logits for tags
            self.logits = self.project_layer_bilstm(model_outputs)  # 还没看实现
        
        elif self.model_type == 'idcnn':
            # apply dropout before feed to idcnn layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # ldcnn layer
            model_outputs = self.IDCNN_layer(model_inputs)  # Tensor("idcnn/Reshape:0", shape=(?, 400), dtype=float32)

            # logits for tags
            self.logits = self.project_layer_idcnn(model_outputs)  # Tensor("project/Reshape:0", shape=(?, ?, 23), dtype=float32)
        
        else:
            raise KeyError

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)  # Tensor("crf_loss/Mean:0", shape=(), dtype=float32)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)  # <class 'list'>: [(<tensorflow.python.framework.ops.IndexedSlices object at 0x0000016DA7CB4EB8>, <tf.Variable 'char_embedding/char_embedding:0' shape=(2677, 100) dtype=float32_ref>), (<tensorflow.python.framework.ops.IndexedSlices object at 0x0000016DA7CD5240>, <tf.Variable 'char_embedding/seg_embedding/seg_embedding:0' shape=(4, 20) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/idcnn/init_layer_grad/tuple/control_dependency_1:0' shape=(1, 3, 120, 100) dtype=float32>, <tf.Variable 'idcnn/idcnn_filter:0' shape=(1, 3, 120, 100) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/AddN_15:0' shape=(1, 3, 100, 100) dtype=float32>, <tf.Variable 'idcnn/atrous-conv-layer-0/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/AddN_14:0' shape=(100,) dtype=float32>, <tf.Variable 'idcnn/atrous-conv-layer-0/filterB:0' shape=(100,) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/AddN_13:0' shape=(1, 3, 100, 100) dtype=float32>, <tf.Variable 'idcnn/atrous-conv-layer-1/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/AddN_12:0' shape=(100,) dtype=float32>, <tf.Variable 'idcnn/atrous-conv-layer-1/filterB:0' shape=(100,) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/AddN_11:0' shape=(1, 3, 100, 100) dtype=float32>, <tf.Variable 'idcnn/atrous-conv-layer-2/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/AddN_10:0' shape=(100,) dtype=float32>, <tf.Variable 'idcnn/atrous-conv-layer-2/filterB:0' shape=(100,) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/project/logits/xw_plus_b/MatMul_grad/tuple/control_dependency_1:0' shape=(400, 23) dtype=float32>, <tf.Variable 'project/logits/W:0' shape=(400, 23) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/project/logits/xw_plus_b_grad/tuple/control_dependency_1:0' shape=(23,) dtype=float32>, <tf.Variable 'project/logits/b:0' shape=(23,) dtype=float32_ref>), (<tf.Tensor 'optimizer/gradients/AddN_5:0' shape=(24, 24) dtype=float32>, <tf.Variable 'crf_loss/transitions:0' shape=(24, 24) dtype=float32_ref>)]
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]  # tf.clip_by_value输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """
        #高:3 血:22 糖:23 和:24 高:3 血:22 压:25 char_inputs=[3,22,23,24,3,22,25]
        #高血糖 和 高血压 seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3]  seg_inputs=[1,2,3,0,1,2,3]
        embedding = []
        self.char_inputs_test=char_inputs
        self.seg_inputs_test=seg_inputs
        with tf.variable_scope("char_embedding" if not name else name):  # 注释掉", tf.device('/cpu')"，用环境变量来设置GPU
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            # 输入char_inputs='常' 对应的字典的索引/编号/value为：8
            # self.char_lookup=[2677*100]的向量，char_inputs字对应在字典的索引/编号/key=[1]   好像共有2677种字符   vec.txt中共3260行
            # embedding_lookup()根据input_ids中的id，寻找embeddings中的第id行。如input_ids=[1,3,5]，则找出embeddings中第1，3，5行，组成一个tensor返回。
            # 同时向量是可训练的
            # 参考 https://blog.csdn.net/laolu1573/article/details/77170407
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))  # <class 'list'>: [<tf.Tensor 'char_embedding/embedding_lookup:0' shape=(?, ?, 100) dtype=float32>]
            # self.embedding1.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"):  # 注释掉", tf.device('/cpu')"
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        #shape=[4*20]
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))  # <class 'list'>: [<tf.Tensor 'char_embedding/embedding_lookup:0' shape=(?, ?, 100) dtype=float32>, <tf.Tensor 'char_embedding/seg_embedding/embedding_lookup:0' shape=(?, ?, 20) dtype=float32>]
            embed = tf.concat(embedding, axis=-1)  # Tensor("char_embedding/concat:0", shape=(?, ?, 120), dtype=float32)
        self.embed_test=embed
        self.embedding_test=embedding
        return embed

    
    #IDCNN layer 
    def IDCNN_layer(self, model_inputs, name=None):
        """

        :param idcnn_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, cnn_output_width]
        """
        # tf.expand_dims会向tensor中插入一个维度，插入位置就是参数代表的位置（维度从0开始）。
        model_inputs = tf.expand_dims(model_inputs, 1)  # 为什么要插入一个维度？？？好像是tf的数据格式要求 Tensor("ExpandDims:0", shape=(?, 1, ?, 120), dtype=float32)
        self.model_inputs_test=model_inputs
        reuse = False
        if self.dropout == 1.0:  # 测试阶段
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            #shape=[1*3*120*100]
            shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter]
            print('idcnn model filter weights\'s shape:')
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            
            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]  # ？？？
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer",use_cudnn_on_gpu=True)  #是否使用cudnn加速，默认为true。Tensor("idcnn/init_layer:0", shape=(?, 1, ?, 100), dtype=float32)
            self.layerInput_test=layerInput
            finalOutFromLayers = []
            
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    #1,1,2
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):  # 表示不同的卷积层
                        #w 卷积核的高度，卷积核的宽度，通道数，卷积核个数
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())  # <tf.Variable 'idcnn/atrous-conv-layer-0/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>
                        if j==1 and i==1:
                            self.w_test_1=w
                        if j==2 and i==1:
                            self.w_test_2=w                            
                        b = tf.get_variable("filterB", shape=[self.num_filter])
#tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）
    #除去name参数用以指定该操作的name，与方法有关的一共四个参数：                  
    #value： 
    #指需要做卷积的输入图像，要求是一个4维Tensor，具有[batch, height, width, channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数] 
    #filters： 
    #相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，同理这里第三维channels，就是参数value的第四维
    #rate： 
    #要求是一个int型的正数，正常的卷积操作应该会有stride（即卷积核的滑动步长），但是空洞卷积是没有stride参数的，
    #这一点尤其要注意。取而代之，它使用了新的rate参数，那么rate参数有什么用呢，它定义为我们在输入
    #图像上卷积时的采样间隔，你可以理解为卷积核当中穿插了（rate-1）数量的“0”，
    #把原来的卷积核插出了很多“洞洞”，这样做卷积时就相当于对原图像的采样间隔变大了。
    #具体怎么插得，可以看后面更加详细的描述。此时我们很容易得出rate=1时，就没有0插入，
    #此时这个函数就变成了普通卷积。  
    #padding： 
    #string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同边缘填充方式。
    #ok，完了，到这就没有参数了，或许有的小伙伴会问那“stride”参数呢。其实这个函数已经默认了stride=1，也就是滑动步长无法改变，固定为1。
    #结果返回一个Tensor，填充方式为“VALID”时，返回[batch,height-2*(filter_width-1),width-2*(filter_height-1),out_channels]的Tensor，填充方式为“SAME”时，返回[batch, height, width, out_channels]的Tensor，这个结果怎么得出来的？先不急，我们通过一段程序形象的演示一下空洞卷积。                        
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        self.conv_test=conv  # Tensor("idcnn/atrous-conv-layer-0/convolution:0", shape=(?, 1, ?, 100), dtype=float32)
                        conv = tf.nn.bias_add(conv, b)

                        # 调试，这里只能输出shape
                        # print('w=\n', w)
                        # print('b=\n', b)

                        conv = tf.nn.relu(conv)
                        if isLast:  # 为什么要判断这个？？？
                            finalOutFromLayers.append(conv)  # <class 'list'>: [<tf.Tensor 'idcnn/atrous-conv-layer-2/Relu:0' shape=(?, ?, ?, 100) dtype=float32>]
                            totalWidthForLastDim += self.num_filter  # 100
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)  # 这是合并4个islast的结果吗？？？  Tensor("idcnn/concat:0", shape=(?, ?, ?, 400), dtype=float32)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, 1.0)  # Tensor("idcnn/dropout/mul:0", shape=(?, ?, ?, 400), dtype=float32)
            #Removes dimensions of size 1 from the shape of a tensor. 
                #从tensor中删除所有大小是1的维度
            
                #Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed. If you don’t want to remove all size 1 dimensions, you can remove specific size 1 dimensions by specifying squeeze_dims. 
            
                #给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。 如果不想删除所有尺寸1尺寸，可以通过指定squeeze_dims来删除特定尺寸1尺寸。
            finalOut = tf.squeeze(finalOut, [1])  # Tensor("idcnn/Squeeze:0", shape=(?, ?, 400), dtype=float32)
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])  # Tensor("idcnn/Reshape:0", shape=(?, 400), dtype=float32)
            self.cnn_output_width = totalWidthForLastDim  # 400
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])

                # 调试，这里只能输出形状信息
                # print('w=\n',W)
                # print('b=\n',b)

                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])
    
    #Project layer for idcnn by crownpku
    #Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param idcnn_outputs: Tensor("idcnn/Reshape:0", shape=(?, 400), dtype=float32)
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)  # <tf.Variable 'project/logits/W:0' shape=(400, 23) dtype=float32_ref>

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)  # Tensor("project/logits/xw_plus_b:0", shape=(?, 23), dtype=float32)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])  # Tensor("project/Reshape:0", shape=(?, ?, 23), dtype=float32)

    def loss_layer(self, project_logits, lengths, name=None):  # Tensor("project/Reshape:0", shape=(?, ?, num_tags), dtype=float32)  Tensor("Sum:0", shape=(?,), dtype=int32)
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # 下面的一顿转换没搞懂？？？ 目的好像是满足某种格式
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)  # Tensor("crf_loss/concat:0", shape=(?, 1, 24), dtype=float32)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)  # Tensor("crf_loss/mul_1:0", shape=(?, ?, 1), dtype=float32)？？？
            logits = tf.concat([project_logits, pad_logits], axis=-1)  # Tensor("crf_loss/concat_1:0", shape=(?, ?, 24), dtype=float32)
            logits = tf.concat([start_logits, logits], axis=1)  # Tensor("crf_loss/concat_2:0", shape=(?, ?, 24), dtype=float32)？？？
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)  # Tensor("crf_loss/concat_3:0", shape=(?, ?), dtype=int32)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)  # 概率转移矩阵
            #crf_log_likelihood在一个条件随机场里面计算标签序列的log-likelihood
            #inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
            #一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入. 
            #tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签. 
            #sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度. 
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵    
            #log_likelihood: 标量,log-likelihood 
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            # 返回的第二个值是形状为[num_tags, num_tags] 的转移矩阵
            # self.trans：一个[num_tags,num_tags]转换矩阵，即转移矩阵
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch  # 
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss,_,char_lookup_out,seg_lookup_out,char_inputs_test,seg_inputs_test,embed_test,embedding_test,\
                model_inputs_test,layerInput_test,conv_test,w_test_1,w_test_2,char_inputs_test= sess.run(
                [self.global_step, self.loss, self.train_op,self.char_lookup,self.seg_lookup,self.char_inputs_test,self.seg_inputs_test,\
                 self.embed_test,self.embedding_test,self.model_inputs_test,self.layerInput_test,self.conv_test,self.w_test_1,self.w_test_2,self.char_inputs],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits


    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence  # batch中样本的长度好像不同？？？
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)  # 应该是维特比算法？？？

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()  # t.eval()等效于sess.run(t).
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                #gold = iob_iobes([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                #pred = iob_iobes([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])                
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results


    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(session=sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
