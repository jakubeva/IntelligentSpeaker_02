#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from asr_tf.general_function.file_wav import *
from asr_tf.general_function.file_dict import *
from asr_tf.general_function.gen_func import *

# LSTM_CNN
import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import random

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization  # , Flatten
from tensorflow.keras.layers import Lambda, TimeDistributed, Activation, Conv2D, MaxPooling2D  # , Merge
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adadelta, Adam

from asr_tf.read_data import DataSpeech

abspath = ''
ModelName = '251bn'


# NUM_GPU = 2


class ModelSpeech():  # 语音模型类
    def __init__(self, datapath):
        """
        初始化
        默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
        """
        self.MS_OUTPUT_SIZE = 1428  # 神经网络最终输出的每一个字符向量维度的大小
        # self.BATCH_SIZE = BATCH_SIZE # 一次训练的batch
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1600
        self.AUDIO_FEATURE_LENGTH = 200
        self._model, self.base_model = self.CreateModel()

        self.datapath = datapath

    def CreateModel(self):
        """
        定义CNN/LSTM/CTC模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

        参数： \\
        input_shape: tuple，默认值(1600, 200, 1) \\
        output_shape: tuple，默认值(200, 1428)
        """
        # 初始化输入参数 Input
        input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv0')(
            input_data)  # 卷积层1600,200,32
        layer_h1 = BatchNormalization(epsilon=0.0002, name='BN0')(layer_h1)
        layer_h1 = Activation('relu', name='Act0')(layer_h1)

        layer_h2 = Conv2D(32, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv1')(
            layer_h1)  # 卷积层
        layer_h2 = BatchNormalization(epsilon=0.0002, name='BN1')(layer_h2)
        layer_h2 = Activation('relu', name='Act1')(layer_h2)

        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层 800,100,32

        layer_h4 = Conv2D(64, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv2')(
            layer_h3)  # 卷积层800，100，64
        layer_h4 = BatchNormalization(epsilon=0.0002, name='BN2')(layer_h4)
        layer_h4 = Activation('relu', name='Act2')(layer_h4)

        layer_h5 = Conv2D(64, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv3')(
            layer_h4)  # 卷积层
        layer_h5 = BatchNormalization(epsilon=0.0002, name='BN3')(layer_h5)
        layer_h5 = Activation('relu', name='Act3')(layer_h5)

        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层 400，50，128

        layer_h7 = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv4')(
            layer_h6)  # 卷积层400，50，128
        layer_h7 = BatchNormalization(epsilon=0.0002, name='BN4')(layer_h7)
        layer_h7 = Activation('relu', name='Act4')(layer_h7)

        layer_h8 = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv5')(
            layer_h7)  # 卷积层
        layer_h8 = BatchNormalization(epsilon=0.0002, name='BN5')(layer_h8)
        layer_h8 = Activation('relu', name='Act5')(layer_h8)

        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层 200，25，128

        layer_h10 = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv6')(
            layer_h9)  # 卷积层 200，25，128
        layer_h10 = BatchNormalization(epsilon=0.0002, name='BN6')(layer_h10)
        layer_h10 = Activation('relu', name='Act6')(layer_h10)

        layer_h11 = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv7')(
            layer_h10)  # 卷积层
        layer_h11 = BatchNormalization(epsilon=0.0002, name='BN7')(layer_h11)
        layer_h11 = Activation('relu', name='Act7')(layer_h11)

        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层200，25，128

        layer_h13 = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv8')(
            layer_h12)  # 卷积层200，25，128
        layer_h13 = BatchNormalization(epsilon=0.0002, name='BN8')(layer_h13)
        layer_h13 = Activation('relu', name='Act8')(layer_h13)

        layer_h14 = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv9')(
            layer_h13)  # 卷积层
        layer_h14 = BatchNormalization(epsilon=0.0002, name='BN9')(layer_h14)
        layer_h14 = Activation('relu', name='Act9')(layer_h14)

        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)  # 池化层200，25，128

        layer_h16 = Reshape((200, 3200))(layer_h15)  # Reshape层200,3200

        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层

        layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(
            layer_h17)  # 全连接层200,1428
        y_pred = Activation('softmax', name='Activation0')(layer_h18)

        model_base = Model(inputs=input_data, outputs=y_pred)

        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        # # Lambda用来做数据变换，不引入任何新的参数
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

        print('[*提示] 创建模型成功，模型编译成功')
        return model, model_base

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args

        y_pred = y_pred[:, :, :]
        # y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def TrainModel(self, datapath, epoch=2, save_step=10, batch_size=32):
        """
        训练模型
        参数：
            datapath: 数据保存的路径
            epoch: 迭代轮数
            save_step: 每多少步保存一次模型
            filename: 默认保存文件名，不含文件后缀名
        """
        data = DataSpeech(datapath, 'train')
        num_data = data.GetDataNum()  # 获取数据的数量
        yielddatas = data.data_genetator(batch_size, self.AUDIO_LENGTH)
        for epoch in range(epoch):  # 迭代轮数
            n_step = 0  # 迭代数据数
            while True:
                try:
                    print('[message] epoch %d . Have train datas %d+' % (epoch, n_step * save_step))
                    if n_step * save_step == num_data:
                        break
                    # self._model.fit_generator(yielddatas, save_step, nb_worker=2)
                    tf_callback = tf.keras.callbacks.TensorBoard(log_dir="data/tensorboard")
                    self._model.fit_generator(yielddatas, save_step, callbacks=[tf_callback])
                    n_step += 1
                except StopIteration:
                    print('[error] generator error. please check data format.')
                    break

                self.SaveModel(comment='_e_' + str(epoch) + '_step_' + str(n_step * save_step))
                # self.TestModel(self.datapath, str_dataset='train', data_count=4)
                # self.TestModel(self.datapath, str_dataset='dev', data_count=4)
        print('end')

    def LoadModel(self, filename=abspath + 'model_speech/m' + ModelName + '/speech_model' + ModelName + '.model'):
        self._model.load_weights(filename)

    def SaveModel(self, filename=abspath + 'model_speech/m' + ModelName + '/speech_model' + ModelName, comment=''):
        self._model.save_weights(filename + comment + '.h5')
        self.base_model.save_weights(filename + comment + '.base.h5')
        # 需要安装 hdf5 模块
        # self._model.save(filename + comment + '.h5')
        # self.base_model.save(filename + comment + '.base.h5')
        f = open('step' + ModelName + '.txt', 'w')
        f.write(filename + comment)
        f.close()

    def TestModel(self, str_dataset='dev', data_count=32, out_report=False, show_ratio=True,
                  io_step_print=100):
        """
        测试检验模型效果
        io_step_print
            为了减少测试时标准输出的io开销，可以通过调整这个参数来实现
        io_step_file
            为了减少测试时文件读写的io开销，可以通过调整这个参数来实现
        """
        data = DataSpeech(self.datapath, str_dataset)
        # data.LoadDataList(str_dataset)
        num_data = data.GetDataNum()  # 获取数据的数量
        if data_count <= 0 or data_count > num_data:  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data
        try:
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数
            words_num = 0
            word_error_num = 0

            nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            if out_report:
                txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8')  # 打开文件并读入
                txt_obj.truncate((data_count + 1) * 300)  # 预先分配一定数量的磁盘空间，避免后期在硬盘中文件存储位置频繁移动，以防写入速度越来越慢
                txt_obj.seek(0)  # 从文件首开始

            txt = '测试报告\n模型编号 ' + ModelName + '\n\n'
            for i in range(data_count):
                data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据

                # 数据格式出错处理 开始
                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                while data_input.shape[0] > self.AUDIO_LENGTH:
                    print('*[Error]', 'wave data lenghth of num', (ran_num + i) % num_data, 'is too long.',
                          '\n A Exception raise when test Speech Model.')
                    i += 1
                    continue
                # 数据格式出错处理 结束

                pre = self.Predict(data_input, data_input.shape[0] // 8)

                words_n = data_labels.shape[0]  # 获取每个句子的字数
                words_num += words_n  # 把句子的总字数加上
                edit_distance = GetEditDistance(data_labels, pre)  # 获取编辑距离
                if edit_distance <= words_n:  # 当编辑距离小于等于句子字数时
                    word_error_num += edit_distance  # 使用编辑距离作为错误字数
                else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                    word_error_num += words_n  # 就直接加句子本来的总字数就好了

                if i % io_step_print == 0 and show_ratio:
                    print('[ASRT Info] Testing: ', i, '/', data_count)

                txt = ''
                if out_report:
                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(pre) + '\n'
                    txt += '\n'
                    txt_obj.write(txt)

                i += 1
            print('*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率：', word_error_num / words_num * 100, '%')
            if out_report:
                txt += '*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率： ' + str(
                    word_error_num / words_num * 100) + ' %'
                txt_obj.write(txt)
                txt_obj.truncate()  # 去除文件末尾剩余未使用的空白存储字节
                txt_obj.close()

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

    def Predict(self, data_input, input_len):
        """
        预测结果
        返回语音识别后的拼音符号列表
        """

        batch_size = 1
        in_len = np.zeros((batch_size), dtype=np.int32)
        in_len[0] = input_len
        x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float64)
        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input
        base_pred = self.base_model.predict(x=x_in)
        # base_pred = base_pred[:, :, :]
        # base_pred =base_pred[:, 2:, :]

        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)
        if tf.__version__[0:2] == '1.':
            r1 = r[0][0].eval(session=tf.compat.v1.Session())
        else:
            r1 = r[0][0].numpy()
        # tf.compat.v1.reset_default_graph()
        # return r1[0]
        p = 0
        while p < len(r1[0]) and r1[0][p] != -1:
            p += 1
        return r1[0][0:p]

        # from ctc_decoder import decode
        # labels, score = decode(base_pred[0])
        # return labels, score

    def RecognizeSpeech(self, wavsignal, fs):
        """
        最终做语音识别用的函数，识别一个wav序列的语音
        """
        data_input = GetFrequencyFeature3(wavsignal, fs)
        input_length = len(data_input)
        input_length = input_length // 8

        data_input = np.array(data_input, dtype=np.float64)
        # print(data_input,data_input.shape)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        # r1 = self.Predict(data_input, input_length)
        r1 = self.Predict(data_input, input_length)
        list_symbol_dic = GetSymbolList(self.datapath)  # 获取拼音列表

        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])

        return r_str

    def RecognizeSpeech_FromFile(self, filename):
        """
        最终做语音识别用的函数，识别指定文件名的语音
        """
        wavsignal, fs = read_wav_data(filename)
        r = self.RecognizeSpeech(wavsignal, fs)
        return r

    @property
    def model(self):
        return self._model
