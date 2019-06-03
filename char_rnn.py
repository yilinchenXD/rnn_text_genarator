# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 12:51 PM
# @Author  : zengyilin
# @Email   : yilinxd@163.com
# @File    : char_rnn.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os
import copy
import tensorflow as tf


class char_rnn():

    def __init__(self, config):
        # 一大堆参数
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.lstm_layer_size = config.lstm_layer_size
        self.lstm_cell_size = config.lstm_cell_size


        # placeholder
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.inputx = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_sequences")
        self.outputy = tf.placeholder(shape=[None, None], dtype=tf.int32, name="output_sequences")
        self.sequence_length = tf.placeholder(shape=[None], dtype=tf.int32, name='sequance_length')

        # word embeding
        with tf.name_scope("word_embed"):
            w = tf.Variable(name="embedding",
                            initial_value=tf.random.uniform([self.vocab_size, self.embed_size], -1, 1))
            embed = tf.nn.embedding_lookup(w, self.inputx)
            #embed = tf.nn.dropout(embed, keep_prob=self.keep_prob)

            #cehck
            self.check = []
            #self.check.append(embed)

        with tf.name_scope("lstm"):
            def make_lstm(hidden_size):
                cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,
                                                    forget_bias=1.0,
                                                    state_is_tuple=True,
                                                    )
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                return cell

            # 多层lstm叠加
            stack_cell = tf.contrib.rnn.MultiRNNCell(
                [make_lstm(self.lstm_cell_size) for i in range(self.lstm_layer_size)], state_is_tuple=True)

            outputs, last_states = tf.nn.dynamic_rnn(cell=stack_cell, dtype=tf.float32,
                                                     sequence_length=self.sequence_length,
                                                     inputs=embed,
                                                     # initial_state=init_state
                                                     )
            #test
            #self.check.append(outputs)

        with tf.name_scope("softmax"):
           seq_output = tf.concat(outputs, 1)
           seq_output = tf.reshape(seq_output, [-1, self.lstm_cell_size])

           # test
           #self.check.append(seq_output)

           softmax_w = tf.Variable(tf.truncated_normal([self.lstm_cell_size, self.vocab_size], stddev=0.1))
           softmax_b = tf.Variable(tf.zeros(self.vocab_size))

           self.logits = tf.matmul(seq_output, softmax_w) + softmax_b

           proba_prediction = tf.nn.softmax(self.logits, name='predictions')
           self.predictions = tf.argmax(proba_prediction, 1, name='predictions')

           #test
           #self.check.append(self.logits)
           self.check.append(self.predictions)


        with tf.name_scope("loss"):

            self.y_reshaped = tf.reshape(self.outputy, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_reshaped )

            self.loss = tf.reduce_mean(loss)
            # test
            #self.check.append(y_reshaped)
            #self.check.append(loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.cast(self.predictions, dtype=tf.int32), self.y_reshaped)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

            #test
            self.check.append(self.accuracy)




if __name__ == "__main__":
    inputx = np.array([[1, 2, 3, 4, 5,9], [3, 4, 5, 6, 7, 0],[3, 4, 5, 6, 7, 0]])
    outputy= np.array([[2, 3, 4, 5, 6,0], [4, 5, 6, 7, 0, 8],[4, 5, 6, 7, 0, 8]])
    outputy = inputx
    #inputx = [np.array([1, 2, 3, 4, 5]), np.array([3, 4, 5, 6, 7, 0])]
    #outputy = [np.array([2, 3, 4, 5, 6]), np.array([4, 5, 6, 7, 0, 8])]
    sequence_length = np.array([6, 6,6])
    batch_size = 3
    keep_prob = 1
    maxlength = 10
    vocab_size = 20
    embed_size = 5
    num_filters = 1
    lstm_layer_size = 1
    lstm_cell_size = 2
    #regula_lambd = 0.1
    #num_classes = 2

    #tf.flags.DEFINE_integer('maxlength', maxlength, 'maxlength')
    tf.flags.DEFINE_integer('vocab_size', vocab_size, 'vocab_size')
    tf.flags.DEFINE_integer('embed_size', embed_size, "embed_size")
    tf.flags.DEFINE_integer('num_filters', num_filters, 'num_filters')
    #tf.flags.DEFINE_integer('num_classes', num_classes, 'Number of classes')
    tf.flags.DEFINE_integer('lstm_cell_size', lstm_cell_size, 'lstm_cell_size')
    tf.flags.DEFINE_integer('lstm_layer_size', lstm_layer_size, 'lstm_layer_size')
    #tf.flags.DEFINE_float('regula_lambd', regula_lambd, 'regula_lambd')

    config = tf.app.flags.FLAGS

    model = char_rnn(config)

    with tf.Session() as sess:


        feeddict = {
            model.inputx: inputx,
            model.outputy: outputy,
            model.batch_size: batch_size,
            model.keep_prob: keep_prob,
            model.sequence_length: sequence_length
        }


        fetches = {}
        for i, v in enumerate(model.check):
            fetches[str(i)] = v

        sess.run(tf.global_variables_initializer())
        vars = sess.run(fetches, feed_dict=feeddict)

        for v in vars.values():
            print(v.shape)

