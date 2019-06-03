# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 2:18 PM
# @Author  : zengyilin
# @Email   : yilinxd@163.com
# @File    : train.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os
import copy
import tensorflow as tf


from text_utils import text_read
from char_rnn import char_rnn
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义参数
tf.flags.DEFINE_integer("batch_size",128,"batch_size")
tf.flags.DEFINE_integer("num_epochs",100,"num_epochs")
tf.flags.DEFINE_integer("save_every_steps",1000,"save_every_steps")
tf.flags.DEFINE_integer("evaluate_every_steps",200,"evaluate_every_steps")

#tf.flags.DEFINE_integer('maxlength', 100, 'maxlength')
tf.flags.DEFINE_integer('vocab_size', 0, 'vocab_size')
tf.flags.DEFINE_integer('embed_size', 128, "embed_size")
tf.flags.DEFINE_integer('lstm_cell_size', 128, 'lstm_cell_size')
tf.flags.DEFINE_integer('lstm_layer_size', 5, 'lstm_layer_size')

tf.flags.DEFINE_float("learning_rate",0.1,"learning_rate")
tf.flags.DEFINE_float("decay_rate",0.9,"decay_rate")
tf.flags.DEFINE_float("decay_steps",1000,"decay_rate")
tf.flags.DEFINE_float("keep_prob",0.6,"keep_prob")

tf.flags.DEFINE_string('text_path', "./data/poetry.txt", 'corpus')

FLAGS = tf.flags.FLAGS

data_helper = text_read(FLAGS)


datax,datay,lengths = data_helper.get_data()

trainx ,testx , trainy ,testy ,trainlen ,testlen = train_test_split(datax,datay,lengths,test_size=0.1,random_state=10)

train_data = data_helper.batch_iter(trainx, trainy, trainlen, FLAGS.batch_size, FLAGS.num_epochs)

FLAGS.vocab_size = len(data_helper.vocab.vocabulary_._mapping)
print("vocab_size {}".format(FLAGS.vocab_size))

# Output files directory
import time
timestamp = str(int(time.time()))
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Save parameters to file
import pickle as pkl
params = FLAGS.flag_values_dict()
params_file = open(os.path.join(outdir, 'params.pkl'), 'wb')
pkl.dump(params, params_file, True)
params_file.close()

with tf.Graph().as_default():
    with tf.Session() as sess:

        model = char_rnn(FLAGS)

        # Train procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Learning rate decay
        starter_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)

        # # optimizer
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # grads_and_vars = optimizer.compute_gradients(model.loss)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)



        # # optimizer
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), 10)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op=  optimizer.apply_gradients(zip(grads, tvars),global_step=global_step)


        # Summaries
        loss_summary = tf.summary.scalar('Loss', model.loss)
        accuracy_summary = tf.summary.scalar('Accuracy', model.accuracy)

        # Train summary
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summary
        valid_summary_op = tf.summary.merge_all()
        valid_summary_dir = os.path.join(outdir, 'summaries', 'valid')
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())

        import datetime
        def run_step(input_data, is_training=True):
            input_x, input_y, sequence_length = input_data


            fetches = {'step': global_step,
                       'loss': model.loss,
                       'accuracy': model.accuracy,
                       'learning_rate': learning_rate}
            feed_dict = {model.inputx: input_x,
                         model.outputy: input_y}


            feed_dict[model.batch_size] = len(input_x)
            feed_dict[model.sequence_length] = sequence_length

            if is_training:
                fetches['train_op'] = train_op
                fetches['summaries'] = train_summary_op
                feed_dict[model.keep_prob] = FLAGS.keep_prob
            else:
                fetches['summaries'] = valid_summary_op
                feed_dict[model.keep_prob] = 1.0

            vars = sess.run(fetches, feed_dict)
            step = vars['step']
            cost = vars['loss']
            accuracy = vars['accuracy']
            summaries = vars['summaries']
            l_rate = vars["learning_rate"]

            # Write summaries to file
            if is_training:
                train_summary_writer.add_summary(summaries, step)
            else:
                valid_summary_writer.add_summary(summaries, step)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, accuracy: {:g}, learning_rate :{}".format(time_str, step, cost, accuracy,
                                                                                       l_rate))
            return accuracy

        def watch_demo(words):

            c  = copy.deepcopy(words)

            with open("temp.txt","w",encoding="utf-8") as f:

                for i in range(20):


                    a = data_helper.words2ids(c)
                    l = len(a)
                    demo_fatches = {
                        "pred":model.predictions
                    }

                    demo_feed_dict = {
                        model.inputx:np.array([a])
                    }
                    demo_feed_dict[model.batch_size] = len(a)
                    demo_feed_dict[model.sequence_length] = np.array([l])
                    demo_feed_dict[model.keep_prob] = 1.0

                    demo_vars = sess.run(demo_fatches, demo_feed_dict)

                    pred_words = data_helper.ids2words(demo_vars["pred"])
                    s = "".join(pred_words)
                    c = s
                    print("demo -- {}".format(s))
                    f.write(s+"\n")
                    f.flush()

        print('Start training ...')
        for train_input in train_data:
            run_step(train_input, is_training=True)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every_steps == 0:
                print('\nValidation')
                run_step((testx, testy, testlen), is_training=False)
                print('')

                watch_demo("雨中黄叶树，")


            if current_step % FLAGS.save_every_steps == 0:
                save_path = saver.save(sess, os.path.join(outdir, 'model/clf'), current_step)



        print('\nAll the files have been saved to {}\n'.format(outdir))






