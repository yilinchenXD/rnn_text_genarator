# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 1:33 AM
# @Author  : zengyilin
# @Email   : yilinxd@163.com
# @File    : text_utils.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os
import copy
import tensorflow as tf
from tensorflow.contrib import learn


class text_read():

    def __init__(self, config):

        self.config = config

        path = config.text_path

        documents = []
        with open(path, 'r', encoding="utf-8") as f:
            line = f.readline()
            while line:
                documents.append(line.strip())
                line = f.readline()

        def tokenizer_v1(docs):
            for doc in docs:
                #yield " ".join(doc)
                yield doc

        vocab = learn.preprocessing.VocabularyProcessor(100, 0, tokenizer_fn=tokenizer_v1)
        vocab.fit(raw_documents=documents)

        self.vocab = vocab
        #vocab.vocabulary_._mapping

    def words2ids(self, words):

         result = []
         for i in words:
             if i in self.vocab.vocabulary_._mapping:
                 result.append(self.vocab.vocabulary_._mapping[i])
             else :
                 result.append(0)

         return result

    def ids2words(self,ids):

        result = []
        for i in ids:

            if i <=  len(self.vocab.vocabulary_._reverse_mapping):
                result.append(self.vocab.vocabulary_.reverse(i))
            else:
                result.append("<UNK>")

        return result

    def get_encoder_data(self):

        path = self.config.text_path

        documents = []
        print("get sparse ont-hot data")
        with open(path, 'r', encoding="utf-8") as f:
            line = f.readline()
            while line:
                documents.append(self.words2ids(line.strip()))
                line = f.readline()
                self.words2ids(line)
        return documents

    def get_data(self):
        documents = self.get_encoder_data()

        data = np.array(documents)

        x = data[:-1]
        y = data[1:]

        assert  y.shape == x.shape
        lens = [len(i )for i in y]

        return x,y,lens

    def batch_iter(self,datax,datay,lens,batch_size ,epochs):

        assert len(datax) == len(datay) == len(lens)

        data_size = len(datax)
        epoch_length = data_size // batch_size

        for i in range(epochs):
            for j in range(epoch_length):
                start_index = i * batch_size
                end_index = start_index + batch_size

                xdata = datax[start_index: end_index]
                ydata = datay[start_index: end_index]
                sequence_length = lens[start_index: end_index]
                yield xdata, ydata, np.array(sequence_length)









if __name__ == "__main__":
    tf.flags.DEFINE_string('text_path', "./data/poetry.txt", 'corpus')

    config = tf.app.flags.FLAGS

    mod = text_read(config)
    word_dict = mod.vocab.vocabulary_._mapping
    size = len(word_dict)
    #print(word_dict )
    print(size )

    w2id = mod.words2ids("我们")
    print(w2id)

    id2w = mod.ids2words([1,2,3,4,20000])
    print(id2w)

    mod.get_data()