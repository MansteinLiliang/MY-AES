#pylint: skip-file
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import time
import numpy as np
import dataset
from model import RNN
# use_gpu(-1) # -1:cpu; 0,1,2,..: gpu


def define_tensor_size(prompt_id):
    tensor_ranges = {
        1: (50, 50),
        2: (50, 50),
        3: (50, 30),
        4: (50, 30),
        5: (50, 30),
        6: (50, 30),
        7: (50, 50),
        8: (50, 100)
    }
    return tensor_ranges[prompt_id]
e = 0.01
lr = 0.1
drop_rate = 0.
prompt_id = 1
vocab_size = 0  # 0 is define to automated infer vocab-size
sent_len, doc_len = define_tensor_size(prompt_id)  # sent_len is batch_size of tensor
hidden_size = [500]
word_embedding_size = 200
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "rmsprop"
train_path, dev_path, test_path = './data/fold_0/train.tsv', './data/fold_0/dev.tsv', './data/fold_0/test.tsv'
(train_x, train_masks, train_y), (dev_x, dev_masks, dev_y), (test_x, test_masks, test_y), vocab, vocab_size =\
    dataset.get_data((train_path, dev_path, test_path), prompt_id, vocab_size, doc_len, sent_len)

print "#word size = ", vocab_size

print "compiling..."
model = RNN(vocab_size, word_embedding_size, hidden_size, cell, optimizer, drop_rate, doc_len)

print "training..."
train_batch = dataset.batch_generator(train_x, train_masks, train_y)
start = time.time()
for i in xrange(2000):
    in_start = time.time()
    X, mask, y = train_batch.next()
    cost, true_cost, pred = model.train(X.astype("int32"), np.asarray(mask, dtype=theano.config.floatX), lr, y)
    # print i, g_error, (batch_id + 1), "/", len(data_xy), cost
    in_time = time.time() - in_start
    print "Iter = " + str(i) + ", Error = " + str(true_cost) + ", Time = " + str(in_time)
    print 'ytrue = ' + str(y) +", ypred = " + str(pred)
    # if cost <= e:
    #     break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/hed.model", model)

