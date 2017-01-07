#pylint: skip-file
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import time
import numpy as np
import dataset
from model import RNN
import cPickle as pkl
from w2vEmbReader import W2VEmbReader
from asap_evaluator import Evaluator
from mylayers.layer_utils import floatX
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


#   configuration
e = 0.01
lr = 0.001
drop_rate = 0.
prompt_id = 1
vocab_size = 4000  # 0 is define to automated infer vocab-size
sent_len, doc_len = define_tensor_size(prompt_id)  # sent_len is batch_size of tensor
doc_num = 32  # defining the doc batch_size to accelerate
hidden_size = [500]
word_embedding_size = 50
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "rmsprop"
train_path, dev_path, test_path = './data/fold_0/train.tsv', './data/fold_0/dev.tsv', './data/fold_0/test.tsv'
(train_x, train_masks, train_y_org), (dev_x, dev_masks, dev_y_org), (test_x, test_masks, test_y_org), vocab, vocab_size =\
    dataset.get_data((train_path, dev_path, test_path), prompt_id, vocab_size, doc_len, sent_len)

train_y = dataset.get_model_friendly_scores(train_y_org, prompt_id)
dev_y = dataset.get_model_friendly_scores(dev_y_org, prompt_id)
test_y = dataset.get_model_friendly_scores(test_y_org, prompt_id)

print 'dev_y_org as integer...'
assert len(vocab) == vocab_size
print "#word size = ", vocab_size

print "compiling..."
# TODO We should try different word-embedding
# word-embedding1
# emb_path = "./data/En_vectors.txt"
# emb_reader = W2VEmbReader(emb_path, emb_dim=word_embedding_size)
# U = np.zeros(shape=(vocab_size, word_embedding_size), dtype=theano.config.floatX)
# U = emb_reader.get_emb_matrix_given_vocab(vocab, U)

# word-embedding2
U = floatX(np.random.uniform(-0.05, 0.05, size=(vocab_size, word_embedding_size)))
U[0] = np.zeros((word_embedding_size,), dtype=theano.config.floatX)

# word-embedding3: zero-embeddings
# U = np.zeros(shape=(vocab_size, word_embedding_size), dtype=theano.config.floatX)


'''Loading google word2vec
# word_vectors = pkl.load(open('./data/word_vectors.pk.'))
# for word in vocab:d
#     U[vocab[word]] = word_vectors[word]
'''

model = RNN(U, vocab_size, word_embedding_size, hidden_size, cell, optimizer, drop_rate, doc_len)
#   Original dev_y and Original test_y should be given as integer
#   dev_y_org is like[8,10,2,10,...] not in range(0,1.0)
#   dev_y is in range(0, 1.0)
evl = Evaluator(dataset, prompt_id, 'None', dev_y_org.astype('int32'), test_y_org.astype('int32'))
print "training..."
train_batch = dataset.train_batch_generator(train_x, train_masks, train_y, doc_num)
start = time.time()
evl_epoch = 0


for i in xrange(300):
    epoch, X, mask, y = train_batch.next()
    if (i+1) % 10 == 0:
        evl_epoch += 1
        print "Starting evaluation: " + str(evl_epoch) + " time"
        in_start = time.time()
        evl.evaluate(dev_x, dev_masks, dev_y, test_x, test_masks, test_y, model, evl_epoch)
        in_time = time.time() - in_start
        print "Evaluation: "+ str(evl_epoch)+ " spent Time = " + str(in_time)[:3]
    in_start = time.time()

    true_cost, pred, real_X= model.train(X, np.asarray(mask, dtype=theano.config.floatX), lr, y, doc_num)
    # print pred.flatten()
    # print real_X
    # exit()
    in_time = time.time() - in_start
    # print pred.flatten()[:10]
    print "Epoch = " + str(epoch) + " Iter = " + str(i) + ", Error = " + str(true_cost)[:6] + ", Time = " + str(in_time)[:3]
    # print 'ytrue = ' + str(y) +", ypred = " + str(pred)
    # if cost <= e:
    #     break

print "Finished. Time = " + str(time.time() - start)[:5]

# print "save model..."
# save_model("./model/hed.model", model)

