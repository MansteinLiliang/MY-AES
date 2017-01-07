# pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from gru_layer import GRULayer
from lstm_layer import LSTMLayer


class SentEncoderLayer(object):
    def __init__(self, layer_name, X, in_size, hidden_size, cell, optimizer, p, is_train, total_sents, mask, rng):
        # TODO sent representation can be pooling the over whole sentence
        """
        Support for dynamic batch, which is specified by num_sens*batch_docs
        :param layer_name:
        :param rng:
        :param X:
        :param in_size:
        :param hidden_size:
        :param cell:
        :param optimizer:
        :param p:
        :param is_train:
        :param batch_size:
        :param mask:
        :return Tensor: shape is (sent_len, sent_num, embedding)
        """

        self.X = X
        self.in_size = in_size  # word_embedding size
        self.hidden_size_list = hidden_size # sent_embedding size
        self.cell = cell
        self.drop_rate = p
        self.is_train = is_train
        self.total_sents = total_sents  # T.scalar
        self.mask = mask
        self.rng = rng
        self.num_hds = len(hidden_size)
        self.layer_name = layer_name
        self.define_layers()

    def define_layers(self):
        self.layers = []
        self.params = []
        # hidden layers
        for i in xrange(self.num_hds):
            if i == 0:
                layer_input = self.X
                shape = (self.in_size, self.hidden_size_list[0])
            else:
                layer_input = self.layers[i - 1].activation
                shape = (self.hidden_size_list[i - 1], self.hidden_size_list[i])

            # if self.cell == "gru":
            hidden_layer = GRULayer(self.rng, str(i), layer_input, shape,
                                    self.mask, self.total_sents, self.is_train, self.drop_rate)
            # elif self.cell == "lstm":
            #     hidden_layer = LSTMLayer(self.rng, str(i), shape, layer_input,
            #                              self.mask, self.is_train, self.batch_size, self.drop_rate)

            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        self.activation = hidden_layer.activation
        # hidden_size is equal to the rnn-cell state size(output a hidden state)
        self.hidden_size = hidden_layer.out_size


class DocEncoderLayer(object):
    def __init__(self, layer_name, rng, X, in_size, hidden_size,
                 cell, optimizer, p, is_train, total_docs, sent_mask):
        """
        :param rng:
        :param X: shape(sent_nums, doc_nums, in_size)
        :param in_size:
        :param hidden_size:
        :param cell:
        :param optimizer:
        :param p:
        :param is_train:
        :param total_docs:
        :param sent_mask: (sent_nums, doc_nums)
        :return Tensor: shape is (sent_num, doc_num, embedding)
        """
        prefix = layer_name + "_"
        '''
        def code(j):
            i = mask[:, j].sum() - 1
            i = T.cast(i, 'int32')
            sent_x = X[i, j * self.in_size : (j + 1) * self.in_size]
            return sent_x
        sent_X, updates = theano.scan(lambda i: code(i), sequences=[T.arange(mask.shape[1])])
        '''
        self._in_size = in_size
        self.hidden_zie = hidden_size
        if cell == "gru":
            self.encoder = GRULayer(rng, prefix, X, (in_size, hidden_size),
                     sent_mask, total_docs, is_train, p)
        # elif cell == "lstm":
        #     self.encoder = LSTMLayer(rng, prefix + layer_id, shape, sent_X, mask, is_train, 1, p)

        self.activation = self.encoder.activation

        self.params = self.encoder.params
