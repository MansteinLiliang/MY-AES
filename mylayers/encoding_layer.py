# pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from gru_layer import GRULayer
from lstm_layer import LSTMLayer


class SentEncoderLayer(object):
    def __init__(self, rng, X, in_size, hidden_size,
                 cell, optimizer, p, is_train, batch_size, mask):
        self.X = X
        self.in_size = in_size
        self.hidden_size_list = hidden_size
        self.cell = cell
        self.drop_rate = p
        self.is_train = is_train
        self.batch_size = batch_size
        self.mask = mask
        self.rng = rng
        self.num_hds = len(hidden_size)

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

            if self.cell == "gru":
                hidden_layer = GRULayer(self.rng, str(i), shape, layer_input,
                                        self.mask, self.is_train, self.batch_size, self.drop_rate)
            elif self.cell == "lstm":
                hidden_layer = LSTMLayer(self.rng, str(i), shape, layer_input,
                                         self.mask, self.is_train, self.batch_size, self.drop_rate)

            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        self.activation = hidden_layer.activation
        # hidden_size is equal to the rnn-cell state size(output a hidden state)
        self.hidden_size = hidden_layer.out_size


class DocEncoderLayer(object):
    def __init__(self, cell, rng, layer_id, shape, X, mask, is_train=1, batch_size=1, p=0.5):
        """
        :param cell:
        :param rng:
        :param layer_id:
        :param shape:(sent_max_len,batch_size, hiddendim_of_sent_rnn)
        :param X: input X tensor from sentence rnn
        :param mask: sentence mask, shape=(max_sent_len,)
        :param is_train:
        :param batch_size:
        :param p: dropout prob
        :return Tensor: (sent_max_len, hidden_dim)
        """
        prefix = "SentEncoder_"
        self.in_size, self.out_size = shape

        '''
        def code(j):
            i = mask[:, j].sum() - 1
            i = T.cast(i, 'int32')
            sent_x = X[i, j * self.in_size : (j + 1) * self.in_size]
            return sent_x
        sent_X, updates = theano.scan(lambda i: code(i), sequences=[T.arange(mask.shape[1])])
        '''
        sent_X = T.reshape(X[X.shape[0] - 1, :], (batch_size, self.in_size))
        # TODO sent representation can be pooling the over whole sentence

        mask = T.reshape(T.ones_like(sent_X)[:, 0], (batch_size, 1))

        if cell == "gru":
            self.encoder = GRULayer(rng, prefix + layer_id, shape, sent_X, mask, is_train, 1, p)
        elif cell == "lstm":
            self.encoder = LSTMLayer(rng, prefix + layer_id, shape, sent_X, mask, is_train, 1, p)

        self.activation = self.encoder.activation[self.encoder.activation.shape[0] - 1, :]
        self.sent_encs = sent_X
        self.params = self.encoder.params
