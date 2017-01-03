# pylint: skip-file
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
from mylayers.encoding_layer import SentEncoderLayer
from mylayers.syntax_attention import SyntaxAttentionLayer
from mylayers.utils import init_weights, init_bias
from optimizers import *


class RNN(object):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 cell="gru", optimizer="rmsprop", p=0.5, num_sents=1):
        '''
        :param in_size: word-embedding dimension
        :param hidden_size: lm-rnn-layer hidden_size
        :param cell:
        :param optimizer:
        :param p:
        :param num_sents:
        '''
        self.idxs = T.imatrix('idxs')
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.vocab_size = vocab_size
        self.drop_rate = p
        self.num_sents = num_sents  # num_sents is doc_len or input_tesor.shape[1]
        self.is_train = T.iscalar('is_train')  # for dropout
        # self.batch_size = T.iscalar('batch_size')  # for mini-batch training
        self.mask = T.matrix("mask")  # dype=None means to use config.floatX
        self.sent_mask = T.switch(self.mask.sum(axis=0)> 0, 1.0, 0.0)
        self.syntax_vector = init_bias(hidden_size[0], 'syntax_vector')
        self.optimizer = optimizer
        self.layers = []
        self.params = []
        self.y = T.scalar('y')
        # Word_Embdding layer
        embeddings = theano.shared(0.2 * np.random.uniform(
            -0.01, 0.01,(self.vocab_size, self.embedding_size)).astype(theano.config.floatX),name='WEmb')  # add one for PADDING at the end
        self.params.append(embeddings)
        self.idxs = T.imatrix()
        # self.X = embeddings[self.idxs].reshape((self.idxs.shape[0], self.idxs.shape[1], self.embedding_size))
        self.X = embeddings[self.idxs]
        self.define_layers()
        self.define_train_test_funcs()

    def define_layers(self):
        rng = RandomStreams(1234)

        # LM layers
        sent_encoder_layer = SentEncoderLayer(rng, self.X, self.embedding_size, self.hidden_size,
                                              self.cell, self.optimizer, self.drop_rate,
                                              self.is_train, self.num_sents, self.mask)
        self.layers += sent_encoder_layer.layers
        self.params += sent_encoder_layer.params

        i = len(self.layers) - 1

        # Doc layer
        layer_input = sent_encoder_layer.activation
        sent_X = layer_input[layer_input.shape[0] - 1, :]
        syntax_att = SyntaxAttentionLayer(str(i+1), (self.num_sents, sent_encoder_layer.hidden_size),
                                          sent_X, self.sent_mask, self.syntax_vector)

        self.layers.append(syntax_att)
        self.params += syntax_att.params

        # codes is a vector, sent_encoder_layer.hidden_size
        codes = syntax_att.activation

        self.activation = codes

        # https://github.com/fchollet/keras/pull/9/files
        self.epsilon = 1.0e-15

    # def categorical_crossentropy(self, y_pred, y_true):
    #     y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
    #     m = T.reshape(self.mask, (self.mask.shape[0] * self.batch_size, 1))
    #     ce = T.nnet.categorical_crossentropy(y_pred, y_true)
    #     ce = T.reshape(ce, (self.mask.shape[0] * self.batch_size, 1))
    #     return T.sum(ce * m) / T.sum(m)

    def mean_squared_error(self, X, y):
        W_out = init_weights((self.hidden_size[0],1), 'mse_W')
        b_out = init_bias(1, 'mse_b')
        self.params.append(W_out)
        self.params.append(b_out)
        y_pred = T.dot(X, W_out)+b_out
        print 'compile the mse'
        # self.cost = T.pow(y_pred-y, 2).sum()
        self.cost = T.pow(y_pred-y,2).mean()
        clip = theano.gradient.grad_clip(self.cost, 0, 2.0)
        return clip, y_pred

    def define_train_test_funcs(self):
        # pYs = T.reshape(self.activation, (self.batch_size, 1))

        # tYs = T.reshape(self.X, (self.batch_size, 1))

        # cost = self.categorical_crossentropy(pYs, tYs)
        # self.activation is a scorer vector
        cost, pred = self.mean_squared_error(self.activation, self.y)

        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)

        # updates = sgd(self.params, gparams, lr)
        # updates = momentum(self.params, gparams, lr)
        # updates = rmsprop(self.params, gparams, lr)
        # updates = adagrad(self.params, gparams, lr)
        # updates = adadelta(self.params, gparams, lr)
        # updates = adam(self.params, gparams, lr)

        self.train = theano.function(inputs=[self.idxs, self.mask, lr, self.y],
                                     givens={self.is_train: np.cast['int32'](1)},
                                     outputs=[cost, self.cost, pred],
                                     updates=updates)

