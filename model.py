# pylint: skip-file
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
from mylayers.encoding_layer import SentEncoderLayer, DocEncoderLayer
from mylayers.attention import SyntaxAttentionLayer, CoherenceAttention, MeaningAttention
from mylayers.gating_layer import SimpleGatingLayer
from mylayers.layer_utils import init_weights, init_bias
from optimizers import *


class RNN(object):
    def __init__(self, U, vocab_size, embedding_size, hidden_size,
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
        self.layers = []
        self.params = []
        self.num_sents = num_sents  # num_sents is doc_len
        self.batch_docs = T.iscalar('batch_docs')  # input_tesor.shape[1] = batch_docs*num_sents
        self.is_train = T.iscalar('is_train')  # for dropout
        # self.batch_size = T.iscalar('batch_size')  # for mini-batch training
        self.mask = T.matrix("mask")  # dype=None means to use config.floatX
        self.sent_mask = T.switch(self.mask.sum(axis=0)> 0, 1.0, 0.0)
        self.syntax_vector = init_weights((hidden_size[0], 1), 'syntax_vector')
        self.params.append(self.syntax_vector)
        # self.coherence_vector = init_weights((hidden_size[0],), 'coherence_vector')
        # self.params.append(self.coherence_vector)

        self.optimizer = optimizer
        self.y = T.fvector('y')
        # TODO Word Embdding matrix initialization
        embeddings = theano.shared(value=U, name='WEmb')
        self.sent_X = embeddings
        # embeddings = theano.shared(0.2 * np.random.uniform(
        #     -0.01, 0.01,(self.vocab_size, self.embedding_size)).astype(theano.config.floatX), name='WEmb')  # add one for PADDING at the end
        # self.params.append(embeddings)
        self.idxs = T.imatrix()
        self.X = embeddings[self.idxs]
        self.define_layers()
        self.define_train_test_funcs()

    def define_layers(self):
        rng = RandomStreams(1234)

        # LM layers
        sent_encoder_layer = SentEncoderLayer(
            'SentEncoder', self.X, self.embedding_size,
            self.hidden_size, self.cell, self.optimizer,
            self.drop_rate, self.is_train,
            self.num_sents * self.batch_docs, self.mask, rng
        )
        self.layers += sent_encoder_layer.layers
        self.params += sent_encoder_layer.params

        i = len(self.layers) - 1

        # Doc layer
        layer_input = sent_encoder_layer.activation
        sent_X = layer_input[layer_input.shape[0] - 1, :]

        #  sent_X shape is: (doc_num * sent_num, dim)
        doc_sent_X = T.reshape(sent_X, (self.batch_docs, self.num_sents, sent_encoder_layer.hidden_size))

        # Annotation over all sentence
        # X_T = doc_sent_X.dimshuffle([1, 0, 2])
        # sent_mask_T = T.transpose(self.sent_mask.reshape((self.batch_docs, -1)))
        # sent_annotate_layer = DocEncoderLayer(
        #     'SentAnnotation', rng, X_T, sent_encoder_layer.hidden_size,
        #     sent_encoder_layer.hidden_size, 'gru', 'None', 0, 1, self.batch_docs, sent_mask_T
        # )
        # self.layers.append(sent_annotate_layer)
        # self.params += sent_annotate_layer.params
        # doc_annotation = sent_annotate_layer.activation

        # syntax attention
        syntax_att = SyntaxAttentionLayer(
            str(i+1),(self.batch_docs, self.num_sents,
                      sent_encoder_layer.hidden_size),
            doc_sent_X, self.sent_mask, self.syntax_vector)
        self.layers.append(syntax_att)
        self.params += syntax_att.params
        syntax_att_output = syntax_att.activation

        # coherence attention
        # coherence_att = CoherenceAttention(
        #     str(i+2), (self.batch_docs, self.num_sents, sent_encoder_layer.hidden_size),
        #     doc_sent_X, self.sent_mask.reshape((self.batch_docs, -1)),
        #     self.coherence_vector
        # )
        # self.layers.append(coherence_att)
        # self.params += coherence_att.params
        # coherence_att_output = coherence_att.activation

        # meaning attention
        X_T = doc_sent_X.dimshuffle([1, 0, 2])
        sent_mask_T = T.transpose(self.sent_mask.reshape((self.batch_docs, -1)))
        doc_encoding_layer = DocEncoderLayer(
            'DocEncoding', rng, X_T, sent_encoder_layer.hidden_size,
            sent_encoder_layer.hidden_size, 'gru', 'None', 0, 1, self.batch_docs, sent_mask_T
        )
        meaning_att = MeaningAttention(
            (self.num_sents, self.batch_docs, doc_encoding_layer.hidden_zie),
            X_T, doc_encoding_layer.activation,
            sent_mask_T
        )
        self.layers.append(doc_encoding_layer)
        self.layers.append(meaning_att)
        self.params += doc_encoding_layer.params
        self.params += meaning_att.params
        meaning_att_output = meaning_att.activation


        # self.activation = syntax_att_output
        self.activation = T.concatenate([syntax_att_output, meaning_att_output], axis=-1)

        # codes is a vector, sent_encoder_layer.hidden_size,
        # https://github.com/fchollet/keras/pull/9/files
        self.epsilon = 1.0e-15

    # def categorical_crossentropy(self, y_pred, y_true):
    #     y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
    #     m = T.reshape(self.mask, (self.mask.shape[0] * self.batch_size, 1))
    #     ce = T.nnet.categorical_crossentropy(y_pred, y_true)
    #     ce = T.reshape(ce, (self.mask.shape[0] * self.batch_size, 1))
    #     return T.sum(ce * m) / T.sum(m)

    def mean_squared_error(self, X, y):
        """
        :param X: if doc = 1, shape=(output_dim,), else shape=(doc_len,output_dim)
        :param y:
        :return:
        """
        W_out = init_weights((2*self.hidden_size[0],1), 'mse_W')
        b_out = init_bias(1, 'mse_b', value=0.0)
        self.params.append(W_out)
        self.params.append(b_out)
        y_pred = T.dot(X, W_out)+b_out
        print 'compile the mse'
        # self.cost = T.pow(y_pred-y, 2).sum()
        self.cost = T.mean(T.square(y_pred.flatten() - y), axis=-1)
        # self.cost = T.mean(T.square((y_pred-y.reshape((self.batch_docs, 1)))))
        # self.cost = T.pow(y_pred-y.reshape((self.batch_docs, 1)), 2).mean()
        clip = theano.gradient.grad_clip(self.cost, -5.0, 5.0)
        return clip, T.clip(y_pred, 0.0, 1.0)

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

        self.train = theano.function(inputs=[self.idxs, self.mask, lr, self.y, self.batch_docs],
                                     givens={self.is_train: np.cast['int32'](1)},
                                     outputs=[self.cost, pred, self.sent_X],
                                     updates=updates,
                                     on_unused_input='ignore',
                                     allow_input_downcast=True
                                     )

        self.predict = theano.function(
            inputs=[self.idxs, self.mask, self.y, self.batch_docs],
            givens={self.is_train: np.cast['int32'](1)},
            outputs=[self.cost, pred],
            on_unused_input='ignore',
            allow_input_downcast=True
        )