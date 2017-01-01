#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from .. import utils
init_weights = utils.init_weights
init_bias = utils.init_bias

class AttentionLayer(object):
    def __init__(self, layer_id, shape, sent_encs, mask, syntax_vector):
        """
        Syntax attention layer attention on syntax_vector
        :param layer_id:
        :param shape:
        :param mask: sentence_mask
        :param sent_encs: input from sentence_rnn layer
        :param syntax_vector: shape=(hidden_dim_sent_rnn,)
        :return Tensor: shape=(output_dim,)
        """
        prefix = "AttentionLayer_"
        layer_id = "_" + layer_id
        self.num_sents, self.out_size = shape
        self.W_a = init_weights([2*self.out_size, self.out_size], prefix+"W_a" + layer_id)
        self.W_u = init_weights([self.out_size, 1], prefix+"W_u"+layer_id)
        self.b = init_bias(self.out_size, prefix+'b_a'+layer_id)
        syntax_matrix = T.reshape(T.tile(syntax_vector, sent_encs.shape[1]),sent_encs.shape)
        concat = T.concatenate([sent_encs, syntax_matrix], axis=1)
        # TODO why their is a bias ?
        strength = T.dot(T.tanh(T.dot(concat, self.W_a)+self.b),self.W_u)
        strength_mask = T.dot(strength, mask)
        a = T.nnet.softmax(strength_mask)
        c = T.dot(a, sent_encs)
        self.activation = T.dot(c, mask[...,None]).sum(axis=0)
        # self.activation = T.tanh(T.dot(sent_decs, self.W_a3) + T.dot(c, self.W_a4))
        # self.params = [self.W_a1, self.W_a2, self.W_a3, self.W_a4, self.U_a]
        self.params = [self.W_a, self.W_u, self.b]
