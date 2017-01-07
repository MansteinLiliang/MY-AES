#pylint: skip-file
import theano.tensor as T
import theano
from mylayers import layer_utils

init_weights = layer_utils.init_weights
init_bias = layer_utils.init_bias



class CoherenceAttention(object):
    def __init__(self, layer_id, shape, sent_encs, mask, coherence_vector, begin_embedding=None, end_embedding=None):
        # TODO begin_embedding and end_embedding will be completed after language model Now use 0-padding
        """
        Syntax attention layer attention on syntax_vector
        :param layer_id:
        :param shape: tuple(num_docs, num_sents, rnn_dim)
        :param mask: sentence_mask is of shape(num_docs, num_sents)
        :param sent_encs: input from sentence_rnn layer, shape is (num_docs, num_sents, rnn_dim)
        :param coherence_vector: shape=(hidden_dim_sent_rnn,)
        :return Tensor: shape=(output_dim,)
        """
        prefix = "Coherence_Attention_"
        layer_id = "_" + layer_id
        self.batch_docs, self.num_sents, self.out_size = shape
        # TODO we need to add precomputed syntax_vector
        self.W_a = init_weights((self.out_size*3, self.out_size), prefix + "W_a" + layer_id)
        self.b_a = init_bias(self.out_size, prefix + "b_a" + layer_id)
        # TODO coherence vector has 2 ways to implement
        # coherence_matrix = T.tile(coherence_vector, (self.batch_docs, 1))
        self.M = mask
        self.X = sent_encs
        X_T = self.X.dimshuffle([1, 0, 2])
        M_T = T.transpose(self.M)
        self.padding_vector = init_bias(self.out_size, "sent_padding", 0.0)
        X_T_PAD = T.concatenate([
            T.tile(self.padding_vector, (1, self.batch_docs)).reshape((1, self.batch_docs, -1)),
            X_T,
            T.tile(self.padding_vector, (1, self.batch_docs)).reshape((1, self.batch_docs, -1))
        ])
        # TODO we need to reshape sent_mask to (num_sents, num_docs) tensor

        def _active_mask(x_pre, x, x_next, coherence_vector):
            #  x is of shape(doc_nums, rnn_dim)
            #
            concat = T.concatenate([x, x_pre, x_next], axis=1)
            h_hat = T.tanh(T.dot(concat, self.W_a)+self.b_a)
            strength = T.dot(h_hat, coherence_vector)
            return strength

        h, updates = theano.scan(_active_mask, sequences=dict(input=X_T_PAD, taps=[-1, 0, 1]),
                                 non_sequences=coherence_vector
                                 )
        # h is of shape(num_sents, num_docs)
        strength_mask = h*M_T
        strength_mask_T = T.transpose(strength_mask)

        # note that softmax will be computed row-wised if X is matrix
        # a is of shape(num_docs, num_sents)
        a = T.nnet.softmax(strength_mask_T)
        c = (a[:, :, None]*sent_encs).sum(axis=1)
        self.activation = c
        self.params = [self.W_a, self.b_a]


class MeaningAttention(object):
    def __init__(self, shape, sent_encs, sent_rnn_att, sent_mask, query_vector=None):
        """
        Attention computing with query
        :param sent_rnn_att: shape is (sent_num, doc_num, embedding)
        :param: shape is (sent_num, doc_num, embedding)
        :param sent_encs:
        :param sent_mask:
        :param query_vector:
        """
        prefix = "Meaning_Attention_"
        self.sent_num, self.doc_num, self.in_size = shape
        concat = T.concatenate([sent_encs, sent_rnn_att], axis=-1)

        self.W_a = init_weights([2*self.in_size, self.in_size], prefix+"W_a")
        if query_vector == None:
            self.W_u = init_weights((self.in_size,), name='query_vector')
        else:
            self.W_u = query_vector
        self.b  = init_bias(self.in_size, prefix+'b_a')
        strength = T.dot(T.tanh(T.dot(concat, self.W_a)+self.b), self.W_u)
        strength_mask = strength * sent_mask
        a = T.nnet.softmax(strength_mask)[:, :, None]
        c = (a*sent_encs).sum(axis=0)
        self.activation = c
        self.params = [self.W_a, self.W_u, self.b]


class SyntaxAttentionLayer(object):
    def __init__(self, layer_id, shape, sent_encs, sent_mask, syntax_vector):
        """
        Syntax attention layer attention on syntax_vector
        :param layer_id:
        :param shape: tuple(num_sents, out_size)
        :param sent_mask: sentence_mask
        :param sent_encs: input from sentence_rnn layer
        :param syntax_vector: shape=(hidden_dim_sent_rnn,)
        :return Tensor: shape=(output_dim,)
        """
        prefix = "Syntax_Attention_"
        layer_id = "_" + layer_id
        self.batch_docs, self.num_sents, self.out_size = shape
        self.W_a = init_weights([2*self.out_size, self.out_size], prefix+"W_a" + layer_id)
        self.W_u = init_weights([self.out_size, 1], prefix+"W_u"+layer_id)
        self.b = init_bias(self.out_size, prefix+'b_a'+layer_id)
        syntax_matrix = T.reshape(T.tile(syntax_vector, self.num_sents*self.batch_docs), shape)
        concat = T.concatenate([sent_encs, syntax_matrix], axis=2)
        # TODO why or should their is a bias in the dot next?
        strength = T.dot(T.tanh(T.dot(concat, self.W_a)+self.b), self.W_u).flatten()
        strength_mask = strength * sent_mask
        a = T.nnet.softmax(strength_mask).reshape((self.batch_docs,self.num_sents))[:, :, None]
        c = (a*sent_encs).sum(axis=1)
        # self.activation = T.dot(c, mask[...,None]).sum(axis=0)
        self.activation = c
        # self.activation = T.tanh(T.dot(sent_decs, self.W_a3) + T.dot(c, self.W_a4))
        # self.params = [self.W_a1, self.W_a2, self.W_a3, self.W_a4, self.U_a]
        self.params = [self.W_a, self.W_u, self.b]
