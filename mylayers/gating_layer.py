import theano.tensor as T
import theano
from mylayers import layer_utils

init_weights = layer_utils.init_weights
init_bias = layer_utils.init_bias


class SimpleGatingLayer(object):
    def __init__(self, vector_len, att_vector, pre_output):
        prefix = "SimpleGatingLayer_"
        concat = T.concatenate([att_vector, pre_output], axis=-1)
        self.W_a = init_weights([2*vector_len, vector_len], prefix+"W_a")
        self.b = init_bias(vector_len, prefix+'b_a')
        gating = T.dot(concat, self.W_a)
        cur_output = gating * pre_output + (1 - gating) * att_vector
        self.activation = cur_output
        self.params = [self.W_a, self.b]