import tensorflow as tf
from constants import *
from layers import *

class MyModel(tf.keras.Model):

    def __init__(self, K, adj_norm_tensor):
        super(MyModel, self).__init__()
        self.K = K
        self.adj = adj_norm_tensor

        # the first layer is a sparse conv layer since the input tensor is sparse
        self.conv_1 = GraphSparseConvolution(adj_norm=adj_norm_tensor, output_size=CONV1_OUT_SIZE, dropout_rate=DROPOUT, act=tf.nn.relu)
        # the second and third conv layer share the same input
        self.conv_2 = GraphConvolution(adj_norm=adj_norm_tensor, output_size=K, dropout_rate=DROPOUT, act=lambda x: x)
        self.conv_3 = GraphConvolution(adj_norm=adj_norm_tensor, output_size=K, dropout_rate=DROPOUT, act=lambda x: x)
        # decoder
        self.top_dec = TopologyDecoder(act=tf.math.sigmoid, dropout_rate=DROPOUT)

    def call(self, inputs):
        # first convolution
        x = self.conv_1(inputs)
        # second and third convs
        self.mu = self.conv_2(x)
        self.logvar = self.conv_3(x)

        # get the encoding from the precedent two layers
        self.encode = self.mu + tf.random.normal([self.adj.shape[0], self.K]) * tf.exp(self.logvar)
        
        # get the reconstruction of the adj
        top = self.top_dec(self.encode)
        # reshape to tensor of shape (n_nodes^2)
        return tf.reshape(top, [-1])
    
    def get_encode(self):
        return self.encode

    def get_mu(self):
        return self.mu
    
    def get_logvar(self):
        return self.logvar