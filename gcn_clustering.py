import os  
import math
import random

from scipy.sparse import csr_matrix
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn import metrics

from constants import LR, DATASET_NAME, DROPOUT, epochs, CONV1_OUT_SIZE
from data import load_data, get_test_edges
from metrics import clustering_metrics
from model import MyModel    
from loss import total_loss

# convert sparse matrix to sparse tensor
def convert_sparse_matrix_to_sparse_tensor(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    indices_matrix = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    sparse_tensor = tf.SparseTensor(indices=indices_matrix, values=values, dense_shape=shape)
    return tf.cast(sparse_tensor, dtype=tf.float32)

def train(features, adj_train, adj_train_norm, train_edges, train_false_edges, clustering_labels , K):
    print("training")

    # intialize the adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    max_acc = 0
    max_f1 = 0
    max_top_acc = 0

    n_nodes = adj_train.shape[0]

    # convert the normalized adj and the features to tensors
    adj_train_norm_tensor = convert_sparse_matrix_to_sparse_tensor(adj_train_norm)
    feature_tensor = convert_sparse_matrix_to_sparse_tensor(features)
    
    # define the model
    model = MyModel(K, adj_train_norm_tensor)

    for i in range(epochs):
        
        with tf.GradientTape() as tape:
            # forward pass
            pred = model(feature_tensor)

            # get the predictions for edges that are not in the adj_train
            train_edges_p_pred = [pred[x[0]*adj_train.shape[0]+x[1]] for x in train_edges]
            train_edges_n_pred = [pred[x[0]*adj_train.shape[0] +x[1]] for x in train_false_edges]

            train_edges_p_l = [1]*len(train_edges_p_pred)
            train_edges_n_l = [0]*len(train_edges_n_pred)

            pred = train_edges_p_pred + train_edges_n_pred

            y_actual = train_edges_p_l+train_edges_n_l
            
            # if you whant to train on the entire original adj use the below line
            #y_actual = adj_train.toarray().flatten()

            # get the embeddings
            embeddings_np = model.get_encode().numpy()

            mu = model.get_mu()
            logvar = model.get_logvar()

            # get loss
            loss = total_loss(y_actual, pred, mu, logvar, n_nodes)

            # get gradient from loss 
            grad = tape.gradient(loss, model.trainable_variables)

            # optimize the weights
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
        

        print("#"*30)
        print("epoch:{}, train loss: {}".format(i, loss))

        # get adj accuracy
        top_acc_function = tf.keras.metrics.BinaryAccuracy()
        top_acc_function.update_state(y_actual, pred)
        top_train_accuracy = top_acc_function.result().numpy()
        
        if(max_top_acc < top_train_accuracy):
            max_top_acc = top_train_accuracy

        print("train top acc: {}".format(top_train_accuracy))

        # get labels accuracy 
        pred_labels_x = embeddings_np.argmax(1)
            
        cm = clustering_metrics(labels, pred_labels_x)
        res = cm.clusteringAcc()
        print("acc:{}, f1:{}".format(res[0], res[2]))

        if(res[0] > max_acc):
            max_acc = res[0]
        if(res[2] > max_f1):
            max_f1 = res[2]
    
    print("max_acc:{}, max_f1:{}, max_top_acc: {}".format(max_acc, max_f1, max_top_acc))

# compute Ã = D^{1/2}(A+I)D^{1/2}
def compute_adj_norm(adj):
    
    adj_I = adj + sp.eye(adj.shape[0])

    D = np.sum(adj_I, axis=1)
    D_power = sp.diags(np.asarray(np.power(D, -0.5)).reshape(-1))

    adj_norm = D_power.dot(adj_I).dot(D_power)

    return adj_norm

if __name__ == "__main__":

    # load data : adj, features, node labels and number of clusters
    data = load_data(DATASET_NAME)

    complete_adj = data[0]
    features = data[1] 
    labels = [x[1] for x in data[2]]
    n_clusters = data[3]
    # get train edges (the ones that are not in the adj used to train but 
    # used in order to compute the graditens)
    adj_train_triu, train_edges, train_false_edges, test_edges, test_false_edges = get_test_edges(complete_adj)
    print("got adj_train")
    
    # since get_test_edges returns a triu, we sum to its transpose 
    adj_train = adj_train_triu + adj_train_triu.T

    # get normalized adj
    adj_train_norm = compute_adj_norm(adj_train)
    print("normalized the adj matrix")

    # start training
    train(features, adj_train, adj_train_norm, train_edges, train_false_edges, labels, K=n_clusters)