# file from https://github.com/tkipf/gcn/blob/master/gcn/utils.py

import sqlite3
import numpy as np
import random 
from scipy import sparse as sp

# get n random edges that have 0 value in adj
def get_false_edges(adj, n):
    print("getting false edges")
    false_edges = []

    while len(false_edges) < n:
        r1 = random.randint(0, adj.shape[0]-1)
        r2 = random.randint(0, adj.shape[0]-1)
        
        # check that the edge is not present in the adj
        # and that is in the triu part
        if(adj[r1, r2] == 0 and r1<r2):
            false_edges.append([r1,r2])
            
    return false_edges

# sparse matrix to (coords of the edges, values of the edges, shape of the matrix)
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# get adj_train, train a and test edges
def get_test_edges(adj, test_size=0.1, train_size=0.1):
    print("getting test edges")
    adj_ = sp.triu(adj)
    coords, _, shape = sparse_to_tuple(adj_)
    
    all_edges = coords
    
    # define the number of edges for train and test
    num_train = int(train_size*all_edges.shape[0])
    num_test = int(test_size*all_edges.shape[0])

    # shuffle the edges
    np.random.shuffle(all_edges)

    # get the first num_test edges (after shuffling)
    # as the test_edges (the positive ones)
    test_edges = all_edges[:num_test]
    # get the first num_train after the first num_test edges (after shuffling)
    # as the train_edges (the positive ones)
    train_edges = all_edges[num_test:num_test+num_train]
    res_edges = all_edges[num_test+num_train:]
    
    # get random false edges
    false_edges = get_false_edges(adj, test_edges.shape[0]+train_edges.shape[0])
    # split them into train and test
    test_false_edges = false_edges[:test_edges.shape[0]]
    train_false_edges = false_edges[test_edges.shape[0]:]
    
    print("got false edges")

    # turn the remaning edges into a sparse matrix
    adj_train = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj.shape)

    return adj_train, train_edges, np.array(train_false_edges), test_edges, np.array(test_false_edges) 

    
# load data: adj_complete, features, labels, num of classes
def load_data(dataset_name):
    if(dataset_name == "cora"):
        return load_cora_data()
    elif (dataset_name == "citeseer"):
        return load_citeseer_data()

def load_citeseer_data():
    content_d = {}
    with open("citeseer/citeseer.content", "r") as fin_c:
        classes = set()
        docs = set()
        for line in fin_c:
            splitted = line.strip().split('\t')
            if(len(splitted)>2):
                content_d[splitted[0]] = {"features":splitted[1:-1], "class":splitted[-1]}
                classes.add(splitted[-1])
                docs.add(splitted[0])
    
    class_to_id = {}
    for i, class_ in enumerate(list(classes)):
        class_to_id[class_] = i
    
    for i, doc in enumerate(list(docs)):
        content_d[doc]["id"] = i
    
    cites = []
    with open("citeseer/citeseer.cites", "r") as fin_c:
        for line in fin_c:
            splitted = line.strip().split("\t")
            if(len(splitted)>1 and splitted[0] in content_d and splitted[1] in content_d):
                cites.append([content_d[splitted[0]]["id"], content_d[splitted[1]]["id"]])
                cites.append([content_d[splitted[1]]["id"], content_d[splitted[0]]["id"]])
    
    cited = np.array(cites).T
    adj_complete = sp.csr_matrix((np.ones(cited.shape[1]), (cited[0], cited[1])))

    labels = []
    for key in content_d.keys():
        labels.append([content_d[key]["id"], class_to_id[content_d[key]["class"]]]) 

    labels = sorted(labels, key= lambda y: y[0])

    features_ = []
    for key in content_d.keys():
        ws = content_d[key]["features"]
        for i, w in enumerate(ws):
            if(w == "1"):
                features_.append([content_d[key]["id"], i])
    
    features_ = np.array(features_).T

    features = sp.csr_matrix((np.ones(features_.shape[1]), (features_[0], features_[1])))

    return adj_complete, features, labels, len(class_to_id.keys())

def load_cora_data():
    # get cursor from db 
    conn = sqlite3.connect('CORA/cora.db')
    cur = conn.cursor()

    # load data from db
    adj_sparse, features_sparse, labels, n_clusters = load_cora_adj_features(cur)
    
    return adj_sparse, features_sparse, labels, n_clusters

def load_cora_adj_features(cur):
    
    cur.execute("SELECT * FROM paper")
    papers = cur.fetchall()
    cur.execute("SELECT * FROM cites")
    cited = cur.fetchall()
    cur.execute("SELECT * FROM content")
    features = cur.fetchall()

    paper_id_to_int = {}
 
    set_labels = set()

    # associate paper_id to integer
    for i, paper in enumerate(papers):
        paper_id_to_int[paper[0]] = i
        set_labels.add(paper[1])
    
    topic_to_label_index = {}
    for i, label in enumerate(list(set_labels)):
        topic_to_label_index[label] = i
    
    labels = [ [paper_id_to_int[x[0]], topic_to_label_index[x[1]]] for x in papers ]

    cited = np.array([[paper_id_to_int[x[0]],paper_id_to_int[x[1]]] for x in cited]).T

    # get sparse matrix 
    adj_s = sp.csr_matrix((np.ones(cited.shape[1]*2), (np.append(cited[0], cited[1]), np.append(cited[1], cited[0])))).tocsr()

    # get set of words
    words = set()
    for feature in features:
        words.add(feature[1])

    # associate word_id to integer
    word_to_int = {}
    for i, word in enumerate(list(words)):
        word_to_int[word] = i

    features = np.array([ [ paper_id_to_int[x[0]], word_to_int[x[1]] ] for x in features ]).T

    # get sparse matrix
    features_s = sp.csr_matrix((np.ones(features.shape[1]), (features[0], features[1])))

    return adj_s, features_s, labels, len(set_labels)