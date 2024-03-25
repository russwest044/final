import dgl
import torch
import random
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from tqdm.notebook import tqdm
import networkx as nx


def batcher():
    def batcher_dev(batch):
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k

    return batcher_dev

def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)

    return batcher_dev

def load_mat(dataset):
    """Load data."""
    data = sio.loadmat("./data/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None
    
    return adj, feat, labels, ano_labels, str_ano_labels, attr_ano_labels


def adj_to_dgl_graph(adj):
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


def normalize_adj(adj: sp.csr_matrix, self_loop=True):
    """
    Symmetrically normalize adjacency matrix: 
        A^ = A + I_n
        D^ = Sigma A^_ii
        D^(-1/2)
        A~ = D^(-1/2) x A^ x D^(-1/2)
    """
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """
    Convert sparse matrix to tuple representation.
    Set insert_batch=True if you want to insert a batch dimension.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def dense_to_one_hot(labels_dense, num_classes):
    """
    Convert class labels from scalars to one-hot vectors.
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot


def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return features.todense(), sparse_to_tuple(features)
    return features.todense()


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    # 每个节点一条trace
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
    subv = []

    for i,trace in enumerate(traces): # 对每一个节点都生成一个子图
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv


def gaussian_noised_feature(X):
    """
    add gaussian noise to the attribute matrix X
    Args:
        X: the attribute matrix
    Returns: the noised attribute matrix X_tilde
    """
    N = torch.Tensor(np.random.normal(1, 0.1, X.shape))
    X_tilde = X * N
    return X_tilde


def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    """
    A^ = A + I_n
    D^ = Sigma A^_ii
    D^(-1/2)
    A~ = D^(-1/2) x A^ x D^(-1/2)
    a(I_n-(1-a)A~)^-1
    """
    N = A.shape[0]

    # Self-loops
    A_loop = sp.eye(N) + A
    
    # Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    T_sym_csc = T_sym.tocsc()
    
    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N).tocsc() - (1 - alpha) * T_sym_csc)
    
    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    
    return sp.csr_matrix(T_S)


def diffusion_adj(adj, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    adj_tmp = adj + np.eye(adj.shape[0])

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    sqrt_d_inv = np.sqrt(d_inv)

    # calculate norm adj
    norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    if mode == "ppr":
        diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))

    return diff_adj


def remove_edge(A, similarity, remove_rate=0.1):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges based on cosine similarity of embedding
    n_node = A.shape[0]
    for i in range(n_node):
        A[i, torch.argsort(similarity[i].cpu())[:int(round(remove_rate * n_node))]] = 0

    # normalize adj
    Am = normalize_adj(A, self_loop=True, symmetry=False)
    Am = numpy_to_torch(Am)
    return Am


def aug_random_edge(input_adj, drop_percent=0.2):
    """
    randomly delect partial edges and
    randomly add the same number of edges in the graph
    """
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()
    num_drop = int(len(row_idx) * percent)

    edge_index = [i for i in range(len(row_idx))]
    edges = dict(zip(edge_index, zip(row_idx, col_idx)))
    drop_idx = random.sample(edge_index, k=num_drop)

    list(map(edges.__delitem__, filter(edges.__contains__, drop_idx)))

    new_edges = list(zip(*list(edges.values())))
    new_row_idx = new_edges[0]
    new_col_idx = new_edges[1]
    data = np.ones(len(new_row_idx)).tolist()

    new_adj = sp.csr_matrix((data, (new_row_idx, new_col_idx)), shape=input_adj.shape)

    row_idx, col_idx = (new_adj.todense() - 1).nonzero()
    no_edges_cells = list(zip(row_idx, col_idx))
    add_idx = random.sample(no_edges_cells, num_drop)
    new_row_idx_1, new_col_idx_1 = list(zip(*add_idx))
    row_idx = new_row_idx + new_row_idx_1
    col_idx = new_col_idx + new_col_idx_1
    data = np.ones(len(row_idx)).tolist()

    new_adj = sp.csr_matrix((data, (row_idx, col_idx)), shape=input_adj.shape)
    return sp.csr_matrix(new_adj)

# def generate_rwr_subgraph(g, seed, trace, features, subgraph_size, entire_graph=False):
#     subv = torch.unique(torch.cat(trace)).tolist()
#     try:
#         subv.remove(seed)
#     except ValueError:
#         pass
#     if len(subv) > subgraph_size-1:
#         subv = subv[:subgraph_size-1]
#     subv = [seed] + subv
#     if entire_graph:
#         subg = g.subgraph(g.nodes())
#     else:
#         subg = g.subgraph(subv)
    
#     n_feat = features[subv]
#     subg.ndata["features"] = n_feat

#     return subg
