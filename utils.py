import dgl
import torch
import torch.nn as nn
import random
import faiss
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from tqdm.notebook import tqdm
import networkx as nx

# from data import get_dataset, HeatDataset, PPRDataset, set_train_val_test_split
# from models import GCN
# from seeds import val_seeds, test_seeds

def load(dataset, train_rate=0.3, val_rate=0.1):
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
    # if 'str_anomaly_label' in data:
    #     str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
    #     attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    # else:
    #     str_ano_labels = None
    #     attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    # return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels
    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels

def create_dgl_graph(adj):
    """Translate data into DGLGraph."""
    g = dgl.DGLGraph()
    g.from_scipy_sparse_matrix(adj)
    # n_feat = preprocess_features(n_feat)
    # g.ndata['features'] = torch.tensor(n_feat)
    # g.readonly()
    return g

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

# def adj_to_dgl_graph(adj):
#     nx_graph = nx.from_scipy_sparse_matrix(adj)
#     dgl_graph = dgl.DGLGraph(nx_graph)
#     return dgl_graph

def dense_to_one_hot(labels_dense, num_classes):
    """
    Convert class labels from scalars to one-hot vectors.
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return features.todense(), sparse_to_tuple(features)
    return features.todense()

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

    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    
    return T_S

# def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
#     a = nx.convert_matrix.to_numpy_array(graph)
#     if self_loop:
#         a = a + np.eye(a.shape[0])                                
#     d = np.diag(np.sum(a, 1))                                     
#     dinv = fractional_matrix_power(d, -0.5)                       
#     at = np.matmul(np.matmul(dinv, a), dinv)                      
#     return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))

# def compute_heat(graph: nx.Graph, t=5, self_loop=True):
#     a = nx.convert_matrix.to_numpy_array(graph)
#     if self_loop:
#         a = a + np.eye(a.shape[0])
#     d = np.diag(np.sum(a, 1))
#     return np.exp(t * (np.matmul(a, inv(d)) - 1))

# def generate_rwr_subgraph(dgl_graph, subgraph_size):
#     all_idx = list(range(dgl_graph.number_of_nodes()))
#     reduced_size = subgraph_size - 1
#     traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
#     subv = []

#     for i,trace in enumerate(traces):
#         subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
#         retry_time = 0
#         while len(subv[i]) < reduced_size:
#             cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
#             subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
#             retry_time += 1
#             if (len(subv[i]) <= 2) and (retry_time >10):
#                 subv[i] = (subv[i] * reduced_size)
#         subv[i] = subv[i][:reduced_size]
#         subv[i].append(i)

#     return subv

def generate_rwr_subgraph(
    g, seed, trace, features, positional_embedding_size, entire_graph=False
):
    subv = torch.unique(torch.cat(trace)).tolist()
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:
        subg = g.subgraph(g.nodes())
    else:
        subg = g.subgraph(subv)

    # subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)
    # subg = _add_masked_node_embedding(subg, features)
    n_feat = features[g.nodes()]
    subg.ndata["features"] = n_feat

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg

# def _add_masked_node_embedding(g, features):
#     b = g.number_of_nodes()
#     n_feat = features[g.nodes()][1:]
#     added_feat_zero_row = torch.zeros((b, n_feat.shape[1]), dtype=torch.long)
#     n_feat = torch.cat((added_feat_zero_row, n_feat), dim=-2)
#     g.ndata["features"] = n_feat
#     return g
# def run_kmeans(x, args):
#     """
#     Args:
#         x: data to be clustered
#     """
    
#     print('performing kmeans clustering')
#     results = {'im2cluster':[],'centroids':[],'density':[]}
    
#     for seed, num_cluster in enumerate(args.num_cluster):
#         # intialize faiss clustering parameters
#         d = x.shape[1]
#         k = int(num_cluster)
#         clus = faiss.Clustering(d, k)
#         clus.verbose = True
#         clus.niter = 20
#         clus.nredo = 5
#         clus.seed = seed
#         clus.max_points_per_centroid = 1000
#         clus.min_points_per_centroid = 10

#         res = faiss.StandardGpuResources()
#         cfg = faiss.GpuIndexFlatConfig()
#         cfg.useFloat16 = False
#         cfg.device = args.gpu    
#         index = faiss.GpuIndexFlatL2(res, d, cfg)  

#         clus.train(x, index)   

#         D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
#         im2cluster = [int(n[0]) for n in I]
        
#         # get cluster centroids
#         centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
#         # sample-to-centroid distances for each cluster 
#         Dcluster = [[] for c in range(k)]          
#         for im,i in enumerate(im2cluster):
#             Dcluster[i].append(D[im][0])
        
#         # concentration estimation (phi)        
#         density = np.zeros(k)
#         for i,dist in enumerate(Dcluster):
#             if len(dist)>1:
#                 d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
#                 density[i] = d     
                
#         #if cluster only has one point, use the max to estimate its concentration        
#         dmax = density.max()
#         for i,dist in enumerate(Dcluster):
#             if len(dist)<=1:
#                 density[i] = dmax 

#         density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
#         density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
#         # convert to cuda Tensors for broadcast
#         centroids = torch.Tensor(centroids).cuda()
#         centroids = nn.functional.normalize(centroids, p=2, dim=1)    

#         im2cluster = torch.LongTensor(im2cluster).cuda()               
#         density = torch.Tensor(density).cuda()
        
#         results['centroids'].append(centroids)
#         results['density'].append(density)
#         results['im2cluster'].append(im2cluster)    
        
#     return results