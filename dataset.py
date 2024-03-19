import math
import torch
import dgl
import dgl.data
import numpy as np
import scipy.sparse as sp
import torch.utils.data
from utils import *


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        adj, 
        features,
        rw_hops=64,
        subgraph_size=4,
        restart_prob=0.8,
        step_dist=[1.0, 0.0, 0.0],
        aug=None
    ):
        super(GraphDataset).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0

        self.aug = aug
        self.features = features
        self.g = adj_to_dgl_graph(adj)
        if self.aug:
            self.g_aug = adj(self._graph_data_argumentation(adj))
        else:
            self.g_aug = None
        self.length = self.g.number_of_nodes()

    def _graph_data_argumentation(self, adj):
        """ Graph data argumentation """
        if self.aug == "gdc":
            adj_hat = gdc(adj, 0.2, 0.1)
        elif self.aug == "em":
            adj_hat = aug_random_edge(adj, 0.2)
        # adj_hat = normalize_adj(adj_hat)
        # adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()
        return adj_hat

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        node_idx = idx

        # step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        # if step == 0:
        #     other_node_idx = node_idx
        # else:
        #     other_node_idx = dgl.contrib.sampling.random_walk(
        #         g=self.g, seeds=[node_idx], num_traces=1, num_hops=step
        #     )[0][0][-1].item()

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                    self.g.out_degree(node_idx)
                    * math.e
                    / (math.e - 1)
                    / self.restart_prob
                )
                + 0.5
            ),
        )

        trace = dgl.contrib.sampling.random_walk_with_restart(
            self.g,
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )

        graph_q = generate_rwr_subgraph(
            g=self.g,
            seed=node_idx,
            trace=trace[0],
            features=self.features,
            subgraph_size=self.subgraph_size, 
        )

        # graph_k = generate_rwr_subgraph(
        #     g=self.g_aug,
        #     seed=other_node_idx,
        #     trace=trace[1],
        #     features=self.features,
        #     subgraph_size=self.subgraph_size, 
        # )

        return graph_q, idx



# class GraphDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         adj, 
#         features,
#         rw_hops=64,
#         subgraph_size=4,
#         restart_prob=0.8,
#         step_dist=[1.0, 0.0, 0.0],
#         aug=None
#     ):
#         super(GraphDataset).__init__()
#         self.rw_hops = rw_hops
#         self.subgraph_size = subgraph_size
#         self.restart_prob = restart_prob
#         self.step_dist = step_dist
#         assert sum(step_dist) == 1.0
#         # assert positional_embedding_size > 1
#         # assert aug in ("gdc", "modify")

#         self.aug = aug
#         self.features = features
#         self.g = create_dgl_graph(adj)
#         if self.aug is not None:
#             self.g_aug = create_dgl_graph(self._graph_data_argumentation(adj))
#         else:
#             self.g_aug = None
#         self.length = self.g.number_of_nodes()

#     def _graph_data_argumentation(self, adj):
#         """ Graph data argumentation """
#         if self.aug == "gdc":
#             adj_hat = gdc(adj, 0.2)
#         else:
#             adj_hat = None
#         adj_hat = normalize_adj(adj_hat)
#         adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()
#         return adj_hat

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         node_idx = idx

#         step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
#         if step == 0:
#             other_node_idx = node_idx
#         else:
#             other_node_idx = dgl.contrib.sampling.random_walk(
#                 g=self.g, seeds=[node_idx], num_traces=1, num_hops=step
#             )[0][0][-1].item()

#         max_nodes_per_seed = max(
#             self.rw_hops,
#             int(
#                 (
#                     self.g.out_degree(node_idx)
#                     * math.e
#                     / (math.e - 1)
#                     / self.restart_prob
#                 )
#                 + 0.5
#             ),
#         )

#         trace = dgl.contrib.sampling.random_walk_with_restart(
#             self.g,
#             seeds=[node_idx],
#             restart_prob=self.restart_prob,
#             max_nodes_per_seed=max_nodes_per_seed,
#         )

#         graph_q = generate_rwr_subgraph(
#             g=self.g,
#             seed=node_idx,
#             trace=trace[0],
#             features=self.features,
#             subgraph_size=self.subgraph_size, 
#         )

#         if self.g_aug:
#             trace_aug = dgl.contrib.sampling.random_walk_with_restart(
#                 self.g_aug,
#                 seeds=[other_node_idx],
#                 restart_prob=self.restart_prob,
#                 max_nodes_per_seed=max_nodes_per_seed,
#             )

#             graph_k = generate_rwr_subgraph(
#                 g=self.g_aug,
#                 seed=other_node_idx,
#                 trace=trace_aug,
#                 features=self.features
#             )

#             return (graph_q, graph_k), idx
#         return graph_q, idx

# class AnomalyGraphDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         dataset,
#         rw_hops=64,
#         subgraph_size=64,
#         restart_prob=0.8,
#         # positional_embedding_size=32,
#         step_dist=[1.0, 0.0, 0.0],
#         cat_prone=False,
#     ):
#         super(AnomalyGraphDataset).__init__()
#         assert len(self.graphs) == 1
#         self.num_classes = self.data.y.shape[1]

#     def __getitem__(self, idx):
#         graph_idx = 0
#         node_idx = idx
#         for i in range(len(self.graphs)):
#             if node_idx < self.graphs[i].number_of_nodes():
#                 graph_idx = i
#                 break
#             else:
#                 node_idx -= self.graphs[i].number_of_nodes()

#         traces = dgl.contrib.sampling.random_walk_with_restart(
#             self.graphs[graph_idx],
#             seeds=[node_idx],
#             restart_prob=self.restart_prob,
#             max_nodes_per_seed=self.rw_hops,
#         )

#         graph_q = generate_rwr_subgraph(
#             g=self.graphs[graph_idx],
#             seed=node_idx,
#             trace=traces[0],
#             positional_embedding_size=self.positional_embedding_size,
#         )
#         return graph_q, self.data.y[idx].argmax().item()
    
#     def _create_dgl_graph(self, data):
#         graph = dgl.DGLGraph()
#         src, dst = data.edge_index.tolist()
#         num_nodes = data.edge_index.max() + 1
#         graph.add_nodes(num_nodes)
#         graph.add_edges(src, dst)
#         graph.add_edges(dst, src)
#         graph.readonly()
#         return graph

# class ReassignedDataset(torch.utils.data.Dataset):
#     """A dataset where the new images labels are given in argument.
#     Args:
#         image_indexes (list): list of data indexes
#         pseudolabels (list): list of labels for each data
#         dataset (list): list of tuples with paths to images
#         transform (callable, optional): a function/transform that takes in
#                                         an PIL image and returns a
#                                         transformed version
#     """

#     def __init__(self, nd_indices, pseudolabels, dataset, transform=None):
#         self.all_nodes = self.make_dataset(nd_indices, pseudolabels, dataset)
#         self.transform = transform

#     def make_dataset(self, nd_indices, pseudolabels, dataset):
#         label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
#         all_nodes = []
#         for j, idx in enumerate(nd_indices):
#             pseudolabel = label_to_idx[pseudolabels[j]]
#             all_nodes.append((idx, pseudolabel))
#         return all_nodes

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): index of data
#         Returns:
#             tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
#         """
#         nd_idx, pseudolabel = self.all_nodes[index]
#         return nd_idx, pseudolabel

#     def __len__(self):
#         return len(self.imgs)