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