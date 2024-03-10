import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set

from models.gat import UnsupervisedGAT
from models.gin import UnsupervisedGIN
from models.mpnn import UnsupervisedMPNN


class GraphEncoder(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 15.
    edge_input_dim : int
        Dimension of input edge feature, default to be 15.
    output_dim : int
        Dimension of prediction, default to be 12.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    num_layer_set2set : int
        Number of set2set layers
    """

    def __init__(
        self,
        positional_embedding_size=32,
        max_node_freq=8,
        max_edge_freq=8,
        max_degree=128,
        freq_embedding_size=32,
        degree_embedding_size=32,
        output_dim=32,
        node_hidden_dim=32,
        edge_hidden_dim=32,
        num_layers=6,
        num_heads=4,
        num_step_set2set=6,
        num_layer_set2set=3,
        norm=False,
        gnn_model="mpnn",
        degree_input=False,
        lstm_as_gate=False,
        target_node=False,
    ):
        super(GraphEncoder, self).__init__()

        if degree_input:
            node_input_dim = positional_embedding_size + degree_embedding_size + 1
        else:
            node_input_dim = positional_embedding_size + 1
        # node_input_dim = (
        #     positional_embedding_size + freq_embedding_size + degree_embedding_size + 3
        # )
        edge_input_dim = freq_embedding_size + 1
        if gnn_model == "mpnn":
            self.gnn = UnsupervisedMPNN(
                output_dim=output_dim,
                node_input_dim=node_input_dim,
                node_hidden_dim=node_hidden_dim,
                edge_input_dim=edge_input_dim,
                edge_hidden_dim=edge_hidden_dim,
                num_step_message_passing=num_layers,
                lstm_as_gate=lstm_as_gate,
            )
        elif gnn_model == "gat":
            self.gnn = UnsupervisedGAT(
                node_input_dim=node_input_dim,
                node_hidden_dim=node_hidden_dim,
                edge_input_dim=edge_input_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        elif gnn_model == "gin":
            self.gnn = UnsupervisedGIN(
                num_layers=num_layers,
                num_mlp_layers=2,
                input_dim=node_input_dim,
                hidden_dim=node_hidden_dim,
                output_dim=output_dim,
                final_dropout=0.5,
                learn_eps=False,
                graph_pooling_type="sum",
                neighbor_pooling_type="sum",
                use_selayer=False,
            )
        self.gnn_model = gnn_model

        self.max_node_freq = max_node_freq
        self.max_edge_freq = max_edge_freq
        self.max_degree = max_degree
        self.degree_input = degree_input

        # self.node_freq_embedding = nn.Embedding(
        #     num_embeddings=max_node_freq + 1, embedding_dim=freq_embedding_size
        # )
        if degree_input:
            self.degree_embedding = nn.Embedding(
                num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size
            )

        # self.edge_freq_embedding = nn.Embedding(
        #     num_embeddings=max_edge_freq + 1, embedding_dim=freq_embedding_size
        # )

        self.set2set = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)
        self.lin_readout = nn.Sequential(
            nn.Linear(2 * node_hidden_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, output_dim),
        )
        self.norm = norm
        self.target_node = target_node

    def forward(self, g):
        """Predict molecule labels

        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.

        Returns
        -------
        res : Predicted labels
        """

        # nfreq = g.ndata["nfreq"]
        if self.degree_input:
            device = g.ndata["seed"].device
            degrees = g.in_degrees()
            if device != torch.device("cpu"):
                degrees = degrees.cuda(device)

            n_feat = torch.cat(
                (  
                    g.ndata["features"], 
                    # g.ndata["pos_undirected"],
                    self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    g.ndata["seed"].unsqueeze(1).float(),
                ),
                dim=-1,
            )
        else:
            n_feat = torch.cat(
                (
                    g.ndata["features"], 
                    # g.ndata["pos_undirected"],
                    # g.ndata["pos_directed"],
                    # self.node_freq_embedding(nfreq.clamp(0, self.max_node_freq)),
                    # self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    g.ndata["seed"].unsqueeze(1).float(),
                    # nfreq.unsqueeze(1).float() / self.max_node_freq,
                    # degrees.unsqueeze(1).float() / self.max_degree,
                ),
                dim=-1,
            )

        # efreq = g.edata["efreq"]
        # e_feat = torch.cat(
        #     (
        #         self.edge_freq_embedding(efreq.clamp(0, self.max_edge_freq)),
        #         efreq.unsqueeze(1).float() / self.max_edge_freq,
        #     ),
        #     dim=-1,
        # )
        e_feat = None

        if self.target_node:
            x = self.gnn(g, n_feat, e_feat, target_node=self.target_node)
            return x
        
        if self.gnn_model == "gin":
            y = self.gnn(g, n_feat, e_feat)
        else:
            y = self.gnn(g, n_feat, e_feat)
            y = self.set2set(g, y)
            y = self.lin_readout(y)
        
        if self.norm:
            y = F.normalize(x, p=2, dim=-1, eps=1e-5)
        return y

class GraphPooling(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



if __name__ == "__main__":
    model = GraphEncoder(gnn_model="gin")
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1, 2], [1, 2, 2, 1])
    g.ndata["pos_directed"] = torch.rand(3, 16)
    g.ndata["pos_undirected"] = torch.rand(3, 16)
    g.ndata["seed"] = torch.zeros(3, dtype=torch.long)
    g.ndata["nfreq"] = torch.ones(3, dtype=torch.long)
    g.edata["efreq"] = torch.ones(4, dtype=torch.long)
    g = dgl.batch([g, g, g])
    y = model(g)
    print(y.shape)
    print(y)
