import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.model_zoo.chem.gnn import GCNLayer
from dgl.nn.pytorch import AvgPooling, Set2Set

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

class UnsupervisedGCN(nn.Module):
    def __init__(
        self,
        node_input_dim, 
        hidden_size=64,
        num_layer=2,
        readout="root",
        layernorm: bool = False,
        set2set_lstm_layer: int = 3,
        set2set_iter: int = 6,
    ):
        super(UnsupervisedGCN, self).__init__()
        self.layers = nn.ModuleList(
            [
                GCNLayer(
                    in_feats=node_input_dim if i == 0 else hidden_size,
                    out_feats=hidden_size,
                    activation=F.relu if i + 1 < num_layer else None,
                    residual=False,
                    batchnorm=False,
                    dropout=0.0,
                )
                for i in range(num_layer-1)
            ]
        )
        if readout == "avg":
            self.readout = AvgPooling()
        elif readout == "set2set":
            self.readout = Set2Set(
                hidden_size, n_iters=set2set_iter, n_layers=set2set_lstm_layer
            )
            self.linear = nn.Linear(2 * hidden_size, hidden_size)
        elif readout == "root":
            # HACK: process outside the model part
            self.readout = lambda _, x: x
        else:
            raise NotImplementedError
        self.layernorm = layernorm
        if layernorm:
            self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
            # self.ln = nn.BatchNorm1d(hidden_size, affine=False)

    def forward(self, g, feats, efeats=None):
        for layer in self.layers:
            feats = layer(g, feats) # (B*S, d)
        feats = self.readout(g, feats)
        if isinstance(self.readout, Set2Set):
            feats = self.linear(feats)
        if self.layernorm:
            feats = self.ln(feats)
        return feats
    

if __name__ == "__main__":
    model = UnsupervisedGCN(node_input_dim=64)
    # print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    feat = torch.rand(3, 64)
    print(model(g, feat).shape)
