import dgl
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from dgl.nn.pytorch import AvgPooling, Set2Set

from models.gat import UnsupervisedGAT
from models.gin import UnsupervisedGIN
from models.mpnn import UnsupervisedMPNN
from models.gcn import UnsupervisedGCN

from utils import adj_to_dgl_graph

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

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
		ft_size,
		output_dim=64,
		node_hidden_dim=64,
		edge_hidden_dim=64,
		num_layers=2,
		num_heads=4,
		num_step_set2set=2,
		num_layer_set2set=2,
		norm=True,
		gnn_model="mpnn",
		lstm_as_gate=False,
		readout='avg',
		nmb_prototypes=0, 
	):
		super(GraphEncoder, self).__init__()

		node_input_dim = ft_size
		if gnn_model == "mpnn":
			self.gnn = UnsupervisedMPNN(
				output_dim=output_dim,
				node_input_dim=node_input_dim,
				node_hidden_dim=node_hidden_dim,
				num_step_message_passing=num_layers,
				lstm_as_gate=lstm_as_gate,
			)
		elif gnn_model == "gat":
			self.gnn = UnsupervisedGAT(
				node_input_dim=node_input_dim,
				node_hidden_dim=node_hidden_dim,
				edge_input_dim=edge_hidden_dim, 
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
		elif gnn_model == "gcn":
			self.gnn = UnsupervisedGCN(
				node_input_dim=node_input_dim, 
				hidden_size=node_hidden_dim,
				num_layer=num_layers,
				readout="root",
			)
		self.gnn_model = gnn_model

		if readout == "avg":
			self.readout = AvgReadout()
		elif readout == "root":
			self.readout = lambda _, x: x
		else:
			raise NotImplementedError

		# self.set2set = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)
		# self.lin_readout = nn.Sequential(
		# 	nn.Linear(2 * node_hidden_dim, node_hidden_dim),
		# 	nn.ReLU(),
		# 	nn.Linear(node_hidden_dim, output_dim),
		# )
		self.norm = norm

		# prototype layer
		self.prototypes = None
		if isinstance(nmb_prototypes, list):
			self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
		elif nmb_prototypes > 0:
			self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

	def forward(self, adj, n_feat, e_feat=None, return_all_outputs=False, get_node=False):
		"""Predict molecule labels

		Parameters
		----------
		adj : (batch_size, subgraph_size, subgraph_size)
		n_feat : (batch_size, subgraph_size, ft_size)

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
		batch = []
		for i in range(adj.shape[0]):
			sparse_adj = sp.coo_matrix(adj[i])
			src, dst = sparse_adj.row, sparse_adj.col
			graph = dgl.DGLGraph()
			graph.add_nodes(adj[i].shape[0])
			graph.add_edges(src, dst)
			batch.append(graph)
		
		batch_size = len(batch)
		# get batch dglGraph
		g = dgl.batch(batch)
		n_feat = torch.flatten(n_feat, 0, 1)
		
		if get_node:
			if self.gnn_model == "gin":
				y, all_outputs = self.gnn(g, n_feat, e_feat)
			else:
				y = self.gnn(g, n_feat, e_feat)
				y = y.view(batch_size, -1, y.shape[-1])
				h = y[:, -1, :].squeeze(1) # (batch_size, d)
				y = y[:, :-1, :] # (batch_size, s, d)
				y = self.readout(y) # (batch_size, d)
				y = torch.cat((y, h), dim=0)
		else:
			if self.gnn_model == "gin":
					y, all_outputs = self.gnn(g, n_feat, e_feat)
			else:
				y = self.gnn(g, n_feat, e_feat)
				# y = self.set2set(g, y)
				# y = self.lin_readout(y)
				y = y.view(batch_size, -1, y.shape[-1])
				y = self.readout(y) # (batch_size, d)
		
		if self.norm:
			y = F.normalize(y, p=2, dim=-1, eps=1e-5)
		
		if self.prototypes is not None:
			return y, self.prototypes(y)

		if return_all_outputs: # gin
			return y, all_outputs
		return y


class AvgReadout(nn.Module):

	def __init__(self):
		super(AvgReadout, self).__init__()

	def forward(self, seq):
		return torch.mean(seq, 1)

class MaxReadout(nn.Module):

	def __init__(self):
		super(MaxReadout, self).__init__()

	def forward(self, seq):
		return torch.max(seq,1).values

class MinReadout(nn.Module):

	def __init__(self):
		super(MinReadout, self).__init__()

	def forward(self, seq):
		return torch.min(seq, 1).values

class WSReadout(nn.Module):

	def __init__(self):
		super(WSReadout, self).__init__()

	def forward(self, seq, query):
		query = query.permute(0,2,1)
		sim = torch.matmul(seq,query)
		sim = F.softmax(sim,dim=1)
		sim = sim.repeat(1, 1, 64)
		out = torch.mul(seq,sim)
		out = torch.sum(out,1)
		return out


class MultiPrototypes(nn.Module):
	def __init__(self, output_dim, nmb_prototypes):
		super(MultiPrototypes, self).__init__()
		self.nmb_heads = len(nmb_prototypes)
		for i, k in enumerate(nmb_prototypes):
			self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

	def forward(self, x):
		out = []
		for i in range(self.nmb_heads):
			out.append(getattr(self, "prototypes" + str(i))(x))
		return out
	

if __name__ == "__main__":
	g = dgl.DGLGraph()
	g.add_nodes(3)
	g.add_edges([0, 0, 1, 2], [1, 2, 2, 1])
	g.ndata["features"] = torch.rand(3, 16)
	g = dgl.batch([g, g, g])
	model = GraphEncoder(ft_size=16, gnn_model="gin")
	# print(model)
	y = model(g)
	print(y.shape)
	print(y)
