import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):

	def __init__(self, in_ft, out_ft, act, bias=True):
		super(GCN, self).__init__()
		self.fc = nn.Linear(in_ft, out_ft, bias=False)
		self.act = nn.PReLU() if act == 'prelu' else act
		self.dropout = nn.Dropout(0.2)
		
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_ft))
			self.bias.data.fill_(0.0)
		else:
			self.register_parameter('bias', None)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, adj, seq, sparse=False):
		seq_fts = self.fc(seq)
		if sparse:
			out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
		else:
			out = torch.bmm(adj, seq_fts)
		if self.bias is not None:
			out += self.bias
		
		out = self.act(out)
		
		return out

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


class Contextual_Discriminator(nn.Module):

	def __init__(self, n_h, negsamp_round):
		super(Contextual_Discriminator, self).__init__()
		self.f_k = nn.Bilinear(n_h, n_h, 1)
		for m in self.modules():
			self.weights_init(m)
		self.negsamp_round = negsamp_round

	def weights_init(self, m):
		if isinstance(m, nn.Bilinear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
		scs = []
		scs.append(self.f_k(h_pl, c))
		c_mi = c
		for _ in range(self.negsamp_round):
			c_mi = torch.cat((c_mi[-2:-1,:], c_mi[:-1,:]),0)
			scs.append(self.f_k(h_pl, c_mi))
		logits = torch.cat(tuple(scs))
		return logits

class Patch_Discriminator(nn.Module):

	def __init__(self, n_h, negsamp_round):
		super(Patch_Discriminator, self).__init__()
		self.f_k = nn.Bilinear(n_h, n_h, 1)
		for m in self.modules():
			self.weights_init(m)
		self.negsamp_round = negsamp_round

	def weights_init(self, m):
		if isinstance(m, nn.Bilinear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, h_ano, h_unano, s_bias1=None, s_bias2=None):
		scs = []
		scs.append(self.f_k(h_unano, h_ano))
		h_mi = h_ano
		for _ in range(self.negsamp_round):
			h_mi = torch.cat((h_mi[-2:-1, :], h_mi[:-1, :]), 0)
			scs.append(self.f_k(h_unano, h_mi))
		logits = torch.cat(tuple(scs))
		return logits


class Model(nn.Module):

	def __init__(self, n_in, n_h, activation='prelu', nmb_prototypes=0, readout='min', alpha=1.0):
		super(Model, self).__init__()

		self.alpha = alpha
		self.read_mode = readout
		self.gcn_context = GCN(n_in, n_h, activation)
		# self.gcn_patch = GCN(n_in, n_h, activation)

		if readout == 'max':
			self.read = MaxReadout()
		elif readout == 'min':
			self.read = MinReadout()
		elif readout == 'avg':
			self.read = AvgReadout()
		elif readout == 'weighted_sum':
			self.read = WSReadout()

		# self.c_disc = Contextual_Discriminator(n_h, negsamp_round_context)
		# self.p_disc = Patch_Discriminator(n_h, negsamp_round_patch)
		# prototype layer
		self.prototypes = None
		if isinstance(nmb_prototypes, list):
			self.prototypes = MultiPrototypes(n_h, nmb_prototypes)
		elif nmb_prototypes > 0:
			self.prototypes = nn.Linear(n_h, nmb_prototypes, bias=False)

	def forward(self, adj, seq1, sparse=False, msk=None, samp_bias1=None, samp_bias2=None):
		h_1 = self.gcn_context(adj, seq1, sparse)
		# h_2 = self.gcn_patch(adj, seq1, sparse)

		if self.read_mode != 'weighted_sum':
			c = self.read(h_1[:, :-1, :])
			h_mv = h_1[:, -1, :]
			# h_unano = h_2[:, -1, :]
			# h_ano = h_2[:, -2, :]
		else:
			c = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
			h_mv = h_1[:, -1, :]
			# h_unano = h_2[:, -1, :]
			# h_ano = h_2[:, -2, :]
		
		out = self.alpha * c + (1-self.alpha) * h_mv
		# out = self.read(h_1)

		if self.prototypes is not None:
			return out, self.prototypes(out)

		return out

		# ret1 = self.c_disc(c, h_mv, samp_bias1, samp_bias2)
		# ret2 = self.p_disc(h_ano, h_unano, samp_bias1, samp_bias2)

		# return ret1, ret2, c, h_mv
		