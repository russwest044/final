import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):

	def __init__(self, in_ft, out_ft, act, bias=True):
		super(GCN, self).__init__()
		self.fc = nn.Linear(in_ft, out_ft, bias=False)
		self.act = nn.PReLU() if act == 'prelu' else act
		
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

	def forward(self, seq, adj, sparse=False):
		seq_fts = self.fc(seq)
		if sparse:
			out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
		else:
			out = torch.bmm(adj, seq_fts)
		if self.bias is not None:
			out += self.bias
		
		return self.act(out)

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