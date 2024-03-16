import torch
from torch import nn
import torch.nn.functional as F
from models.graph_encoder import GraphEncoder

# def get_Mahalanobis_score(x, n_clusters, centroids, precision):
# 	"""
# 	Compute the proposed Mahalanobis confidence score on input dataset
# 	return: Mahalanobis score
# 	"""
# 	x = x.cuda() # (batch_size, d)
# 	x = x.detach()
	
# 	# compute Mahalanobis score
# 	gaussian_score = 0
# 	for i in range(n_clusters):
# 		batch_sample_mean = centroids[i]
# 		zero_f = x - batch_sample_mean # (batch_size, d)
# 		# (batch_size, d) @ (d, d) @ (d, batch_size)
# 		term_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag() # (1, batch_size)
# 		if i == 0:
# 			gaussian_score = term_gau.view(-1,1) # (batch_size, 1)
# 		else:
# 			gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1) # (batch_size, k)
	
# 	# Input_processing
# 	# pred = gaussian_score.max(1)[1] # (batch_size, )
# 	# batch_sample_mean = centroids.index_select(0, pred) # (batch_size, d)
# 	# zero_f = x - batch_sample_mean # (batch_size, d)
# 	# pure_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag() # (1, batch_size)
# 	# loss = torch.mean(-pure_gau)
# 	# loss.backward()
# 	return gaussian_score

# @torch.no_grad()
# def distributed_sinkhorn(out, world_size, sinkhorn_iterations=3, epsilon=0.05):
# 	Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
# 	B = Q.shape[1] * world_size # number of samples to assign
# 	K = Q.shape[0] # how many prototypes

# 	# make the matrix sums to 1
# 	sum_Q = torch.sum(Q)
# 	Q /= sum_Q

# 	for it in range(sinkhorn_iterations):
# 		# normalize each row: total weight per prototype must be 1/K
# 		sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
# 		Q /= sum_of_rows
# 		Q /= K

# 		# normalize each column: total weight per sample must be 1/B
# 		Q /= torch.sum(Q, dim=0, keepdim=True)
# 		Q /= B

# 	Q *= B # the colomns must sum to 1 so that Q is an assignment
# 	return Q.t()

# class ClusteringLayer(nn.Module):
# 	def __init__(self, n_clusters, hidden_size, centroids=None, precision=None, T=1.0):
# 		super(ClusteringLayer, self).__init__()
# 		self.k = n_clusters 
# 		self.T = T 
# 		self.d = hidden_size 
# 		if centroids is None:
# 			# (k, d)
# 			initial_centroids = torch.zeros(
# 				 self.k,
# 				 self.d, 
# 				 dtype=torch.float
# 			).cuda()
# 			nn.init.xavier_uniform_(initial_centroids)
# 		else:
# 			initial_centroids = centroids
		
# 		if precision is None:
# 			# (d, d)
# 			initial_precision = torch.zeros(
# 					self.d,
# 					self.d, 
# 					dtype=torch.float
# 			).cuda()
# 			nn.init.xavier_uniform_(initial_precision)
# 		else:
# 			initial_precision = precision
# 		self.centroids = nn.Parameter(initial_centroids)
# 		self.precision = nn.Parameter(initial_precision)
	
# 	def forward(self, x):
# 		gaussian_score = get_Mahalanobis_score(x, self.k, self.centroids, self.precision) # (batch_size, k)
# 		score = gaussian_score.exp() / self.T
# 		norm = torch.sum(score, dim=-1)
# 		score = score / norm
# 		return score

class DMC_v1(nn.Module):
	def __init__(self, n_clusters, base_encoder, hidden_size, centroids, transformer):
		"""
		dim: feature dimension (default: 256)
		mlp_dim: hidden dimension in MLPs (default: 4096)
		T: softmax temperature (default: 1.0)
		"""
		super(DMC_v1, self).__init__()

		self.d = hidden_size
		self.k = n_clusters
		# build encoders
		self.base_encoder = base_encoder()
		# define loss function
		self.criterion = nn.CrossEntropyLoss()
		self.centroids = centroids
		self.transformer = transformer.detach()

		# TODO: projection head
		# if output_dim == 0:
		# 	self.projection_head = None
		# else:
		# 	self.projection_head = nn.Linear(hidden_size, output_dim)
	
	# def forward_head(self, x):
	# 	if self.projection_head is not None:
	# 		return self.projection_head(x)
	# 	return x

	def forward(self, x, idx, is_eval=False):
		"""
		Input:
			x1: first views of target nodes
			x2: second views of target nodes
			m: moco momentum
		Output:
			loss
		"""
		if is_eval:
			output = self.base_encoder(x)
			pred = output.data.max(1)[1]
			diff = self.transformer @ (output - self.centroids[pred])
			return torch.mm(diff, diff.t())
		
		output = self.base_encoder(x)
		y = self.centroids[idx]
		diff = self.transformer @ (output-y)
		loss = torch.mm(diff, diff.t())
		return loss
	
# class DMC_v2(nn.Module):
# 	def __init__(self, n_clusters, base_encoder, hidden_size, centroids, output_dim=0, T=1.0):
# 		"""
# 		dim: feature dimension (default: 256)
# 		mlp_dim: hidden dimension in MLPs (default: 4096)
# 		T: softmax temperature (default: 1.0)
# 		"""
# 		super(DMC, self).__init__()

# 		self.T = T
# 		self.d = hidden_size
# 		self.k = n_clusters
# 		# build encoders
# 		self.base_encoder = base_encoder()
# 		self.momentum_encoder = base_encoder()

# 		self.centroids = centroids
# 		self.clusteringlayer = ClusteringLayer(self.k, self.d, self.centroids)

# 		# TODO: projection head
# 		if output_dim == 0:
# 			self.projection_head = None
# 		else:
# 			self.projection_head = nn.Linear(hidden_size, output_dim)

# 	@torch.no_grad()
# 	def _update_momentum_encoder(self, m):
# 		"""Momentum update of the momentum encoder"""
# 		for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
# 			param_m.data = param_m.data * m + param_b.data * (1. - m)

# 	def forward_head(self, x):
# 		if self.projection_head is not None:
# 			return self.projection_head(x)
# 		return x

# 	def forward(self, x1, x2, is_eval=False):
# 		"""
# 		Input:
# 			x1: first views of target nodes
# 			x2: second views of target nodes
# 			m: moco momentum
# 		Output:
# 			loss
# 		"""
# 		if is_eval:
# 			eavl_out = self.clusteringlayer(self.forward_head(self.base_encoder(x1)))
# 			return eavl_out
		
# 		z1 = self.base_encoder(x1)
# 		z2 = self.base_encoder(x2)

# 		# projection
# 		p1 = self.clusteringlayer(self.forward_head(z1))
# 		p2 = self.clusteringlayer(self.forward_head(z2))

# 		# get assignments
# 		q1 = distributed_sinkhorn(z1)
# 		q2 = distributed_sinkhorn(z2)

# 		def log_product(a, b):
# 			return torch.sum(a * F.log_softmax(b, dim=1))

# 		# cluster assignment prediction
# 		loss = - 0.5 * torch.mean(log_product(q1, p2) + log_product(q2, p1))
# 		return loss


def model1(**kwargs):
    base_encoder = GraphEncoder(**kwargs)
    model = DMC_v1(base_encoder)
    return model
