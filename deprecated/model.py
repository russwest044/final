import torch
from torch import nn
from sklearn.manifold import TSNE
# from matplotlib import pyplot as plt
import torch.nn.functional as F


def get_Mahalanobis_score(x, n_clusters, centroids, precision):
	"""
	Compute the proposed Mahalanobis confidence score on input dataset
	return: Mahalanobis score
	"""
	# compute Mahalanobis score
	gaussian_score = 0
	for i in range(n_clusters):
		batch_sample_mean = centroids[i].detach()
		zero_f = x - batch_sample_mean # (batch_size, d)
		# (batch_size, d) @ (d, d) @ (d, batch_size)
		term_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag() # (1, batch_size)
		if i == 0:
			gaussian_score = term_gau.view(-1,1) # (batch_size, 1)
		else:
			gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1) # (batch_size, k)
	
	# Input_processing
	pred = gaussian_score.max(1)[1] # (batch_size, )
	batch_sample_mean = centroids.index_select(0, pred) # (batch_size, d)
	zero_f = x - batch_sample_mean # (batch_size, d)
	pure_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()# (1, batch_size)
	return -pure_gau


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

class DMC(nn.Module):
	def __init__(self, n_clusters, base_encoder, hidden_size):
		"""
		dim: feature dimension (default: 256)
		mlp_dim: hidden dimension in MLPs (default: 4096)
		T: softmax temperature (default: 1.0)
		"""
		super(DMC, self).__init__()

		self.d = hidden_size
		self.k = n_clusters
		# build encoders
		self.base_encoder = base_encoder
		# define loss function
		# self.cluster_result = cluster_result
		# self.transformer = transformer.detach()

		# TODO: projection head
		# if output_dim == 0:
		# 	self.projection_head = None
		# else:
		# 	self.projection_head = nn.Linear(hidden_size, output_dim)
	
	# def forward_head(self, x):
	# 	if self.projection_head is not None:
	# 		return self.projection_head(x)
	# 	return x

	def forward(self, x, cluster_result=None, precision=None, is_eval=False):
		"""
		Input:
			x1: first views of target nodes
			x2: second views of target nodes
			m: moco momentum
		Output:
			loss
		"""
		output = self.base_encoder(x)
		guassian_score = get_Mahalanobis_score(output, self.k, cluster_result['centroids'], precision)
		if is_eval:
			# pred = guassian_score.max(1)[1] # (batch_size, )
			# pred_score = guassian_score[torch.arange(guassian_score.shape[0]), pred].unsqueeze(dim=-1) # (batch_size, 1)
			return F.sigmoid(guassian_score)
		loss = torch.sum(guassian_score)
		return loss
	
	# def visualize(self, epoch,x):
	# 	fig = plt.figure()
	# 	ax = plt.subplot(111)
	# 	x = self.base_encoder.detach() 
	# 	x = x.cpu().numpy()[:2000]
	# 	x_embedded = TSNE(n_components=2).fit_transform(x)
	# 	plt.scatter(x_embedded[:,0], x_embedded[:,1])
	# 	fig.savefig('plots/mnist_{}.png'.format(epoch))
	# 	plt.close(fig)


class SwAV(nn.Module):
	def __init__(
			self, 
			encoder, 
			hidden_size, 
			nmb_prototypes=0, 
			output_dim=0, 
			norm=True, 
			T=1.0
	):
		"""
		dim: feature dimension (default: 256)
		mlp_dim: hidden dimension in MLPs (default: 4096)
		T: softmax temperature (default: 1.0)
		"""
		super(SwAV, self).__init__()

		self.T = T
		self.norm = norm
		self.d = hidden_size
		self.k = nmb_prototypes
		# build encoders
		self.encoder = encoder

		# projection head
		# if output_dim == 0:
		# 	self.projection_head = None
		# elif hidden_mlp == 0:
		# 	self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
		# else:
		# 	self.projection_head = nn.Sequential(
		# 		nn.Linear(num_out_filters * block.expansion, hidden_mlp),
		# 		nn.BatchNorm1d(hidden_mlp),
		# 		nn.ReLU(inplace=True),
		# 		nn.Linear(hidden_mlp, output_dim),
		# 	)

		# prototype layer
		self.prototypes = None
		if isinstance(nmb_prototypes, list):
			self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
		elif nmb_prototypes > 0:
			# self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
			self.prototypes = nn.Linear(hidden_size, nmb_prototypes, bias=False)

		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
		# 	elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
		# 		nn.init.constant_(m.weight, 1)
		# 		nn.init.constant_(m.bias, 0)

		# TODO: projection head
		if output_dim == 0:
			self.projection_head = None
		else:
			self.projection_head = nn.Linear(hidden_size, output_dim)

	def forward_head(self, x):
		if self.projection_head is not None:
			return self.projection_head(x)
		
		if self.norm:
			x = nn.functional.normalize(x, dim=-1, p=2)
	
		if self.prototypes is not None:
			return x, self.prototypes(x)
		
		return x

	def forward(self, x1, x2, is_eval=False):
		"""
		Input:
			x1: first views of target nodes
			x2: second views of target nodes
			m: moco momentum
		Output:
			loss
		"""
		_, p1 = self.forward_head(self.encoder(x1)) # (batch_size, k)
		_, p2 = self.forward_head(self.encoder(x2))

		if is_eval:
			def get_score(x):
				x = F.normalize(x, dim=-1)
				pred = x.max(1)[1] # (batch_size, )
				pred_score = x[torch.arange(x.shape[0]), pred].unsqueeze(dim=-1) # (batch_size, 1)
				# anomaly = (x.sum(dim=-1).unsqueeze(dim=-1) - pred_score) / (self.k-1) - pred_score
				anomaly = pred_score
				return anomaly.squeeze(dim=-1)
			scores = 0.8*get_score(p1) + 0.2*get_score(p2)
			return scores
		
		# get assignments
		q1 = distributed_sinkhorn(p1)
		q2 = distributed_sinkhorn(p2)

		def log_product(a, b):
			return torch.sum(a * F.log_softmax(b / self.T, dim=1), dim=1)
		
		# cluster assignment prediction
		loss = -0.5 * torch.mean(log_product(q1, p2) + log_product(q2, p1))
		return loss


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

@torch.no_grad()
def distributed_sinkhorn(out, world_size=-1, sinkhorn_iterations=3, epsilon=0.05):
	Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
	B = Q.shape[1] * world_size # number of samples to assign
	K = Q.shape[0] # how many prototypes

	# make the matrix sums to 1
	sum_Q = torch.sum(Q)
	Q /= sum_Q

	for it in range(sinkhorn_iterations):
		# normalize each row: total weight per prototype must be 1/K
		sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
		Q /= sum_of_rows
		Q /= K

		# normalize each column: total weight per sample must be 1/B
		Q /= torch.sum(Q, dim=0, keepdim=True)
		Q /= B

	Q *= B # the colomns must sum to 1 so that Q is an assignment
	return Q.t()