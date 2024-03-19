import numpy as np
import torch
import faiss
from tqdm import tqdm
import torch.nn.functional as F
# from dataset import ReassignedDataset


def sample_estimator(nd_lists, features, centroids, args):
	"""
	compute precision (inverse of covariance)

	Input:
		nd_lists (k, N_k): for each cluster (list), 
					the list of node indices belonging to this cluster
		features (N, d): output of model
		centroids (k, d): centroids of each cluster
	Return: 
		precision (d, d): precision
	"""
	import sklearn.covariance
	group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

	k = int(args.num_clusters)
	X = 0
	for i in range(k):
		if i == 0:
			X = features[nd_lists[i]] - centroids[i] # (N_k, d)
		else:
			X = torch.cat((X, features[nd_lists[i]] - centroids[i]), 0) # (N, d)
			
	# find inverse            
	group_lasso.fit(X.cpu().numpy())
	precision = group_lasso.precision_ # (d, d)
	precision = torch.from_numpy(precision).float()

	return precision


def get_Mahalanobis_score(x, n_clusters, centroids, precision):
	"""
	Compute the proposed Mahalanobis confidence score on input dataset
	return: Mahalanobis score
	"""
	# compute Mahalanobis score
	# centroids = centroids.detach()
	# precision = precision.detach()
	gaussian_score = 0
	for i in range(n_clusters):
		batch_sample_mean = centroids[i]
		zero_f = x - batch_sample_mean # (batch_size, d)
		# (batch_size, d) @ (d, d) @ (d, batch_size)
		term_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag() # (1, batch_size)
		if i == 0:
			gaussian_score = term_gau.view(-1,1) # (batch_size, 1)
		else:
			gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1) # (batch_size, k)
	
	# gaussian_score = F.normalize(gaussian_score, dim=-1)
	# print(gaussian_score[:3])
	# gaussian_score = normalize(gaussian_score)
	# print(gaussian_score[:3])
	# Input_processing
	# pred = gaussian_score.min(1)[1] # (batch_size, )
	# pure_gau = gaussian_score[torch.arange(gaussian_score.shape[0]), pred].unsqueeze(dim=-1)
	# print(pure_gau[:3])
	# batch_sample_mean = centroids.index_select(0, pred) # (batch_size, d)
	# zero_f = x - batch_sample_mean # (batch_size, d)
	# pure_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()# (1, batch_size)
	# print(pure_gau[:3])
	# return pure_gau.squeeze(1)
	return gaussian_score


def compute_features(eval_loader, model, embedding_dim, device):
	print('Computing features...')
	model.eval()
	features = torch.zeros(len(eval_loader.dataset), embedding_dim).to(device)
	for data, index in tqdm(eval_loader):
		data = data
		data = data.to(device)
		feat = model(data) # (batch_size, d)
		features[index] = feat 
	return features.detach()


def preprocess_features(npdata, pca=256):
	"""Preprocess an array of features.
	Args:
		npdata (np.array N * ndim): features to preprocess
		pca (int): dim of output
	Returns:
		np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
	"""
	_, ndim = npdata.shape
	npdata =  npdata.astype('float32')

	# Apply PCA-whitening with Faiss
	mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
	mat.train(npdata)
	assert mat.is_trained
	npdata = mat.apply_py(npdata)

	# L2 normalization
	row_sums = np.linalg.norm(npdata, axis=1)
	npdata = npdata / row_sums[:, np.newaxis]

	return npdata


def learn_metric(X, y):
	from metric_learn import LMNN

	lmnn = LMNN(n_neighbors=3, max_iter=2)
	lmnn.fit(X, y)

	# X_transformed = lmnn.transform(X)
	W = lmnn.components_

	return torch.tensor(W, dtype=torch.float32)
	

def run_kmeans(x, args):
	"""
	Args:
		x: data to be clustered
	"""
	
	print('performing kmeans clustering')
	results = {'nd2cluster': None,'centroids': None,'density': None}
	
	# intialize faiss clustering parameters
	d = x.shape[1]
	k = int(args.num_clusters)

	# faiss implementation of k-means
	clus = faiss.Clustering(d, k)
	clus.verbose = True
	clus.niter = 20
	clus.nredo = 5
	clus.seed = args.seed
	clus.max_points_per_centroid = 1000
	clus.min_points_per_centroid = 20

	# res = faiss.StandardGpuResources()
	# cfg = faiss.GpuIndexFlatConfig()
	# cfg.useFloat16 = False
	# cfg.device = args.gpu    
	# index = faiss.GpuIndexFlatL2(res, d, cfg)
	index = faiss.IndexFlatL2(d)

	# perform clustering
	clus.train(x, index)   

	D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
	nd2cluster = [int(n[0]) for n in I]
	
	# get cluster centroids
	centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
	
	# sample-to-centroid distances for each cluster 
	# Dcluster = [[] for c in range(k)]          
	# for im,i in enumerate(nd2cluster):
	#     Dcluster[i].append(D[im][0])
	
	# # concentration estimation (phi)        
	# density = np.zeros(k)
	# for i,dist in enumerate(Dcluster):
	#     if len(dist)>1:
	#         d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
	#         density[i] = d     
			
	# #if cluster only has one point, use the max to estimate its concentration        
	# dmax = density.max()
	# for i,dist in enumerate(Dcluster):
	#     if len(dist)<=1:
	#         density[i] = dmax 

	# density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
	# density = args.temperature*density/density.mean()  #scale the mean to temperature 
	
	# convert to cuda Tensors for broadcast
	centroids = torch.Tensor(centroids)
	centroids = F.normalize(centroids, p=2, dim=1)    

	nd2cluster = torch.LongTensor(nd2cluster)          
	# density = torch.Tensor(density)
	
	results['centroids'] = centroids
	# results['density'] = density
	results['nd2cluster'] = nd2cluster 
		
	return results

# def cluster_assign(nd_lists, dataset):
#     """Creates a dataset from clustering, with clusters as labels.
#     Args:
#         images_lists (list of list): for each cluster, the list of image indexes
#                                     belonging to this cluster
#         dataset (list): initial dataset
#     Returns:
#         ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
#                                                      labels
#     """
#     assert nd_lists is not None
#     pseudolabels = []
#     nd_indices = []
#     for cluster, nodes in enumerate(nd_lists): # (k, N_k)
#         nd_indices.extend(nodes)
#         pseudolabels.extend([cluster] * len(nodes))

#     return ReassignedDataset(nd_indices, pseudolabels, dataset)