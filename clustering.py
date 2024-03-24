import numpy as np
import torch
import faiss
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


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
	
	# print('performing kmeans clustering')
	results = {'nd2cluster': None,'centroids': None,'density': None}
	
	# intialize faiss clustering parameters
	d = x.shape[1]
	k = int(args.num_clusters)

	# faiss implementation of k-means
	clus = faiss.Clustering(d, k)
	clus.verbose = False
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


def visualize(X, labels, ano_labels, savepath='./fig/cluster.png'):
	"""
	T-SNE Visualization 
	Args:
		X : feature vectors
		labels: predicted labels
	"""
	tsne = TSNE()
	X_tsne = tsne.fit_transform(X)

	plt.figure(figsize=(10, 8))
	sns.set_theme(style="whitegrid", rc={"axes.linewidth": 1, "axes.edgecolor": "black"})

	for label in np.unique(labels):
		plt.scatter(
			X_tsne[labels == label, 0], 
			X_tsne[labels == label, 1], 
			label=f'Cluster {label}', 
			cmap='viridis'
		)
	# plot anomalies
	anomaly_mask = ano_labels == 1
	plt.scatter(
			X_tsne[anomaly_mask, 0], 
			X_tsne[anomaly_mask, 1], 
			label=f'Anomaly', 
			c='black'
		)

	plt.legend(loc='lower right')
	# plt.title("Test Visualization")
	# hide axis
	plt.xticks([])
	plt.yticks([])
	# plt.axis('off')
	
	if savepath:
		plt.savefig(savepath, dpi=300, bbox_inches='tight')


def run_cbof(x, centroids, labels, mode):
	large_clusters, _ = get_clusters(labels)
	distances = []
	for p, label in zip(x, labels):
		if label in large_clusters:
			center = centroids[label]
			d = get_distance(p, center, mode)
		else:
			d = min([get_distance(p, center, mode) for center in centroids[large_clusters]])
		distances.append(d)
	return torch.cat(distances)


def get_clusters(labels):
	sizes = np.unique(labels.numpy(), return_counts=True)[1]
	n_clusters = len(sizes)

	large_clusters = []
	small_clusters = []
	count = 0
	MAX_N_POINT_IN_LARGE_CLUSTER = 0.9 * sum(sizes)
	BETA = 5

	satisfy_alpha = False
	satisfy_beta = False
	for i in range(n_clusters):
		if satisfy_alpha and satisfy_beta:
			small_clusters.append(i)
			continue
	
		count += sizes[i]
		if count > MAX_N_POINT_IN_LARGE_CLUSTER:
			satisfy_alpha = True

		if i < n_clusters-1:
			ratio = sizes[i] / sizes[i + 1]
			if ratio > BETA:
				satisfy_beta = True
	
		large_clusters.append(i)
	return large_clusters, small_clusters


def get_distance(a, b, mode='cos', precision=None):
	diff = a-b
	if mode == 'eu':
		return torch.mm(diff, diff.T)
	if mode == 'ma' and precision:
		return torch.mm(torch.mm(diff, precision), diff.T).diag()
	if mode == 'cos':
		return torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))