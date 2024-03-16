import numpy as np
import torch
import torch.nn as nn
import faiss
import tqdm
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
    precision = torch.from_numpy(precision).float().cuda()

    return precision


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    for _, (data, index) in enumerate(tqdm(eval_loader)):
        data = data.detach()
        data = data.cuda(non_blocking=True)
        feat = model(data, is_eval=True) # (batch_size, d)
        features[index] = feat 
    return features.cpu()


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

    lmnn = LMNN(k=3, max_iter=10)
    lmnn.fit(X, y)

    # X_transformed = lmnn.transform(X)
    W = lmnn.transformer_weight_

    return W
    

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
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = args.gpu    
    index = faiss.GpuIndexFlatL2(res, d, cfg)  

    # perform clustering
    clus.train(x, index)   

    D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
    nd2cluster = [int(n[0]) for n in I]
    
    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
    
    # sample-to-centroid distances for each cluster 
    Dcluster = [[] for c in range(k)]          
    for im,i in enumerate(nd2cluster):
        Dcluster[i].append(D[im][0])
    
    # concentration estimation (phi)        
    density = np.zeros(k)
    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
            density[i] = d     
            
    #if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max()
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax 

    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = args.temperature*density/density.mean()  #scale the mean to temperature 
    
    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)    

    nd2cluster = torch.LongTensor(nd2cluster).cuda()               
    density = torch.Tensor(density).cuda()
    
    results['centroids'] = centroids
    results['density'] = density
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