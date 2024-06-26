import os
import math
import dgl
import time
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from tqdm import tqdm
from graph_model import Model
from scipy.sparse import csr_matrix
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics.cluster import normalized_mutual_info_score

# from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # for macbook

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', type=int, default=39)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda:0')
# dataset
parser.add_argument('--dataset', type=str, default='cora', 
					choices=["cora", "citeseer", "blogcatalog", 
							 "flickr", "citation", "acm", "pubmed"])
# model definition
parser.add_argument("--model", type=str, default="gcn",
					choices=["gat", "mpnn", "gin", "gcn"])
# num of clusters
parser.add_argument('--nmb_prototypes', type=int, default=9)
parser.add_argument('--num_clusters', type=int, default=9)
# hyperparameters
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
# model parameters
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--test_rounds', type=int, default=100)
# ratio
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument("--freeze_prototypes_niters", default=1e10, type=int,
					help="freeze the prototypes during this many iterations from the start")
args = parser.parse_args()


def train():
	pass
	
def test():
	pass

def compute_features(model, subgraphs, embedding_dim, device):
	model.eval()

	all_feat = torch.zeros(nb_nodes, embedding_dim).to(device)
	# with tqdm(total=batch_num) as pbar_eval:
	#     pbar_eval.set_description('Generating Features')
	for batch_idx in range(batch_num):
		is_final_batch = (batch_idx == (batch_num - 1))
		if not is_final_batch:
			idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
		else:
			idx = all_idx[batch_idx * batch_size:]

		cur_batch_size = len(idx)
		ba = []
		bf = []
		added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
		added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
		added_adj_zero_col[:, -1, :] = 1.
		added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

		for i in idx:
			cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
			cur_feat = features[:, subgraphs[i], :]  # (1, subg_size, ft_size)
			ba.append(cur_adj)
			bf.append(cur_feat)

		ba = torch.cat(ba)
		ba = torch.cat((ba, added_adj_zero_row), dim=1)
		ba = torch.cat((ba, added_adj_zero_col), dim=2)
		bf = torch.cat(bf)
		bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

		feat = model(ba, bf)[0]
		all_feat[idx] = feat
			# pbar_eval.update(1)
	return all_feat.detach()


def cluster_memory(model, local_memory_embeddings, nb_nodes, nmb_kmeans_iters=10):
	K = args.num_clusters
	assignments = -100 * torch.ones(nb_nodes).long()
	with torch.no_grad():
		# run k-means

		# init centroids
		centroids = torch.empty(K, args.embedding_dim).to(device) # (K, d)
		random_idx = torch.randperm(len(local_memory_embeddings))[:K].to(device)
		assert len(random_idx) >= K, "please reduce the number of centroids"
		centroids = local_memory_embeddings[random_idx].to(device) # (K, d)

		for n_iter in range(nmb_kmeans_iters + 1):

			# E step
			# (N, d) @ (d, K) -> (N, K)
			dot_products = torch.mm(local_memory_embeddings, centroids.t())
			_, local_assignments = dot_products.max(dim=1) # max cluster index: (N, 1)

			# finish
			if n_iter == nmb_kmeans_iters:
				break

			# M step
			where_helper = get_indices_sparse(local_assignments.cpu().numpy())
			counts = torch.zeros(K).to(device).int()
			emb_sums = torch.zeros(K, args.embedding_dim).to(device)
			for k in range(len(where_helper)): 
				# iterate every cluster
				if len(where_helper[k][0]) > 0:
					emb_sums[k] = torch.sum(
						local_memory_embeddings[where_helper[k][0]],
						dim=0,
					)
					counts[k] = len(where_helper[k][0])
			mask = counts > 0
			centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

			# normalize centroids
			centroids = nn.functional.normalize(centroids, dim=1, p=2)
		
		# initialize classifier
		model.prototypes.weight.copy_(centroids)

		# gather the assignments
		assignments = local_assignments

	return assignments.detach(), centroids.detach()


def get_indices_sparse(data): # (N, 1)
	cols = np.arange(data.size) # [0, ..., N-1]
	M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
	return [np.unravel_index(row.data, data.shape) for row in M]


def adjust_learning_rate(optimizer, epoch, args):
	"""Decays the learning rate with half-cycle cosine after warmup"""
	if epoch < args.warmup_epochs:
		lr = args.lr * epoch / args.warmup_epochs
	else:
		lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch -
							  args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr


if __name__ == '__main__':
	print('Dataset: {}'.format(args.dataset), flush=True)
	print('k: {}'.format(args.num_clusters), flush=True)
	print('lr: {}'.format(args.lr), flush=True)
	print('ratio: {}'.format(args.alpha), flush=True)
	print('embedding_dim: {}'.format(args.embedding_dim), flush=True)
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	dgl.random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	random.seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	batch_size = args.batch_size
	subgraph_size = args.subgraph_size

	# data preprocess
	adj, features, labels, ano_labels, str_ano_labels, attr_ano_labels = load_mat(
		args.dataset)
	features = preprocess_features(features)
	dgl_graph = adj_to_dgl_graph(adj)

	nb_nodes = features.shape[0]
	all_idx = list(range(nb_nodes))
	ft_size = features.shape[1]

	adj = normalize_adj(adj)
	adj = (adj + sp.eye(adj.shape[0])).todense()

	# adj_hat = aug_random_edge(adj, 0.2)
	# adj = adj.todense()
	# adj_hat = adj_hat.todense()

	features = torch.FloatTensor(features[np.newaxis]).to(device)
	# add perturbations
	# features = gaussian_noised_feature(features)
	adj = torch.FloatTensor(adj[np.newaxis]).to(device)
	# adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(device)
	labels = torch.FloatTensor(labels[np.newaxis]).to(device)

	# build model
	model = Model(ft_size, args.embedding_dim, nmb_prototypes=args.num_clusters, alpha=args.alpha).to(device)
	
	# build optimizer
	if args.optimizer == "sgd":
		optimizer = torch.optim.SGD(
			model.parameters(),
			lr=args.lr,
			momentum=0.9,
			weight_decay=args.weight_decay,
		)
	elif args.optimizer == "adam":
		optimizer = torch.optim.Adam(
			model.parameters(),
			lr=args.lr,
			weight_decay=args.weight_decay,
		)
	elif args.optimizer == "adagrad":
		optimizer = torch.optim.Adagrad(
			model.parameters(),
			lr=args.lr,
			weight_decay=args.weight_decay,
		)
	else:
		raise NotImplementedError

	cnt_wait = 0
	best = 1e9
	best_t = 0
	# iteration length
	batch_num = nb_nodes // batch_size + 1

	# store NMI
	pre_assignments = None
	nmis = []

	end = time.time()
	# train()
	for epoch in range(args.epochs+1):
		total_loss = 0.
		subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
		criterion = nn.CrossEntropyLoss(ignore_index=-100)

		# get features from frozen encoder
		local_memory_embeddings = compute_features(model, subgraphs, args.embedding_dim, device)
		assignments, _ = cluster_memory(model, local_memory_embeddings, nb_nodes)
		
		# calculate NMI between epochs
		cur_assignments = assignments.cpu().numpy()
		if pre_assignments is not None:
			nmi = normalized_mutual_info_score(cur_assignments, pre_assignments)
			nmis.append(nmi)
		pre_assignments = cur_assignments

		if epoch == args.epochs:
			break

		# switch to train mode
		model.train()

		all_idx = list(range(nb_nodes))
		random.shuffle(all_idx)
		
		end = time.time()
		for batch_idx in range(batch_num):
			is_final_batch = (batch_idx == (batch_num - 1))
			if not is_final_batch:
				idx = all_idx[batch_idx *
							  batch_size: (batch_idx + 1) * batch_size]
			else:
				idx = all_idx[batch_idx * batch_size:]

			cur_batch_size = len(idx)
			ba = []
			bf = []

			added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
			added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
			added_adj_zero_col[:, -1, :] = 1.
			added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

			for i in idx:
				cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
				# (1, subg_size, ft_size)
				cur_feat = features[:, subgraphs[i], :]
				ba.append(cur_adj)
				bf.append(cur_feat)

			ba = torch.cat(ba)
			ba = torch.cat((ba, added_adj_zero_row), dim=1)
			ba = torch.cat((ba, added_adj_zero_col), dim=2)
			bf = torch.cat(bf)
			bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

			# adjust learning rate
			# lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
			# learning_rates.update(args.lr)

			# model forward
			emb, output = model(ba, bf) # (2*batch_size, K)
			emb = emb.detach()

			# calculate score
			score = output / args.temperature
			
			loss = criterion(score, assignments[idx])
			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()

			# cancel some gradients
			if epoch < args.freeze_prototypes_niters:
				for name, p in model.named_parameters():
					if "prototypes" in name:
						p.grad = None
			optimizer.step()
			# scaler.scale(loss).backward()
			# scaler.step(optimizer)
			# scaler.update()

			# ============ update memory banks ... ============
			local_memory_embeddings[idx] = emb
		
		mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
		print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)
		# if mean_loss < best:
		# 	best = mean_loss
		# 	best_t = epoch
		# 	cnt_wait = 0
		# 	torch.save(model.state_dict(), './pretrained2/{}.pkl'.format(args.dataset))
		# else:
		# 	cnt_wait += 1

	# ============ Test ============
	# print('Testing AUC!', flush=True)
	# print('Loading {}th epoch'.format(best_t), flush=True)
	# model.load_state_dict(torch.load('./pretrained2/{}.pkl'.format(args.dataset)))

	# switch to eval mode
	model.eval()

	subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
	local_memory_embeddings = compute_features(model, subgraphs, args.embedding_dim, device) # (N, d)
	# initialize prototypes for classification
	_, _ = cluster_memory(model, local_memory_embeddings, nb_nodes)
	
	multi_round_ano_score = np.zeros((args.test_rounds, nb_nodes))
	for round in range(args.test_rounds):
		all_idx = list(range(nb_nodes))
		random.shuffle(all_idx)
		subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

		for batch_idx in range(batch_num):

			is_final_batch = (batch_idx == (batch_num - 1))
			if not is_final_batch:
				idx = all_idx[batch_idx *
							batch_size: (batch_idx + 1) * batch_size]
			else:
				idx = all_idx[batch_idx * batch_size:]

			cur_batch_size = len(idx)
			ba = []
			bf = []

			added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
			added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
			added_adj_zero_col[:, -1, :] = 1.
			added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

			for i in idx:
				cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
				cur_feat = features[:, subgraphs[i], :]
				ba.append(cur_adj)
				bf.append(cur_feat)

			ba = torch.cat(ba)
			ba = torch.cat((ba, added_adj_zero_row), dim=1)
			ba = torch.cat((ba, added_adj_zero_col), dim=2)
			bf = torch.cat(bf)
			bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

			# ===================forward=====================
			with torch.no_grad():
				output = model(ba, bf)[1]
				# score = F.softmax(output / args.temperature, dim=1)
				score = output / args.temperature
				pred = score.max(1)[1]
				pure_gau = score[torch.arange(score.shape[0]), pred].unsqueeze(dim=-1) # (batch_size, 1)
				ano_score = -pure_gau.squeeze(1).cpu().numpy()

			multi_round_ano_score[round, idx] = ano_score

	ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
	auc = roc_auc_score(ano_labels, ano_score_final)
	print('Testing AUC:{:.4f}'.format(auc), flush=True)
	# print('Total Time: {0:.0f} s'.format(time.time() - end))
