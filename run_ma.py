import os
import math
import dgl
import time
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
# from tqdm import tqdm
from utils import *
import clustering
from graph_model import Model
# from models.graph_encoder import GraphEncoder
from sklearn.metrics import roc_auc_score

# from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', type=int, default=39)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda:0')
# dataset
parser.add_argument('--dataset', type=str, default='citeseer', 
					choices=["cora", "citeseer", "blogcatalog", 
							 "flickr", "citation", "acm", "pubmed"])
# model definition
parser.add_argument("--model", type=str, default="gcn",
					choices=["gat", "mpnn", "gin", "gcn"])
# num of clusters
parser.add_argument('--num_clusters', type=int, default=9)
# hyperparameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--sgd_momentum', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--test_rounds', type=int, default=100)

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

		feat = model(ba, bf)
		all_feat[idx] = feat
			# pbar_eval.update(1)
	return all_feat.detach().cpu().numpy()


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
	device = args.device

	print('Dataset: {}'.format(args.dataset), flush=True)
	print('k: {}'.format(args.num_clusters), flush=True)
	print('lr: {}'.format(args.lr), flush=True)
	print('ratio: {}'.format(args.alpha), flush=True)
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	random.seed(args.seed)
	dgl.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
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

	adj_hat = aug_random_edge(adj, 0.2)
	adj = adj.todense()
	adj_hat = adj_hat.todense()

	features = torch.FloatTensor(features[np.newaxis]).to(device)
	# add perturbations
	# features = gaussian_noised_feature(features)
	adj = torch.FloatTensor(adj[np.newaxis]).to(device)
	adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(device)
	labels = torch.FloatTensor(labels[np.newaxis]).to(device)

	# build model
	gnn_args = {
		'ft_size': ft_size,
		'output_dim': args.embedding_dim,
		'gnn_model': args.model
	}

	# model = GraphEncoder(**gnn_args)
	model = Model(ft_size, args.embedding_dim, nmb_prototypes=0, alpha=args.alpha).to(device)
	
	# build optimizer
	if args.optimizer == "sgd":
		optimizer = torch.optim.SGD(
			model.parameters(),
			lr=args.lr,
			momentum=args.sgd_momentum,
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

	# all_auc = [] 
	# loss function
	criterion = nn.CrossEntropyLoss()

	cnt_wait = 0
	best = 1e9
	best_t = 0
	# iteration length
	batch_num = nb_nodes // batch_size + 1

	# training
	for epoch in range(args.epochs):
		# print("==> training...")
		# start_time = time.time()
		total_loss = 0.
		subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

		# placeholder for clustering result
		cluster_result = {
			# (N, )
			'nd2cluster': torch.zeros(nb_nodes, dtype=torch.long),
			# (k, d)
			'centroids': torch.zeros(int(args.num_clusters), args.embedding_dim),
			# (k, )
			# 'density': torch.zeros(int(args.num_clusters))
		}

		# get features from frozen encoder
		eval_feat = compute_features(model, subgraphs, args.embedding_dim, device)

		# Kmeans
		cluster_result = clustering.run_kmeans(eval_feat, args)
		# print(np.unique(cluster_result['nd2cluster'].numpy(), return_counts=True))

		eval_feat = torch.FloatTensor(eval_feat).detach()
		nd_lists = [[] for i in range(args.num_clusters)]  # (k, N_k)
		for i in range(nb_nodes):
			nd_lists[cluster_result['nd2cluster'][i]].append(i)  # (k, N_k)

		centroids = cluster_result['centroids'].detach()
		precision = clustering.sample_estimator(nd_lists, eval_feat, centroids, args)
		precision = precision.detach()
		assignments = cluster_result['nd2cluster'].to(device)
		centroids = centroids.to(device)
		precision = precision.to(device)

		# cluster visualization
		# if epoch % args.print_cluster_freq == 0:
		#     clustering.visualize(eval_feat, cluster_result['nd2cluster'], 
		#                          ano_labels, savepath='./fig/' + f'cluster_{epoch}.png')

		# switch to train mode
		model.train()
		
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
			output = model(ba, bf)
			guassian_score = clustering.get_Mahalanobis_score(
				output, args.num_clusters, centroids, precision)
			
			loss = criterion(guassian_score, assignments[idx])
			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# scaler.scale(loss).backward()
			# scaler.step(optimizer)
			# scaler.update()
		
		mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
		print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)
		# if mean_loss < best:
		# 	best = mean_loss
		# 	best_t = epoch
		# 	cnt_wait = 0
		# 	torch.save(model.state_dict(), './pretrained1/{}.pkl'.format(args.dataset))
		# else:
		# 	cnt_wait += 1
	
	# ============ Test ============
	print('Testing AUC!', flush=True)
	# print('Loading {}th epoch'.format(best_t), flush=True)
	# model.load_state_dict(torch.load('./pretrained1/{}.pkl'.format(args.dataset)))

	# switch to eval mode
	model.eval()

	multi_round_ano_score = np.zeros((args.test_rounds, nb_nodes))
	# with tqdm(total=args.test_rounds) as pbar_test:
	#     pbar_test.set_description('Testing')
	for round in range(args.test_rounds):
		all_idx = list(range(nb_nodes))
		random.shuffle(all_idx)
		subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
		# Kmeans
		eval_feat = compute_features(model, subgraphs, args.embedding_dim, device)
		cluster_result = clustering.run_kmeans(eval_feat, args)

		nd_lists = [[] for i in range(args.num_clusters)]  # (k, N_k)
		for i in range(nb_nodes):
			nd_lists[cluster_result['nd2cluster'][i]].append(i)  # (k, N_k)

		eval_feat = torch.FloatTensor(eval_feat).detach()
		centroids = cluster_result['centroids'].detach()
		precision = clustering.sample_estimator(nd_lists, eval_feat, centroids, args)
		precision = precision.detach()
		
		gaussian_score = clustering.get_Mahalanobis_score(
			eval_feat, args.num_clusters, centroids, precision)
		pred = gaussian_score.max(1)[1]
		pure_gau = gaussian_score[torch.arange(
			gaussian_score.shape[0]), pred].unsqueeze(dim=-1)
		ano_score = -pure_gau.squeeze(1)

		multi_round_ano_score[round] = ano_score
		# pbar_test.update(1)

	ano_score_final = np.mean(multi_round_ano_score,
							  axis=0) + np.std(multi_round_ano_score, axis=0)
	auc = roc_auc_score(ano_labels, ano_score_final)
	# all_auc.append(auc)
	print('Testing AUC:{:.4f}'.format(auc), flush=True)
