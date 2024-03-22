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
from utils import *
import clustering
from models.graph_encoder import GraphEncoder
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
parser.add_argument('--nmb_prototypes', type=int, default=7)
parser.add_argument('--num_clusters', type=int, default=7)
# hyperparameters
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--warmup_epochs', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--sgd_momentum', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--alpha', type=float, default=0.9)
# print
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--print_cluster_freq', type=int, default=33)
parser.add_argument('--test_rounds', type=int, default=10)

parser.add_argument('--temperature', type=float, default=100.0)
args = parser.parse_args()


def train():
    # training
    for epoch in range(args.epochs):
        print("==> training...")
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

        eval_feat = torch.FloatTensor(eval_feat).detach()
        nd_lists = [[] for i in range(args.num_clusters)]  # (k, N_k)
        for i in range(nb_nodes):
            nd_lists[cluster_result['nd2cluster'][i]].append(i)  # (k, N_k)

        centroids = cluster_result['centroids'].detach()
        precision = clustering.sample_estimator(nd_lists, eval_feat, centroids, args)
        precision = precision.detach()

        # cluster visualization
        # if epoch % args.print_cluster_freq == 0:
        #     clustering.visualize(eval_feat, cluster_result['nd2cluster'], 
        #                          savepath='./fig/' + f'cluster_{epoch}.png')

        # train for one epoch
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        learning_rates = AverageMeter('LR', ':.4e')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            batch_num,
            [batch_time, data_time, learning_rates, losses],
            prefix="Epoch: [{}]".format(epoch)
        )

        # switch to train mode
        model.train()
        
        end = time.time()
        for batch_idx in range(batch_num):
            # measure data loading time
            data_time.update(time.time() - end) 

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
            learning_rates.update(args.lr)

            # model forward
            output = model(ba, bf, get_node=True) # (2*batch_size, d)
            output = output.view(2, -1, args.embedding_dim) # (2, batch_size, d)
            output = args.alpha * output[0] + (1-args.alpha) * output[1] # (batch_size, d)
            # output = proj(output)
            score = output / args.temperature
            
            loss = criterion(score, cluster_result['nd2cluster'][idx])
            # cur_labels = cluster_result['nd2cluster'][idx].repeat(2)
            # loss = criterion(guassian_score, cur_labels)

            losses.update(loss.item(), cur_batch_size)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            batch_time.update(time.time() - end)

            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)
            
        # save state
        # checkpoint = {
        #     "model_state_dict": model.state_dict(),
        #     "cluster_result": cluster_result,
        #     "transformer": transformer
        # }
        # torch.save(checkpoint, './pretrained/{}.pth'.format(args.dataset))
        # # torch.save(model.state_dict(), './pretrained/{}.pkl'.format(args.dataset))
        # end_time = time.time()
        # print("epoch {}, total time {:.2f}".format(
        # 	epoch, end_time - start_time))


def test():
    print('Testing AUC!', flush=True)
    # print('Loading {}th epoch'.format(best_t), flush=True)
    # checkpoint = torch.load('./pretrained/{}.pth'.format(args.dataset))

    # model.load_state_dict(checkpoint["model_state_dict"])
    # cluster_result = checkpoint["cluster_result"]
    # transformer = checkpoint["transformer"]
    # model.load_state_dict(torch.load('./pretrained/{}.pkl'.format(args.dataset)))

    # switch to eval mode
    model.eval()

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

    multi_round_ano_score = np.zeros((args.test_rounds, nb_nodes))
    # with tqdm(total=args.test_rounds) as pbar_test:
    #     pbar_test.set_description('Testing')
    for round in tqdm(range(args.test_rounds)):
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
                output = model(ba, bf, get_node=True)
                output = output.view(2, -1, args.embedding_dim) # (2, batch_size, d)
                output = args.alpha * output[0] + (1-args.alpha) * output[1] # (batch_size, d)
                # output = proj(output)
                score = output / args.temperature
                pred = score.max(1)[1]  # (2*batch_size, )
                pure_gau = score[torch.arange(score.shape[0]), pred].unsqueeze(dim=-1)
                ano_score = pure_gau.squeeze(1)

            multi_round_ano_score[round, idx] = ano_score

        # pbar_test.update(1)

    ano_score_final = np.mean(multi_round_ano_score,
                              axis=0) + np.std(multi_round_ano_score, axis=0)
    auc = roc_auc_score(ano_labels, ano_score_final)
    # all_auc.append(auc)
    print('Testing AUC:{:.4f}'.format(auc), flush=True)


def compute_features(model, subgraphs, embedding_dim, device):
    """
    for data, index in eval_loader:
        data = data
        data = data.to(device)
        feat = model(data)
        features[index] = feat 
    """
    print('Computing features...', flush=True)
    model.eval()
    # proj.eval()

    all_feat = torch.zeros(nb_nodes, embedding_dim).to(device)
    with tqdm(total=batch_num) as pbar_eval:
        pbar_eval.set_description('Generating Features')
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
                cur_feat = features[:, subgraphs[i], :]
                ba.append(cur_adj)
                bf.append(cur_feat)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
        
            feat = model(ba, bf, get_node=True) # (2*batch_size, d)
            feat = feat.view(2, -1, embedding_dim) # (2, batch_size, d)
            feat = args.alpha * feat[0] + (1-args.alpha) * feat[1] # (batch_size, d)
            all_feat[idx] = feat
            pbar_eval.update(1)
    return all_feat.detach().numpy()


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    device = args.device

    print('Dataset: {}'.format(args.dataset), flush=True)
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
    features = gaussian_noised_feature(features)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)

    # build model
    gnn_args = {
        'ft_size': ft_size,
        'output_dim': args.embedding_dim,
        'gnn_model': args.model,
        'nmb_prototypes': args.num_clusters, 
    }

    model = GraphEncoder(**gnn_args)
    # proj = nn.Linear(args.embedding_dim, args.num_clusters)
    
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
    # iteration length
    batch_num = nb_nodes // batch_size + 1

    train()
    test()
