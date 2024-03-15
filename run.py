import math
import dgl
import time
import torch
import numpy as np
import random
import warnings
import argparse
# from functools import partial
from .utils import *
import clustering
from .dataset import GraphDataset
from dmc import dmc
from sklearn.metrics import roc_auc_score

# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='')
args = parser.parse_args()


def main(args):
    device = args.device

    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random.seed(args.seed)
        dgl.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # data preprocess
    adj, features, labels = load(args.dataset)
    features = preprocess_features(features)
    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    
    # build dataset
    train_dataset = GraphDataset(adj=adj, features=features, aug=args.augmentation)
    eval_dataset = GraphDataset(adj=adj, features=features, aug=None)

    if args.sampler:
        train_sampler = args.sampler
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    # build model
    gnn_args = {
        'positional_embedding_size': args.positional_embedding_size,
        'max_node_freq': args.max_node_freq,
        'max_edge_freq': args.max_edge_freq,
        'max_degree': args.max_degree,
        'freq_embedding_size': args.freq_embedding_size,
        'degree_embedding_size': args.degree_embedding_size,
        'output_dim': args.hidden_size,
        'node_hidden_dim': args.hidden_size,
        'edge_hidden_dim': args.hidden_size,
        'num_layers': args.num_layer,
        'num_step_set2set': args.set2set_iter,
        'num_layer_set2set': args.set2set_lstm_layer,
        'norm': args.norm,
        'gnn_model': args.model,
        'degree_input': True
    }
    model = dmc(gnn_args)
    transformer = torch.eye(args.hidden_size, requires_grad=False)
    

    # build optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum if args.momentum else 0.9,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=args.learning_rate,
            lr_decay=args.lr_decay_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError

    # training
    for epoch in range(args.num_epoch):
        print("==> training...")
        start_time = time.time()

        # placeholder for clustering result
        cluster_result = {'nd2cluster': torch.zeros(len(eval_dataset), dtype=torch.long).cuda(), # (N, )
                        'centroids': torch.zeros(int(args.num_clusters), args.low_dim).cuda(), # (k, d)
                        'density': torch.zeros(int(args.num_clusters)).cuda() # (k, )
                        }
        
        # get features from frozen encoder
        features = clustering.compute_features(eval_loader, model, args)
        # projection
        features = torch.mm(transformer, features)

        # Kmeans
        cluster_result = clustering.run_kmeans(features, args)
        nd_lists = [[] for i in range(args.num_clusters)] # (k, N_k)
        for i in range(len(eval_dataset)):
            nd_lists[cluster_result['nd2cluster'][i]].append(i) # (k, N_k)
        # get centroids and precision
        centroids = cluster_result['centroids'] # (k, d)
        # precision = clustering.sample_estimator(nd_lists, features, centroids, args)
        # add centroids and precision to model
        centroids = torch.tensor(centroids, dtype=torch.float).cuda()
        model.centroids = centroids
        W = clustering.learn_metric(features, cluster_result['nd2cluster'])
        model.transformer = W*transformer
        # # precision = torch.tensor(precision, dtype=torch.float).cuda()
        # model.clusteringlayer.precision = torch.nn.Parameter(precision)

        # assign pseudolabels
        # train_dataset = clustering.cluster_assign(nd_lists, dataset.imgs)
        
        # train for one epoch
        train(train_loader, model, optimizer, epoch, args, device)
        # save model
        torch.save(model.state_dict(), './pretrained/{}.pkl'.format(args.dataset))
        end_time = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, end_time - start_time))
    
    # testing
    test_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    test_anomaly(test_loader, model, device)


def train(train_loader, model, optimizer, epoch, args, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    for i, (batch, idx) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)

        # graph_q, graph_k = batch
        graph_q = batch

        graph_q = graph_q.to(device)
        # graph_k = graph_k.to(device)
        idx = idx.to(device)

        bsz = graph_q.batch_size

        # model forward
        # loss = model(graph_q, graph_k)
        loss = model(graph_q, idx)
        losses.update(loss.item(), bsz)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def test_anomaly(test_loader, model, device):
    # Testing
    # print('Loading {}th epoch'.format(best_t), flush=True)
    model.load_state_dict(torch.load('{}.pkl'.format(args.dataset)))
    multi_round_ano_score = []
    print('Testing AUC!', flush=True)
    
    # switch to eval mode
    model.eval()

    with tqdm(total=args.test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        for _ in range(args.test_rounds):
            cur_round_ano_score = []
            labels = []
            for g, label in test_loader:
                g = g.to(device)
                # ===================forward=====================
                with torch.no_grad():
                    scores_final = model(g, is_eval=True).cpu().numpy()
                cur_round_ano_score.append(scores_final)
                labels.append(label)
            pbar_test.update(1)

            auc = roc_auc_score(labels, cur_round_ano_score)
            multi_round_ano_score.append(auc)
            print('Testing AUC:{:.4f}'.format(auc), flush=True)

        final_auc = np.mean(multi_round_ano_score)
        print('FINAL TESTING AUC:{:.4f}'.format(final_auc))


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

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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
    main()
