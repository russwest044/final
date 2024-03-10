import torch
from torch import nn
import torch.nn.functional as F
from models.graph_encoder import GraphEncoder
from loss import hypersphere_loss


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, nmb_prototypes, dim=128, mlp_dim=256, hidden_dim=32, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        # build encoders
        self.base_encoder = base_encoder(outputdim=hidden_dim)
        self.momentum_encoder = base_encoder(outputdim=hidden_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        
        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(hidden_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(hidden_dim, nmb_prototypes, bias=False)
    
    def _mask_target_node_feature(self, feat):
        """Set target node feature to tensor 0."""
        B, _, C = feat.shape
        added_feat_zero_row = torch.zeros((B, 1, C), dtype=torch.long)
        x_masked = torch.cat((added_feat_zero_row, feat[:, 1:, :]), dim=1)
        return x_masked

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        # projectors
        self.base_encoder.head = self._build_mlp(3, self.hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, self.hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
    
    # def forward_pred(self, f_graph, f_node):
    #     # normalize
    #     logits = nn.functional.normalize(f_graph, dim=1) / self.T
    #     labels = nn.functional.normalize(f_node, dim=1)
    #     return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def contrastive_loss(self, q, k):
        """
        MoCo-v3 contrastive loss
        https://arxiv.org/abs/2104.02057
        def ctr(q, k):
            logits = mm(q, k.t()) # [N, N] pairs
            labels = range(N) # positives are in diagonal
            loss = CrossEntropyLoss(logits/tau, labels)
            return 2 * tau * loss
        """

        # normalize
        q = nn.functional.normalize(q, dim=1) # (N, C)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T # (N, C), (M, C) --> (N, M)
        N = logits.shape[0]  # batch size per GPU
        # labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        labels = torch.arange(N, dtype=torch.long)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m, alpha=None):
        """
        Input:
            x1: first views of target nodes
            x2: second views of target nodes
            m: moco momentum
            alpha: coefficient of node-graph prediction loss
        Output:
            loss
        """
        # compute features
        # =================== Node-Graph prediction forward =====================
        # feat_q, feat_k = x1.ndata["features"], x2["features"]
        # x1.ndata["features"] = self._mask_target_node_feature(feat_q)
        # x1_masked = self.base_encoder(target_node=True)
        # x2.ndata["features"] = self._mask_target_node_feature(feat_k)
        # x2_masked = self.base_encoder(target_node=True)

        # =================== MoCo forward =====================
        # x1.ndata["features"], x2.ndata["features"] = feat_q, feat_k
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
        
        # pred_loss = self.pred_loss(q1, x1_masked) + self.pred_loss(q2, x2_masked)
        loss = self.moco_loss(q1, k2) + self.moco_loss(q2, k1)

        # contrastive_loss = alpha * pred_loss + (1-alpha) * moco_loss
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


def moco_for_graph(**kwargs):
    base_encoder = GraphEncoder(**kwargs)
    model = MoCo(base_encoder)
    return model
