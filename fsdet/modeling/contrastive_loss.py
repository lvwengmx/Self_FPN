import torch.nn as nn
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from .graph.gat_v2 import GAT
from .graph.gnn_base import GNNReID
from .graph.graph_generator import GraphGenerator
from .graph.latentgnn import LatentGNNV1
from PIL import ImageFilter
import random
import os
import numpy as np
from fvcore.common.file_io import PathManager

class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim):  # dim=1024, feat-dim=128
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):

        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GNN(nn.Module):

    def __init__(self, emb_dim: int, out_dim: int, mpnn_dev: str, mpnn_opts: dict, gnn_type: str = "gat",
                 final_relu: bool = False):
        super(GNN, self).__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.mpnn_opts = mpnn_opts
        self.gnn_type = gnn_type
        mpnn_dev = mpnn_dev  # torch.cuda.is_available()
        if gnn_type == "gat_v2":
            self.gnn = GAT(in_channels=emb_dim, hidden_channels=emb_dim // 4, out_channels=out_dim,
                           num_layers=mpnn_opts["gnn_params"]["gnn"]["num_layers"],
                           heads=mpnn_opts["gnn_params"]["gnn"]["num_heads"],
                           v2=True, )
        elif gnn_type == "gat":
            self.gnn = GNNReID(emb_dim, mpnn_dev, mpnn_opts["gnn_params"],)
        elif gnn_type == "latentgnn":
            self.gnn = LatentGNNV1(in_channels=64, latent_dims=[16, 16], channel_stride=2,
                                   num_kernels=2, mode="asymmetric",
                                   graph_conv_flag=False)
        self.graph_generator = GraphGenerator(mpnn_dev, **mpnn_opts["graph_params"])


        if final_relu:
            self.relu_final = nn.ReLU()
        else:
            self.relu_final = nn.Identity()

    def forward(self, x):
        if "gat" in self.gnn_type:
            z = x
            z_cnn = z.clone()
            z = z.flatten(1)
            edge_attr, edge_index, z = self.graph_generator.get_graph(z)
        else:
            z = x
            z_cnn = z.clone()
        if self.gnn_type == "gat_v2":
            z = self.gnn(z, edge_index.t().contiguous())
        elif self.gnn_type == "gat":
            _, (z,) = self.gnn(z, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
        elif self.gnn_type == "latentgnn":
            z = self.gnn(z)
            z = z.flatten(1)
        z = self.relu_final(z)
        return z


def getBinaryTensor(imgTensor, boundary=0.5):
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)


def calculate_cosine_similarty(h_emb, eps=1e-8):
    a_n = h_emb.norm(dim=1).unsqueeeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))
    sim_matrix = torch.einsum('bc,ad->bd', a_norm, a_norm.transpose(0, 1))
    return sim_matrix


# Contrastive LOSS
class SupConLoss(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_func='none'):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func



    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [1024, 128]
            labels (tensor): shape of [M].  e.g., [1024]
        """

        #
        # 计算样本对之间的距离
        pairwise_distance = torch.cdist(features, features, p=2)
        # 创建一个布尔矩阵，指示样本是否属于同一类
        label_matrix = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        # 计算同类样本的距离
        positive_distance = pairwise_distance * label_matrix
        # 计算不同类样本的距离
        negative_distance = positive_distance * (1-label_matrix)
        # 计算对比损失
        loss = torch.mean(torch.max(positive_distance - negative_distance, torch.zeros_like(pairwise_distance)))



        # assert features.shape[0] == labels.shape[0] == ious.shape[0]
        #
        #
        #
        # if len(labels.shape) == 1:
        #     labels = labels.reshape(-1, 1) # lables=(1024,1)
        #
        # # # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        # label_mask = torch.eq(labels, labels.T).float().cuda() # (1024, 1024) 相同=1, otherwise=0
        #
        #
        # #voc_txt = np.loadtxt('/home/lab-202/FSCE/VOC_att.txt', dtype=np.float, delimiter=',')
        # # voc_txt = torch.from_numpy(voc_txt).T
        # # index = labels.shape[0]
        # # new_voc = []
        # # for i in range(index):
        # #     labels_index = labels[i]  # 8
        # #     if labels_index == 0:
        # #         coco_re = voc_txt[0:1]
        # #     else:
        # #         coco_re = voc_txt[labels_index-1:labels_index]
        # #
        # #     new_voc.append(coco_re)
        # #
        # # new_word_node = torch.stack(new_voc)
        # # new_word_node = torch.reshape(new_word_node, (-1, 64))
        # # new_word_node = torch.matmul(new_word_node, new_word_node.T).float().cuda()
        # # new_word_node = F.normalize(new_word_node, dim=0)
        #
        #
        #
        #
        #
        # # coco_txt = np.loadtxt('/home/lab-202/FSCE/word_w2v.txt', dtype=np.float, delimiter=',')
        # # coco_tensor = torch.from_numpy(coco_txt).T  # (80, 300)
        # # coco_class = np.loadtxt('/home/lab-202/FSCE/coco_class.csv', dtype=np.int, delimiter=',')
        # # class_i = coco_class.shape[0]
        # # coco_word = []
        # # for j in range(class_i):
        # #     class_index = coco_class[j]
        # #     index_word = coco_tensor[class_index]
        # #     coco_word.append(index_word)
        # # coco_word = torch.stack(coco_word)
        #
        # # index = labels.shape[0]
        # # new_coco = []
        # # for i in range(index):
        # #     labels_index = labels[i]  # 8
        # #     if labels_index == 0:
        # #         coco_re = coco_tensor[0:1]
        # #     else:
        # #         coco_re = coco_tensor[labels_index-1:labels_index]
        # #
        # #     new_coco.append(coco_re)
        # #
        # # new_word_node = torch.stack(new_coco)
        # # new_word_node = torch.reshape(new_word_node, (-1, 300))
        # # label_mask = torch.matmul(new_word_node, new_word_node.T).float().cuda()
        # # label_mask = vis_simi * label_mask
        #
        # # similarity = z{i}*z{j}/T T=0.2
        # similarity = torch.div(
        #     torch.matmul(features, features.T), self.temperature) #(1024, 1024)
        # # for numerical stability 行最大值+行最大索引 (1024, 1)
        # sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        # similarity = similarity - sim_row_max.detach()
        #
        # # mask out self-contrastive
        # logits_mask = torch.ones_like(similarity)
        # logits_mask.fill_diagonal_(0) # 对角线位置填充为0，解除自身相关
        #
        # exp_sim = torch.exp(similarity) * logits_mask  # exp(z{i}*z{j}/T T=0.2)
        #
        # log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        #
        # per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1) # L(z{i})=1024

        # per_semantic_log_prob = (log_prob * logits_mask * new_word_node).sum(1) / new_word_node.sum(1)  #L
        #
        # per = per_label_log_prob + per_semantic_log_prob

        # per = per_label_log_prob
        #
        # keep = ious >= self.iou_threshold
        # per = per[keep]
        # loss = -per
        #
        # coef = self._get_reweight_func(self.reweight_func)(ious)
        # coef = coef[keep]  # f(u{i})
        #
        # loss = loss * coef
        # return loss.mean()
        return loss

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay







class SupConLossV2(nn.Module):
    def __init__(self, temperature=0.2, iou_threshold=0.5):
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold

    def forward(self, features, labels, ious):
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)


        exp_sim = torch.exp(similarity)
        mask = logits_mask * label_mask
        keep = (mask.sum(1) != 0 ) & (ious >= self.iou_threshold)

        log_prob = torch.log(
            (exp_sim[keep] * mask[keep]).sum(1) / (exp_sim[keep] * logits_mask[keep]).sum(1)
        )

        loss = -log_prob
        return loss.mean()


class SupConLossWithStorage(nn.Module):
    def __init__(self, temperature=0.2, iou_threshold=0.5):
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold

    def forward(self, features, labels, ious, queue, queue_label):
        fg = queue_label != -1
        # print('queue', torch.sum(fg))
        queue = queue[fg]
        queue_label = queue_label[fg]

        keep = ious >= self.iou_threshold
        features = features[keep]
        feat_extend = torch.cat([features, queue], dim=0)

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        labels = labels[keep]
        queue_label = queue_label.reshape(-1, 1)
        label_extend = torch.cat([labels, queue_label], dim=0)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, label_extend.T).float().cuda()

        # print('# companies', label_mask.sum(1))

        similarity = torch.div(
            torch.matmul(features, feat_extend.T), self.temperature)
        # print('logits range', similarity.max(), similarity.min())

        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)
        loss = -per_label_log_prob
        return loss.mean()


class SupConLossWithPrototype(nn.Module):
    '''TODO'''

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, protos, proto_labels):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
            proto (tensor): shape of [B, 128]
            proto_labels (tensor), shape of [B], where B is number of prototype (base) classes
        """
        assert features.shape[0] == labels.shape[0]
        fg_index = labels != self.num_classes

        features = features[fg_index]  # [m, 128]
        labels = labels[fg_index]      # [m, 128]
        numel = features.shape[0]      # m is named numel

        # m  =  n  +  b
        base_index = torch.eq(labels, proto_labels.reshape(-1,1)).any(axis=0)  # b
        novel_index = ~base_index  # n
        if torch.sum(novel_index) > 1:
            ni_pk = torch.div(torch.matmul(features[novel_index], protos.T), self.temperature)  # [n, B]
            ni_nj = torch.div(torch.matmul(features[novel_index], features[novel_index].T), self.temperature)  # [n, n]
            novel_numer_mask = torch.ones_like(ni_nj)  # mask out self-contrastive
            novel_numer_mask.fill_diagonal_(0)
            exp_ni_nj = torch.exp(ni_nj) * novel_numer_mask  # k != i
            novel_label_mask = torch.eq(labels[novel_index], labels[novel_index].T)
            novel_log_prob = ni_nj - torch.log(exp_ni_nj.sum(dim=1, keepdim=True) + ni_pk.sum(dim=1, keepdim=True))
            loss_novel = -(novel_log_prob * novel_numer_mask * novel_label_mask).sum(1) / (novel_label_mask * novel_numer_mask).sum(1)
            loss_novel = loss_novel.sum()
        else:
            loss_novel = 0

        if torch.any(base_index):
            bi_pi = torch.div(torch.einsum('nc,nc->n', features[base_index], protos[labels[base_index]]), self.temperature) # shape = [b]
            bi_nk = torch.div(torch.matmul(features[base_index], features[novel_index].T), self.temperature)  # [b, n]
            bi_pk = torch.div(torch.matmul(features[base_index], protos.T), self.temperature)  # shape = [b, B]
            # bi_pk_mask = torch.ones_like(bi_pk)
            # bi_pk_mask.scatter_(1, labels[base_index].reshape(-1, 1), 0)
            # base_log_prob = bi_pi - torch.log(torch.exp(bi_nk).sum(1) + (torch.exp(bi_pk) * bi_pk_mask).sum(1))
            base_log_prob = bi_pi - torch.log(torch.exp(bi_nk).sum(1) + torch.exp(bi_pk).sum(1))
            loss_base = -base_log_prob
            loss_base = loss_base.sum()
        else:
            loss_base = 0

        loss = (loss_novel + loss_base) / numel
        try:
            assert loss >= 0
        except:
            print('novel', loss_novel)
            print('base', loss_base)
            exit('loss become negative.')
        return loss