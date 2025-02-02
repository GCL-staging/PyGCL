import copy
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple
from torch_geometric.nn import global_add_pool, GCNConv, GINConv

import GCL.augmentors as A
from GCL.losses import BootstrapLoss


class Normalize(nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class BGRL(torch.nn.Module):
    def __init__(self,
                 encoder: torch.nn.Module,
                 augmentor: Tuple[A.Augmentor, A.Augmentor],
                 hidden_dim: int,
                 dropout: float = 0.2,
                 predictor_norm='batch',
                 mode='L2L',
                 loss=BootstrapLoss()):
        super(BGRL, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.mode = mode
        self.loss = loss

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None, batch=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        if self.mode == 'L2L':
            h1_pred = self.predictor(h1_online)
            h2_pred = self.predictor(h2_online)

            with torch.no_grad():
                _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
                _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

            return h1, h2, h1_pred, h2_pred, h1_target, h2_target
        elif self.mode == 'G2G':
            g1 = global_add_pool(h1, batch)
            g1_online = global_add_pool(h1_online, batch)
            g1_pred = self.predictor(g1_online)

            g2 = global_add_pool(h2, batch)
            g2_online = global_add_pool(h2_online, batch)
            g2_pred = self.predictor(g2_online)

            with torch.no_grad():
                _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
                _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)
                g1_target = global_add_pool(h1_target, batch)
                g2_target = global_add_pool(h2_target, batch)

            return g1, g2, g1_pred, g2_pred, g1_target, g2_target
        else:  # self.mode == 'G2L'
            g1 = global_add_pool(h1, batch)
            h1_pred = self.predictor(h1_online)
            g2 = global_add_pool(h2, batch)
            h2_pred = self.predictor(h2_online)

            with torch.no_grad():
                _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
                _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)
                g1_target = global_add_pool(h1_target, batch)
                g2_target = global_add_pool(h2_target, batch)

            return g1, g2, h1_pred, h2_pred, g1_target, g2_target


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, activation,
                 num_layers: int, dropout: float = 0.2,
                 encoder_norm='batch', projector_norm='batch', base_conv='GCNConv'):
        super(Encoder, self).__init__()
        self.activation = activation()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        if base_conv == 'GINConv':
            self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
            for _ in range(num_layers - 1):
                self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))
        else:  # base_conv == 'GCNConv'
            self.layers.append(make_gin_conv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            nn.PReLU(),
            nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


def make_gin_conv(input_dim: int, out_dim: int) -> GINConv:
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
