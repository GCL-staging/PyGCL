import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, GINEConv

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def make_gin_conv(input_dim: int, out_dim: int) -> GINConv:
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


def make_gine_conv(input_dim: int, out_dim: int) -> GINEConv:
    return GINEConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int, hidden_dim: int, activation,
                 num_layers: int,
                 batch_norm: bool = False,
                 use_atom_encoder: bool = False,
                 use_bond_encoder: bool = False,
                 base_conv: str = 'GINConv'):
        super(Encoder, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if batch_norm else None
        self.base_conv = base_conv

        if base_conv == 'GINConv':
            if use_bond_encoder:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
        elif base_conv == 'GINEConv':
            if use_bond_encoder:
                self.layers.append(make_gine_conv(hidden_dim, hidden_dim))
            else:
                self.layers.append(make_gine_conv(input_dim, hidden_dim))
        else:
            self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))

        for _ in range(num_layers - 1):
            # add batch norm layer if batch norm is used
            if self.batch_norms is not None:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            if base_conv == 'GINConv':
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            elif base_conv == 'GINEConv':
                if use_bond_encoder:
                    self.layers.append(make_gine_conv(hidden_dim, hidden_dim))
                else:
                    self.layers.append(make_gine_conv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

        if use_atom_encoder:
            self.atom_encoder = AtomEncoder(hidden_dim)
        else:
            self.atom_encoder = None

        if use_bond_encoder:
            self.bond_encoder = BondEncoder(hidden_dim)
        else:
            self.bond_encoder = None

    def forward(self, x, edge_index, edge_weight=None):
        if self.atom_encoder is not None:
            x = self.atom_encoder(x)
        if self.bond_encoder is not None:
            edge_weight = self.bond_encoder(edge_weight)
        z = x
        num_layers = len(self.layers)
        for i, conv in enumerate(self.layers):
            if self.base_conv == 'GINConv':
                z = conv(z, edge_index)
            else:
                z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            if self.batch_norms is not None and i != num_layers - 1:
                z = self.batch_norms[i](z)
        return z
