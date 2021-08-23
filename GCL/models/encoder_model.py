import torch
import GCL.augmentors as A

from torch import nn
from typing import Optional, Tuple, List, Union
from torch_geometric.nn import global_add_pool


class EncoderModel(nn.Module):
    def __init__(self, encoder: torch.nn.Module,
                 augmentor: Union[Tuple[A.Augmentor, A.Augmentor], List[A.Augmentor]], num_views: int = 2):
        super(EncoderModel, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.num_views = num_views

    def forward(self, x: torch.FloatTensor, batch: torch.LongTensor,
                edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None):
        num_nodes = x.size()[0]

        assert self.num_views >= 2
        if self.num_views == 2:
            aug1, aug2 = self.augmentor
            x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
            x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
            shuffled_x1 = x1[torch.randperm(num_nodes)]
            shuffled_x2 = x2[torch.randperm(num_nodes)]

            z = self.encoder(x, edge_index, edge_weight)
            z1 = self.encoder(x1, edge_index1, edge_weight1)
            z2 = self.encoder(x2, edge_index2, edge_weight2)
            z3 = self.encoder(shuffled_x1, edge_index1, edge_weight1)
            z4 = self.encoder(shuffled_x2, edge_index2, edge_weight2)

            g = global_add_pool(z, batch)
            g1 = global_add_pool(z1, batch)
            g2 = global_add_pool(z2, batch)

            return z, g, z1, z2, g1, g2, z3, z4
        else:
            z = self.encoder(x, edge_index, edge_weight)
            g = global_add_pool(z, batch)

            z_list = []
            g_list = []

            for aug in self.augmentor:
                x1, edge_index1, edge_weight1 = aug(x, edge_index, edge_weight)
                z1 = self.encoder(x1, edge_index1, edge_weight1)
                g1 = global_add_pool(z1, batch)

                z_list.append(z1)
                g_list.append(g1)

            return z, g, z_list, g_list
