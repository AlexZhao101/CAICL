import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import List, Optional, Tuple
from collections import OrderedDict
import torch.distributed as dist
import pocket

from fnda import TransformerEncoderLayer as TELayer
from fnda import BertConnectionLayer as CrossAttentionEncoderLayer
import torchvision.ops.boxes as box_ops
import numpy as np

from hoi_mixup import TokenMixer


class LongShortDistanceEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 512, dropout: float = 0.1, use_sp=False):
        super().__init__()
        self.use_sp = use_sp

        self.encoder_layer_a = TELayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_feedforward, dropout_prob=dropout, use_sp=self.use_sp)

        if not use_sp:
            self.encoder_layer_b = TELayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout_prob=dropout, use_sp=False)

    def forward(self, x, dist, boxes, hi, oi, mask_a, mask_b):
        if self.use_sp:
            x, attn = self.encoder_layer_a(x, dist, boxes, hi, oi, None)
            attn = [attn]
        else:
            x, attn1 = self.encoder_layer_a(x, dist, boxes, hi, oi, mask_a)
            x, attn2 = self.encoder_layer_b(x, dist, boxes, hi, oi, mask_b)
            attn = [attn1, attn2]
        return x, attn


class ModifiedEncoder(nn.Module):
    def __init__(self,
                 hidden_size: int = 256, representation_size: int = 512,
                 num_heads: int = 8, num_layers: int = 2,
                 dropout_prob: float = .1, return_weights: bool = False,
                 ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.mod_enc = nn.ModuleList([LongShortDistanceEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=representation_size,
            dropout=dropout_prob,
            use_sp=i == 0
        ) for i in range(num_layers)])

    def forward(self, x: Tensor, dist: Tensor, boxes: Tensor, hi: Tensor, oi: Tensor) -> Tuple[
        Tensor, List[Optional[Tensor]]]:
        attn_weights = []
        x = x.unsqueeze(0)

        mask_a, mask_b = self.generate_mask(dist)

        for i, layer in enumerate(self.mod_enc):
            x, attn = layer(x, dist, boxes, hi, oi, mask_a, mask_b)

            if isinstance(attn, list):
                attn_weights.extend(attn)

        x = x.squeeze(0)
        return x, attn_weights

    def generate_mask(self, pairwise_dist):
        n = pairwise_dist.shape[0]

        mask_a = torch.ones_like(pairwise_dist)
        mask_b = torch.ones_like(pairwise_dist)

        sorted, indices = torch.sort(pairwise_dist, dim=-1)

        split = n // 2

        tau_index = indices[:, split]
        tau_dist = pairwise_dist[torch.arange(n), tau_index].unsqueeze(1)

        x_near, y_near = torch.nonzero(pairwise_dist <= tau_dist).unbind(1)
        x_far, y_far = torch.nonzero(pairwise_dist > tau_dist).unbind(1)

        mask_a[x_far, y_far] = 0
        mask_b[x_near, y_near] = 0

        # can always attend to itself
        mask_a[torch.arange(n), torch.arange(n)] = 1
        mask_b[torch.arange(n), torch.arange(n)] = 1

        return mask_a, mask_b


class CompEncoder(nn.Module):
    def __init__(self, hidden_size, return_weights, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            pocket.models.TransformerEncoderLayer(
                hidden_size=hidden_size,
                return_weights=return_weights
            ) for _ in range(num_layers)])

    def forward(self, x):
        weights = []
        for i in range(self.num_layers):
            x, w = self.layers[i](x)
            weights.append(w)
        return x, weights[-1]


class ObjectRelation(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8, dim_feedforward=512
            ) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x