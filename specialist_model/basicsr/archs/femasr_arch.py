import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY



from .fema_utils import ResBlock, CombineQuantBlock
from timm.models.layers import DropPath

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)




## AdaMean
def adaptive_mean_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size))
    return normalized_feat + style_mean.expand(size)


## AdaStd
def adaptive_std_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat) / content_std.expand(size)
    return normalized_feat * style_std.expand(size)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pdb import set_trace as stx
import numbers
from torchvision.models import vgg19
from einops import rearrange

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0) # B

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False) # B, C, 1

    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)


    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )

    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)
    # indices = (
    #     torch.arange(0, topk_values.size(1))
    #     .unsqueeze(0)
    #     .repeat(batch_size, 1)
    #     .to(topk_values.device)
    # )
    # indices_selected = indices.masked_select(mask)
    # indices_selected = indices_selected.reshape(batch_size, num_matches)
    # filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    # return filtered_input_maps, filtered_candidate_maps
    return filtered_candidate_maps


def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, C, H*W)
    candidate_maps: (B, C, H*W)
    """
    distances = torch.cdist(input_maps, candidate_maps) # B,C,C
    
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)

class Matching(nn.Module):
    def __init__(self, dim=32, match_factor=1):
        super(Matching, self).__init__()
        self.num_matching = int(dim/match_factor)
    def forward(self, x, perception):
        b, c, h, w = x.size()
        x = x.flatten(2, 3)
        perception = perception.flatten(2, 3)
        # print('x, perception1', x.size(), perception.size())
        filtered_candidate_maps = neirest_neighbores_on_l2(x, perception, self.num_matching)
        # filtered_input_maps = filtered_input_maps.reshape(b, self.num_matching, h, w)
        filtered_candidate_maps = filtered_candidate_maps.reshape(b, self.num_matching, h, w)
        return filtered_candidate_maps


class Matching_transformation(nn.Module):
    def __init__(self, dim=32, match_factor=1, ffn_expansion_factor=2, scale_factor=8, bias=True):
        super(Matching_transformation, self).__init__()
        self.num_matching = int(dim / match_factor)
        self.channel = dim
        hidden_features = int(self.channel * ffn_expansion_factor)
        self.matching = Matching(dim=dim, match_factor=match_factor)
        # self.matching = Matching(dim=dim)


        self.perception = nn.Conv2d(3 * dim, dim, 1, bias=bias)
        self.max = nn.MaxPool2d(scale_factor, scale_factor)
        self.mean = nn.AvgPool2d(scale_factor, scale_factor)

        self.dwconv = nn.Sequential(nn.Conv2d(2 * self.num_matching, hidden_features, 1, bias=bias),
                                    nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                              groups=hidden_features, bias=bias), nn.GELU(),
                                    nn.Conv2d(hidden_features, 2 * self.num_matching, 1, bias=bias))
        self.conv12 = nn.Conv2d(2 * self.num_matching, self.channel, 1, bias=bias)

    def forward(self, x, perception):
        perception = self.perception(perception)
        perception1, perception2 = self.max(perception), self.mean(perception)
        filtered_candidate_maps1 = self.matching(x, perception1)
        filtered_candidate_maps2 = self.matching(x, perception2)
        concat = torch.cat([filtered_candidate_maps1, filtered_candidate_maps2], dim=1)
        # conv11 = self.conv11(concat)
        dwconv = self.dwconv(concat)
        out = self.conv12(dwconv * concat)

        return out

class FeedForward(nn.Module):
    def __init__(self, dim=32, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True, ffn_matching=True):
        super(FeedForward, self).__init__()
        self.num_matching = int(dim/match_factor)
        self.channel = dim
        self.matching = ffn_matching
        hidden_features = int(self.channel * ffn_expansion_factor)

        self.project_in = nn.Sequential(
            nn.Conv2d(self.channel, hidden_features, 1, bias=bias),
            nn.Conv2d(hidden_features, self.channel, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=bias)
        )
        if self.matching is True:
            self.matching_transformation = Matching_transformation(dim=dim,
                                                                   match_factor=match_factor,
                                                                   ffn_expansion_factor=ffn_expansion_factor,
                                                                   scale_factor=scale_factor,
                                                                   bias=bias)

        self.project_out = nn.Sequential(
            nn.Conv2d(self.channel, hidden_features, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=bias),
            # nn.GELU(),
            nn.Conv2d(hidden_features, self.channel, 1, bias=bias))

    def forward(self, x, perception):
        project_in = self.project_in(x)
        if perception is not None:
            out = self.matching_transformation(project_in, perception)
        else:
            out = project_in
        project_out = self.project_out(out)
        return project_out



##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, match_factor=2,ffn_expansion_factor=2,scale_factor=8, bias=True, attention_matching=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.matching = attention_matching
        if self.matching is True:
            self.matching_transformation = Matching_transformation(dim=dim,
                                                                   match_factor=match_factor,
                                                                   ffn_expansion_factor=ffn_expansion_factor,
                                                                   scale_factor=scale_factor,
                                                                   bias=bias)

    def forward(self, x, perception):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # perception = self.LayerNorm(perception)
        if self.matching is True:
            q = self.matching_transformation(q, perception)
        else:
            q = q
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward_Restormer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_Restormer, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention_restormer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_restormer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out




# class NormLayer(nn.Module):
#     """Normalization Layers.
#     ------------
#     # Arguments
#         - channels: input channels, for batch norm and instance norm.
#         - input_size: input shape without batch size, for layer norm.
#     """
#
#     def __init__(self, channels, norm_type='in'):
#         super(NormLayer, self).__init__()
#         norm_type = norm_type.lower()
#         self.norm_type = norm_type
#         self.channels = channels
#         if norm_type == 'bn':
#             self.norm = nn.BatchNorm2d(channels, affine=True)
#         elif norm_type == 'in':
#             self.norm = nn.InstanceNorm2d(channels, affine=False)
#         elif norm_type == 'gn':
#             self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
#         elif norm_type == 'none':
#             self.norm = lambda x: x * 1.0
#         else:
#             assert 1 == 0, 'Norm type {} not support.'.format(norm_type)
#
#     def forward(self, x):
#         return self.norm(x)




import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


class SFTLayer_torch(nn.Module):
    def __init__(self, dim):
        super(SFTLayer_torch, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(dim, dim, 1)
        self.SFT_scale_conv1 = nn.Conv2d(dim, dim, 1)
        self.SFT_shift_conv0 = nn.Conv2d(dim, dim, 1)
        self.SFT_shift_conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x0, feature):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(feature), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(feature), 0.01, inplace=True))
        return x0 * scale + shift



class Downsample(nn.Module):
    def __init__(self, n_feat,scale):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=scale, stride=scale, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat,scale):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * (scale*scale), kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.body(x)

class ConvNeXtBlockLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1
        )  # depthwise conv
        # self.norm = ConvNeXtBlockLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        # x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


# class ConvNeXtBlock(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#
#     def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(
#             dim, dim, kernel_size=3, padding=1
#         )  # depthwise conv
#         # self.norm = ConvNeXtBlockLayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Sequential(nn.Conv2d(dim,dim,3,1,1),nn.GELU(),nn.Conv2d(dim,dim,3,1,1)) # pointwise/1x1 convs, implemented with linear layers
#         # self.act = nn.GELU()
#         # self.pwconv2 = nn.Linear(dim, dim)
#         # self.gamma = (
#         #     nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#         #     if layer_scale_init_value > 0
#         #     else None
#         # )
#         # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#
#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = input + x
#         return x


# class ConvNeXtBlock(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#
#     def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
#         super().__init__()
#         # self.dwconv = nn.Conv2d(
#         #     dim, dim, kernel_size=3, padding=1, groups=dim
#         # )  # depthwise conv
#         # # self.norm = ConvNeXtBlockLayerNorm(dim, eps=1e-6)
#         # self.pwconv1 = nn.Linear(
#         #     dim, dim
#         # )  # pointwise/1x1 convs, implemented with linear layers
#         # self.act = nn.GELU()
#         # self.pwconv2 = nn.Linear(dim, dim)
#         # self.gamma = (
#         #     nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#         #     if layer_scale_init_value > 0
#         #     else None
#         # )
#         # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#
#         # self.res = nn.Sequential(nn.Conv2d(dim,dim,3,1,1),
#         #                          nn.GELU(),
#         #                          nn.Conv2d(dim,dim,3,1,1))
#         self.res = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1))
#
#     def forward(self, x):
#         input = x
#         # x = self.dwconv(x)
#         # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         # # x = self.norm(x)
#         # x = self.pwconv1(x)
#         # x = self.act(x)
#         # x = self.pwconv2(x)
#         # if self.gamma is not None:
#         #     x = self.gamma * x
#         # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#         # x = input + self.res(x)
#         x = self.res(x)
#         return x

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim=32, num_heads=1, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True,
                 LayerNorm_type='WithBias', attention_matching=True, ffn_matching=True, ffn_restormer=False):
        super(TransformerBlock, self).__init__()
        self.dim =dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              match_factor=match_factor,
                              ffn_expansion_factor=ffn_expansion_factor,
                              scale_factor=scale_factor,
                              bias=bias,
                              attention_matching=attention_matching)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_restormer = ffn_restormer
        if self.ffn_restormer is False:
            self.ffn = FeedForward(dim=dim,
                                   match_factor=match_factor,
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   scale_factor=scale_factor,
                                   bias=bias,
                                   ffn_matching=ffn_matching)
        else:
            self.ffn = FeedForward_Restormer(dim=dim,
                                             ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias)
        self.LayerNorm = LayerNorm(dim * 3)

    def forward(self, x, perception):
        percetion = self.LayerNorm(perception)
        x = x + self.attn(self.norm1(x), percetion)
        if self.ffn_restormer is False:
            x = x + self.ffn(self.norm2(x), percetion)
        else:
            x = x + self.ffn(self.norm2(x))
        return x


class ResBlock_TransformerBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """

    def __init__(self, dim=32, num_heads=1, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True,
                 LayerNorm_type='WithBias', attention_matching=True, ffn_matching=True, ffn_restormer=False, unit_num=3):
        super(ResBlock_TransformerBlock, self).__init__()
        self.unit_num = unit_num
        self.TransformerBlock = nn.ModuleList()

        for i in range(self.unit_num):
            self.TransformerBlock.append(TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                match_factor=match_factor,
                ffn_expansion_factor=ffn_expansion_factor,
                scale_factor=scale_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                attention_matching=attention_matching,
                ffn_matching=ffn_matching,
                ffn_restormer=ffn_restormer))

    def forward(self, input, perception):
        tmp = input
        for i in range(self.unit_num):
            tmp = self.TransformerBlock[i](tmp, perception)

        out = 0.2 * tmp + input
        return out

# class ResBlock_new(nn.Module):
#     """
#     Use preactivation version of residual block, the same as taming
#     """
#     def __init__(self, channel):
#         super(ResBlock_new, self).__init__()
#         self.channel = channel
#         self.conv = nn.Sequential(
#             # NormLayer(in_channel, norm_type),
#             # ActLayer(channel, act_type),
#             nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=True),
#             # NormLayer(out_channel, norm_type),
#             nn.GELU(),
#             nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=True),
#         )
#
#     def forward(self, input):
#         res = self.conv(input)
#         out = res + input
#         return out

class Perception_fusion(nn.Module):
    def __init__(self, dim=32):
        super(Perception_fusion, self).__init__()
        self.channel = dim
        self.conv11 = nn.Conv2d(3 * self.channel, 3 * self.channel, 1, 1)
        self.dwconv = nn.Conv2d(3 * self.channel, 6 * self.channel, kernel_size=3, stride=1, padding=1,
                                groups=3 * self.channel)
    def forward(self, feature1, feature2, feature3):
        concat = torch.cat([feature1, feature2, feature3], dim=1)
        conv11 = self.conv11(concat)
        dwconv1, dwconv2 = self.dwconv(conv11).chunk(2, dim=1)
        b, c, h, w = dwconv1.size()
        dwconv1 = dwconv1.flatten(2, 3)
        dwconv1 = F.softmax(dwconv1, dim=1)
        dwconv1 = dwconv1.reshape(b, c, h, w)
        perception = torch.mul(dwconv1, concat) + dwconv2
        return perception



class Net(nn.Module):
    def __init__(self, channel_query_dict, number_block, num_heads=8, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True,
                 LayerNorm_type='WithBias', attention_matching=True, ffn_matching=True, ffn_restormer=False,unit_num=3):
        super().__init__()
        self.channel_query_dict = channel_query_dict
        self.enter = nn.Sequential(nn.Conv2d(3, channel_query_dict[256], 3, 1, 1))
        self.shallow = ConvNeXtBlock(channel_query_dict[256])
        self.middle = ConvNeXtBlock(channel_query_dict[256])
        self.deep = ConvNeXtBlock(channel_query_dict[256])
        self.perception_fusion = Perception_fusion(channel_query_dict[256])
        self.block = nn.ModuleList()
        self.number_block = number_block
        for i in range(self.number_block):
            self.block.append(ResBlock_TransformerBlock(dim=channel_query_dict[256],
                                                        num_heads=num_heads,
                                                        match_factor=match_factor,
                                                        ffn_expansion_factor=ffn_expansion_factor,
                                                        scale_factor=scale_factor,
                                                        bias=bias,
                                                        LayerNorm_type=LayerNorm_type,
                                                        attention_matching=attention_matching,
                                                        ffn_matching=ffn_matching,
                                                        ffn_restormer=ffn_restormer,
                                                        unit_num=unit_num))
        self.downsample = Downsample(channel_query_dict[256],scale_factor)
        self.upsample = Upsample(channel_query_dict[256],scale_factor)
        self.fusion = nn.Conv2d(2*channel_query_dict[256], channel_query_dict[256], 1)
        self.out = nn.Sequential(ConvNeXtBlock(channel_query_dict[256]),
                                 ConvNeXtBlock(channel_query_dict[256]),
                                 nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1))

    def forward(self, x):
        ori = x
        enter = self.enter(x)
        shallow = self.shallow(enter)
        middle = self.middle(shallow)
        deep = self.deep(middle)
        perception = self.perception_fusion(shallow, middle, deep)
        # print('enter', enter.size())
        block = self.downsample(enter)
        block_input = block
        # print('block_input',block_input.size())
        for i in range(self.number_block):
            block = self.block[i](block, perception)
        block = block_input + block
        upsample = self.upsample(block)
        fusion = self.fusion(torch.cat([upsample, deep], dim=1))
        out = self.out(fusion) + ori
        return out


#@ARCH_REGISTRY.register()
class FeMaSRNet(nn.Module):
    def __init__(self,
                 *,
                 number_block,
                 num_heads=8,
                 match_factor=1,
                 ffn_expansion_factor=3,
                 scale_factor=8,
                 bias=True,
                 LayerNorm_type='WithBias',
                 attention_matching=True,
                 ffn_matching=True,
                 ffn_restormer=False,
                 **ignore_kwargs):
        super().__init__()
        channel_query_dict = {8: 256, 16: 256, 32: 384, 64: 192, 128: 96, 256: 16, 512: 32}
        self.restoration_network = Net(channel_query_dict=channel_query_dict,
                                       number_block=number_block,
                                       num_heads=num_heads,
                                       match_factor=match_factor,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       scale_factor=scale_factor,
                                       bias=bias,
                                       LayerNorm_type=LayerNorm_type,
                                       attention_matching=attention_matching,
                                       ffn_matching=ffn_matching,
                                       ffn_restormer=ffn_restormer)

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def encode_and_decode(self, input, current_iter=None):

        restoration = self.restoration_network(input)
        return restoration

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    def check_image_size(self, x, window_size=16):
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % (window_size)) % (
            window_size)
        mod_pad_w = (window_size - w % (window_size)) % (
            window_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
        return x

    @torch.no_grad()
    def test(self, input):
        _, _, h_old, w_old = input.shape

        # input = self.check_image_size(input)
        # if lq_equalize is not None:
        #     lq_equalize = self.check_image_size(lq_equalize)

        restoration = self.encode_and_decode(input)

        output = restoration
        # output = output[:,:, :h_old, :w_old]

        # self.use_semantic_loss = org_use_semantic_loss
        return output

    def forward(self, input):
        # print('**********************************************************')
        # print('input',input.size())
        # print('###########################################################')

        # if gt_indices is not None:
        #     # in LQ training stage, need to pass GT indices for supervise.
        #     out_img_structure, feature_structure, out_img_ill_radio, feature_ill_radio, out_img_input_equalize, feature_input_equalize, enhanced, feature_enhanced = self.encode_and_decode(input, lq_equalize=lq_equalize, deep_feature=deep_feature)
        # else:
        # in HQ stage, or LQ test stage, no GT indices needed.
        restoration = self.encode_and_decode(input)

        return restoration


if __name__== '__main__': 
    import os
    import time
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    x = torch.randn(1, 3, 2160, 3840).cuda()
    channel_query_dict = {8: 256, 16: 256, 32: 384, 64: 192, 128: 96, 256: 16, 512: 32}
    model = Net(channel_query_dict=channel_query_dict,
                number_block=5,
                num_heads=8,
                match_factor=4,
                ffn_expansion_factor=3,
                scale_factor=8,
                bias=True,
                LayerNorm_type='WithBias',
                attention_matching=True,
                ffn_matching=True,
                ffn_restormer=False).cuda()
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    inp_shape=(3,1024,1024)
    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=True)

    params = float(params[:-4])
    print(macs)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    with torch.no_grad():
        start_time = time.time()
        output = model(x)
        end_time = time.time()
    running_time = end_time - start_time
    print(output.shape)
    print(running_time)