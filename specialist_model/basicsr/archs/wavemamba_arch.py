import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.io import savemat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
from typing import Optional, Callable
import math
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys
from basicsr.utils.registry import ARCH_REGISTRY
import torch.autograd


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.relu = nn.GELU()

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch,int(in_channel/(r**2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:,out_channel:out_channel * 2, :, :] / 2
    x3 = x[:,out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:,out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class ffn(nn.Module):
    def __init__(self, num_feat, ffn_expand=2):
        super(ffn, self).__init__()

        dw_channel = num_feat * ffn_expand
        self.conv1 = nn.Conv2d(num_feat, dw_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel//2, num_feat, kernel_size=1, padding=0, stride=1)
        
        self.sg = SimpleGate()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1)*x2
        # x = x * self.sca(x)
        x = self.conv3(x)
        return x

# FC
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)


# Local feature
class Local(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden_dim = int(dim // growth_rate)

        self.weight = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.weight(y)
        return x*y


# Gobal feature
class Gobal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape
        y = F.interpolate(x, size=[C, C], mode='bilinear', align_corners=True)
        # b c w h -> b c h w
        y = self.act1(self.conv1(y)).permute(0, 1, 3, 2)
        # b c h w -> b w h c
        y = self.act2(self.conv2(y)).permute(0, 3, 2, 1)
        # b w h c -> b c w h
        y = self.act3(self.conv3(y)).permute(0, 3, 1, 2)
        y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)
        return x*y
    

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        self.local = Local(dim, ffn_scale)
        self.gobal = Gobal(dim)
        self.conv = nn.Conv2d(2*dim, dim, 1, 1, 0)
        # Feedforward layer
        self.fc = FC(dim, ffn_scale) 

    def forward(self, x):
        y = self.norm1(x)
        y_l = self.local(y)
        y_g = self.gobal(y)
        y = self.conv(torch.cat([y_l, y_g], dim=1)) + x

        y = self.fc(self.norm2(y)) + y
        return y

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class LFSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = ffn(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x



class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)
        

    def forward(self, x):
        x_list = []
        x_h_list=[]
        x_v_list=[]
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
            x_h_list.append(x_i_h)
            x_v_list.append(x_i_v)

        x = torch.cat(x_list, dim = 1)
        x_h = torch.cat(x_h_list, dim=1)
        x_v = torch.cat(x_v_list, dim=1)
        
        return x_h, x_v, x


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


class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf//2, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out

class Matching_transformation(nn.Module):
    def __init__(self, dim=32, match_factor=1, ffn_expansion_factor=1, bias=True):
        super(Matching_transformation, self).__init__()
        self.num_matching = int(dim / match_factor)
        self.channel = dim
        hidden_features = int(self.channel * ffn_expansion_factor)
        self.matching = Matching(dim=dim, match_factor=match_factor)
        # self.matching = Matching(dim=dim)

        self.paconv =  PAConv(dim*2)

    def forward(self, x, perception):
        filtered_candidate_maps = self.matching(x, perception)
        # conv11 = self.conv11(concat)
        concat = torch.cat([x, filtered_candidate_maps], dim=1)
        out = self.paconv(concat)

        return out

class FeedForward(nn.Module):
    def __init__(self, dim=32, match_factor=4, ffn_expansion_factor=1, bias=True, ffn_matching=True):
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
                                                                   bias=bias)

        self.project_out = nn.Sequential(
            nn.Conv2d(self.channel, hidden_features, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=bias),
            nn.GELU(),
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
class CMTAttention(nn.Module):
    def __init__(self, dim, num_heads, match_factor=4,ffn_expansion_factor=1,scale_factor=8, bias=True, attention_matching=True):
        super(CMTAttention, self).__init__()
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
    def __init__(self, dim, ffn_expansion_factor=1, bias=True):
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


class HFEBlock(nn.Module):
    def __init__(self, dim=48, num_heads=1, match_factor=4, ffn_expansion_factor=1, bias=True, attention_matching=True, ffn_matching=True, ffn_restormer=False):
        super(HFEBlock, self).__init__()
        self.dim =dim
        self.norm1 = LayerNorm2d(dim)
        self.attn = CMTAttention(dim=dim,
                              num_heads=num_heads,
                              match_factor=match_factor,
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias,
                              attention_matching=attention_matching)
        self.norm2 = LayerNorm2d(dim)
        self.ffn_restormer = ffn_restormer
        if self.ffn_restormer is False:
            self.ffn = FeedForward(dim=dim,
                                   match_factor=match_factor,
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias,
                                   ffn_matching=ffn_matching)
        else:
            self.ffn = FeedForward_Restormer(dim=dim,
                                             ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias)
        self.LayerNorm = LayerNorm2d(dim)

    def forward(self, x, perception):
        percetion = self.LayerNorm(perception)
        x = x + self.attn(self.norm1(x), percetion)
        if self.ffn_restormer is False:
            x = x + self.ffn(self.norm2(x), percetion)
        else:
            x = x + self.ffn(self.norm2(x))
        return x


class Frequency_fusion(nn.Module):
    def __init__(self, in_c=3, dim=48):
        super(Frequency_fusion, self).__init__()
        self.channel = in_c
        self.conv11 = nn.Conv2d(3 * self.channel, dim, 1, 1)
        self.dwconv = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1,
                                groups=dim)

    def forward(self, feature1, feature2, feature3):
        concat = torch.cat([feature1, feature2, feature3], dim=1)
        conv11 = self.conv11(concat)
        dwconv1, dwconv2 = self.dwconv(conv11).chunk(2, dim=1)
        b, c, h, w = dwconv1.size()
        dwconv1 = dwconv1.flatten(2, 3)
        dwconv1 = F.softmax(dwconv1, dim=1)
        dwconv1 = dwconv1.reshape(b, c, h, w)
        perception = torch.mul(dwconv1, conv11) + dwconv2

        return perception


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


class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V        


class DownFRG(nn.Module):
    def __init__(self, dim, n_l_blocks=1, n_h_blocks=1, expand=2):
        super().__init__()
        self.dwt = DWT()
        self.l_conv = nn.Conv2d(dim*2, dim, 3, 1, 1)
        self.l_blk = nn.Sequential(*[LFSSBlock(dim, expand=expand) for _ in range(n_l_blocks)])

        self.h_fusion = SKFF(dim, height=3, reduction=8)
        self.h_blk = nn.Sequential(*[HFEBlock(dim, match_factor=1, ffn_expansion_factor=1) for _ in range(n_h_blocks)])
    
    def forward(self, x, x_d):
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)
        b, c, h, w = x_LL.shape
        x_LL = self.l_conv(torch.cat([x_LL, x_d], dim=1))
        x_LL = rearrange(x_LL, "b c h w -> b (h w) c").contiguous()
        for l_layer in self.l_blk:
            x_LL = l_layer(x_LL, [h, w])
        x_LL = rearrange(x_LL, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        x_h = self.h_fusion([x_HL, x_LH, x_HH])
        for h_layer in self.h_blk:
            x_h = h_layer(x_h, x_LL)
        
        return x_LL, x_h

class upFRG(nn.Module):
    def __init__(self, dim, n_l_blocks=1, n_h_blocks=1, expand=2):
        super().__init__()
        self.iwt = IWT()
        self.l_blk = nn.Sequential(*[LFSSBlock(dim, expand=expand) for _ in range(n_l_blocks)])

        self.h_out_conv = nn.Conv2d(dim, dim*3, 3, 1, 1)
        self.h_blk = nn.Sequential(*[HFEBlock(dim, match_factor=1, ffn_expansion_factor=1) for _ in range(n_h_blocks)])
    
    def forward(self, x_l, x_h):
        b, c, h, w = x_l.shape
        x_l = rearrange(x_l, "b c h w -> b (h w) c").contiguous()
        for l_layer in self.l_blk:
            x_l = l_layer(x_l, [h, w])
        x_l = rearrange(x_l, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        for h_layer in self.h_blk:
            x_h = h_layer(x_h, x_l)
        x_h = self.h_out_conv(x_h)
        x_l = self.iwt(torch.cat([x_l, x_h], dim=1))

        return x_l


class UNet(nn.Module):
    def __init__(self, in_chn=3, wf=48, n_l_blocks=[1,1,2], n_h_blocks=[1,1,1], ffn_scale=2):
        super(UNet, self).__init__()
        self.ps_down1 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d((2**2) * in_chn, wf, 1, 1, 0)
        )
        self.ps_down2 = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d((4**2) * in_chn, wf, 1, 1, 0)
        )
        self.ps_down3 = nn.Sequential(
            nn.PixelUnshuffle(8),
            nn.Conv2d((8**2) * in_chn, wf, 1, 1, 0)
        )
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        # encoder of UNet-64
        prev_channels = 0
        self.down_group1 = DownFRG(wf, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], expand=ffn_scale)
        self.down_group2 = DownFRG(wf, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], expand=ffn_scale)
        self.down_group3 = DownFRG(wf, n_l_blocks=n_l_blocks[2], n_h_blocks=n_h_blocks[2], expand=ffn_scale)

        # decoder of UNet-64
        self.up_group3 = upFRG(wf, n_l_blocks=n_l_blocks[2], n_h_blocks=n_h_blocks[2], expand=ffn_scale)
        self.up_group2 = upFRG(wf, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], expand=ffn_scale)
        self.up_group1 = upFRG(wf, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], expand=ffn_scale)

        self.last = nn.Conv2d(wf, in_chn, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        img = x
        img_down1 = self.ps_down1(x)
        img_down2 = self.ps_down2(x)
        img_down3 = self.ps_down3(x)

        ##### shallow conv #####
        x1 = self.conv_01(img)
        ######## UNet-64 ########
        # Down-path (Encoder)
        x_l, x_H1 = self.down_group1(x1, img_down1)
        x_l, x_H2 = self.down_group2(x_l, img_down2)
        x_l, x_H3 = self.down_group3(x_l, img_down3)

        # Up-path (Decoder)
        x_l = self.up_group3(x_l, x_H3)      
        x_l = self.up_group2(x_l, x_H2)
        x_l = self.up_group1(x_l, x_H1)

        ##### Reconstruct #####
        out_1 = self.last(x_l) + img

        return out_1

    
@ARCH_REGISTRY.register()
class WaveMamba(nn.Module):
    def __init__(self,
                 *,
                 in_chn,
                 wf,
                 n_l_blocks=[1,1,2],
                 n_h_blocks=[1,1,1], 
                 ffn_scale=2.0, 
                 **ignore_kwargs):
        super().__init__()
        self.restoration_network = UNet(in_chn=in_chn, wf=wf, n_l_blocks=n_l_blocks, n_h_blocks=n_h_blocks, ffn_scale=ffn_scale)

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

    def check_image_size(self, x, window_size=8):
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % (window_size)) % (
            window_size)
        mod_pad_w = (window_size - w % (window_size)) % (
            window_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    @torch.no_grad()
    def test(self, input):
        _, _, h_old, w_old = input.shape

        restoration = self.encode_and_decode(input)

        output = restoration

        return output

    def forward(self, input):

        restoration = self.encode_and_decode(input)

        return restoration


if __name__== '__main__': 
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 1920, 1280).to(device)
    model = UNet(in_chn=3, wf=32, n_l_blocks=[1,2,4], n_h_blocks=[1,1,1], ffn_scale=2).to(device)
#    print(model)
    inp_shape=(3,512, 512)
    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=True)

    params = float(params[:-4])
    print('mac', macs)
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated(device)
        start_time = time.time()
        output = model(x)
        end_time = time.time()
        memory_used = torch.cuda.max_memory_allocated(device)
    running_time = end_time - start_time
    print(output.shape)
    print(running_time)
    print(f"Memory used: {memory_used / 1024**3:.3f} GB")