import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers

from einops import rearrange
import math

NEG_INF = -1000000
# Layer Norm
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


#########################################
class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


##########################################################################
class TransformerBlock(nn.Module):
    r""" ART Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size: window size of dense attention
        interval: interval size of sparse attention
        ds_flag (int): use Dense Attention or Sparse Attention, 0 for DAB and 1 for SAB.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        # act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=8,
                 interval=16,
                 ds_flag=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.interval = interval
        self.ds_flag = ds_flag
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.fusion = nn.Conv2d(dim*2, dim, 1, 1, 0)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        if min(H, W) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.ds_flag = 0
            self.window_size = min(H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # padding
        size_par_d = self.window_size
        pad_l_d = pad_t_d = 0
        pad_r_d = (size_par_d - W % size_par_d) % size_par_d
        pad_b_d = (size_par_d - H % size_par_d) % size_par_d
        x_d = F.pad(x, (0, 0, pad_l_d, pad_r_d, pad_t_d, pad_b_d))
        _, Hd, Wd, _ = x_d.shape

        mask_d = torch.zeros((1, Hd, Wd, 1), device=x_d.device)
        if pad_b_d > 0:
            mask_d[:, -pad_b_d:, :, :] = -1
        if pad_r_d > 0:
            mask_d[:, :, -pad_r_d:, :] = -1

        # partition the whole feature map into several groups
        # Dense Attention
        G = Gh_d = Gw_d = self.window_size
        x_d = x_d.reshape(B, Hd // G, G, Wd // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x_d = x_d.reshape(B * Hd * Wd // G ** 2, G ** 2, C)
        nP_d = Hd * Wd // G ** 2  # number of partitioning groups
        # attn_mask
        if pad_r_d > 0 or pad_b_d > 0:
            mask_d = mask_d.reshape(1, Hd // G, G, Wd // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
            mask_d = mask_d.reshape(nP_d, 1, G * G)
            attn_mask_d = torch.zeros((nP_d, G * G, G * G), device=x_d.device)
            attn_mask_d = attn_mask_d.masked_fill(mask_d < 0, NEG_INF)
        else:
            attn_mask_d = None

        x_d = self.attn(x_d, Gh_d, Gw_d, mask=attn_mask_d)  # nP*B, Gh*Gw, C
        # merge embeddings
        x_d = x_d.reshape(B, Hd // G, Wd // G, G, G, C).permute(0, 1, 3, 2, 4,
                                                                5).contiguous()  # B, Hd//G, G, Wd//G, G, C
        x_d = x_d.reshape(B, Hd, Wd, C)
        # remove padding
        if pad_r_d > 0 or pad_b_d > 0:
            x_d = x_d[:, :H, :W, :].contiguous()
        x_d = x_d.permute(0,3,1,2).contiguous()
        
        # Sparse Attention
        size_par_s = self.interval
        pad_l_s = pad_t_s = 0
        pad_r_s = (size_par_s - W % size_par_s) % size_par_s
        pad_b_s = (size_par_s - H % size_par_s) % size_par_s
        x_s = F.pad(x, (0, 0, pad_l_s, pad_r_s, pad_t_s, pad_b_s))
        _, Hd, Wd, _ = x_s.shape
        mask_s = torch.zeros((1, Hd, Wd, 1), device=x_s.device)

        if pad_b_s > 0:
            mask_s[:, -pad_b_s:, :, :] = -1
        if pad_r_s > 0:
            mask_s[:, :, -pad_r_s:, :] = -1

        I, Gh_s, Gw_s = self.interval, Hd // self.interval, Wd // self.interval
        x_s = x_s.reshape(B, Gh_s, I, Gw_s, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()
        x_s = x_s.reshape(B * I * I, Gh_s * Gw_s, C)
        nP_s = I ** 2  # number of partitioning groups
        # attn_mask
        if pad_r_s > 0 or pad_b_s > 0:
            mask_s = mask_s.reshape(1, Gh_s, I, Gw_s, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
            mask_s = mask_s.reshape(nP_s, 1, Gh_s * Gw_s)
            attn_mask_s = torch.zeros((nP_s, Gh_s * Gw_s, Gh_s * Gw_s), device=x_s.device)
            attn_mask_s = attn_mask_s.masked_fill(mask_s < 0, NEG_INF)
        else:
            attn_mask_s = None

        # MSA
        x_s = self.attn(x_s, Gh_s, Gw_s, mask=attn_mask_s)  # nP*B, Gh*Gw, C
        # merge embeddings
        x_s = x_s.reshape(B, I, I, Gh_s, Gw_s, C).permute(0, 3, 1, 4, 2, 5).contiguous() # B, Gh, I, Gw, I, C
        x_s = x_s.reshape(B, Hd, Wd, C)
        # remove padding
        if pad_r_s > 0 or pad_b_s > 0:
            x_s = x_s[:, :H, :W, :].contiguous()
        x_s = x_s.permute(0,3,1,2).contiguous()

        x = self.fusion(torch.cat((x_d, x_s), dim=1))
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, ds_flag={self.ds_flag}, mlp_ratio={self.mlp_ratio}"
        

##########################################################################

class STransformerBlock(nn.Module):
    r""" ART Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size: window size of dense attention
        interval: interval size of sparse attention
        ds_flag (int): use Dense Attention or Sparse Attention, 0 for DAB and 1 for SAB.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        # act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 interval=8,
                 ds_flag=1,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.interval = interval
        self.ds_flag = ds_flag
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        if min(H, W) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.ds_flag = 0
            self.window_size = min(H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # padding
        size_par = self.interval if self.ds_flag == 1 else self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - W % size_par) % size_par
        pad_b = (size_par - H % size_par) % size_par
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape

        mask = torch.zeros((1, Hd, Wd, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1

        # partition the whole feature map into several groups
        if self.ds_flag == 0:  # Dense Attention
            G = Gh = Gw = self.window_size
            x = x.reshape(B, Hd // G, G, Wd // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.reshape(B * Hd * Wd // G ** 2, G ** 2, C)
            nP = Hd * Wd // G ** 2  # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Hd // G, G, Wd // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                mask = mask.reshape(nP, 1, G * G)
                attn_mask = torch.zeros((nP, G * G, G * G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        if self.ds_flag == 1:  # Sparse Attention
            I, Gh, Gw = self.interval, Hd // self.interval, Wd // self.interval
            x = x.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.reshape(B * I * I, Gh * Gw, C)
            nP = I ** 2  # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
                mask = mask.reshape(nP, 1, Gh * Gw)
                attn_mask = torch.zeros((nP, Gh * Gw, Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        # MSA
        x = self.attn(x, Gh, Gw, mask=attn_mask)  # nP*B, Gh*Gw, C

        # merge embeddings
        if self.ds_flag == 0:
            x = x.reshape(B, Hd // G, Wd // G, G, G, C).permute(0, 1, 3, 2, 4,
                                                                5).contiguous()  # B, Hd//G, G, Wd//G, G, C
        else:
            x = x.reshape(B, I, I, Gh, Gw, C).permute(0, 3, 1, 4, 2, 5).contiguous()  # B, Gh, I, Gw, I, C
        x = x.reshape(B, Hd, Wd, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))


        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, ds_flag={self.ds_flag}, mlp_ratio={self.mlp_ratio}"

## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x



class Net(nn.Module):
    def __init__(self, dim, n_blocks=8, num_heads=6, window_size=8, interval=16, mlp_ratio=2, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Sequential(
            nn.Conv2d(3, dim // upscaling_factor, 3, 1, 1),
            nn.PixelUnshuffle(upscaling_factor)
        )
        out_dim = upscaling_factor * dim 
        self.feats = nn.Sequential(*[TransformerBlock(out_dim, num_heads=num_heads, window_size=window_size, interval=interval, mlp_ratio=mlp_ratio) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(out_dim, 3*upscaling_factor*upscaling_factor, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        x0 = x
        for layer in self.feats:
            x = layer(x, [h, w])
        x = x + x0
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x = self.to_img(x)
        return x
    
# for LOL dataset
# class SAFMN(nn.Module):
#     def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
#         super().__init__()
#         self.to_feat = nn.Sequential(
#             nn.Conv2d(3, dim, 3, 1, 1),
#             ResBlock(dim, 3, 1, 1)
#         )

#         self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

#         self.to_img = nn.Sequential(
#             ResBlock(dim, 3, 1, 1),
#             nn.Conv2d(dim, 3, 3, 1, 1)
#         )

#     def forward(self, x):
#         x = self.to_feat(x)
#         x = self.feats(x) + x
#         x = self.to_img(x)
#         return x

@ARCH_REGISTRY.register()
class ART(nn.Module):
    def __init__(self,
                 *,
                 dim, 
                 n_blocks=8, 
                 num_heads=6, 
                 window_size=8, 
                 interval=16, 
                 mlp_ratio=2, 
                 upscaling_factor=4,
                 **ignore_kwargs):
        super().__init__()
        self.restoration_network = Net(dim=dim, n_blocks=n_blocks, num_heads=num_heads, window_size=window_size, interval=interval, mlp_ratio=mlp_ratio, upscaling_factor=upscaling_factor)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    x = torch.randn(1, 3, 2000, 3000).cuda()
    model = Net(dim=64, n_blocks=8, num_heads=8, window_size=8, interval=8, mlp_ratio=2, upscaling_factor=8).cuda()
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    with torch.no_grad():
        start_time = time.time()
        output = model(x)
        end_time = time.time()
    running_time = end_time - start_time
    print(output.shape)
    print(running_time)