# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
import collections
from torch import Tensor
from itertools import repeat

from .model_utils import bilinear_sampler


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        # self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LlamaMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.act_fn = act_layer()
        # self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1, kernel_size=3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, padding=1, stride=stride, padding_mode="zeros"
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=1, padding_mode="zeros")
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BasicEncoder, self).__init__()

        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2

        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode="zeros")
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4, output_dim * 2, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        a = _bilinear_intepolate(a, self.stride, H, W)
        b = _bilinear_intepolate(b, self.stride, H, W)
        c = _bilinear_intepolate(c, self.stride, H, W)
        d = _bilinear_intepolate(d, self.stride, H, W)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


def _bilinear_intepolate(x, stride, H, W):
    return F.interpolate(x, (H // stride, W // stride), mode="bilinear", align_corners=True)


class ShallowEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=64, stride=1, norm_fn="instance", dropout=0.0, residual=True, cfg=None):
        super(ShallowEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn

        self.in_planes = output_dim

        self.residual = residual
        self.shallower = cfg.shallower

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=3, stride=1, padding=1, padding_mode="zeros")
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, stride=2)

        if self.shallower:
            self.layer2 = self._make_layer(64, stride=2)
            self.conv2 = nn.Conv2d(64, output_dim, kernel_size=1)
        else:
            self.layer2 = self._make_layer(96, stride=2)
            self.conv2 = nn.Conv2d(64 + 96, output_dim, kernel_size=1)

        self.dropout = -1
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # A LOT TO REDUCE HERE

        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        if self.shallower:
            tmp = self.layer1(x)
            x = x + F.interpolate(tmp, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)
            tmp = self.layer2(tmp)
            x = x + F.interpolate(tmp, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)
            tmp = None
            x = self.conv2(x) + x
        else:
            a = self.layer1(x)
            b = self.layer2(a)

            a = F.interpolate(a, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)
            b = F.interpolate(b, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)

            if self.residual:
                x = self.conv2(torch.cat([a, b], dim=1)) + x
            else:
                x = self.conv2(torch.cat([a, b], dim=1))

        if self.training and self.dropout > 0:
            x = self.dropout(x)

        return x


class ShallowShallow(nn.Module):
    def __init__(self, input_dim=3, output_dim=64, stride=1, norm_fn="instance", dropout=0.0, residual=True, cfg=None):
        super(ShallowShallow, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn

        self.in_planes = output_dim

        self.residual = residual
        self.shallower = cfg.shallower

        self.sssingle = cfg.sssingle

        # import pdb;pdb.set_trace()

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=3, stride=2, padding=1, padding_mode="zeros")
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(output_dim, stride=2)

        self.layer2 = self._make_layer(output_dim, stride=2)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        self.dropout = -1
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        self.in_planes = dim

        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)

        if self.sssingle:
            # layers = (layer1)
            return layer1
        else:
            layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
            layers = (layer1, layer2)
            return nn.Sequential(*layers)

    def forward(self, x):
        # A LOT TO REDUCE HERE

        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        tmp = self.layer1(x)
        x = x + F.interpolate(tmp, (x.shape[-2:]), mode="bilinear", align_corners=True)
        tmp = self.layer2(tmp)
        x = x + F.interpolate(tmp, (x.shape[-2:]), mode="bilinear", align_corners=True)
        tmp = None
        x = self.conv2(x) + x

        x = F.interpolate(x, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)

        if self.training and self.dropout > 0:
            x = self.dropout(x)

        return x


class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4, multiple_track_feats=False, padding_mode="zeros"):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.multiple_track_feats = multiple_track_feats

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            *_, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(coords.device)

            # checklevel
            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(corrs.reshape(B * S * N, 1, H, W), coords_lvl, padding_mode=self.padding_mode)
            corrs = corrs.view(B, S, N, -1)

            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        # out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out

    def corr(self, targets):
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            targets_split = targets.split(C // self.num_levels, dim=-1)
            B, S, N, C = targets_split[0].shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            if self.multiple_track_feats:
                fmap1 = targets_split[i]
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


################################################################################################


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False, rope=False):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.rope = rope
        if self.rope:
            self.rotary_emb = LlamaRotaryEmbedding(dim_head, max_position_embeddings=2048, base=10000)

    def forward(self, x, context=None, dummy=None, attn_mask=None, cotrack=False):
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        if self.rope:
            cos_q, sin_q = self.rotary_emb(v, seq_len=N1)
            cos_k, sin_k = self.rotary_emb(v, seq_len=N2)

            q = (q * cos_q) + (rotate_half(q) * sin_q)
            k = (k * cos_k) + (rotate_half(k) * sin_k)

        sim = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            import pdb

            pdb.set_trace()
            sim = sim + attn_mask

        attn = sim.softmax(dim=-1)

        # torch.Size([576, 25, 8, 48])
        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        if cotrack:
            return self.to_out(x)
        return self.to_out(x), None


class LinearAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, num_heads=8, qkv_bias=False, rope=False):
        super().__init__()
        inner_dim = query_dim
        context_dim = default(context_dim, query_dim)
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, dummy=None, attn_mask=None):
        print("chipchpi")
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)

        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        sim = torch.einsum("bhqd,bhkd->bhqk", q, k)
        Z = 1 / (torch.einsum("bhqd,bhd->bhq", q, k.sum(dim=2)) + 1e-6).unsqueeze(-1)
        sim = sim * Z

        # (attn @ v).transpose(1, 2).shape
        # torch.Size([576, 25, 8, 48])

        x = torch.einsum("bhqk, bhkd -> bhqd", sim, v).transpose(1, 2).reshape(B, N1, C)

        if attn_mask is not None:
            import pdb

            pdb.set_trace()

        return self.to_out(x), None


class AttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = nn.MultiheadAttention,
        mlp_ratio=4.0,
        cfg=None,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # self.attn = attn_class(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        if cfg.linear_att:
            attn_class = LinearAttention
            self.attn = attn_class(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            self.attn = attn_class(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        if cfg.llamamlp:
            self.mlp = LlamaMLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.SiLU)
        else:
            # approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                # act_layer=approx_gelu,
                drop=0,
            )

    def forward(self, x, mask=None):
        # Prepare the mask for PyTorch's attention (it expects a different format)
        # attn_mask = mask if mask is not None else None
        # Normalize before attention
        x = self.norm1(x)

        # PyTorch's MultiheadAttention returns attn_output, attn_output_weights
        # attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)

        attn_output, _ = self.attn(x, x, x)

        # Add & Norm
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, cfg=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)

        # if cfg.rope:
        #     raise NotImplementedError("rope not enabled now")
        #     self.cross_attn = Attention(hidden_size, context_dim=context_dim, num_heads=num_heads, qkv_bias=False, rope=cfg.rope,  **block_kwargs)
        # else:

        if cfg.linear_att:
            attn_class = LinearAttention
            self.cross_attn = attn_class(
                hidden_size, context_dim=context_dim, num_heads=num_heads, qkv_bias=True, **block_kwargs
            )
        else:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, batch_first=True, **block_kwargs
            )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        if cfg.llamamlp:
            self.mlp = LlamaMLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.SiLU)
        else:
            # approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                # act_layer=approx_gelu,
                drop=0,
            )

    def forward(self, x, context, mask=None):
        # Normalize inputs
        x = self.norm1(x)
        context = self.norm_context(context)

        # Apply cross attention
        # Note: nn.MultiheadAttention returns attn_output, attn_output_weights
        attn_output, _ = self.cross_attn(x, context, context, attn_mask=mask)

        # Add & Norm
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlockCotracker(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(context_dim)
        self.cross_attn = Attention(
            hidden_size, context_dim=context_dim, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, context, mask=None):
        # if mask is not None:
        #     if mask.shape[1] == x.shape[1]:
        #         mask = mask[:, None, :, None].expand(
        #             -1, self.cross_attn.heads, -1, context.shape[1]
        #         )
        #     else:
        #         mask = mask[:, None, None].expand(-1, self.cross_attn.heads, x.shape[1], -1)

        #     max_neg_value = -torch.finfo(x.dtype).max
        #     attn_bias = (~mask) * max_neg_value
        # else:
        #     attn_bias = None

        x = x + self.cross_attn(self.norm1(x), context=self.norm_context(context), cotrack=True)
        x = x + self.mlp(self.norm2(x))
        return x


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # init it by 64
        # self.max_seq_len_cached = 256

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
#     # sin = sin[position_ids].unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# class EfficientCorrBlock:
#     def __init__(
#         self,
#         fmaps,
#         num_levels=4,
#         radius=4,
#         padding_mode="zeros",
#     ):
#         B, S, C, H, W = fmaps.shape
#         self.padding_mode = padding_mode
#         self.num_levels = num_levels
#         self.radius = radius
#         self.fmaps_pyramid = []
#         self.fmaps_pyramid.append(fmaps)
#         for i in range(self.num_levels - 1):
#             fmaps_ = fmaps.reshape(B * S, C, H, W)
#             fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
#             _, _, H, W = fmaps_.shape
#             fmaps = fmaps_.reshape(B, S, C, H, W)
#             self.fmaps_pyramid.append(fmaps)
#     def sample(self, coords, target):
#         r = self.radius
#         device = coords.device
#         B, S, N, D = coords.shape
#         assert D == 2
#         target = target.permute(0,1,3,2).unsqueeze(-1)

#         out_pyramid = []
#         for i in range(self.num_levels):
#             pyramid = self.fmaps_pyramid[i]
#             C, H, W = pyramid.shape[2:]
#             centroid_lvl = torch.cat([torch.zeros_like(coords[...,:1], device=device), coords],dim=-1).reshape(B * S, N, 1, 1, 3) / 2**i

#             dx = torch.linspace(-r, r, 2 * r + 1, device=device)
#             dy = torch.linspace(-r, r, 2 * r + 1, device=device)
#             xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
#             zgrid = torch.zeros_like(xgrid, device=device)
#             delta = torch.stack([zgrid, xgrid, ygrid] , axis=-1)
#             delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
#             coords_lvl = centroid_lvl + delta_lvl
#             pyramid_sample = bilinear_sampler(pyramid.reshape(B * S, C, 1, H, W), coords_lvl)

#             corr = torch.sum(target * pyramid_sample.reshape(B, S, C, N, -1), dim=2)
#             corr = corr / torch.sqrt(torch.tensor(C).float())
#             out_pyramid.append(corr)

#         out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
#         out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
#         return out
# # Inside the tracker forward funciton:
# if self.efficient_corr:
#     corr_block = EfficientCorrBlock(
#         fmaps,
#         num_levels=4,
#         radius=3,
#         padding_mode="border",
#     )
# else:
#     corr_block = CorrBlock(
#         fmaps,
#         num_levels=4,
#         radius=3,
#         padding_mode="border",
#     )
# if self.efficient_corr:
#     fcorrs = corr_block.sample(coords, track_feat)
# else:
#     corr_block.corr(track_feat)
#     fcorrs = corr_block.sample(coords)
