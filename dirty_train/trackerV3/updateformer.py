import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.nn.init import trunc_normal_

from .blocks import (
    # Mlp,
    AttnBlock,
    CorrBlock,
    CrossAttnBlock,
)


class UpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self, time_depth=6, input_dim=320, hidden_size=384, num_heads=8, output_dim=130, mlp_ratio=4.0, cfg=None
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.cfg = cfg
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.time_blocks = TransformerEncoderWrapper(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=time_depth,
            dim_feedforward=mlp_hidden_dim,
            dropout=0,
            norm_first=True,
            batch_first=True,  # Set to True if your input data is (batch_size, seq_length, feature)
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None):
        tokens = self.input_transform(input_tensor)
        B, NNN, T, C = tokens.shape
        tokens = tokens.reshape((B * NNN, T, C))
        tokens = self.time_blocks(tokens)
        tokens = tokens.reshape((B, NNN, T, C))
        flow = self.flow_head(tokens)
        return flow


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
        cfg=None,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks

        if self.add_space_attn:
            self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        else:
            self.virual_tracks = None
        self.cfg = cfg
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention, cfg=cfg)
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention, cfg=cfg)
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, cfg=cfg)
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, cfg=cfg)
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights(trunk_init=cfg.trunk_init)

    def initialize_weights(self, trunk_init):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def init_weights_vit_timm(module: nn.Module, name: str = ""):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if trunk_init:
            self.apply(init_weights_vit_timm)
        else:
            self.apply(_basic_init)

    def forward(self, input_tensor, mask=None):
        tokens = self.input_transform(input_tensor)

        if self.cfg.resi_uformer:
            init_tokens = tokens

        B, _, T, _ = tokens.shape

        if self.add_space_attn:
            virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
            tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape

        j = 0
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if self.add_space_attn and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0):
                space_tokens = tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)  # B N T C -> (B T) N C
                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](virtual_tokens, point_tokens, mask=mask)
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](point_tokens, virtual_tokens, mask=mask)
                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C
                j += 1

        if self.add_space_attn:
            tokens = tokens[:, : N - self.num_virtual_tracks]

        if self.cfg.resi_uformer:
            tokens = tokens + init_tokens

        flow = self.flow_head(tokens)
        return flow


def TransformerEncoderWrapper(
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    norm_first: bool = True,
    batch_first: bool = True,
):
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
        norm_first=norm_first,
    )

    _trunk = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)
    return _trunk


def TransformerDecoderWrapper(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    norm_first: bool = True,
    batch_first: bool = True,
):
    decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
        norm_first=norm_first,
    )

    _trunk = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers)
    return _trunk
