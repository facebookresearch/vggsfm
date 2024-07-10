import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


from .losses import sequence_loss, balanced_ce_loss

from .model_utils import (
    sample_features4d,
    sample_features5d,
    get_2d_embedding,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

from .updateformer import EfficientUpdateFormer, UpdateFormer
from .blocks import CorrBlock, ShallowEncoder
from .extract_patch import extract_glimpse, extract_glimpse_forloop

# from .refine_track import refine_track_softmax, refine_track
# from kornia.geometry.subpix import dsnt


class FineTracker(nn.Module):
    def __init__(self, stride=1, corr_levels=3, corr_radius=3, latent_dim=128, cfg=None):
        super(FineTracker, self).__init__()

        self.stable_scale = cfg.fine_stable_scale
        self.normalize_transformer = cfg.fine_normalize
        self.stride = stride

        self.hidden_dim = hdim = 64

        self.latent_dim = latent_dim

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.cfg = cfg

        self.transformer_dim = 32 + self.corr_levels * (self.corr_radius * 2 + 1) ** 2 + self.latent_dim

        if self.transformer_dim % 2 == 0:
            self.transformer_dim += 4
            self.pad_transformer_dim = False
        else:
            self.transformer_dim += 5
            self.pad_transformer_dim = True

        # No space attention for fine patch

        if self.cfg.sample_ffeat:
            outdim = 2
        else:
            outdim = self.latent_dim + 2

        self.updateformer = EfficientUpdateFormer(
            space_depth=0,
            time_depth=cfg.fine_depth,
            input_dim=self.transformer_dim,
            hidden_size=cfg.fine_hid,
            output_dim=outdim,
            mlp_ratio=4.0,
            add_space_attn=False,
            cfg=cfg,
        )

        # ckpt_updatef

        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeat_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())
        # self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

        if self.cfg.sample_ffeat:
            self.ffeat_updater = None
            self.norm = None

    def forward(
        self,
        xys,
        rgbs=None,
        fmaps=None,
        extra_fmaps=None,
        coords_init=None,
        query_track_init=None,
        iters=4,
        trajs_g=None,
        vis_g=None,
        valids=None,
        return_feat=False,
        is_train=False,
    ):
        B, N, D = xys.shape
        assert D == 2

        B, S, C, H, W = fmaps.shape
        device = fmaps.device

        H8 = H // self.stride
        W8 = W // self.stride

        xys_ = xys.clone() / float(self.stride)

        if coords_init is None:
            coords = xys_.reshape(B, 1, N, 2).repeat(1, S, 1, 1)  # init with zero vel
        else:
            coords = coords_init.clone() / self.stride

        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        if query_track_init is None:
            query_track_feat = sample_features4d(fmaps[:, 0], coords[:, 0])
        else:
            query_track_feat = query_track_init

        # init track feats by query feats
        track_feats = query_track_feat.unsqueeze(1).repeat(1, S, 1, 1).clone()  # B, S, N, C

        coords_bak = coords.clone()

        coord_preds = []

        delta_coords_ = None

        # Iterative Refinement
        for itr in range(iters):
            coords = coords.detach()
            fcorr_fn.corr(track_feats)

            fcorrs = fcorr_fn.sample(coords)  # B, S, N, LRR
            LRR = fcorrs.shape[3]

            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)

            flows_emb = get_2d_embedding(flows_, 16, cat_coords=False)
            flows_emb = torch.cat([flows_emb, flows_ / self.stable_scale], dim=-1)

            track_feats__ = track_feats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            #############
            pos_embed = get_2d_sincos_pos_embed(self.transformer_dim, grid_size=(H8, W8)).to(coords.device)
            sampled_pos_emb = sample_features4d(pos_embed.expand(B, -1, -1, -1), coords[:, 0])
            sampled_pos_emb = rearrange(sampled_pos_emb, "b n c -> (b n) c").unsqueeze(1)
            #############

            pad = torch.zeros_like(flows_emb[..., 0:3])  # zero padding

            if self.pad_transformer_dim:
                transformer_input = torch.cat([flows_emb, fcorrs_, track_feats__, pad], dim=2)
            else:
                transformer_input = torch.cat([flows_emb, fcorrs_, track_feats__, pad[..., 0:2]], dim=2)

            x = transformer_input + sampled_pos_emb

            x = rearrange(x, "(b n) t d -> b n t d", b=B)
            # B, N, T, C

            # if is_train and self.cfg.ckpt_updatef:
            #     delta = torch.utils.checkpoint.checkpoint(self.updateformer, x, use_reentrant=False)
            # else:
            delta = self.updateformer(x)

            # BN, T, C
            delta = rearrange(delta, " b n t d -> (b n) t d", b=B)

            delta_coords_ = delta[:, :, :2]
            delta_coords_ = delta_coords_ / self.stable_scale

            if self.normalize_transformer:
                delta_coords_ = F.tanh(delta_coords_)
                delta_coords_ = (delta_coords_ * 0.6 + 0.5) * H8

            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)

            if not is_train:
                coords[:, 0] = coords_bak[:, 0]  # lock coord0 for target

            if self.cfg.train.fix_first_cor:
                coords[:, 0] = coords_bak[:, 0]  # lock coord0 for target

            if self.cfg.sample_ffeat:
                track_feats = sample_features5d(fmaps, coords)
            else:
                delta_feats_ = delta[:, :, 2:]

                track_feats__ = track_feats__.reshape(B * N * S, self.latent_dim)
                delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
                track_feats__ = self.ffeat_updater(self.norm(delta_feats_)) + track_feats__
                track_feats = track_feats__.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)  # B,S,N,C

            coord_preds.append(coords * self.stride)

        return coord_preds, track_feats, query_track_feat
