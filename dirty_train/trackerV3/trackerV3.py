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
from .blocks import CorrBlock, ShallowEncoder, ShallowShallow
from .extract_patch import extract_glimpse, extract_glimpse_forloop
from .refine_track import refine_track_softmax, refine_track
from .fine_tracker import FineTracker

# from kornia.geometry.subpix import dsnt


class TrackerV3(nn.Module):
    def __init__(self, stride=8, corr_levels=4, corr_radius=3, latent_dim=128, cfg=None):
        super(TrackerV3, self).__init__()

        # import pdb;pdb.set_trace()
        self.stable_scale = cfg.stable_scale
        self.stride = stride

        self.hidden_dim = hdim = 256

        self.latent_dim = latent_dim

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.cfg = cfg

        self.transformer_dim = 128 + self.corr_levels * (self.corr_radius * 2 + 1) ** 2 + self.latent_dim

        if self.cfg.adapad:
            remainder = self.transformer_dim % 4

            self.topadnum = 0 if remainder == 0 else 4 - remainder
            print(self.topadnum)
            self.pad_transformer_dim = False
            self.transformer_dim += self.topadnum
        else:
            if self.transformer_dim % 2 == 0:
                self.transformer_dim += 2
                self.pad_transformer_dim = False
            else:
                self.transformer_dim += 3
                self.pad_transformer_dim = True

        # import pdb;pdb.set_trace()

        if not self.cfg.space_att:
            self.updateformer = UpdateFormer(
                time_depth=6,
                input_dim=self.transformer_dim,
                hidden_size=cfg.uformerdim,
                output_dim=self.latent_dim + 2,
                mlp_ratio=4.0,
                cfg=cfg,
            )
        else:
            self.updateformer = EfficientUpdateFormer(
                space_depth=6,
                time_depth=6,
                input_dim=self.transformer_dim,
                hidden_size=cfg.uformerdim,
                output_dim=self.latent_dim + 2,
                mlp_ratio=4.0,
                add_space_attn=True,
                num_virtual_tracks=64,
                cfg=cfg,
            )

        self.norm = nn.GroupNorm(1, self.latent_dim)

        self.ffeat_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())
        self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

        if self.cfg.track_conf:
            import pdb

            pdb.set_trace()
            self.conf_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

        if self.cfg.softmax_refine or self.cfg.fine_tracker:
            shallow_dim = self.cfg.shallow_dim
            shallow_input_dim = (self.cfg.MODEL.ENCODER.output_dim + 3) if cfg.refine_with_f else 3
            if self.cfg.downfneat:
                shallow_input_dim = 32 + 3

            if self.cfg.shallowshallow:
                self.shallowfnet = ShallowShallow(
                    input_dim=shallow_input_dim,
                    output_dim=shallow_dim,
                    norm_fn="instance",
                    dropout=0,
                    stride=1,
                    cfg=cfg,
                )
            else:
                self.shallowfnet = ShallowEncoder(
                    input_dim=shallow_input_dim,
                    output_dim=shallow_dim,
                    norm_fn="instance",
                    dropout=0,
                    stride=1,
                    cfg=cfg,
                )

        if self.cfg.fine_tracker:
            self.finetracker = FineTracker(latent_dim=shallow_dim, cfg=cfg)
        else:
            self.finetracker = None

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

        B, S, C, H8, W8 = fmaps.shape

        device = rgbs.device

        # fmaps

        # import pdb;pdb.set_trace();m=1

        if self.cfg.track_downr > 1:
            xys = xys / float(self.cfg.track_downr)

        xys_ = xys.clone() / float(self.stride)

        if coords_init is None:
            coords = xys_.reshape(B, 1, N, 2).repeat(1, S, 1, 1)  # init with zero vel
        else:
            raise NotImplementedError
            coords = coords_init.clone() / self.stride

        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        if query_track_init is None:
            query_track_feat = sample_features4d(fmaps[:, 0], coords[:, 0])
        else:
            query_track_feat = query_track_init

        # init track feats by query feats
        track_feats = query_track_feat.unsqueeze(1).repeat(1, S, 1, 1)  # B, S, N, C

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

            flows_emb = get_2d_embedding(flows_, 64, cat_coords=False)

            pad = torch.zeros_like(flows_emb[..., 0:1])  # zero padding

            # HOW ABOUT REMOVING flows_ HERE?
            if self.cfg.adapad:
                if self.topadnum > 0:
                    flows_emb = torch.cat([flows_emb, pad.expand(-1, -1, self.topadnum)], dim=-1)
            else:
                flows_emb = torch.cat([flows_emb, flows_ / self.stable_scale], dim=-1)

            track_feats__ = track_feats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            #############
            pos_embed = get_2d_sincos_pos_embed(self.transformer_dim, grid_size=(H8, W8)).to(coords.device)
            sampled_pos_emb = sample_features4d(pos_embed.expand(B, -1, -1, -1), coords[:, 0])
            sampled_pos_emb = rearrange(sampled_pos_emb, "b n c -> (b n) c").unsqueeze(1)
            #############

            # if self.cfg.scale_tcorr:
            #     fcorrs_ = fcorrs_ / fcorrs_.shape[-1]

            if self.pad_transformer_dim:
                transformer_input = torch.cat([flows_emb, fcorrs_, track_feats__, pad], dim=2)
            else:
                transformer_input = torch.cat([flows_emb, fcorrs_, track_feats__], dim=2)

            x = transformer_input + sampled_pos_emb

            x = rearrange(x, "(b n) t d -> b n t d", b=B)
            # B, N, T, C

            if is_train and self.cfg.ckpt_updatef:
                delta = torch.utils.checkpoint.checkpoint(self.updateformer, x, use_reentrant=False)
            else:
                delta = self.updateformer(x)

            # BN, T, C
            delta = rearrange(delta, " b n t d -> (b n) t d", b=B)

            delta_coords_ = delta[:, :, :2]
            delta_coords_ = delta_coords_ / self.stable_scale

            delta_feats_ = delta[:, :, 2:]

            track_feats__ = track_feats__.reshape(B * N * S, self.latent_dim)
            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
            track_feats__ = self.ffeat_updater(self.norm(delta_feats_)) + track_feats__
            track_feats = track_feats__.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)  # B,S,N,C

            # How about really sampling features here?

            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)

            if not is_train:
                coords[:, 0] = coords_bak[:, 0]  # lock coord0 for target

            if self.cfg.train.fix_first_cor:
                coords[:, 0] = coords_bak[:, 0]  # lock coord0 for target

            if self.cfg.track_downr > 1:
                coord_preds.append(coords * self.stride * self.cfg.track_downr)
            else:
                coord_preds.append(coords * self.stride)

        # B, S, N
        vis_e = self.vis_predictor(track_feats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
        vis_e = torch.sigmoid(vis_e)

        if self.cfg.track_conf:
            conf_e = self.conf_predictor(track_feats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
            conf_e = torch.sigmoid(conf_e)
        patch_score = None

        refine_loss = None
        if self.cfg.softmax_refine and not self.cfg.fine_tracker:
            assert self.cfg.train.fix_first_cor
            full_refined_tracks, refine_loss, refined_tracks = refine_track_softmax(
                rgbs, fmaps, self.shallowfnet, coord_preds, trajs_g, vis_g, is_train=is_train, cfg=self.cfg
            )
        elif self.cfg.fine_tracker:
            assert self.cfg.train.fix_first_cor
            full_refined_tracks, refine_loss, refined_tracks, patch_score = refine_track(
                rgbs,
                fmaps,
                self.shallowfnet,
                coord_preds,
                trajs_g,
                vis_g,
                is_train=is_train,
                finetracker=self.finetracker,
                compute_score=True,
                cfg=self.cfg,
            )

        if trajs_g is not None:
            valids = torch.ones(vis_g.shape).to(vis_g.device)

            # filter out the tracks not visible in the first frame
            mask = vis_g[:, 0, :] == True
            valids = valids * mask.unsqueeze(1)

            seq_loss = sequence_loss(
                coord_preds,
                trajs_g,
                vis_g,
                valids,
                0.8,
                vis_aware=self.cfg.train.vis_aware,
                vis_aware_w=self.cfg.train.vis_aware_w,
                huber=self.cfg.train.huber,
                max_thres=self.cfg.clip_trackL,
            )
            
            if not torch.isfinite(seq_loss).all():
                print("seq_loss goes NaN or inf")
                
            vis_loss, _ = balanced_ce_loss(vis_e, vis_g, valids)

            if not torch.isfinite(vis_loss).all():
                print("vis_loss goes NaN or inf")

            if self.cfg.track_conf:
                final_dis = torch.sqrt(torch.sum((coord_preds[-1] - trajs_g) ** 2, dim=-1))
                conf_loss, _ = balanced_ce_loss(conf_e, final_dis < 1, valids)
            else:
                conf_loss = vis_loss * 0

            # loss for refinement
            if refine_loss is None:
                refine_loss = seq_loss * 0

            # if self.cfg.debug:
            #     import pdb;pdb.set_trace()
            losses = (seq_loss, vis_loss, conf_loss, refine_loss)
        else:
            losses = None

        if self.cfg.softmax_refine or self.cfg.fine_tracker:
            coord_preds.append(full_refined_tracks)

        if return_feat:
            return coord_preds, None, vis_e, track_feats, patch_score, losses
        else:
            return coord_preds, None, vis_e, patch_score, losses
