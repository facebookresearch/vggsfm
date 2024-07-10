import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from kornia.geometry.subpix import dsnt
from .extract_patch import extract_glimpse, extract_glimpse_forloop
from .losses import sequence_loss
from PIL import Image
import os
from kornia.utils.grid import create_meshgrid
from .model_utils import sample_features4d


def get_content_to_extract(fmaps, rgbs, B, S, C, H8, W8, H, W):
    # Perform the interpolation
    fmaps_reshape = F.interpolate(fmaps.reshape(B * S, C, H8, W8), (H, W), mode="bilinear", align_corners=True)
    # Concatenate the results with rgbs
    content_to_extract = torch.cat([rgbs.reshape(B * S, 3, H, W), fmaps_reshape], dim=1)
    return content_to_extract


def save_images(tensor, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Denormalize the images
    tensor = tensor * 255
    # Loop over the images
    for i, img in enumerate(tensor):
        # Convert to PIL Image
        img = Image.fromarray(img.permute(1, 2, 0).byte().numpy())
        # Save the image
        img.save(os.path.join(folder, f"image_{i}.png"))


def refine_track(
    rgbs,
    fmaps,
    fnet,
    coord_preds,
    trajs_g=None,
    vis_g=None,
    track_ratio=0.2,
    is_train=False,
    finetracker=None,
    compute_score=False,
    cfg=None,
):
    # patch raidus and patch size
    pradius = cfg.pradius
    psize = pradius * 2 + 1

    # coarse_pred: BxSxNx2, where B is the batch, S is the video length, and N is the number of tracks
    # now we are going to extract patches with the center at coarse_pred
    # please note that the last dimension indicates x and y, and hence has a dim number of 2
    coarse_pred = coord_preds[-1].detach().clone()
    query_points = coarse_pred[:, 0]

    if trajs_g is not None and vis_g is not None:
        # trajs_g: ground truth track, BxSxNx2
        patch_trajs_g = (trajs_g - coarse_pred) + pradius

        # if the ground truth point is outside the patch (centered at coarse_pred), it is outsider
        # both x and y should be within the range
        inside_flag = (patch_trajs_g >= 0) & (patch_trajs_g <= (psize - 1))
        inside_flag = inside_flag.sum(dim=-1) == 2

        # note that outsiders are not visible to the patches
        patch_vis_g = torch.logical_and(inside_flag, vis_g)
    else:
        patch_vis_g = None

    if is_train:
        # Prepare the valid patches
        # During training, we only want the tracks that have most insiders
        # Therefore, we sample top track_ratio (e.g., 20%) of tracks from all

        B, S, N, _ = coarse_pred.shape

        valid_patches_count = patch_vis_g.sum(dim=1)
        M = int(track_ratio * N)
        _, track_idx = valid_patches_count.topk(M, dim=1)
        track_idx_expand = track_idx.unsqueeze(1).expand(-1, S, -1)

        B_indices = torch.arange(B).view(B, 1, 1).expand(-1, S, M)
        S_indices = torch.arange(S).view(1, S, 1).expand(B, -1, M)

        # selection by indexing
        selected_tracks = coarse_pred[B_indices, S_indices, track_idx_expand, :]  # BxSxMx2
        selected_tracks_gt = trajs_g[B_indices, S_indices, track_idx_expand, :]  # BxSxMx2
        selected_tracks_vis = patch_vis_g[B_indices, S_indices, track_idx_expand]  # BxSxM
    else:
        selected_tracks = coarse_pred
        selected_tracks_gt = trajs_g
        selected_tracks_vis = patch_vis_g

    # Now, extract patches
    # update the B, S, N since we have updated the tracks
    B, S, N, _ = selected_tracks.shape
    _, _, _, H, W = rgbs.shape
    # feature maps are in the 1/8 resolution of rgbs

    if cfg.downfneat:
        # dimension average pooling
        _, _, C, H8, W8 = fmaps.shape
        fmaps = fmaps.reshape(B, S, C // 4, 4, H8, W8)
        fmaps = fmaps.max(dim=3)[0].detach().clone()

    _, _, C, H8, W8 = fmaps.shape

    with torch.no_grad():
        if cfg.refine_with_f:
            # upsample to full resolution
            # note we need to combine batch and video length so images are treated as individual ones
            # import pdb;pdb.set_trace()

            # if is_train and cfg.ckpt_updatef:
            #     content_to_extract = torch.utils.checkpoint.checkpoint(get_content_to_extract, fmaps, rgbs, B, S, C, H8, W8, H, W)
            # else:
            # content_to_extract = get_content_to_extract(fmaps, rgbs, B, S, C, H8, W8, H, W)
            fmaps_reshape = F.interpolate(fmaps.reshape(B * S, C, H8, W8), (H, W), mode="bilinear", align_corners=True)
            rgbs_reshape = rgbs.reshape(B * S, 3, H, W)
            content_to_extract = torch.cat([rgbs_reshape, fmaps_reshape], dim=1)
            # get_content_to_extract
            # fmaps_reshape = F.interpolate(fmaps.reshape(B*S, C, H8, W8), (H, W), mode="bilinear", align_corners=True)
            # content_to_extract = torch.cat([rgbs.reshape(B*S, 3, H, W), fmaps_reshape],dim=1)
        else:
            content_to_extract = rgbs_reshape = rgbs.reshape(B * S, 3, H, W)

        C_in = content_to_extract.shape[1]

        # if cfg.debug:
        #     tmp = extract_glimpse(content_to_extract, (psize, psize), selected_tracks.reshape(B*S, N, 2))

        # (B*S)x C_in x H x W -> (B*S)x C_in x H_new x W_new x P
        # moving sliding windows (psize x psize) to build patches
        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1)

    # note point positions are float here
    # we can use grid_sample to extract patches but it takes too much memory
    # instead, we use the floored track xy to sample patches
    # and save the left numbers
    track_int = selected_tracks.floor().int()
    track_frac = selected_tracks - track_int

    # get the location of the  top left corner of patches
    # because the ouput of pytorch unfold are indexed by top left corner
    adjusted_points = track_int - pradius
    adjusted_points_BSN = adjusted_points.clone()

    # clamp the values so that we will not go out of indexes (VERY IMPORTANT: WE ASSUME H=W here)
    adjusted_points = adjusted_points.clamp(0, H - psize)

    # reshape from BxSxNx2 -> (B*S)xNx2
    adjusted_points = adjusted_points.reshape(B * S, N, 2)

    # prepare batches for indexing, (B*S)xN
    batch_indices = torch.arange(B * S)[:, None].expand(-1, N).to(content_to_extract.device)

    # get the selected patches
    # selected_patches: (B*S) x N x C_in x P x P
    selected_patches = content_to_extract[batch_indices, :, adjusted_points[..., 1], adjusted_points[..., 0]]

    # Can it reduce GPU consumption?
    # content_to_extract = None

    ### Feed selected_patches to network
    # (B*S*N) x C_out x P x P

    # if is_train and cfg.ckpt_updatef:
    #     patch_feat = torch.utils.checkpoint.checkpoint(fnet, selected_patches.reshape(B*S*N, C_in, psize, psize), use_reentrant=False)
    # else:

    patch_feat = fnet(selected_patches.reshape(B * S * N, C_in, psize, psize))

    C_out = patch_feat.shape[1]

    # get back to BxSxNx C_out xPxP
    patch_feat_reshape = patch_feat.reshape(B, S, N, C_out, psize, psize)

    refined_tracks = None
    fine_level_topleft = None
    refine_loss_tracker = None
    refine_loss_softmax = None
    query_point_feat = None
    refine_loss = 0

    if finetracker is not None:
        # Refine the coarse tracks by finetracker
        patch_feat_reshape = rearrange(patch_feat_reshape, "b s n c p q -> (b n) s c p q")

        # get the point locations at the first frame for all the tracks
        # and move them relative to the top left of the patch
        patch_xyz = track_frac[:, 0] + pradius
        patch_xyz = patch_xyz.reshape(B * N, 2).unsqueeze(1)

        # get the refined tracks, which is a list
        # the coord_preds_fine[-1] is the final prediction of finetracker
        # it provides 2d positions with the top left corner of patch as zero

        # TODO: wait, we can change here right?
        if is_train:
            coord_preds_fine, _, query_point_feat = finetracker(
                xys=patch_xyz, fmaps=patch_feat_reshape, iters=3, is_train=is_train
            )
        else:
            coord_preds_fine, _, query_point_feat = finetracker(
                xys=patch_xyz, fmaps=patch_feat_reshape, iters=6, is_train=is_train
            )
            # coord_preds_fine, _, query_point_feat = finetracker(xys = patch_xyz, fmaps = patch_feat_reshape, iters=3, is_train = is_train)

        fine_prediction = coord_preds_fine[-1].clone()

        # from (relative to the patch top left) to (relative to the image top left)
        # therefore, we need to add the coordinates of the top left corners to the fine_level_topleft
        for idx in range(len(coord_preds_fine)):
            # several predictions
            fine_level = rearrange(coord_preds_fine[idx], "(b n) s u v -> b s n u v", b=B, n=N)
            fine_level = fine_level.squeeze(-2)
            fine_level = fine_level + adjusted_points_BSN
            coord_preds_fine[idx] = fine_level

        # now both selected_tracks_gt and coord_preds_fine are defined relative to the image top left
        # compute the loss
        if trajs_g is not None:
            refine_loss_tracker = sequence_loss(
                coord_preds_fine,
                selected_tracks_gt,
                selected_tracks_vis,
                selected_tracks_vis,
                0.8,
                vis_aware=False,
                ignore_first=True,
            )
            refine_loss_tracker = refine_loss_tracker * 3
            refine_loss = refine_loss + refine_loss_tracker
        else:
            refine_loss = 0

        # the refined tracks from finetracker
        refined_tracks = coord_preds_fine[-1]

        score = None

        if compute_score:
            # BxNxC_out, query_point_feat indicates the feat of points at the first frame (for all the tracks)
            # Therefore we don't have S dimension here
            query_point_feat = query_point_feat.reshape(B, N, C_out)
            # reshape to B x (S-1) x N x C_out, and then to (B*(S-1)*N) x C_out
            query_point_feat = query_point_feat.unsqueeze(1).expand(-1, S - 1, -1, -1)
            query_point_feat = query_point_feat.reshape(B * (S - 1) * N, C_out)

            # Radius and size for softmax refinement
            sradius = 2
            ssize = sradius * 2 + 1

            # get patch_feat_reshape back
            patch_feat_reshape = rearrange(patch_feat_reshape, "(b n) s c p q -> b s n c p q", b=B, n=N)

            # Again, we unfold the patches to smaller patches
            # so that we can then focus on smaller patches
            # patch_feat_unfold: B x S x N x C_out x (P - 2*sradius) x (P - 2*sradius) x ssize x ssize
            patch_feat_unfold = patch_feat_reshape.unfold(4, ssize, 1).unfold(5, ssize, 1)

            # As mentioned above, fine_prediction is the output of finetracker
            # which is the predicted point location relative to the patch (PxP) top left
            # here again, we first floor it,
            # and get the top left corner of the smaller patch (ssize x ssize) by fine_level_topleft_floor - sradius
            # fine_prediction: (B*N) x S x 1 x 2
            fine_prediction_floor = fine_prediction.floor().int()
            fine_level_topleft_floor = fine_prediction_floor - sradius
            # clamp to ensure the smaller patch is valid
            fine_level_topleft_floor = fine_level_topleft_floor.clamp(0, psize - ssize)
            fine_level_topleft_floor = fine_level_topleft_floor.squeeze(2)

            # prepare the batch indices and xy locations
            # BxSxN
            batch_indices_softmax = torch.arange(B)[:, None, None].expand(-1, S, N)
            # (B*S*N)
            batch_indices_softmax = batch_indices_softmax.reshape(-1).to(patch_feat_unfold.device)
            y_indices = fine_level_topleft_floor[..., 0].flatten()  # Flatten H indices
            x_indices = fine_level_topleft_floor[..., 1].flatten()  # Flatten W indices

            # note again x_indices cooresponds to [..., 1] and y_indices cooresponds to [..., 0]
            reference_frame_feat = patch_feat_unfold.reshape(
                B * S * N, C_out, psize - sradius * 2, psize - sradius * 2, ssize, ssize
            )[batch_indices_softmax, :, x_indices, y_indices]
            reference_frame_feat = reference_frame_feat.reshape(B, S, N, C_out, ssize, ssize)
            # pick the frames other than the first one, so we have S-1 frames here
            reference_frame_feat = reference_frame_feat[:, 1:].reshape(B * (S - 1) * N, C_out, ssize * ssize)

            # compute similarity
            sim_matrix = torch.einsum("mc,mcr->mr", query_point_feat, reference_frame_feat)

            softmax_temp = 1.0 / C_out**0.5  # C_out=128/64

            ########################################################################################
            # if debug:
            #   sim_matrix[:,12] = 1e6
            ########################################################################################

            heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
            # 2D heatmaps
            heatmap = heatmap.reshape(B * (S - 1) * N, ssize, ssize)  # B x ssize x ssize

            if False:
                # Dropped
                # # # # # # # # # # # # # # # # # #
                # heatmap[:, sradius, sradius] = 1
                # compute the 2D expectations of 2D heatmaps
                # they are relative to the top left of (ssize x ssize) patches
                # hence their values are within [0, ssize-1]
                coords_softmax = dsnt.spatial_expectation2d(heatmap[None], False)[0]
                coords_softmax = coords_softmax.reshape(B, S - 1, N, 2)

                # we need to get back to the image top left corner
                # so we add
                # xy of the top left corner of smaller patches (fine_level_topleft_floor)
                # xy of the top left corner of patches (adjusted_points_BSN)
                # xy of the 2D expectations by softmax

                # fine_level_topleft_floor (B*N, S, 2)
                fine_level_topleft_floor = fine_level_topleft_floor.reshape(B, N, S, 2)
                fine_level_topleft_floor = fine_level_topleft_floor.permute(0, 2, 1, 3)
                refined_tracks_softmax = fine_level_topleft_floor[:, 1:] + coords_softmax
                refined_tracks_softmax = refined_tracks_softmax + adjusted_points_BSN[:, 1:]

                # fine_level_topleft_floor
                # Bx(S-1) to BxS, use the query points to pad the shape
                refined_tracks_softmax = torch.cat([selected_tracks[:, 0:1], refined_tracks_softmax], dim=1)

                refined_tracks = refined_tracks_softmax

            coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
            grid_normalized = create_meshgrid(ssize, ssize, normalized_coordinates=True, device=heatmap.device).reshape(
                1, -1, 2
            )  # [1, ssize x ssize, 2]

            # (B*S*N, ssize, ssize)
            var = (
                torch.sum(grid_normalized**2 * heatmap.view(-1, ssize * ssize, 1), dim=1) - coords_normalized**2
            )  # [(BxS-1xN), 2]
            std = torch.sum(
                torch.sqrt(torch.clamp(var, min=1e-10)), -1
            )  # [(BxS-1xN)]  clamp needed for numerical stability

            score = std.reshape(B, S - 1, N)

    if cfg.softmax_refine:
        if query_point_feat is None:
            # if we do not get query_point_feat from finetracker, compute it here
            query_frame_feat = patch_feat_reshape[:, 0].reshape(B * N, C_out, psize, psize)
            relative_point_loc = track_frac[:, 0] + pradius
            relative_point_loc = relative_point_loc / (psize - 1)
            relative_point_loc = 2 * relative_point_loc - 1
            relative_point_loc = relative_point_loc.reshape(B * N, 1, 1, 2)
            query_point_feat = F.grid_sample(
                query_frame_feat, relative_point_loc, mode="bilinear", align_corners=True, padding_mode="zeros"
            )

        # BxNxC_out, query_point_feat indicates the feat of points at the first frame (for all the tracks)
        # Therefore we don't have S dimension here
        query_point_feat = query_point_feat.reshape(B, N, C_out)
        # reshape to B x (S-1) x N x C_out, and then to (B*(S-1)*N) x C_out
        query_point_feat = query_point_feat.unsqueeze(1).expand(-1, S - 1, -1, -1)
        query_point_feat = query_point_feat.reshape(B * (S - 1) * N, C_out)

        # Radius and size for softmax refinement
        sradius = 2
        ssize = sradius * 2 + 1

        # Again, we unfold the patches to smaller patches
        # so that we can then focus on smaller patches
        # patch_feat_unfold: B x S x N x C_out x (P - 2*sradius) x (P - 2*sradius) x ssize x ssize
        patch_feat_unfold = patch_feat_reshape.unfold(4, ssize, 1).unfold(5, ssize, 1)

        if fine_level_topleft is None:
            import pdb

            pdb.set_trace()

        # As mentioned above, fine_level_topleft is the output of finetracker
        # which is the predicted point location relative to the patch (PxP) top left
        # here again, we first floor it,
        # and get the top left corner of the smaller patch (ssize x ssize) by fine_level_topleft_floor - sradius
        fine_level_topleft_floor = fine_level_topleft.floor().int()
        fine_level_topleft_floor = fine_level_topleft_floor - sradius
        # clamp to ensure the smaller patch is valid
        fine_level_topleft_floor = fine_level_topleft_floor.clamp(0, psize - ssize)
        fine_level_topleft_floor = fine_level_topleft_floor.squeeze(2)

        # prepare the batch indices and xy locations
        batch_indices_softmax = torch.arange(B)[:, None, None].expand(-1, S, N)
        batch_indices_softmax = batch_indices_softmax.reshape(-1).to(patch_feat_unfold.device)
        y_indices = fine_level_topleft_floor[..., 0].flatten()  # Flatten H indices
        x_indices = fine_level_topleft_floor[..., 1].flatten()  # Flatten W indices

        # note again x_indices cooresponds to [..., 1] and y_indices cooresponds to [..., 0]
        reference_frame_feat = patch_feat_unfold.reshape(
            B * S * N, C_out, psize - sradius * 2, psize - sradius * 2, ssize, ssize
        )[batch_indices_softmax, :, x_indices, y_indices]
        reference_frame_feat = reference_frame_feat.reshape(B, S, N, C_out, ssize, ssize)
        # pick the frames other than the first one, so we have S-1 frames here
        reference_frame_feat = reference_frame_feat[:, 1:].reshape(B * (S - 1) * N, C_out, ssize * ssize)

        # compute similarity
        sim_matrix = torch.einsum("mc,mcr->mr", query_point_feat, reference_frame_feat)

        softmax_temp = 1.0 / C_out**0.5  # C_out=128/64
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
        # 2D heatmaps
        heatmap = heatmap.reshape(B * (S - 1) * N, ssize, ssize)  # B x ssize x ssize

        # compute the 2D expectations of 2D heatmaps
        # they are relative to the top left of (ssize x ssize) patches
        # hence their values are within [0, ssize-1]
        coords_softmax = dsnt.spatial_expectation2d(heatmap[None], False)[0]
        coords_softmax = coords_softmax.reshape(B, S - 1, N, 2)

        # we need to get back to the image top left corner
        # so we add
        # xy of the top left corner of smaller patches (fine_level_topleft_floor)
        # xy of the top left corner of patches (adjusted_points_BSN)
        # xy of the 2D expectations by softmax
        refined_tracks_softmax = fine_level_topleft_floor.reshape(B, S, N, 2)[:, 1:] + coords_softmax
        refined_tracks_softmax = refined_tracks_softmax + adjusted_points_BSN[:, 1:]

        refine_loss_softmax = (selected_tracks_gt[:, 1:] - refined_tracks_softmax).abs()

        # do not consider those outsiders of large patches, but maybe consider smaller patches?
        lossmask = selected_tracks_vis[:, 1:].unsqueeze(-1).expand_as(refine_loss_softmax).bool()

        if lossmask.any():
            refine_loss_softmax = refine_loss_softmax[lossmask]
        else:
            refine_loss_softmax = refine_loss_softmax * selected_tracks_vis[:, 1:].unsqueeze(-1)

        # loss for softmax refinement
        refine_loss_softmax = refine_loss_softmax.mean() * 0.5

        # Bx(S-1) to BxS, use the query points to pad the shape
        refined_tracks_softmax = torch.cat([selected_tracks[:, 0:1], refined_tracks_softmax], dim=1)

        refined_tracks = refined_tracks_softmax

        refine_loss = refine_loss + refine_loss_softmax

    if is_train:
        full_output = coarse_pred.clone()
        full_output[B_indices, S_indices, track_idx_expand, :] = refined_tracks
    else:
        full_output = refined_tracks

    if score is not None:
        score = torch.cat([torch.ones_like(score[:, 0:1]), score], dim=1)

    # full_output[:,0] - query_points
    # force the first frame to be query points
    full_output[:, 0] = query_points

    if compute_score:
        return full_output.detach(), refine_loss, refined_tracks, score

    return full_output.detach(), refine_loss, refined_tracks


def refine_track_softmax(rgbs, fmaps, fnet, coord_preds, trajs_g, vis_g, track_ratio=0.2, is_train=False, cfg=None):
    pradius = cfg.pradius
    psize = pradius * 2 + 1

    DEBUG = False

    if DEBUG:
        coarse_pred = trajs_g.detach().clone()
    else:
        coarse_pred = coord_preds[-1].detach().clone()

    patch_trajs_g = (trajs_g - coarse_pred) + pradius
    inside_flag = (patch_trajs_g >= 0) & (patch_trajs_g <= (psize - 1))
    inside_flag = inside_flag.sum(dim=-1) == 2
    patch_vis_g = torch.logical_and(inside_flag, vis_g)

    if is_train:
        # Prepare the valid patches
        B, S, N, _ = coarse_pred.shape

        valid_patches_count = patch_vis_g.sum(dim=1)

        M = int(track_ratio * N)
        _, track_idx = valid_patches_count.topk(M, dim=1)
        track_idx_expand = track_idx.unsqueeze(1).expand(-1, S, -1)

        B_indices = torch.arange(B).view(B, 1, 1).expand(-1, S, M)
        S_indices = torch.arange(S).view(1, S, 1).expand(B, -1, M)

        selected_tracks = coarse_pred[B_indices, S_indices, track_idx_expand, :]
        selected_tracks_gt = trajs_g[B_indices, S_indices, track_idx_expand, :]
        selected_tracks_vis = patch_vis_g[B_indices, S_indices, track_idx_expand]
    else:
        selected_tracks = coarse_pred
        selected_tracks_gt = trajs_g
        selected_tracks_vis = patch_vis_g

    # Now, extract patches
    B, S, N, _ = selected_tracks.shape
    _, _, _, H, W = rgbs.shape
    _, _, C, H8, W8 = fmaps.shape

    with torch.no_grad():
        if cfg.refine_with_f:
            fmaps_reshape = F.interpolate(fmaps.reshape(B * S, C, H8, W8), (H, W), mode="bilinear", align_corners=True)
            rgbs_reshape = rgbs.reshape(B * S, 3, H, W)
            content_to_extract = torch.cat([rgbs_reshape, fmaps_reshape], dim=1)
        else:
            content_to_extract = rgbs_reshape = rgbs.reshape(B * S, 3, H, W)

    C_in = content_to_extract.shape[1]

    content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1)

    if DEBUG:
        selected_tracks = selected_tracks.int()

    # point positions are float here
    track_int = selected_tracks.floor().int()
    track_frac = selected_tracks - track_int

    # # BxSxNx2 -> BSxNx2
    # adjusted_points = track_int.reshape(B*S, N, 2)
    # adjusted_points = adjusted_points - pradius
    # # VERY IMPORTANT: WE ASSUME H=W here
    # adjusted_points = adjusted_points.clamp(0, H-psize)
    # # indexing:
    # batch_indices = torch.arange(B*S)[:, None].expand(-1, N).to(content_to_extract.device)

    adjusted_points = track_int - pradius
    # VERY IMPORTANT: WE ASSUME H=W here
    adjusted_points = adjusted_points.clamp(0, H - psize)

    adjusted_points_BSN = adjusted_points.clone()
    adjusted_points = adjusted_points.reshape(B * S, N, 2)
    batch_indices = torch.arange(B * S)[:, None].expand(-1, N).to(content_to_extract.device)

    # get the selected patches
    selected_patches = content_to_extract[batch_indices, :, adjusted_points[..., 1], adjusted_points[..., 0]]

    ### Feed selected_patches to network
    # BSN x C_in x P xP
    patch_feat = fnet(selected_patches.reshape(B * S * N, C_in, psize, psize))
    C_out = patch_feat.shape[1]

    # get query frame feat
    patch_feat_reshape = patch_feat.reshape(B, S, N, C_out, psize, psize)

    # if DEBUG:
    #     patch_feat_reshape[0,0,0,:,0,0] = patch_feat_reshape[0,0,0,:,0,0] * 100
    #     patch_feat_reshape[0,1,0,:,9,0] = patch_feat_reshape[0,0,0,:,0,0]

    query_frame_feat = patch_feat_reshape[:, 0].reshape(B * N, C_out, psize, psize)

    reference_frame_feat = patch_feat_reshape[:, 1:].reshape(B * (S - 1) * N, C_out, psize, psize)
    reference_frame_feat = reference_frame_feat.reshape(B * (S - 1) * N, C_out, psize * psize)

    # sample the point feat at the first frame
    relative_point_loc = track_frac[:, 0] + pradius
    relative_point_loc = relative_point_loc / (psize - 1)
    relative_point_loc = 2 * relative_point_loc - 1
    relative_point_loc = relative_point_loc.reshape(B * N, 1, 1, 2)

    # Really careful with this: adjusted_points[..., 1], adjusted_points[..., 0]
    # yx
    # Finally we get the query point feat
    query_point_feat = F.grid_sample(
        query_frame_feat, relative_point_loc, mode="bilinear", align_corners=True, padding_mode="zeros"
    )

    query_point_feat = query_point_feat.reshape(B, N, C_out)
    query_point_feat = query_point_feat.unsqueeze(1).expand(-1, S - 1, -1, -1)
    query_point_feat = query_point_feat.reshape(B * (S - 1) * N, C_out)

    sim_matrix = torch.einsum("mc,mcr->mr", query_point_feat, reference_frame_feat)
    softmax_temp = 1.0 / C_out**0.5  # C_out=128
    heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
    heatmap = heatmap.reshape(B * (S - 1) * N, psize, psize)  # BxPxP

    coords_softmax = dsnt.spatial_expectation2d(heatmap[None], False)[0]
    coords_softmax = coords_softmax.reshape(B, S - 1, N, 2)

    if cfg.reverse_soft:
        coords_softmax = coords_softmax[..., [1, 0]]
    # (B*S, N, 2)
    # adjusted_points
    # haha = track_int[:,1:] - pradius + coords_softmax
    refined_tracks = adjusted_points_BSN[:, 1:] + coords_softmax

    refine_loss = (selected_tracks_gt[:, 1:] - refined_tracks).abs()

    lossmask = selected_tracks_vis[:, 1:].unsqueeze(-1).expand_as(refine_loss).bool()

    if lossmask.any():
        refine_loss = refine_loss[lossmask]
    else:
        refine_loss = refine_loss * selected_tracks_vis[:, 1:].unsqueeze(-1)

    refine_loss = refine_loss.mean()

    if is_train:
        refined_tracks = torch.cat([selected_tracks[:, 0:1], refined_tracks], dim=1)
        full_output = coarse_pred.clone()
        # get it back
        full_output[B_indices, S_indices, track_idx_expand, :] = refined_tracks
    else:
        refined_tracks = torch.cat([selected_tracks[:, 0:1], refined_tracks], dim=1)
        full_output = refined_tracks

    return full_output.detach(), refine_loss, refined_tracks

    # patch_vis_g
    # selected_tracks_gt

    # refined_tracks

    # relative_point_loc = relative_point_loc.reshape(B*S, N, 2)

    #

    # relative_point_loc = relative_point_loc / (psize - 1)
    #
    # haha = F.grid_sample(query_feat, relative_point_loc, mode='bilinear', align_corners=False, padding_mode='zeros')

    # haha - tmp.reshape()
    # relative_point_loc
    # tmp = extract_glimpse_forloop(content_to_extract, (psize, psize), selected_tracks.reshape(B*S, N, 2))
    # tmp_to_check = tmp.reshape(B, S, N, C_in, psize, psize)
    # tmp_to_check = tmp_to_check[:, 0]
    # tmp_to_check = tmp_to_check[:,:,:,pradius,pradius]
    # haha.reshape(B, N, C_in)

    # track_int

    # visualzing
    visualize = False

    if visualize:
        from util.track_visual import Visualizer

        save_dir = "/data/home/jianyuan/vggsfm/VGGSfM/tmp/debug_patch"
        track_visualizer = Visualizer(save_dir=save_dir, fps=2, show_first_frame=0, linewidth=25, mode="rainbow")
        image_subset = rgbs[0:1]

        gt_tracks_subset = track_int[0:1][:, :, 1:2]
        gt_tracks_vis_subset = vis_g[0:1][:, :, 1:2]
        from pytorch3d.implicitron.tools import model_io, vis_utils

        res_video_gt = track_visualizer.visualize(
            255 * image_subset, gt_tracks_subset, gt_tracks_vis_subset, save_video=False
        )
        env_name = f"debug_patch"
        viz = vis_utils.get_visdom_connection(
            server=f"http://{cfg.viz_ip}", port=int(os.environ.get("VISDOM_PORT", 10088))
        )
        viz.images((res_video_gt[0] / 255).clamp(0, 1), env=env_name, win="tmp")

        # /data/home/jianyuan/vggsfm/VGGSfM/tmp
        image_patches = selected_patches_reshape[0, :, 1, :3, :, :]
        save_images(image_patches.cpu(), "/data/home/jianyuan/vggsfm/VGGSfM/tmp/visual_patch")

    m = 1
    return None
