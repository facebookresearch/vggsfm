# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

# from cotracker.models.core.model_utils import reduce_masked_mean

EPS = 1e-9


def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean


def balanced_ce_loss(pred, gt, valid=None):
    # pred and gt are the same shape
    for a, b in zip(pred.size(), gt.size()):
        assert a == b  # some shape mismatch!

    if valid is not None:
        for a, b in zip(pred.size(), valid.size()):
            assert a == b  # some shape mismatch!
    else:
        valid = torch.ones_like(gt)

    pos = (gt > 0.95).to(gt.dtype)
    neg = (gt < 0.05).to(gt.dtype)

    label = pos * 2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

    pos_loss = reduce_masked_mean(loss, pos * valid)
    neg_loss = reduce_masked_mean(loss, neg * valid)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss, loss


def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).to(diff.dtype)
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def sequence_loss(
    flow_preds,
    flow_gt,
    vis,
    valids,
    gamma=0.8,
    vis_aware=False,
    huber=False,
    delta=10,
    vis_aware_w=0.1,
    ignore_first=False,
    max_thres=-1,
):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    if ignore_first:
        flow_gt = flow_gt[:, 1:]
        vis = vis[:, 1:]
        valids = valids[:, 1:]

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        if ignore_first:
            flow_pred = flow_pred[:, 1:]

        if huber:
            i_loss = huber_loss(flow_pred, flow_gt, delta)  # B, S, N, 2
        else:
            i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2

        i_loss = torch.nan_to_num(i_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if max_thres > 0:
            valids = torch.logical_and(valids, (i_loss < max_thres).any(dim=-1))

        i_loss = torch.mean(i_loss, dim=3)  # B, S, N

        if vis_aware:
            if vis_aware_w==0:
                valids = torch.logical_and(valids, vis)
                # i_loss = reduce_masked_mean(i_loss, vis, dim=3)
            else:
                i_loss = i_loss * (vis.to(i_loss.dtype) + vis_aware_w)


        flow_loss += i_weight * reduce_masked_mean(i_loss, valids)

    # clip_trackL
    flow_loss = flow_loss / n_predictions

    return flow_loss
