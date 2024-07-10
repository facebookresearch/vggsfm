import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from typing import Union, Tuple


def extract_glimpse_forloop(
    tensor: torch.Tensor, size: Tuple[int, int], offsets, mode="bilinear", padding_mode="zeros", debug=False, orib=None
):
    B, C, W, H = tensor.shape
    h, w = size
    xs = torch.arange(0, w, dtype=tensor.dtype, device=tensor.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=tensor.dtype, device=tensor.device) - (h - 1) / 2.0
    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2

    B, N, _ = offsets.shape

    # Pre-allocate a tensor to hold the results
    sampled_tensor = torch.empty((B, N, C, h, w), dtype=tensor.dtype, device=tensor.device)

    for i in range(N):
        current_offset = offsets[:, i, :]
        current_offset = current_offset.reshape((B, 1, 1, 2))
        offsets_grid = current_offset + grid

        # BxPxPx2
        # W=512

        # must be careful that W==H here
        offsets_grid = offsets_grid / (W - 1)
        offsets_grid = 2 * offsets_grid - 1
        # sampled_tensor[:, i] = torch.nn.functional.grid_sample(tensor, offsets_grid, mode=mode, align_corners=False, padding_mode=padding_mode).squeeze(3).squeeze(3)
        sampled_tensor[:, i] = (
            torch.nn.functional.grid_sample(
                tensor, offsets_grid, mode=mode, align_corners=False, padding_mode=padding_mode
            )
            .squeeze(3)
            .squeeze(3)
        )

    return sampled_tensor


def extract_glimpse(
    tensor: torch.Tensor, size: Tuple[int, int], offsets, mode="bilinear", padding_mode="zeros", debug=False, orib=None
):
    # W, H = tensor.size(-1), tensor.size(-2)
    B, C, W, H = tensor.shape

    h, w = size
    xs = torch.arange(0, w, dtype=tensor.dtype, device=tensor.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=tensor.dtype, device=tensor.device) - (h - 1) / 2.0

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2
    grid = grid[None]

    B, N, _ = offsets.shape

    offsets = offsets.reshape((B * N), 1, 1, 2)
    offsets_grid = offsets + grid
    # offsets_grid = offsets[:, None, None, :] + grid[None, ...]

    # normalised grid  to [-1, 1]
    offsets_grid = (offsets_grid - offsets_grid.new_tensor([W / 2, H / 2])) / offsets_grid.new_tensor([W / 2, H / 2])

    # BxCxHxW -> Bx1xCxHxW
    tensor = tensor[:, None]

    # Bx1xCxHxW -> BxNxCxHxW
    tensor = tensor.expand(-1, N, -1, -1, -1)

    # BxNxCxHxW -> (B*N)xCxHxW
    tensor = tensor.reshape((B * N), C, W, H)

    # offsets_grid.shape: torch.Size([6144, 17, 17, 2])
    sampled = torch.nn.functional.grid_sample(
        tensor, offsets_grid, mode=mode, align_corners=False, padding_mode=padding_mode
    )

    # NOTE: I am not sure it should be h, w or w, h here
    # but okay for sqaures
    sampled = sampled.reshape(B, N, C, h, w)

    return sampled
