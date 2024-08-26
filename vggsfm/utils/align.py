# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def random_rotation_matrix(batch_size=1):
    """Generate random rotation matrices using QR decomposition for orthogonality."""
    random_matrix = torch.randn(batch_size, 3, 3)
    q, r = torch.qr(random_matrix)
    return q


def random_translation(batch_size=1):
    """Generate random translation vectors."""
    return torch.randn(batch_size, 3)


def random_scale(batch_size=1):
    """Generate random scale values."""

    return torch.rand(batch_size, 1) * 100  # Randomly sample from 0 to 100
    # return torch.rand(batch_size, 1) + 0.5  # Avoiding very small scales


def _align_camera_extrinsics_PT3D(
    cameras_src: torch.Tensor,  # Bx3x4 tensor representing [R | t]
    cameras_tgt: torch.Tensor,  # Bx3x4 tensor representing [R | t]
    estimate_scale: bool = True,
    eps: float = 1e-9,
):
    # ASSUME PYTORCH3D CONVENTION
    """
    Get the global rotation R_A with SVD of cov(RR^T):
        ```
        R_A R_i = R_i' for all i
        R_A [R_1 R_2 ... R_N] = [R_1' R_2' ... R_N']
        U, _, V = svd([R_1 R_2 ... R_N]^T [R_1' R_2' ... R_N'])
        R_A = (U V^T)^T
        ```
    """
    R_src = cameras_src[
        :, :, :3
    ]  # Extracting the rotation matrices from [R | t]
    R_tgt = cameras_tgt[
        :, :, :3
    ]  # Extracting the rotation matrices from [R | t]

    RRcov = torch.bmm(R_src, R_tgt.transpose(2, 1)).mean(0)
    U, _, V = torch.svd(RRcov)
    align_t_R = V @ U.t()

    """
    The translation + scale `T_A` and `s_A` is computed by finding
    a translation and scaling that aligns two tensors `A, B`
    defined as follows:
        ```
        T_A R_i + s_A T_i   = T_i'        ;  for all i    // · R_i^T
        s_A T_i R_i^T + T_A = T_i' R_i^T  ;  for all i
            ^^^^^^^^^         ^^^^^^^^^^
                A_i                B_i

        A_i := T_i R_i^T
        A = [A_1 A_2 ... A_N]
        B_i := T_i' R_i^T
        B = [B_1 B_2 ... B_N]
        ```
    The scale s_A can be retrieved by matching the correlations of
    the points sets A and B:
        ```
        s_A = (A-mean(A))*(B-mean(B)).sum() / ((A-mean(A))**2).sum()
        ```
    The translation `T_A` is then defined as:
        ```
        T_A = mean(B) - mean(A) * s_A
        ```
    """
    T_src = cameras_src[
        :, :, 3
    ]  # Extracting the translation vectors from [R | t]
    T_tgt = cameras_tgt[
        :, :, 3
    ]  # Extracting the translation vectors from [R | t]

    A = torch.bmm(R_src, T_src[:, :, None])[:, :, 0]
    B = torch.bmm(R_src, T_tgt[:, :, None])[:, :, 0]
    Amu = A.mean(0, keepdim=True)
    Bmu = B.mean(0, keepdim=True)

    if estimate_scale and A.shape[0] > 1:
        # get the scaling component by matching covariances
        # of centered A and centered B
        Ac = A - Amu
        Bc = B - Bmu
        align_t_s = (Ac * Bc).mean() / (Ac**2).mean().clamp(eps)
    else:
        # set the scale to identity
        align_t_s = 1.0

    # get the translation as the difference between the means of A and B
    align_t_T = Bmu - align_t_s * Amu

    return align_t_R, align_t_T, align_t_s


def align_and_transform_cameras_PT3D(
    cameras_src: torch.Tensor,  # Bx3x4 tensor representing [R | t]
    align_t_R: torch.Tensor,  # 1x3x3 rotation matrix
    align_t_T: torch.Tensor,  # 1x3 translation vector
    align_t_s: float,  # Scaling factor
) -> torch.Tensor:
    """
    # ASSUME PYTORCH3D CONVENTION
    Align and transform the source cameras with the given rotation, translation, and scale.

    Args:
        cameras_src (torch.Tensor): Bx3x4 tensor representing [R | t].
        align_t_R (torch.Tensor): 3x3 rotation matrix for alignment.
        align_t_T (torch.Tensor): 3x1 translation vector for alignment.
        align_t_s (float): Scaling factor for alignment.

    Returns:
        torch.Tensor: Aligned and transformed Bx3x4 tensor.
    """

    # Extract the rotation and translation parts from the source cameras
    R_src = cameras_src[:, :, :3]
    T_src = cameras_src[:, :, 3]

    # Apply the rotation alignment to the source rotations
    aligned_R = torch.bmm(align_t_R.expand(R_src.shape[0], 3, 3), R_src)

    # Apply the translation alignment to the source translations
    aligned_T = (
        torch.bmm(align_t_T[:, None].repeat(R_src.shape[0], 1, 1), R_src)[:, 0]
        + T_src * align_t_s
    )

    return aligned_R, aligned_T


def align_camera_extrinsics(
    cameras_src: torch.Tensor,  # Bx3x4 tensor representing [R | t]
    cameras_tgt: torch.Tensor,  # Bx3x4 tensor representing [R | t]
    estimate_scale: bool = True,
    eps: float = 1e-9,
):
    """
    Align the source camera extrinsics to the target camera extrinsics.
    NOTE Assume OPENCV convention

    Args:
        cameras_src (torch.Tensor): Bx3x4 tensor representing [R | t] for source cameras.
        cameras_tgt (torch.Tensor): Bx3x4 tensor representing [R | t] for target cameras.
        estimate_scale (bool, optional): Whether to estimate the scale factor. Default is True.
        eps (float, optional): Small value to avoid division by zero. Default is 1e-9.

    Returns:
        align_t_R (torch.Tensor): 1x3x3 rotation matrix for alignment.
        align_t_T (torch.Tensor): 1x3 translation vector for alignment.
        align_t_s (float): Scaling factor for alignment.
    """

    R_src = cameras_src[
        :, :, :3
    ]  # Extracting the rotation matrices from [R | t]
    R_tgt = cameras_tgt[
        :, :, :3
    ]  # Extracting the rotation matrices from [R | t]

    RRcov = torch.bmm(R_tgt.transpose(2, 1), R_src).mean(0)
    U, _, V = torch.svd(RRcov)
    align_t_R = V @ U.t()

    T_src = cameras_src[
        :, :, 3
    ]  # Extracting the translation vectors from [R | t]
    T_tgt = cameras_tgt[
        :, :, 3
    ]  # Extracting the translation vectors from [R | t]

    A = torch.bmm(T_src[:, None], R_src)[:, 0]
    B = torch.bmm(T_tgt[:, None], R_src)[:, 0]

    Amu = A.mean(0, keepdim=True)
    Bmu = B.mean(0, keepdim=True)

    if estimate_scale and A.shape[0] > 1:
        # get the scaling component by matching covariances
        # of centered A and centered B
        Ac = A - Amu
        Bc = B - Bmu
        align_t_s = (Ac * Bc).mean() / (Ac**2).mean().clamp(eps)
    else:
        # set the scale to identity
        align_t_s = 1.0

    # get the translation as the difference between the means of A and B
    align_t_T = Bmu - align_t_s * Amu

    align_t_R = align_t_R[None]
    return align_t_R, align_t_T, align_t_s


def apply_transformation(
    cameras_src: torch.Tensor,  # Bx3x4 tensor representing [R | t]
    align_t_R: torch.Tensor,  # 1x3x3 rotation matrix
    align_t_T: torch.Tensor,  # 1x3 translation vector
    align_t_s: float,  # Scaling factor
    return_extri: bool = True,
) -> torch.Tensor:
    """
    Align and transform the source cameras using the provided rotation, translation, and scaling factors.
    NOTE Assume OPENCV convention

    Args:
        cameras_src (torch.Tensor): Bx3x4 tensor representing [R | t] for source cameras.
        align_t_R (torch.Tensor): 1x3x3 rotation matrix for alignment.
        align_t_T (torch.Tensor): 1x3 translation vector for alignment.
        align_t_s (float): Scaling factor for alignment.

    Returns:
        aligned_R (torch.Tensor): Bx3x3 tensor representing the aligned rotation matrices.
        aligned_T (torch.Tensor): Bx3 tensor representing the aligned translation vectors.
    """

    R_src = cameras_src[:, :, :3]
    T_src = cameras_src[:, :, 3]

    aligned_R = torch.bmm(R_src, align_t_R.expand(R_src.shape[0], 3, 3))

    # Apply the translation alignment to the source translations
    # aligned_T = (
    #     torch.bmm(
    #         R_src,
    #         align_t_T[..., None].repeat(R_src.shape[0], 1, 1)
    #     )[..., 0] + T_src * align_t_s
    # )

    # Apply the translation alignment to the source translations
    align_t_T_expanded = align_t_T[..., None].repeat(R_src.shape[0], 1, 1)
    transformed_T = torch.bmm(R_src, align_t_T_expanded)[..., 0]
    aligned_T = transformed_T + T_src * align_t_s

    if return_extri:
        extri = torch.cat([aligned_R, aligned_T.unsqueeze(-1)], dim=-1)
        return extri

    return aligned_R, aligned_T


def test_align_camera_extrinsics(num_tests=10000):
    """Test _align_camera_extrinsics function multiple times with random data."""
    for test_idx in range(num_tests):
        # Randomly generate source cameras
        batch_size = 10
        R_src = random_rotation_matrix(batch_size=batch_size)
        T_src = random_translation(batch_size=batch_size)
        cameras_src = torch.cat([R_src, T_src.unsqueeze(-1)], dim=-1)

        # Generate random transformations
        R_transform = random_rotation_matrix(batch_size=1)
        T_transform = random_translation(batch_size=1)
        scale_transform = random_scale(batch_size=1)

        # Apply transformations to source cameras to create target cameras
        R_tgt, T_tgt = apply_transformation(
            cameras_src,
            R_transform,
            T_transform,
            scale_transform,
            return_extri=False,
        )
        cameras_tgt = torch.cat([R_tgt, T_tgt.unsqueeze(-1)], dim=-1)

        # Test the alignment function
        align_t_R, align_t_T, align_t_s = align_camera_extrinsics(
            cameras_src, cameras_tgt, estimate_scale=True
        )

        R_verify, T_verify = apply_transformation(
            cameras_src, align_t_R, align_t_T, align_t_s, return_extri=False
        )

        # Verify the results
        print(test_idx)
        if not torch.allclose(R_tgt, R_verify, atol=1e-3):
            import pdb

            pdb.set_trace()
        if not torch.allclose(T_tgt, T_verify, atol=1e-3):
            import pdb

            pdb.set_trace()


if __name__ == "__main__":
    test_align_camera_extrinsics()
