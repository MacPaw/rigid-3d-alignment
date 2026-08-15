"""Rotation conventions shared by training and evaluation.

The model emits rotations as a 6D vector holding the first two rows of a 3x3 matrix
(Zhou et al., "On the Continuity of Rotation Representations in Neural Networks").
Both the torch and numpy paths below must agree on that convention, so they live here
together rather than being reimplemented per script.
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def orthonormalize_6d_rotation(rot_6d: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt a batch of 6D rotations into (..., 3, 3) matrices, as rows."""
    v1 = rot_6d[..., :3]
    v2 = rot_6d[..., 3:]

    u1 = F.normalize(v1, dim=-1)

    # Project v2 onto u1 and subtract, so the second row is orthogonal to the first.
    projection = torch.sum(u1 * v2, dim=-1, keepdim=True)
    u2 = F.normalize(v2 - projection * u1, dim=-1)
    u3 = torch.cross(u1, u2, dim=-1)

    return torch.stack([u1, u2, u3], dim=-2)


def reconstruct_rotation(rot_6d) -> np.ndarray:
    """Numpy counterpart of `orthonormalize_6d_rotation` for a single 6D vector."""
    r1 = np.asarray(rot_6d[:3], dtype=np.float64)
    r2 = np.asarray(rot_6d[3:6], dtype=np.float64)
    r1 = r1 / (np.linalg.norm(r1) + 1e-12)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-12)
    r3 = np.cross(r1, r2)
    r3 = r3 / (np.linalg.norm(r3) + 1e-12)
    return np.stack([r1, r2, r3], axis=0)


def geodesic_degrees(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Geodesic angle in degrees between two batches of 6D rotations, shape (B, H, 6)."""
    pred_mat = orthonormalize_6d_rotation(pred)
    target_mat = orthonormalize_6d_rotation(target)

    # trace(pred @ target^T), without materializing the product.
    trace = torch.sum(pred_mat * target_mat, dim=(-2, -1))

    # Clamp before arccos to avoid NaNs from floating point drift outside [-1, 1].
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_theta) * (180.0 / torch.pi)


def geodesic_distance_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    """Geodesic angle in degrees between two 3x3 rotation matrices."""
    cos_val = np.clip((np.trace(r1.T @ r2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))
