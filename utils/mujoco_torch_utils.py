# MuJoCo utilities to replace isaacgym.torch_utils

import torch
import numpy as np
from typing import Tuple


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.
    
    Args:
        a: First quaternion (N, 4) or (4,) - (x, y, z, w)
        b: Second quaternion (N, 4) or (4,) - (x, y, z, w)
    
    Returns:
        Product quaternion (N, 4) or (4,)
    """
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    return quat


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate.
    
    Args:
        q: Quaternion (N, 4) or (4,) - (x, y, z, w)
    
    Returns:
        Conjugate quaternion (N, 4) or (4,)
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1).view(shape)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion.
    
    Args:
        q: Quaternion (N, 4) - (x, y, z, w)
        v: Vector (N, 3)
    
    Returns:
        Rotated vector (N, 3)
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    
    # v' = q * v * q^-1
    # Efficiently compute using: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    qvec = q[:, :3]
    qw = q[:, 3:4]
    
    cross1 = torch.cross(qvec, v, dim=-1)
    cross2 = torch.cross(qvec, cross1 + qw * v, dim=-1)
    
    return v + 2.0 * cross2


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.
    
    Args:
        q: Quaternion (N, 4) - (x, y, z, w)
        v: Vector (N, 3)
    
    Returns:
        Rotated vector (N, 3)
    """
    return quat_rotate(quat_conjugate(q), v)


def get_euler_xyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        q: Quaternion (N, 4) - (x, y, z, w)
    
    Returns:
        Tuple of (roll, pitch, yaw) tensors, each (N,)
    """
    q = q.reshape(-1, 4)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * np.pi / 2,
        torch.asin(sinp)
    )
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def torch_rand_float(lower: float, upper: float, shape: tuple, device: str) -> torch.Tensor:
    """Generate random floats uniformly distributed between lower and upper.
    
    Args:
        lower: Lower bound
        upper: Upper bound
        shape: Shape of output tensor
        device: Device to create tensor on
    
    Returns:
        Random tensor
    """
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def tensor_clamp(t: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clamp tensor values.
    
    Args:
        t: Input tensor
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clamped tensor
    """
    return torch.clamp(t, min=min_val, max=max_val)


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize a vector.
    
    Args:
        x: Input tensor (N, D)
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized tensor
    """
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to quaternion.
    
    Args:
        roll: Roll angle (N,)
        pitch: Pitch angle (N,)
        yaw: Yaw angle (N,)
    
    Returns:
        Quaternion (N, 4) - (x, y, z, w)
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([x, y, z, w], dim=-1)


def quat_identity(n: int, device: str) -> torch.Tensor:
    """Create identity quaternions.
    
    Args:
        n: Number of quaternions
        device: Device to create tensor on
    
    Returns:
        Identity quaternions (n, 4) - (0, 0, 0, 1)
    """
    quat = torch.zeros(n, 4, device=device)
    quat[:, 3] = 1.0
    return quat


def copysign(mag: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Copy sign from one tensor to another.
    
    Args:
        mag: Magnitude tensor
        sign: Sign tensor
    
    Returns:
        Tensor with magnitude of mag and sign of sign
    """
    return torch.abs(mag) * torch.sign(sign)