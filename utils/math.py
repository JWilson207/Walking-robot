# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
from torch import Tensor
import numpy as np
from typing import Tuple


def quat_apply(quat: Tensor, vec: Tensor) -> Tensor:
    """Apply quaternion rotation to a vector.
    
    Args:
        quat: Quaternion (N, 4) - (x, y, z, w)
        vec: Vector (N, 3)
    
    Returns:
        Rotated vector (N, 3)
    """
    # Ensure correct shapes
    quat_shape = quat.shape
    vec_shape = vec.shape
    
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    
    # Extract quaternion components
    qvec = quat[:, :3]
    qw = quat[:, 3:4]
    
    # Compute: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    cross1 = torch.cross(qvec, vec, dim=-1)
    cross2 = torch.cross(qvec, cross1 + qw * vec, dim=-1)
    
    result = vec + 2.0 * cross2
    
    # Restore original batch dimensions if needed
    if len(vec_shape) > 2:
        result = result.reshape(vec_shape)
    
    return result


def normalize(x: Tensor, eps: float = 1e-9) -> Tensor:
    """Normalize a vector or batch of vectors.
    
    Args:
        x: Input tensor (N, D) or (..., N, D)
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized tensor
    """
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def quat_apply_yaw(quat: Tensor, vec: Tensor) -> Tensor:
    """Apply only the yaw component of a quaternion rotation to a vector.
    
    This zeros out the roll and pitch components of the quaternion before
    applying the rotation, keeping only the yaw (z-axis) rotation.
    
    Args:
        quat: Quaternion (N, 4) - (x, y, z, w)
        vec: Vector (N, 3)
    
    Returns:
        Rotated vector (N, 3)
    """
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.  # Zero out x and y components (roll and pitch)
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def wrap_to_pi(angles: Tensor) -> Tensor:
    """Wrap angles to the range [-pi, pi].
    
    Args:
        angles: Input angles (any shape)
    
    Returns:
        Wrapped angles in [-pi, pi]
    """
    angles = angles % (2 * np.pi)
    angles = angles - 2 * np.pi * (angles > np.pi)
    return angles


def torch_rand_sqrt_float(lower: float, upper: float, shape: Tuple[int, ...], device: str) -> Tensor:
    """Generate random floats with square root distribution.
    
    This generates random values that follow a square root distribution,
    which can be useful for sampling that favors values near the bounds.
    
    Args:
        lower: Lower bound
        upper: Upper bound
        shape: Shape of output tensor
        device: Device to create tensor on
    
    Returns:
        Random tensor with square root distribution
    """
    # Generate uniform random values in [-1, 1]
    r = 2 * torch.rand(*shape, device=device) - 1
    
    # Apply square root transformation
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    
    # Map to [0, 1]
    r = (r + 1.) / 2.
    
    # Scale to [lower, upper]
    return (upper - lower) * r + lower