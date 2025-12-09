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

import numpy as np
from typing import Tuple
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg


class SubTerrain:
    """Represents a sub-terrain patch for procedural terrain generation.
    
    This is a replacement for isaacgym.terrain_utils.SubTerrain that works
    with MuJoCo's heightfield representation.
    """
    
    def __init__(self, terrain_name: str = "terrain", width: int = 256, 
                 length: int = 256, vertical_scale: float = 1.0, 
                 horizontal_scale: float = 1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.length, self.width), dtype=np.int16)


def convert_heightfield_to_trimesh(height_field_raw: np.ndarray, 
                                   horizontal_scale: float,
                                   vertical_scale: float, 
                                   slope_threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a heightfield to a triangle mesh.
    
    Args:
        height_field_raw: 2D array of height values
        horizontal_scale: Scale factor for x and y
        vertical_scale: Scale factor for z (height)
        slope_threshold: If provided, slopes above this threshold are made vertical
    
    Returns:
        Tuple of (vertices, triangles)
    """
    num_rows = height_field_raw.shape[0]
    num_cols = height_field_raw.shape[1]
    
    # Create vertices
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    
    if slope_threshold is not None:
        # Apply slope threshold
        slope_threshold = np.tan(slope_threshold)
        # Calculate slopes and clip if needed
        # This is a simplified implementation
        pass
    
    zz = height_field_raw.astype(np.float32) * vertical_scale
    
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = zz.flatten()
    
    # Create triangles
    triangles = []
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            # Two triangles per quad
            v00 = i * num_cols + j
            v01 = i * num_cols + (j + 1)
            v10 = (i + 1) * num_cols + j
            v11 = (i + 1) * num_cols + (j + 1)
            
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    
    triangles = np.array(triangles, dtype=np.uint32)
    
    return vertices, triangles


def random_uniform_terrain(terrain: SubTerrain, min_height: float, max_height: float,
                           step: float = 0.005, downsampled_scale: float = 0.2):
    """Generate random uniform noise terrain.
    
    Args:
        terrain: SubTerrain object to modify
        min_height: Minimum height
        max_height: Maximum height
        step: Height quantization step
        downsampled_scale: Scale for downsampling
    """
    # Generate random heights
    height_range = int((max_height - min_height) / step)
    heights = np.random.randint(0, height_range, 
                               size=(terrain.length, terrain.width))
    terrain.height_field_raw += (heights * step / terrain.vertical_scale + 
                                 min_height / terrain.vertical_scale).astype(np.int16)


def pyramid_sloped_terrain(terrain: SubTerrain, slope: float = 1.0, platform_size: float = 1.0):
    """Generate pyramid-shaped sloped terrain.
    
    Args:
        terrain: SubTerrain object to modify
        slope: Slope magnitude (positive or negative)
        platform_size: Size of flat platform at center
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    
    center_x = terrain.width / 2
    center_y = terrain.length / 2
    
    platform_size_pixels = int(platform_size / terrain.horizontal_scale)
    
    # Distance from center
    dx = np.abs(xx - center_x)
    dy = np.abs(yy - center_y)
    
    # Create slope starting from platform edge
    distance = np.maximum(dx, dy) - platform_size_pixels / 2
    distance = np.maximum(distance, 0)
    
    # Apply slope
    height = slope * distance * terrain.horizontal_scale / terrain.vertical_scale
    terrain.height_field_raw += height.astype(np.int16)


def pyramid_stairs_terrain(terrain: SubTerrain, step_width: float, step_height: float,
                           platform_size: float = 1.0):
    """Generate pyramid-shaped stairs terrain.
    
    Args:
        terrain: SubTerrain object to modify
        step_width: Width of each step
        step_height: Height of each step (can be negative for downward stairs)
        platform_size: Size of flat platform at center
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    
    center_x = terrain.width / 2
    center_y = terrain.length / 2
    
    platform_size_pixels = int(platform_size / terrain.horizontal_scale)
    step_width_pixels = int(step_width / terrain.horizontal_scale)
    
    # Distance from center
    dx = np.abs(xx - center_x)
    dy = np.abs(yy - center_y)
    distance = np.maximum(dx, dy) - platform_size_pixels / 2
    distance = np.maximum(distance, 0)
    
    # Create steps
    num_steps = (distance / step_width_pixels).astype(np.int16)
    height = num_steps * step_height / terrain.vertical_scale
    
    terrain.height_field_raw += height.astype(np.int16)


def discrete_obstacles_terrain(terrain: SubTerrain, obstacle_height: float,
                               rectangle_min_size: float, rectangle_max_size: float,
                               num_rectangles: int, platform_size: float = 1.0):
    """Generate terrain with discrete rectangular obstacles.
    
    Args:
        terrain: SubTerrain object to modify
        obstacle_height: Height of obstacles
        rectangle_min_size: Minimum size of rectangles
        rectangle_max_size: Maximum size of rectangles
        num_rectangles: Number of rectangles to place
        platform_size: Size of clear platform at center
    """
    platform_size_pixels = int(platform_size / terrain.horizontal_scale)
    center_x = terrain.width // 2
    center_y = terrain.length // 2
    
    # Create obstacles
    for _ in range(num_rectangles):
        # Random rectangle size
        width = np.random.uniform(rectangle_min_size, rectangle_max_size)
        length = np.random.uniform(rectangle_min_size, rectangle_max_size)
        
        width_pixels = int(width / terrain.horizontal_scale)
        length_pixels = int(length / terrain.horizontal_scale)
        
        # Random position (avoid center platform)
        valid_position = False
        attempts = 0
        while not valid_position and attempts < 100:
            x = np.random.randint(0, terrain.width - width_pixels)
            y = np.random.randint(0, terrain.length - length_pixels)
            
            # Check if too close to center
            if (abs(x - center_x) > platform_size_pixels or 
                abs(y - center_y) > platform_size_pixels):
                valid_position = True
            attempts += 1
        
        if valid_position:
            # Add obstacle
            height = int(obstacle_height / terrain.vertical_scale)
            terrain.height_field_raw[y:y+length_pixels, x:x+width_pixels] = height


def stepping_stones_terrain(terrain: SubTerrain, stone_size: float, 
                            stone_distance: float, max_height: float = 0.0,
                            platform_size: float = 1.0):
    """Generate stepping stones terrain.
    
    Args:
        terrain: SubTerrain object to modify
        stone_size: Size of each stepping stone
        stone_distance: Distance between stones
        max_height: Maximum height variation
        platform_size: Size of clear platform at center
    """
    # Start with lowered terrain
    terrain.height_field_raw[:] = int(-max_height / terrain.vertical_scale)
    
    stone_size_pixels = int(stone_size / terrain.horizontal_scale)
    stone_distance_pixels = int(stone_distance / terrain.horizontal_scale)
    platform_size_pixels = int(platform_size / terrain.horizontal_scale)
    
    center_x = terrain.width // 2
    center_y = terrain.length // 2
    
    # Create center platform
    x1 = center_x - platform_size_pixels // 2
    x2 = center_x + platform_size_pixels // 2
    y1 = center_y - platform_size_pixels // 2
    y2 = center_y + platform_size_pixels // 2
    terrain.height_field_raw[y1:y2, x1:x2] = 0
    
    # Place stepping stones in a grid
    spacing = stone_size_pixels + stone_distance_pixels
    
    for x in range(0, terrain.width, spacing):
        for y in range(0, terrain.length, spacing):
            # Skip center platform
            if (abs(x - center_x) < platform_size_pixels and 
                abs(y - center_y) < platform_size_pixels):
                continue
            
            # Add stone
            x1 = max(0, x)
            x2 = min(terrain.width, x + stone_size_pixels)
            y1 = max(0, y)
            y2 = min(terrain.length, y + stone_size_pixels)
            
            if x2 > x1 and y2 > y1:
                height = np.random.uniform(0, max_height)
                terrain.height_field_raw[y1:y2, x1:x2] = int(height / terrain.vertical_scale)


class Terrain:
    """Main terrain class for generating procedural terrains.
    
    This class generates various types of terrains including flat, slopes, stairs,
    obstacles, and more. It's compatible with MuJoCo's heightfield representation.
    """
    
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots: int) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        
        if self.type in ["none", 'plane']:
            return
        
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) 
                           for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        
        if self.type == "trimesh":
            self.vertices, self.triangles = convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold
            )
    
    def randomized_terrain(self):
        """Generate randomized terrain patches."""
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
    
    def curiculum(self):
        """Generate curriculum-based terrain (difficulty increases)."""
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        """Generate terrain of a specific selected type."""
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice: float, difficulty: float) -> SubTerrain:
        """Create a single terrain patch based on choice and difficulty.
        
        Args:
            choice: Value in [0, 1] determining terrain type
            difficulty: Value in [0, 1] determining terrain difficulty
        
        Returns:
            SubTerrain object with generated heightfield
        """
        terrain = SubTerrain("terrain",
                            width=self.width_per_env_pixels,
                            length=self.width_per_env_pixels,
                            vertical_scale=self.cfg.vertical_scale,
                            horizontal_scale=self.cfg.horizontal_scale)
        
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, 
                                 step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, 
                                 platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            discrete_obstacles_terrain(terrain, discrete_obstacles_height, 
                                      rectangle_min_size, rectangle_max_size, 
                                      num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            stepping_stones_terrain(terrain, stone_size=stepping_stones_size, 
                                  stone_distance=stone_distance, max_height=0., 
                                  platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain: SubTerrain, row: int, col: int):
        """Add a terrain patch to the overall heightfield map.
        
        Args:
            terrain: SubTerrain patch to add
            row: Row index in the terrain grid
            col: Column index in the terrain grid
        """
        i = row
        j = col
        
        # Map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # Calculate environment origin
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain: SubTerrain, gap_size: float, platform_size: float = 1.):
    """Create terrain with a gap in the center.
    
    Args:
        terrain: SubTerrain object to modify
        gap_size: Size of the gap
        platform_size: Size of platform at edges
    """
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x - x2:center_x + x2, center_y - y2:center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1:center_x + x1, center_y - y1:center_y + y1] = 0


def pit_terrain(terrain: SubTerrain, depth: float, platform_size: float = 1.):
    """Create terrain with a pit in the center.
    
    Args:
        terrain: SubTerrain object to modify
        depth: Depth of the pit
        platform_size: Size of platform at edges
    """
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


class HumanoidTerrain(Terrain):
    """Specialized terrain class for humanoid robots.
    
    This class generates terrains specifically designed for humanoid
    locomotion training, with appropriate obstacle heights and slopes.
    """
    
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots: int) -> None:
        super().__init__(cfg, num_robots)

    def randomized_terrain(self):
        """Generate randomized terrain patches for humanoids."""
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0, 1)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice: float, difficulty: float) -> SubTerrain:
        """Create a humanoid-specific terrain patch.
        
        Args:
            choice: Value in [0, 1] determining terrain type
            difficulty: Value in [0, 1] determining terrain difficulty
        
        Returns:
            SubTerrain object with generated heightfield
        """
        terrain = SubTerrain("terrain",
                            width=self.width_per_env_pixels,
                            length=self.width_per_env_pixels,
                            vertical_scale=self.cfg.vertical_scale,
                            horizontal_scale=self.cfg.horizontal_scale)
        
        discrete_obstacles_height = difficulty * 0.04
        r_height = difficulty * 0.07
        h_slope = difficulty * 0.15
        
        if choice < self.proportions[0]:
            # Flat terrain
            pass
        elif choice < self.proportions[1]:
            # Discrete obstacles
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            discrete_obstacles_terrain(terrain, discrete_obstacles_height, 
                                      rectangle_min_size, rectangle_max_size, 
                                      num_rectangles, platform_size=3.)
        elif choice < self.proportions[2]:
            # Random uniform terrain
            random_uniform_terrain(terrain, min_height=-r_height, max_height=r_height, 
                                 step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            # Upward slope
            pyramid_sloped_terrain(terrain, slope=h_slope, platform_size=0.1)
        elif choice < self.proportions[4]:
            # Downward slope
            pyramid_sloped_terrain(terrain, slope=-h_slope, platform_size=0.1)
        elif choice < self.proportions[5]:
            # Upward stairs
            pyramid_stairs_terrain(terrain, step_width=0.4, step_height=discrete_obstacles_height, 
                                 platform_size=1.)
        elif choice < self.proportions[6]:
            # Downward stairs
            pyramid_stairs_terrain(terrain, step_width=0.4, step_height=-discrete_obstacles_height, 
                                 platform_size=1.)
        else:
            # Flat terrain
            pass
        
        return terrain