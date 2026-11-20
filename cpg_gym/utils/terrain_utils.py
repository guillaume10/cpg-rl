# License: see [LICENSE, LICENSES/legged_gym_license]

import numpy as np

# Base terrain primitives are from NVIDIA Isaac Gym's terrain_utils and are
# imported (not redistributed) so their NVIDIA license is not re-licensed here:
from isaacgym.terrain_utils import (
    random_uniform_terrain, sloped_terrain, pyramid_sloped_terrain,
    discrete_obstacles_terrain, wave_terrain, stairs_terrain,
    pyramid_stairs_terrain, stepping_stones_terrain,
    convert_heightfield_to_trimesh, SubTerrain,
)

# --- Custom terrains for this project ---


def continued_sloped_terrain(terrain, slope=1, starting_heights=None):
    """
    Continues slope from given starting height (array of shape (length,))
    """
    width = terrain.width
    length = terrain.length

    if starting_heights is None:
        starting_heights = np.zeros(length, dtype=np.int16)

    units_per_step = slope * terrain.horizontal_scale / terrain.vertical_scale

    for x in range(width):
        h = x * units_per_step
        terrain.height_field_raw[x, :] += starting_heights + int(round(h))

    return terrain


def discrete_obstacles_terrain_x(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
    """
    Generate a terrain with obstacles that span the full y-direction and vary only in x

    Parameters:
        terrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum width of an obstacle in x-direction [meters]
        max_size (float): maximum width of an obstacle in x-direction [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        start_i = np.random.choice(range(0, i-width, 4))
        # Span the full depth in y-direction, only vary in x
        terrain.height_field_raw[start_i:start_i+width, :] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def discrete_obstacles_terrain_structured(terrain, difficulty=1.0, idx=0,platform_size=1.):
    """
    Generate a structured parkour terrain with specific obstacle sequence:
    1. Drop off a ledge
    2. Series of varied height boxes
    3. Big jump up
    
    Parameters:
        terrain (terrain): the terrain
        difficulty (float): multiplier for obstacle heights (1.0 = standard, 1.5 = harder)
        platform_size (float): size of the flat starting platform at the center [meters]
    Returns:
        terrain (SubTerrain): updated terrain
    """
    # Convert to discrete units
    platform_size = int(platform_size / terrain.horizontal_scale)
    
    (i, j) = terrain.height_field_raw.shape
    
    # Define obstacle sequence (heights in meters, widths in meters)
    # Format: (x_start [m], width [m], height [m])
    obstacles = [
        (0.0, 0.5, 0.0),           # Flat start
        # (0.5, 0.3, -0.2),          # Drop down
        (0.5, 0.7, -0.2),          # Stay low
        (1.2, 1.0, 0.1),           # Step up to ground
        (2.2, 0.6, -0.5),           # Small box up
        (2.8, 0.2, 0.0),         # Small drop
        # (2.1, 0.3, 0.15),          # Medium box
        # (2.4, 0.2, 0.05),          # Small box
        # (2.6, 0.4, 0.35),          # BIG JUMP UP!
    ]
    # VARIED DIFFICULTY COURSES

    obstacles_basic = [
        (0.0, 0.5, 0.0),           # Flat start
        (0.5, 0.7, -0.2),          # Drop down
        (1.2, 1.0, 0.1),           # Step up
        (2.2, 0.6, -0.3),          # Medium pit
        (2.8, 0.2, 0.0),           # Flat landing
    ]

    # obstacles_gaps = [
    #     (0.0, 0.5, 0.0),           # Flat start
    #     (0.5, 0.4, 0.1),           # Small platform
    #     (0.9, 0.6, -0.5),  # GAP of 0.6m here!
    #     (1.5, 0.5, 0.1),           # Landing platform (requires jump)
    #     (2.0, 0.4, -0.2),          # Drop after gap
    #     (2.4, 0.6, 0.2),           # Final platform
    # ]

    obstacles_gaps = [
        (0.0, 0.5, 0.0),           # Flat start
        (0.5, 0.8, 0.2),           # Small platform
        # (0.9, 0.5, -0.5),  # GAP of 0.6m here!
        # (0.9, 0.5, 0.2),           # Landing platform (requires jump)
        (1.3, 1.65, -0.2),          # Drop after gap
        # (1.7, 1.0, 0.2),           # Final platform
        (2.7, 0.29, -0.2),           # Final platform
        # (2.7, 0.28, 0.15),           # Final platform
    ]

    # obstacles_gaps2 = [
    #     (0.0, 0.7, -0.3),           # Flat start
    #     (0.7, 0.5, 0.15),           # Small platform
    #     # GAP of 0.6m here!
    #     (1.2, 0.5, 0.35),           # Landing platform (requires jump)
    #     (1.4, 0.4, -0.2),          # Drop after gap
    #     (1.7, 1.0, 0.2),           # Final platform
    #     (2.7, 0.3, -0.3),           # Final platform
    # ]
    obstacles_gaps2 = [
        (0.0, 2.9, 0.15),           # Flat start

    ]

    obstacles_stairs = [
        (0.0, 0.4, 0.0),           # Flat start
        (0.4, 0.4, 0.1),           # Step 1
        (0.8, 0.4, 0.2),           # Step 2
        (1.2, 0.4, 0.3),           # Step 3 (stairs up!)
        (1.6, 0.6, 0.3),           # Platform at top
        (2.2, 0.4, 0.15),          # Step down 1
        (2.6, 0.4, 0.0),           # Ground level
    ]

    obstacles_narrow = [
        (0.0, 0.5, 0.0),           # Flat start
        (0.5, 0.8, 0.2),           # Elevated platform
        (1.3, 0.2, 0.2),           # NARROW BEAM (balance!)
        (1.5, 0.8, 0.2),           # Landing platform
        (2.3, 0.7, -0.1),          # Lower section
    ]

    obstacles_extreme = [
        (0.0, 0.4, 0.0),           # Flat start
        (0.4, 0.3, 0.3),           # High box
        # GAP of 0.5m
        (1.2, 0.3, -0.5),          # Deep pit after gap!
        (1.5, 0.5, 0.0),           # Recover to ground
        (2.0, 0.4, 0.4),           # BIG JUMP UP
        (2.4, 0.6, 0.4),           # High platform finish
    ]

    obstacles_mixed = [
        (0.0, 0.5, 0.0),           # Flat start
        (0.5, 0.4, -0.3),          # Drop
        # GAP of 0.4m
        (1.3, 0.3, 0.1),           # Small platform (jump from pit!)
        (1.6, 0.5, 0.3),           # High platform
        (2.1, 0.3, 0.0),           # Step down
        (2.4, 0.6, 0.5),           # MAX HEIGHT FINISH
    ]

    if idx == 1:
        obstacles = obstacles_gaps 
    elif idx == 2:
        obstacles = obstacles_gaps2 #obstacles_stairs
    elif idx == 3:
        obstacles = obstacles_narrow
    elif idx == 4:
        obstacles = obstacles_extreme
    elif idx == 5:
        obstacles = obstacles_mixed 
    elif idx == 6:
        obstacles = obstacles_mixed

    # obstacles = obstacles_gaps
    # Apply difficulty scaling
    obstacles = [(x, w, h * difficulty) for x, w, h in obstacles]
    
    # Convert obstacles to discrete units and place them
    for x_start, width, height in obstacles:
        x_start_discrete = int(x_start / terrain.horizontal_scale)
        width_discrete = int(width / terrain.horizontal_scale)
        height_discrete = int(height / terrain.vertical_scale)
        
        # Make sure we don't go out of bounds
        if x_start_discrete + width_discrete < i:
            terrain.height_field_raw[x_start_discrete:x_start_discrete+width_discrete, :] = height_discrete
    
    # Clear spawn platform in the center
    # x1 = (terrain.width - platform_size) // 2
    # x2 = (terrain.width + platform_size) // 2
    # y1 = (terrain.length - platform_size) // 2
    # y2 = (terrain.length + platform_size) // 2
    # terrain.height_field_raw[x1:x2, y1:y2] = 0
    
    return terrain


def flat_then_stairs_terrain(terrain, step_width, step_height):
    """
    Flat for first 2/3 in x, then stairs in x direction only.
    Returns final height and step phase offset (remainder from full step).
    """
    step_width_units = int(step_width / terrain.horizontal_scale)
    step_height_units = int(step_height / terrain.vertical_scale)

    start_x = int(terrain.width * 2 / 3)
    current_height = 0

    for x in range(start_x, terrain.width):
        relative_x = x - start_x
        if relative_x % step_width_units == 0 and relative_x != 0:
            current_height += step_height_units
        terrain.height_field_raw[x, :] += current_height

    # Return offset for where we are in the step pattern
    phase_offset = (terrain.width - start_x) % step_width_units
    return terrain, current_height, phase_offset


def continued_stairs_terrain(terrain, step_width, step_height, starting_height=0, step_offset=0):
    """
    Continue stairs in x-direction, matching step pattern from previous terrain.
    """
    step_width_units = int(step_width / terrain.horizontal_scale)
    step_height_units = int(step_height / terrain.vertical_scale)

    current_height = starting_height

    for x in range(terrain.width):
        adjusted_x = x + step_offset
        if adjusted_x % step_width_units == 0 and adjusted_x != 0:
            current_height += step_height_units
        terrain.height_field_raw[x, :] += current_height

    last_heights = terrain.height_field_raw[-1, :].copy()  
    return terrain, last_heights


def trees_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=0):
    """
    Generate a simulated trees terrain (protruding objects)

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    # height_range = np.arange(max_height-1, max_height, step=1)
    height_range = np.arange(20, max_height, step=1)
    print('height range', height_range)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain
