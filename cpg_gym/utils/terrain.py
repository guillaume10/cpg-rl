# License: see [LICENSE, LICENSES/legged_gym_license]

import numpy as np

from cpg_gym.utils import terrain_utils
from cpg_gym.envs.base.quadruped_gaits_config import QuadrupedGaitsCfg

class Terrain:
    def __init__(self, cfg: QuadrupedGaitsCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.data_collect:
            self._terrain_num = cfg.terrain_num
            self._terrain_cfg_a = cfg.terrain_cfg_a
            self.data_collect_terrain()
        elif cfg.parkour:
            self.parkour()
        elif cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = (i+1) / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def data_collect_terrain(self):
        terrain_num = self._terrain_num
        terrain_cfg_a = self._terrain_cfg_a 
        prev_terrain = None
        prev_terrain2 = None
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
                if terrain_num == 0:
                    # flat 
                    if terrain_cfg_a > 0:
                        import sys; print('[EXIT] Flat terrain does not need multiple terrain_cfg_a.');
                        sys.exit()
                    terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.)
                if terrain_num == 1 or terrain_num == 2:
                    # slope up  or slope down 
                    slopes = [0.1, 0.2, 0.3, 0.4, 0.5]
                    slp = slopes[terrain_cfg_a]
                    if terrain_num == 2:
                        slp *= -1.0
                    if i == 0:
                        terrain = terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.)
                    else:
                        terrain_utils.continued_sloped_terrain(terrain, slope=slp, starting_heights=0)

                if terrain_num == 3 or terrain_num == 4:
                    # stairs up  or stairs down 
                    stair_heights = [0.05, 0.1, 0.15, 0.2, 0.23]
                    stair_height = stair_heights[terrain_cfg_a]
                    if terrain_num == 4:
                        stair_height *= -1.0
                    if i == 0:
                        terrain = terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.)
                    elif i==1:
                        terrain, prev_terrain = terrain_utils.continued_stairs_terrain(
                            terrain, step_width=0.31, step_height=stair_height, starting_height=0)
                    else:
                        terrain = terrain_utils.continued_sloped_terrain(terrain, slope=0, starting_heights=prev_terrain)

                if terrain_num == 5:
                    # gaps
                    difficulties = [0.2, 0.4, 0.6, 0.8, 1.0]
                    difficulty = difficulties[terrain_cfg_a]
                    gap_size = 0.05 + difficulty/2 # 0.05 to 0.55
                    gap_distance = 0.7 - 0.45*difficulty # 0.7 to 0.25
                    if i == 0:
                        terrain = terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.)
                    elif i==1:
                        gap_terrainDeepTransition_DATA_COLLECT(terrain, gap_size=gap_size,gap_distance=gap_distance)

                if terrain_num == 6:
                    # rough terrain (uneven) 
                    rough_heights = [0.02, 0.04, 0.06, 0.08, 0.1]
                    rough_height = rough_heights[terrain_cfg_a]
                    if i == 0:
                        terrain = terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.)
                    else: 
                        terrain_utils.random_uniform_terrain(terrain, min_height=-rough_height, max_height=rough_height, step=0.005, downsampled_scale=0.2)
                if terrain_num == 7:
                    # unstructured terrain (varying large boxes )
                    num_rectangles = 20
                    rectangle_min_size = 1.
                    rectangle_max_size = 2.
                    discrete_obstacles_heights = [0.05, 0.1, 0.15, 0.2, 0.23]
                    discrete_obstacles_height = discrete_obstacles_heights[terrain_cfg_a]
                    if i == 0:
                        terrain = terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.)
                    else:
                        terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=0.5)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)

        # note: primarily for GO1/GO2
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.18

        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-difficulty*0.1, max_height=difficulty*0.1, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            stepping_stones_size = 1 * (1.8-difficulty)
            stepping_stones_size = difficulty
            if difficulty > 0.1:
                stone_width = 1.7 
                terrain_utils.trees_terrain(terrain, 
                                        stone_size=stone_width - difficulty, 
                                        stone_distance=max(0.4, stone_width - difficulty), 
                                        max_height=0.5, 
                                        platform_size=3.3,  
                                        depth=0.)


        elif choice < self.proportions[6]:
            max_gap_length = 0.2 
            gap_size = difficulty*max_gap_length
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            tree_height = 0.5
            wall_terrain(terrain,wall_height=tree_height, wall_width=6, gap_size=1.4,platform_size=3, sides=difficulty>0.5)
        elif choice < self.proportions[8]:
            stepping_stones_size = difficulty
            stone_distance = 0.05 if difficulty==0 else 0.1
            gap_size = 0.05 + difficulty/2
            gap_distance = 0.7- 0.45*difficulty
            gap_terrainDeepTransition(terrain, gap_size=gap_size,gap_distance=gap_distance, platform_size=2.)

        elif choice < self.proportions[9]:
            stepping_stones_size = difficulty +0.05
            stone_distance = difficulty*0.2 
            terrain_utils.stepping_stones_terrain(terrain, 
                                                  stone_size=stepping_stones_size, 
                                                  stone_distance=stone_distance, 
                                                  max_height=0,
                                                  platform_size=3.)
        elif choice < self.proportions[10]:
            step_down(terrain, gap_size=0.5, platform_size=np.random.uniform(1, 3), step_down_height=difficulty * 30)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


    def parkour(self):
        """ Flowing parkour using continued_* terrains (no platforms). """
        # Parkour column - can be set to any column number
        parkour_col = getattr(self.cfg, 'parkour_col', 0)  # Default to column 0
        # Initialize starting height array for the first terrain
        starting_heights = np.zeros(self.length_per_env_pixels, dtype=np.int16)

        blocks_count = 0
        # Define parkour sequence 
        parkour_sequence = [
            ("upslope", 0),
            
            ("downslope", -0.5),
            ("upslope", 0.5),
            ("stairs_down", (-0.15,0.31)), 
            ("stairs_up", (0.15,0.31)),
            ("gaps", (0.55, 0.25)),
            ("rough_heavy", (-0.1, 0.1)),

            ("blocks",(0.2,10)),
            ("blocks",(0.2,10)),
            ("blocks",(0.2,10))
        ]

        if 1:# 
            # for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = terrain_utils.SubTerrain(
                    "terrain",
                    width=self.width_per_env_pixels,
                    length=self.width_per_env_pixels,
                    vertical_scale=self.cfg.vertical_scale,
                    horizontal_scale=self.cfg.horizontal_scale,
                )
                
                # Use sequence index, cycling if needed
                seq_idx = i % len(parkour_sequence)
                terrain_type, params = parkour_sequence[seq_idx]
                
                if terrain_type == "upslope":
                    terrain = terrain_utils.continued_sloped_terrain(
                        terrain, slope=params, starting_heights=starting_heights)
                    # Update starting_heights for next terrain - get the end heights from this terrain
                    starting_heights = terrain.height_field_raw[-1, :]  # Get center column heights
                if terrain_type == "stairs_up":
                    terrain, starting_heights = terrain_utils.continued_stairs_terrain(
                        terrain, step_width=params[1], step_height=params[0], starting_height=starting_heights) #0.31
                    # starting_heights = terrain.height_field_raw[-1, 0]
                elif terrain_type == "gaps":
                    terrain.height_field_raw = np.zeros((terrain.length, terrain.width), dtype=np.int16)
                    for x in range(terrain.length):
                        terrain.height_field_raw[x, :] = starting_heights[x]
                    gap_terrainDeepTransition_DATA_COLLECT(terrain, gap_size=params[0],gap_distance=params[1])
                elif terrain_type == "rough_heavy":
                    # Apply rough terrain on top of current heights
                    terrain.height_field_raw = np.zeros((terrain.length, terrain.width), dtype=np.int16)
                    for x in range(terrain.length):
                        terrain.height_field_raw[x, :] = starting_heights[x]
                    terrain_utils.random_uniform_terrain(
                        terrain, min_height=params[0], max_height=params[1], step=0.005, downsampled_scale=0.15)
                    # Update starting_heights for next terrain
                    # starting_heights = terrain.height_field_raw[:, terrain.width//2]
                elif terrain_type == "stairs_down":
                    terrain, starting_heights = terrain_utils.continued_stairs_terrain(
                        terrain, step_width=params[1], step_height=params[0], starting_height=starting_heights)
                elif terrain_type == "downslope":
                    terrain = terrain_utils.continued_sloped_terrain(
                        terrain, slope=params, starting_heights=starting_heights)
                    # Update starting_heights for next terrain
                    starting_heights = terrain.height_field_raw[-1, 0]
                elif terrain_type == "blocks":
                    num_rectangles = params[1] #20
                    rectangle_min_size = 1
                    rectangle_max_size = 2
                    #terrain_utils.discrete_obstacles_terrain(terrain, params[0], rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=0)
                    # terrain_utils.discrete_obstacles_terrain_x(terrain, params[0], rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=0)
                    terrain_utils.discrete_obstacles_terrain_structured(terrain, difficulty=1.0, idx=blocks_count%3,platform_size=0)
                    blocks_count += 1
                elif terrain_type == "step_down":
                    step_down(terrain, gap_size=0, platform_size=params[0]*2, step_down_height=params[1])

                # Place parkour terrain in specified column
                self.add_terrain_to_map(terrain, i, parkour_col)


def gap_terrainDeepTransition(terrain, gap_size, gap_distance, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    gap_distance = int(gap_distance / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    depth = - int(0.5 / terrain.vertical_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    for i in range(terrain.length // (gap_size+gap_distance)):
        terrain.height_field_raw[i*gap_distance + i*gap_size : i*gap_distance + (i+1)*gap_size , center_y-y2 : center_y + y2] = depth

    if platform_size >= 0.1:
        x1 = (terrain.width - platform_size) // 2
        x2 = (terrain.width + platform_size) // 2
        y1 = (terrain.length - platform_size) // 2
        y2 = (terrain.length + platform_size) // 2
        terrain.height_field_raw[x1:x2, y1:y2] = 0

def gap_terrainDeepTransition_DATA_COLLECT(terrain, gap_size, gap_distance):
    gap_size = int(gap_size / terrain.horizontal_scale)
    gap_distance = int(gap_distance / terrain.horizontal_scale)
    depth = - int(0.5 / terrain.vertical_scale) 

    for i in range(terrain.length // (gap_size+gap_distance)):
        terrain.height_field_raw[i*gap_distance + i*gap_size : i*gap_distance + (i+1)*gap_size , :] = depth


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    pit_depth = int(0.5 / terrain.vertical_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    print('PIT DEPTH', pit_depth)
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = - pit_depth #-10 #-1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0


def step_down(terrain, gap_size, platform_size=1., step_down_height=0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    pit_depth = int(0.05 / terrain.vertical_scale)
    gap_size =  2 
    pit_depth = 5 # this multiplies with vertical_scale to get actual height. for test we have vertical_scale = 0.1 
    pit_depth = step_down_height # 10= 0.1m, etc. 

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[-platform_size//2 :, :] = -pit_depth 


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def wall_terrain(terrain, wall_height, wall_width=6, gap_size=1.4, platform_size=2, sides=True):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    wall_height = int(wall_height / terrain.vertical_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2

    x1 = terrain.length - wall_width 
    x2 = terrain.length
    gap_center = center_y + np.random.uniform(-3, 3)
    y_left  = int(gap_center + gap_size/2)
    y_right = int(gap_center - gap_size/2)

    gap_center = center_x + np.random.uniform(-3, 3)
    x_left  = int(gap_center + gap_size/2)
    x_right = int(gap_center - gap_size/2)

    y1 = terrain.width - wall_width
    y2 = terrain.width

    terrain.height_field_raw[:] = 0
    # FRONT
    terrain.height_field_raw[x1:x2, y_left : terrain.width] = wall_height 
    terrain.height_field_raw[x1:x2, 0: y_right] = wall_height 
    # REAR
    terrain.height_field_raw[wall_width:2*wall_width, y_left : terrain.width] = wall_height 
    terrain.height_field_raw[wall_width:2*wall_width, 0: y_right] = wall_height 
    if sides:
        # LEFT
        terrain.height_field_raw[x_left:terrain.length, 0:wall_width] = wall_height 
        terrain.height_field_raw[0:x_right, 0:wall_width] = wall_height 
        # RIGHT
        terrain.height_field_raw[x_left:terrain.length, y1:y2] = wall_height 
        terrain.height_field_raw[0:x_right, y1:y2] = wall_height 