# License: see [LICENSE, LICENSES/legged_gym_license]

from .base_config import BaseConfig
ROUGH_TERRAIN = True
ADD_NOISE = False

ACTION_GAIT = True 
ACTION_GAIT_NUM = 3 
OBS_SPACE = "FULL_STYLE" 

PARKOUR = False    
# PARKOUR = True            

OBS_HISTORY = 1 

class QuadrupedGaitSelectorCfg(BaseConfig):
    
    class pretrained:
        num_observations = 258
        num_actions = 11
        gait_policy_paths = {}
        nn_path = '{CPG_GYM_ROOT_DIR}/resources/models/stage1/model_0.pt'

        num_stacked_observations = OBS_HISTORY
        runner_class_name = 'OnPolicyRunner'
        class policy:
            init_noise_std = 1.0
            actor_hidden_dims = [512, 256, 128]
            critic_hidden_dims = [512, 256, 128]
            activation = 'elu' 

    class env:
        num_envs = 4096
        num_privileged_obs = None 
        env_spacing = 3.  
        send_timeouts = True 
        episode_length_s = 20 
        if PARKOUR:
            episode_length_s = 200
        num_bodies = 17
        num_observations = 70
        obs_space = OBS_SPACE
        
        if OBS_SPACE == "FULL":
            num_observations = 56  
        num_actions = 12

        if "FULL" in OBS_SPACE:
            num_actions = 11 
            num_observations = 12*3 + num_actions + 4 + 4*4 + 1 
            if "STYLE" in OBS_SPACE:
                num_observations += 3

        if ROUGH_TERRAIN: 
            num_observations += 17*11 
        
        if OBS_HISTORY > 0:
            num_observations *= OBS_HISTORY
            num_stacked_observations = OBS_HISTORY
    
    
    class terrain:
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 1 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        measure_heights = False

        rough_terrain = ROUGH_TERRAIN
        test_time = False
        handstand = False 

        curriculum_start = -1

        if ROUGH_TERRAIN:
            num_rows = 10
            num_cols = 20
            mesh_type = 'trimesh'
            measure_heights = True
            terrain_proportions = [0.1, 0.1, 0.15, 0.15, 0.25, 0., 0., 0., 0.25]
            for i,x in enumerate(measured_points_x):
                measured_points_x[i] += 0.3 

        push_force_idx = 0
        vel_idx = 0
        step_down_height = 0
        test_mode = ''
        data_collect = False 
        gait_num = 0
        terrain_num = 0
        terrain_cfg_a = 0


        parkour = PARKOUR    
        if parkour:
            terrain_width = 3
            terrain_length = 3
            num_rows = 10 #7
            num_cols = 1 # 2 if you want a wall 
            parkour_col = 0
            max_init_terrain_level = 1
            border_size = 0

            # horizontal_scale = 0.005 # [m]
            # vertical_scale = 0.001 # [m] 
            # horizontal_scale = 0.05
            vel = 1.5 
            gait_num = 8
    
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5 # time before command are changed[s] (was 5)
        heading_command = False#True #True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.4, 2]       
            lin_vel_y = [-1e-7, 1e-7]  
            ang_vel_yaw = [-1e-7, 1e-7] 
            heading = [-1e-7, 1e-7]

    class init_state:
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'BFL_hip_joint': 0.,   # [rad]
            'DRL_hip_joint': 0.,   # [rad]
            'AFR_hip_joint': 0. ,  # [rad]
            'CRR_hip_joint': 0.,   # [rad]

            'BFL_thigh_joint': 0.785,   # [rad]
            'DRL_thigh_joint': 0.785,   # [rad]
            'AFR_thigh_joint': 0.785,   # [rad]
            'CRR_thigh_joint': 0.785,   # [rad]

            'BFL_calf_joint': -1.57,   # [rad]
            'DRL_calf_joint': -1.57,   # [rad]
            'AFR_calf_joint': -1.57,   # [rad]
            'CRR_calf_joint': -1.57,   # [rad]
        }


    class control:
        control_type =  'CPG' 

        CPG_TEST = False
        cpg_gait_num = 0
        cpg_body_height = 0
        cpg_ground_clearance = 0
        cpg_x_offset = 0
        cpg_save_dir = 'logs/cpg_data'
        cpg_body_height_idx = 0
        cpg_ground_clearance_idx = 0
        cpg_x_offset_idx = 0
        random_gait = False 

        add_cpg_actions = 0
        if ACTION_GAIT:
            add_cpg_actions = ACTION_GAIT_NUM 
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        cpg_decimation = 6
        decimation = 18 

        variable_gains = False 
        variable_stiffness = [25, 100]
        variable_damping = [0.3, 2.]
        stiffness = {'joint': 100.}  # [N*m/rad]
        damping = {'joint': 2.}     # [N*m*s/rad]

    class asset:
        disable_gravity = False
        collapse_fixed_joints = True 
        fix_base_link = False 
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01 
        thickness = 0.01

        robot_name = "GO1"
        hip_y = 0.08
        file = '{CPG_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_fix.urdf'

        foot_name = "foot"
        penalize_contacts_on = ["base", "hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base","thigh"]
        self_collisions = 1

    class domain_rand:
        randomize_friction = True 
        friction_range = [-0.7, 1.5]
        randomize_base_mass = True
        added_mass_range = [0., 8.] 
        push_robots = True 
        push_interval_s = 15
        max_push_vel_xy = 0.5
        motor_latency = True 
        motor_latency_factor_min = 0.3 
        if not ADD_NOISE:
            randomize_friction = False
            randomize_base_mass = False
            motor_latency = False
        
    class rewards:
        class scales:
            action_clip_penalty = -1
            action_rate = -0.05
            constant = 3
            # Choose specific criteria to optimize
            if 1: # for energy 
                energy = -0.02 
            if 0: # for stability 
                stability = -10
            if 0: # joint dof acc 
                dof_acc = -1e-3 

            # Other example tests from paper 
            # lin_vel_z = 10
            # orientation_pitch=-2
            # orientation_roll=-100
            # ang_vel_x = 20
            # ang_vel_y = 20
            # ang_vel_z = 20
            # lin_vel_y = 20
            # energy = 0.01
            # stability = 10 
            # feet_contact_forces = -0.1
            # bad_tracking_lin_vel_x = 20

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_vx = 2
        tracking_sigma_pos = 0.05
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100 

        soft_dof_pos_limit = 0.9
        base_height_target = 0.3 

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1 
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 2.0
        clip_observations = 100.
        clip_actions = 4.

    class noise:
        add_noise = True
        if not ADD_NOISE:
            add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            height_measurements = 0.1
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.1
            gravity = 0.05

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

        pos = [15, -12, 4]  # [m]
        lookat = [15., 0, 0.]  # [m]

    class sim:
        dt = 0.005 
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 # [m/s] 
            max_depenetration_velocity = 1.0 
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class QuadrupedGaitSelectorCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' 

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1e-4 
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.


    class runner:
        policy_class_name = 'ActorCritic' # 'ActorCriticRecurrent'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates
        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'quadruped_gait_selector' # 'quadruped' #'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt