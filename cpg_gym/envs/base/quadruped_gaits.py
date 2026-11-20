# License: see [LICENSE, LICENSES/legged_gym_license]

from cpg_gym import CPG_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from cpg_gym import CPG_GYM_ROOT_DIR
from cpg_gym.envs.base.base_task import BaseTask
from cpg_gym.utils.terrain import Terrain
from cpg_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_rpy_tensor
from cpg_gym.utils.helpers import class_to_dict
from .quadruped_gaits_config import QuadrupedGaitsCfg
import random
from .kinematics import ComputeIK
from .cpg import *

class QuadrupedGaits(BaseTask):
    def __init__(self, cfg: QuadrupedGaitsCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        # self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.preclip_actions = actions

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            if self.cfg.terrain.parkour or self.cfg.terrain.test_time:
                self._refresh_physics_between_steps()
                self.set_vel_parkour()
                self.last_dof_vel[:] = self.dof_vel[:]
        self.post_physics_step()
        
        if not self.headless:
            self.set_robot_colors(self.env_indices,self._cpg.curr_gait_nums.to(dtype=torch.int))

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.env.num_stacked_observations > 1:
            stacked_obs = self.obs_buf_stacked[:,:,0]
            for i in range(1,self.cfg.env.num_stacked_observations):
                stacked_obs = torch.cat((stacked_obs, self.obs_buf_stacked[:,:,i]), dim=-1)
            return stacked_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        if self.cfg.env.num_stacked_observations > 1:
            stacked_obs = self.obs_buf_stacked[:,:,0]
            for i in range(1,self.cfg.env.num_stacked_observations):
                stacked_obs = torch.cat((stacked_obs, self.obs_buf_stacked[:,:,i]), dim=-1)
            return stacked_obs
        else:
            return self.obs_buf
    
    def _refresh_physics_between_steps(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.use_force_sensors:
            self.gym.refresh_force_sensor_tensor(self.sim)
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # compute observations, rewards, resets, ...
        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.last_actions[:] = self.actions[:]
        self.last_root_pos[:] = self.root_states[:, 0:3]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def set_vel_parkour(self):
        # ramp up desired velocity based on time 
        vel_array = np.array([0, 0.25, 0.5, 0.75, 1.0])
        ramp_vel_index_vector = [ 1, 100, 200, 300, 400]
        if self._ep_counter >= ramp_vel_index_vector[0]:
            self.commands[:, 0] = vel_array[0] * self.cfg.terrain.vel
        if self._ep_counter >= ramp_vel_index_vector[1]:
            self.commands[:, 0] = vel_array[1] * self.cfg.terrain.vel
        if self._ep_counter >= ramp_vel_index_vector[2]:
            self.commands[:, 0] = vel_array[2] * self.cfg.terrain.vel
        if self._ep_counter >= ramp_vel_index_vector[3]:
            self.commands[:, 0] = vel_array[3] * self.cfg.terrain.vel
        if self._ep_counter >= ramp_vel_index_vector[4]:
            self.commands[:, 0] = self.cfg.terrain.vel
        self._ep_counter +=1 

    def set_robot_colors(self, env_idxs, c):
        """ Set color based on gait """
        colors = [
            gymapi.Vec3(0.2, 0.2, 0.2),                # lswalk (black)
            gymapi.Vec3(1.0, 1.0, 1.0),                # dswalk (white)
            gymapi.Vec3(0.0,    0.4470,    0.7410),    # amble  (dark blue)
            gymapi.Vec3(0.8500,    0.3250,    0.0980), # trot (orange) 
            gymapi.Vec3(0.9290,    0.6940,    0.1250), # pace (yellow)
            gymapi.Vec3(0.4660,    0.6740,    0.1880), # pronk (green moss)
            gymapi.Vec3(0.9490,    0.4030,    0.7720), # halfbound (magenta-pink)
            gymapi.Vec3(0.4940,    0.1840,    0.5560), # bound (purple)
            gymapi.Vec3(0.3010,    0.7450,    0.9330), # canter (light blue)
            gymapi.Vec3(0.6350,    0.0780,   0.1840),  # transverse gallop (red-ish)
            gymapi.Vec3(0.1, 0.8, 0.5),                # rotary gallop (green)    
        ]

        if self.cfg.control.control_type == "P":
            for i in env_idxs:
                for j in range(self.cfg.env.num_bodies): 
                    self.gym.set_rigid_body_color(self.envs[i], self.actor_handles[i], 
                                        j, gymapi.MESH_VISUAL_AND_COLLISION, 
                                        gymapi.Vec3(0.4, 0.0, 0.05))
            return
        if self._cpg._couple == False:
            for i in env_idxs:
                for j in range(self.cfg.env.num_bodies): 
                    self.gym.set_rigid_body_color(self.envs[i], self.actor_handles[i], 
                                        j, gymapi.MESH_VISUAL_AND_COLLISION, 
                                        gymapi.Vec3(0.5, 0.5, 0.5))
            return

        if "TEST1" in self._cpg.gaits or "TEST2" in self._cpg.gaits:
            for i in env_idxs:
                self.gym.set_rigid_body_color(self.envs[i], self.actor_handles[i], 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))
                for j in [1,2,3,4, 9,10,11,12, 13,14,15,16]: # FR, RR, RL 
                    self.gym.set_rigid_body_color(self.envs[i], self.actor_handles[i], j, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1,  0.7,    0.2))
                for j in [5,6,7,8]: # FL leg 
                    self.gym.set_rigid_body_color(self.envs[i], self.actor_handles[i], j, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1,  0,    0))
            return 

        for i in env_idxs:
            self.gym.set_rigid_body_color(self.envs[i], self.actor_handles[i], 0, gymapi.MESH_VISUAL_AND_COLLISION, colors[c[i]])

            for j in range(self.cfg.env.num_bodies): 
                self.gym.set_rigid_body_color(self.envs[i], self.actor_handles[i], 
                                    j, gymapi.MESH_VISUAL_AND_COLLISION, colors[c[i]])
 
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.use_force_sensors:
            self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_pos[:] = self.root_states[:, 0:3]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            self.last_rest_indices = env_ids
            return
        # update curriculum
        if self.cfg.terrain.curriculum and self.common_step_counter > self.cfg.terrain.curriculum_start:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        if "CPG" in self.cfg.control.control_type:
            self._cpg.reset(env_ids)
        if not self.cfg.terrain.test_time and not self.cfg.terrain.parkour:
            if "CPG" in self.cfg.control.control_type:
                self._cpg.resample_gaits(env_ids)
            self._resample_commands(env_ids)
        self._resample_gains(env_ids)

        if self.cfg.env.num_stacked_observations > 1:
            self.obs_buf_stacked[env_ids,:,:] = 0.

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        if "CPG" in self.cfg.control.control_type:
            self.obs_buf = torch.cat(( 
                                    self.base_lin_vel * self.obs_scales.lin_vel,  
                                    self.base_ang_vel  * self.obs_scales.ang_vel, 
                                    wrap_to_pi(get_rpy_tensor(self.base_quat)), 
                                    self.commands[:, :3] * self.commands_scale,   
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, 
                                    self.dof_vel * self.obs_scales.dof_vel,                          
                                    self.actions,                                                    
                                    self.base_quat,
                                    (self._cpg.X[:,0,:] - ((MU_UPP+ MU_LOW) / 2)) * 2,  
                                    (self._cpg.X[:,1,:] - np.pi) * 1/np.pi,             
                                    self._cpg.X_dot[:,0,:] *  1/self._cpg._dr_factor, 
                                    (self._cpg.X_dot[:,1,:]) / self._cpg._omega_max, 
                                    ( (self._cpg.curr_gait_nums[:] - 5 ) / 3).unsqueeze(-1)
                                    ),dim=-1)
            
            if "STYLE" in self.cfg.env.obs_space:

                self.obs_buf = torch.cat(( 
                                    self.base_lin_vel * self.obs_scales.lin_vel, 
                                    self.base_ang_vel  * self.obs_scales.ang_vel, 
                                    wrap_to_pi(get_rpy_tensor(self.base_quat)), 
                                    self.commands[:, :3] * self.commands_scale,  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, 
                                    self.dof_vel * self.obs_scales.dof_vel,                         
                                    self.actions,                                                    
                                    self.base_quat,                                                  
                                    (self._cpg.X[:,0,:] - ((MU_UPP+ MU_LOW) / 2)) * 2,  
                                    (self._cpg.X[:,1,:] - np.pi) * 1/np.pi,             
                                    self._cpg.X_dot[:,0,:] *  1/self._cpg._dr_factor,
                                    (self._cpg.X_dot[:,1,:]) / self._cpg._omega_max, 
                                    ( (self._cpg.curr_gait_nums[:] - 5 ) / 3).unsqueeze(-1),
                                    self._cpg._robot_height[:,0,0].unsqueeze(-1) / MAX_BODY_HEIGHT,
                                    self._cpg._ground_clearance[:,0,0].unsqueeze(-1) / MAX_STEP_HEIGHT,
                                    self._cpg._x_offset[:,0,0].unsqueeze(-1) / MIN_FOOT_X_OFFSET,
                                    ),dim=-1)
                                    
            if "Y" in self.cfg.control.control_type:
                self.obs_buf = torch.cat((self.obs_buf,
                                          (self._cpg.Y - ((MU_UPP+ MU_LOW) / 2)) * 2,
                                          self._cpg.Y_dot * 1/self._cpg._dy_factor, # 1/30
                                          ), dim=-1)

            if "PSI" in self.cfg.control.control_type:
                self.obs_buf = torch.cat((self.obs_buf, 
                                         (self._cpg.PSI - np.pi) * 1/np.pi,
                                         self._cpg.PSI_dot * 1/self._cpg._psi_max, 
                                         ), dim=-1)


        else:
            if self.cfg.env.obs_space == "FULL":
                self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        wrap_to_pi(get_rpy_tensor(self.base_quat)),
                                        self.commands[:, :3] * self.commands_scale,
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        self.base_quat,
                                        ),dim=-1)
        
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            if self.cfg.env.num_stacked_observations <= 1:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            else:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[0:int(self.cfg.env.num_observations / self.cfg.env.num_stacked_observations)]

        if self.cfg.env.num_stacked_observations > 1:
            self.obs_buf_stacked[:,:,1:] = self.obs_buf_stacked[:,:,:-1].clone()  # shift older frames back one slot
            self.obs_buf_stacked[:,:,0] = self.obs_buf                            # slot 0 = current frame

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 400
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
                props["stiffness"][i] = 0.0
                props["damping"][i] = 0.0

        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1]) 
            for i in range(1,len(props)):
                props[i].mass = np.random.uniform(props[i].mass * 0.7, props[i].mass * 1.3)
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        if not self.cfg.terrain.test_time:
            self._resample_commands(env_ids)

        if ("CPG" in self.cfg.control.control_type and 
                not self.cfg.terrain.test_time and 
                self.cfg.control.add_cpg_actions < 4 and
                not self.cfg.terrain.parkour):
            env_ids_gaits = (self.episode_length_buf % int(self.cfg.control.cpg_gait_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._cpg.resample_gaits(env_ids_gaits)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if "CPG" in control_type:
            torques = torch.zeros_like(self.torques,device=self.device)
            xs,ys,zs = self._cpg.get_cpg_rl_commands(actions_scaled)
            sideSign = np.array([-1,1,-1,1]) 
            foot_y = torch.ones(self.num_envs,device=self.device,requires_grad=False) * self.cfg.asset.hip_y
            LEG_INDICES = np.array([0,1,2,3])
            for ig_idx,i in enumerate(LEG_INDICES):
                x = xs[:,i]
                z = zs[:,i]
                y = sideSign[i] * foot_y  + ys[:,i]
                torques[:, 3*ig_idx:3*ig_idx+3] = ComputeIK(i,x,y,z,robot_name=self.cfg.asset.robot_name)
                if (self.cfg.terrain.handstand == 2) and (i==2 or i==3):
                    torques[:, 3*ig_idx+1] += np.pi/2
            torques = self.p_gains*(torques - self.dof_pos) - self.d_gains*self.dof_vel 

        elif control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.cfg.domain_rand.motor_latency:
            torques = self.motor_latency * torques + (1 - self.motor_latency) * self.torques

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _resample_gains(self, env_ids):
        """ Sample PD gains for each environment at reset. """
        if self.cfg.control.variable_gains:
            p_range = self.cfg.control.variable_stiffness
            d_range = self.cfg.control.variable_damping
            self.p_gains[env_ids,:] = torch_rand_float( p_range[0], p_range[1], (len(env_ids), 12), device=self.device)
            self.d_gains[env_ids,:] = torch_rand_float( d_range[0], d_range[1], (len(env_ids), 12), device=self.device)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.9, 1.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        if self.cfg.terrain.test_time:
            self.dof_pos[env_ids] = self.default_dof_pos 

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if not self.cfg.terrain.parkour:
                if not self.cfg.terrain.test_time:
                    self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
                else:
                    self.root_states[env_ids, 1:2] += torch_rand_float(-4., 4., (len(env_ids), 1), device=self.device) # xy position within 4m of the center
                    self.root_states[env_ids, 0:1] = 7 
            else:
                self.root_states[env_ids, 0:1] -= 0.8 
                
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.last_root_pos[env_ids] = self.root_states[env_ids, :3]
        # base velocities
        if not self.cfg.terrain.test_time:
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        if self.cfg.terrain.test_time:
            self.root_states[env_ids, 7:13] = 0

        # random orientation 
        num_ids = len(env_ids)
        if num_ids > 1 and not self.cfg.terrain.test_time and self.cfg.noise.add_noise:
            num_ids_2 = int(np.floor(num_ids/2))
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(torch.zeros(len(env_ids),1, dtype=torch.float, device=self.device, requires_grad=False),
                                                                torch.zeros(len(env_ids),1, dtype=torch.float, device=self.device, requires_grad=False),
                                                                torch_rand_float(-np.pi/2, np.pi/2, (len(env_ids),1), device=self.device)).squeeze(1) # +/- np.pi
        
        if self.cfg.terrain.handstand == 1 and num_ids > 1 and not self.cfg.terrain.test_time and not self.cfg.terrain.parkour:
            # for handstand (front)
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(torch.zeros(len(env_ids),1, dtype=torch.float, device=self.device, requires_grad=False),
                                                                torch_rand_float(-np.pi/2, np.pi/2, (len(env_ids),1), device=self.device),
                                                                torch_rand_float(-np.pi/2, np.pi/2, (len(env_ids),1), device=self.device)).squeeze(1) # +/- np.pi
        
        if self.cfg.terrain.handstand == 2 and num_ids > 1 and not self.cfg.terrain.test_time and not self.cfg.terrain.parkour:
            # for rearstand 
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(torch.zeros(len(env_ids),1, dtype=torch.float, device=self.device, requires_grad=False),
                                                                torch_rand_float(-np.pi/2, np.pi/2, (len(env_ids),1), device=self.device),
                                                                torch_rand_float(-np.pi/2, np.pi/2, (len(env_ids),1), device=self.device)).squeeze(1) # +/- np.pi
            
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if torch.mean(self.episode_sums["tracking_lin_vel_x"][env_ids]) / self.max_episode_length > 0.85 * self.reward_scales["tracking_lin_vel_x"]:
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., 3)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_vec = torch.zeros(self.cfg.env.num_observations,dtype=torch.float,device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        noise_vec[48:] = 0. # no noise needed for anything CPG related (careful here with measured heights ) (already 0)
        if self.cfg.terrain.measure_heights:
            noise_vec[-187:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        if self.use_force_sensors:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.use_force_sensors:
            self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.foot_pos = torch.zeros_like(self.dof_pos)
        self.foot_vel = torch.zeros_like(self.dof_vel)

        # Setup contact forces based on terrain type
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # Start with net forces
        
        if self.use_force_sensors:
            # Replace foot forces with reliable sensor data for trimesh terrain
            force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
            num_feet = len(self.feet_names)
            # Reshape: total_sensors = num_envs * num_feet, each sensor has 6 components [fx,fy,fz,tx,ty,tz]
            sensor_forces = force_sensor_readings.view(self.num_envs, num_feet, 6)[..., :3]  # Extract [fx, fy, fz]
            
            # Replace the foot contact forces with sensor forces
            self.contact_forces[:, self.feet_indices, :] = sensor_forces
            print(f"[INFO] Using force sensors for {num_feet} feet per environment ({self.num_envs} envs) on trimesh terrain")

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1)) # [0,0,-1] ? 
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False) 
        self.p_gains = torch.zeros(self.num_envs,12, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs,12, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.preclip_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_pos = torch.zeros_like(self.root_states[:, 0:3])
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        if self.cfg.domain_rand.motor_latency:
            self.motor_latency = torch_rand_float(self.cfg.domain_rand.motor_latency_factor_min, 1, (self.num_envs, 1), device=self.device)

        if self.cfg.env.num_stacked_observations > 1:
            self.obs_buf_stacked = torch.zeros(self.num_envs, int(self.cfg.env.num_observations / self.cfg.env.num_stacked_observations), 
                                                self.cfg.env.num_stacked_observations, 
                                                dtype=torch.float,
                                                device=self.device, 
                                                requires_grad=False)

        if "CPG" in self.cfg.control.control_type:
            # init CPG 
            if self.cfg.control.CPG_TEST:
                self._cpg = CPG(time_step=self.sim_params.dt,
                                  num_envs=self.num_envs,device=self.device,
                                    rl_task_string=self.cfg.control.control_type,
                                    ground_clearance=self.cfg.control.cpg_ground_clearance,
                                    robot_height=self.cfg.control.cpg_body_height,
                                    x_offset=self.cfg.control.cpg_x_offset,
                                    test_time=True 
                                    )
            else:
                self._cpg = CPG(time_step=self.sim_params.dt,
                                    num_envs=self.num_envs,device=self.device,
                                        rl_task_string=self.cfg.control.control_type,
                                        add_cpg_actions=self.cfg.control.add_cpg_actions, couple=self.cfg.control.couple
                                        )
        else:
            self._cpg = CPG()

        self.env_indices = torch.arange(self.num_envs, device=self.device, requires_grad=False)
        if self.cfg.terrain.test_time or self.cfg.terrain.parkour:
            # initialize and save buffers 
            self.vel_array_index = -1
            self._one_second_terminations = 0
            self._early_terminations = 0
            self._completions_half = 0
            self._completions_full = 0
            self.COT_vec = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.vel_vec = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.ang_vel_vec = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.cpg_freq_vec = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.cpg_amp_vec = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.cpg_gait = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.cpg_bh = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.cpg_gc = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.cpg_xoff = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

            self.mean_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.mean_ang_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.completed_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.fallen_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.ep_collect_max = 500
            self.power_vec = torch.zeros(self.ep_collect_max, self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.power_vec_abs = torch.zeros(self.ep_collect_max, self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.mean_vel_vec = torch.zeros(self.ep_collect_max, self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.mean_ang_vel_vec = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.first_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.feet_counter = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
            self.curr_feet =    torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
            self.all_feet =     torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
            self.joint_jerk = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.joint_jerk_vec = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self._ep_counter = 0
            self.vel_array_times = np.linspace(50,int(6300*2),32) 
            self.vel_array = np.linspace(0.1, 3.2, 32)
            self.cpg_save_arr = np.zeros((1, 35))
            self.cpg_save_arr_idx = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:,i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:,i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:,i] = 0.
                self.d_gains[:,i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_foot_pos = to_torch([0., 0.0838, -0.3, 0., -0.0838, -0.3, 0., 0.0838, -0.3, 0., -0.0838, -0.3], device=self.device)
        self.const_ones = torch.ones(self.num_envs,dtype=torch.float, device=self.device, requires_grad=False)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(CPG_GYM_ROOT_DIR=CPG_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.feet_names = feet_names  # Store for use in _init_buffers
        
        # Add force sensors for trimesh terrain (README Known Issues fix)
        self.use_force_sensors = self.cfg.terrain.mesh_type == "trimesh"
        if self.use_force_sensors:
            self.create_foot_force_sensors(robot_asset, feet_names)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def create_foot_force_sensors(self, robot_asset, foot_names):
        """Create force sensors at the feet following PHC's approach.
        
        Args:
            robot_asset: The robot asset to add sensors to
            foot_names: List of foot body names to add sensors to
        """
        for foot_name in foot_names:
            foot_idx = self.gym.find_asset_rigid_body_index(robot_asset, foot_name)
            sensor_pose = gymapi.Transform()
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_constraint_solver_forces = True  # for contacts
            sensor_options.enable_forward_dynamics_forces = False  # exclude gravity
            sensor_options.use_world_frame = True  # report forces in world frame
            
            self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose, sensor_options)
        return

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            if self.cfg.terrain.test_time or self.cfg.terrain.parkour:
                self.env_origins[:] = self.terrain_origins[0, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        # parse num_actions based on if using CPG... [TODO]

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            # print( self.measured_heights) # actual z heights (187 values)
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            # print(height_points) # (x,y) locations to query in map (rotated only by yaw)
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------

    # angular velocities 
    def _reward_ang_vel_x(self):
        # Penalize x axis base angular velocity
        return torch.square(self.base_ang_vel[:, 0])
    def _reward_ang_vel_y(self):
        # Penalize y axis base angular velocity
        return torch.square(self.base_ang_vel[:, 1])
    def _reward_ang_vel_z(self):
        # Penalize z axis base angular velocity
        return torch.square(self.base_ang_vel[:, 2])
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    def _reward_ang_vel_xyz(self):
        # Penalize xyz axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1)

    # orientation 
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_constant(self):
        return self.const_ones

    # linear velocities 
    def _reward_lin_vel_x(self):
        # Penalize x axis base linear velocity
        return torch.square(self.base_lin_vel[:, 0])
    def _reward_lin_vel_y(self):
        # Penalize y axis base linear velocity
        return torch.square(self.base_lin_vel[:, 1])
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    def _reward_lin_vel_yz(self):
        # Penalize yz axis base linear velocity
        return torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)

    def _reward_orientation_roll(self):
        # Penalize roll only 
        return torch.square(wrap_to_pi(get_rpy_tensor(self.base_quat)[:,0]))
    
    def _reward_orientation_pitch(self):
        # Penalize pitch only 
        return torch.square(wrap_to_pi(get_rpy_tensor(self.base_quat)[:,1]))

    def _reward_orientation_yaw(self):
        # Penalize yaw only, and base pos y so it goes straight
        return torch.square(wrap_to_pi(get_rpy_tensor(self.base_quat)[:,2])) \
                +  torch.square(self.root_states[:, 1] -  self.env_origins[:,1])

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_energy(self):
        # Penalize energy
        return torch.sum(torch.abs(self.torques*self.dof_vel), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        # torch.mean(torch.abs((self.last_dof_vel - self.dof_vel) / 0.005), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_clip_penalty(self):
        # Penalize actions outside of clip_action range
        out_of_limits =  (self.preclip_actions - self.cfg.normalization.clip_actions) > 0.
        out_of_limits += (self.preclip_actions + self.cfg.normalization.clip_actions) < 0.
        return torch.sum(out_of_limits, dim=1)


    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)


    # Tracking
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axis)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axis)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # Handstand 
    def _reward_tracking_pitch_pos(self):
        # Track pitch handstand/rearstand)
        pitch_error = torch.square(np.pi/2 - wrap_to_pi(get_rpy_tensor(self.base_quat)[:,1]))
        return torch.exp(-pitch_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_pitch_neg(self):
        # Track pitch handstand/rearstand)
        pitch_error = torch.square(-np.pi/2 - wrap_to_pi(get_rpy_tensor(self.base_quat)[:,1]))
        return torch.exp(-pitch_error/self.cfg.rewards.tracking_sigma) 
    
    def _reward_tracking_lin_vel_x_handstand(self):
        # Tracking of linear velocity commands (becomes vz when in handstand)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 2])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_x_handstand_neg(self):
        # Tracking of linear velocity commands (becomes vz when in handstand)
        lin_vel_error = torch.square(self.commands[:, 0] + self.base_lin_vel[:, 2])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z_handstand(self):
        # Penalize z axis base linear velocity during handstand 
        return torch.square(self.base_lin_vel[:, 0])

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)


    def _reward_gait_one_hot(self):

        gait_outputs = self.actions[:,-self.cfg.control.add_cpg_actions:-3]
    
        # Calculate how "one-hot" the output is
        # Perfect one-hot would have one value at 1.0 and others at 0.0
        max_val = torch.max(gait_outputs, dim=1)[0]
        sum_others = torch.sum(gait_outputs, dim=1) - max_val
        one_hot_quality = max_val - sum_others
        
        # Reward clear, distinct gait selections
        reward = torch.sigmoid(one_hot_quality * 5.0)  # Scale factor to make it more sensitive
        
        return reward

