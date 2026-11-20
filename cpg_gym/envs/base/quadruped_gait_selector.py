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
from cpg_gym.envs.base.quadruped_gaits import *
from cpg_gym.envs.base.cpg import CPG
from datetime import datetime 

curr_time = datetime.now().strftime("%m%d%y%H%M%S")

class MultiGaitPolicyManager:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.policies = {}
        self.gait_names = []
        self.gait_to_idx = {}
        self._load_gait_policies()
    
    def _load_gait_policies(self):
        """Load 1st stage policy from the configuration."""
        from rsl_rl.modules import ActorCritic
        if not self.policies and hasattr(self.cfg.pretrained, 'nn_path'):
            try:
                actor_critic = ActorCritic(
                    num_actor_obs=self.cfg.pretrained.num_observations,
                    num_critic_obs=self.cfg.pretrained.num_observations,
                    num_actions=self.cfg.pretrained.num_actions,
                    actor_hidden_dims=self.cfg.pretrained.policy.actor_hidden_dims,
                    critic_hidden_dims=self.cfg.pretrained.policy.critic_hidden_dims,
                    activation=self.cfg.pretrained.policy.activation,
                    init_noise_std=self.cfg.pretrained.policy.init_noise_std
                ).to(self.device)
                
                nn_path = self.cfg.pretrained.nn_path.format(CPG_GYM_ROOT_DIR=CPG_GYM_ROOT_DIR)
                state_dict = torch.load(nn_path, map_location=self.device)
                if 'model_state_dict' in state_dict:
                    actor_critic.load_state_dict(state_dict['model_state_dict'])
                else:
                    actor_critic.load_state_dict(state_dict)
                
                actor_critic.eval()
                self.policies['SINGLE'] = actor_critic
                self.gait_names = ['SINGLE']
                self.gait_to_idx['SINGLE'] = 0
                
            except Exception as e:
                print(f"Error loading policy: {e}")
                raise
    
    def get_policy(self, gait_name):
        """Get the policy for a specific gait."""
        if 'SINGLE' in self.policies:
            return self.policies['SINGLE']
        else:
            raise ValueError(f"No policy available for gait: {gait_name}")
          

class QuadrupedGaitSelector(QuadrupedGaits):
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
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # measure heights ahead of time, needed for pretrained policy 
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.preclip_actions = actions

        # set gait based on action 
        a = torch.clip(self.actions * self.cfg.control.action_scale, -1, 1)  
        # set the gait based on the action 
        num_gaits = len(self._cpg.gaits)
        if self.num_actions == 1: 
            # Scale gait_nums from float (-1 to 1) to integer indices (0 to num_gaits-1), this is correct
            gait_nums_scaled = ((a + 1) / 2 * (num_gaits - 1)).long().squeeze(-1)  # Remove extra dimension
        else:
            gait_nums_scaled = torch.argmax(a[:,0:num_gaits], dim=1)
        
        # Create an array to store the gait matrices (e.g., trot, pace, bound) for each env_id
        gait_matrices = self._cpg.gait_matrices 
        # Use scaled gait_nums to select the appropriate gait matrix for each environment
        selected_gait_matrices = gait_matrices[gait_nums_scaled]
        self._cpg.curr_gait_nums[:] = gait_nums_scaled # num_gaits = len(self.gaits)
        # Now, broadcast the selected matrices to the correct environment ids
        self._cpg.PHI_FULL[:, :, :] = selected_gait_matrices

        # Initialize accumulators
        self.energy_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.stability_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.lin_vel_z_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.lin_vel_y_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.ang_vel_x_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.ang_vel_y_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.ang_vel_z_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.dof_acc_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.tracking_lin_vel_x_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.bad_tracking_lin_vel_x_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.torques_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.feet_contact_forces_accumulator = torch.zeros(self.num_envs, device=self.device)

        if not self.headless:
            self.set_robot_colors(self.env_indices,self._cpg.curr_gait_nums.to(dtype=torch.int))

        # step physics and render each frame
        self.render()

        record_video = False # note play.py should not have active flags
        if record_video:
            root_path = os.path.join('logs', curr_time, 'frames')
            if not os.path.exists( root_path ):
                os.makedirs( root_path )

        # cpg_action = torch.zeros(self.num_envs, self.cfg.pretrained.num_actions, device=self.device)
        for _ in range(self.cfg.control.decimation):
            if (_ % self.cfg.control.cpg_decimation) == 0:
                self.render()
                if record_video:
                    filename = os.path.join(root_path, f"{self.image_step_counter:04d}.png")
                    self.gym.write_viewer_image_to_file(self.viewer, filename)
                    self.image_step_counter += 1
                    robot_pos = np.array(self.root_states[0,0:3].cpu())
                    camera_position = robot_pos + np.array([0,-2.,0.25])
                    self.set_camera(camera_position, robot_pos)

                # this is inspired from refresh_physics_between_steps, otherwise we go too long and physics not accurate any more 
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)
                self.base_quat[:] = self.root_states[:, 3:7]
                self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
                self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                self.measured_heights = self._get_heights()
                
                # compute obs 
                obs_buf_pretrained = self.get_observations_pretrained()
                # Get pretrained actions using gait-specific policies
                pretrained_actions = self._get_pretrained_actions(obs_buf_pretrained, clip_actions)
                
                # scale action appropriately as above
                self.actions_pretrained = torch.clip(pretrained_actions, -clip_actions, clip_actions).to(self.device)

            self.torques = self._compute_torques(self.actions_pretrained.detach()).view(self.torques.shape)


            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            if self.cfg.terrain.parkour:
                self.set_vel_parkour()

            self._refresh_physics_between_steps()
            instantaneous_power = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
            self.energy_accumulator += instantaneous_power / self.cfg.control.decimation
            instantaneous_stability = torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1)
            self.stability_accumulator += instantaneous_stability  / self.cfg.control.decimation
            self.lin_vel_z_accumulator += torch.square(self.base_lin_vel[:, 2]) / self.cfg.control.decimation
            self.lin_vel_y_accumulator += torch.square(self.base_lin_vel[:, 1]) / self.cfg.control.decimation
            self.ang_vel_x_accumulator += torch.square(self.base_ang_vel[:, 0]) / self.cfg.control.decimation
            self.ang_vel_y_accumulator += torch.square(self.base_ang_vel[:, 1]) / self.cfg.control.decimation
            self.ang_vel_z_accumulator += torch.square(self.base_ang_vel[:, 2]) / self.cfg.control.decimation
            self.dof_acc_accumulator += torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1) / self.cfg.control.decimation
            inst_lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
            self.tracking_lin_vel_x_accumulator += torch.exp(-inst_lin_vel_error/self.cfg.rewards.tracking_sigma) / self.cfg.control.decimation
            self.bad_tracking_lin_vel_x_accumulator += inst_lin_vel_error / self.cfg.control.decimation
            self.last_dof_vel[:] = self.dof_vel[:]
            self.torques_accumulator += torch.sum(torch.square(self.torques), dim=1) / self.cfg.control.decimation
            self.feet_contact_forces_accumulator += (torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)) / self.cfg.control.decimation
        
        self.post_physics_step()


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

    def _refresh_physics_between_steps(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def get_observations(self):
        if self.cfg.env.num_stacked_observations > 1:
            stacked_obs = self.obs_buf_stacked[:,:,0]
            for i in range(1,self.cfg.env.num_stacked_observations):
                stacked_obs = torch.cat((stacked_obs, self.obs_buf_stacked[:,:,i]), dim=-1)
            return stacked_obs
        else:
            return self.obs_buf

    def get_observations_pretrained(self):
        self.compute_cpg_obs()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf_pretrained = torch.clip(self.obs_buf_pretrained, -clip_obs, clip_obs)

        if self.cfg.pretrained.num_stacked_observations > 1:
            stacked_obs = self.obs_buf_stacked_pretrained[:,:,0]
            for i in range(1,self.cfg.pretrained.num_stacked_observations):
                stacked_obs = torch.cat((stacked_obs, self.obs_buf_stacked_pretrained[:,:,i]), dim=-1)
            return stacked_obs
        else:
            return self.obs_buf_pretrained
    

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
        if not self.cfg.terrain.test_time:
            self._resample_commands(env_ids)
        self._resample_gains(env_ids)

        if self.cfg.env.num_stacked_observations > 1:
            self.obs_buf_stacked[env_ids,:,:] = 0.
            self.obs_buf_stacked_pretrained[env_ids,:,:] = 0.

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
    
    def _get_gait_obs_component(self):
        return ((self._cpg.curr_gait_nums[:] - 5) / 3).unsqueeze(-1)

    def compute_cpg_obs(self):
        if "CPG" in self.cfg.control.control_type:
            self.obs_buf_pretrained = torch.cat(( 
                                    self.base_lin_vel * self.obs_scales.lin_vel,  
                                    self.base_ang_vel  * self.obs_scales.ang_vel, 
                                    wrap_to_pi(get_rpy_tensor(self.base_quat)), 
                                    self.commands[:, :3] * self.commands_scale,  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, 
                                    self.dof_vel * self.obs_scales.dof_vel, 
                                    self.actions_pretrained, 
                                    self.base_quat,
                                    (self._cpg.X[:,0,:] - ((MU_UPP+ MU_LOW) / 2)) * 2,  
                                    (self._cpg.X[:,1,:] - np.pi) * 1/np.pi, 
                                    self._cpg.X_dot[:,0,:] *  1/self._cpg._dr_factor, 
                                    (self._cpg.X_dot[:,1,:]) / self._cpg._omega_max, 
                                    self._get_gait_obs_component()
                                    ),dim=-1)
            if "STYLE" in self.cfg.env.obs_space:
               self.obs_buf_pretrained = torch.cat(( 
                                    self.base_lin_vel * self.obs_scales.lin_vel,  
                                    self.base_ang_vel  * self.obs_scales.ang_vel, 
                                    wrap_to_pi(get_rpy_tensor(self.base_quat)), 
                                    self.commands[:, :3] * self.commands_scale,   
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, 
                                    self.dof_vel * self.obs_scales.dof_vel, 
                                    self.actions_pretrained,   
                                    self.base_quat,   
                                    (self._cpg.X[:,0,:] - ((MU_UPP+ MU_LOW) / 2)) * 2,  
                                    (self._cpg.X[:,1,:] - np.pi) * 1/np.pi,   
                                    self._cpg.X_dot[:,0,:] *  1/self._cpg._dr_factor,
                                    (self._cpg.X_dot[:,1,:]) / self._cpg._omega_max, 
                                    self._get_gait_obs_component(),
                                    self._cpg._robot_height[:,0,0].unsqueeze(-1) / MAX_BODY_HEIGHT,
                                    self._cpg._ground_clearance[:,0,0].unsqueeze(-1) / MAX_STEP_HEIGHT,
                                    self._cpg._x_offset[:,0,0].unsqueeze(-1) / MIN_FOOT_X_OFFSET,
                                    ),dim=-1)
            
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf_pretrained = torch.cat((self.obs_buf_pretrained, heights), dim=-1)

        if self.add_noise: 
            if self.cfg.pretrained.num_stacked_observations <= 1:
                self.obs_buf_pretrained += (2 * torch.rand_like(self.obs_buf_pretrained) - 1) * self.noise_scale_vec_pretrained
            else:
                self.obs_buf_pretrained += (2 * torch.rand_like(self.obs_buf_pretrained) - 1) * self.noise_scale_vec_pretrained[0:int(self.cfg.pretrained.num_observations / self.cfg.pretrained.num_stacked_observations)]

        if self.cfg.pretrained.num_stacked_observations > 1:
            self.obs_buf_stacked_pretrained[:,:,1:] = self.obs_buf_stacked_pretrained[:,:,:-1].clone()  # shift older frames back one slot
            self.obs_buf_stacked_pretrained[:,:,0] = self.obs_buf_pretrained


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
            
        else: 
            if self.cfg.env.obs_space == "FULL":
                self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        self.commands[:, :3] * self.commands_scale,
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        self.base_quat,
                                        (self.contact_forces[:, self.feet_indices, 2] > 1.) * 2 - 1,  
                                        ),dim=-1)

        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            if self.cfg.env.num_stacked_observations <= 1:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            else:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[0:int(self.cfg.env.num_observations / self.cfg.env.num_stacked_observations)]

        # stack observations if needed 
        if self.cfg.env.num_stacked_observations > 1:
            self.obs_buf_stacked[:,:,1:] = self.obs_buf_stacked[:,:,:-1].clone()  # shift older frames back one slot
            self.obs_buf_stacked[:,:,0] = self.obs_buf

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        if not self.cfg.terrain.test_time:
            self._resample_commands(env_ids)

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
            torques = self.p_gains*(torques - self.dof_pos) - self.d_gains*self.dof_vel 

        elif control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.cfg.domain_rand.motor_latency:
            torques = self.motor_latency * torques + (1 - self.motor_latency) * self.torques

        return torch.clip(torques, -self.torque_limits, self.torque_limits)



    def _get_noise_scale_vec_pretrained(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_vec = torch.zeros(self.cfg.pretrained.num_observations,dtype=torch.float,device=self.device)
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
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.foot_pos = torch.zeros_like(self.dof_pos)
        self.foot_vel = torch.zeros_like(self.dof_vel)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.image_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1)) 
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
        # ?self.motor_latency = 0
        if self.cfg.domain_rand.motor_latency:
            # self.motor_latency = np.random.uniform(self.cfg.domain_rand.motor_latency_factor_min,1)
            self.motor_latency = torch_rand_float(self.cfg.domain_rand.motor_latency_factor_min, 1, (self.num_envs, 1), device=self.device)

        if self.cfg.env.num_stacked_observations > 1:
            self.obs_buf_stacked = torch.zeros(self.num_envs, int(self.cfg.env.num_observations / self.cfg.env.num_stacked_observations), 
                                                self.cfg.env.num_stacked_observations, 
                                                dtype=torch.float,
                                                device=self.device, 
                                                requires_grad=False)

        
        # pretrained buffers
        self.noise_scale_vec_pretrained = self._get_noise_scale_vec_pretrained(self.cfg)
        self.actions_pretrained = torch.zeros(self.num_envs, self.cfg.pretrained.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_buf_stacked_pretrained = torch.zeros(self.num_envs, int(self.cfg.pretrained.num_observations / self.cfg.pretrained.num_stacked_observations), 
                                            self.cfg.pretrained.num_stacked_observations, 
                                            dtype=torch.float,
                                            device=self.device, 
                                            requires_grad=False)
        self.obs_buf_pretrained = torch.zeros(self.num_envs, self.cfg.pretrained.num_observations, device=self.device, dtype=torch.float,requires_grad=False)
        
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
                                        add_cpg_actions=self.cfg.control.add_cpg_actions,
                                        )

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
            # Track gait usage as categories (one per gait type)
            # At this point in _init_buffers, _cpg should already be initialized (it's initialized earlier)
            num_gaits = len(self._cpg.gaits) if hasattr(self, '_cpg') and hasattr(self._cpg, 'gaits') else 11
            self.cpg_gait = torch.zeros(self.num_envs, num_gaits, dtype=torch.float, device=self.device, requires_grad=False)
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
            self.vel_array_times = np.linspace(50,int(6300*2),32) #range(50,550,50)
            self.vel_array = np.linspace(0.1, 3.2, 32)
            self.cpg_save_arr = np.zeros((1, 66))
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

        # Load multi-gait pretrained policies
        if hasattr(self.cfg, 'pretrained'):
            self.policy_manager = MultiGaitPolicyManager(self.cfg, self.device)
            self.pretrained_policy = self.policy_manager.get_policy('SINGLE')
            print("Single policy mode enabled")


    #------------ reward functions----------------
    def _reward_energy(self):
        energy = self.energy_accumulator.clone()
        self.energy_accumulator.zero_()  # Reset for next step
        return energy

    def _reward_stability(self):
        stability = self.stability_accumulator.clone()
        self.stability_accumulator.zero_()  # Reset for next step
        return stability
    
    def _reward_lin_vel_z(self):
        vel_z = self.lin_vel_z_accumulator.clone()
        self.lin_vel_z_accumulator.zero_()  # Reset for next step
        return vel_z
    
    def _reward_lin_vel_y(self):
        vel_y = self.lin_vel_y_accumulator.clone()
        self.lin_vel_y_accumulator.zero_()  # Reset for next step
        return vel_y

    def _reward_ang_vel_x(self):
        vel_x = self.ang_vel_x_accumulator.clone()
        self.ang_vel_x_accumulator.zero_()  # Reset for next step
        return vel_x

    def _reward_ang_vel_y(self):
        vel_y = self.ang_vel_y_accumulator.clone()
        self.ang_vel_y_accumulator.zero_()  # Reset for next step
        return vel_y

    def _reward_ang_vel_z(self):
        vel_z = self.ang_vel_z_accumulator.clone()
        self.ang_vel_z_accumulator.zero_()  # Reset for next step
        return vel_z
    
    def _reward_dof_acc(self):
        dof_acc = self.dof_acc_accumulator.clone()
        self.dof_acc_accumulator.zero_()  # Reset for next step
        return dof_acc

    def _reward_tracking_lin_vel_x(self):
        tracking_lin_vel_x = self.tracking_lin_vel_x_accumulator.clone()
        self.tracking_lin_vel_x_accumulator.zero_()  # Reset for next step
        return tracking_lin_vel_x

    def _reward_bad_tracking_lin_vel_x(self):
        bad_tracking_lin_vel_x = self.bad_tracking_lin_vel_x_accumulator.clone()
        self.bad_tracking_lin_vel_x_accumulator.zero_()  # Reset for next step
        return bad_tracking_lin_vel_x

    def _reward_torques(self):
        torques = self.torques_accumulator.clone()
        self.torques_accumulator.zero_()  # Reset for next step
        return torques
    
    def _reward_feet_contact_forces(self):
        feet_contact_forces = self.feet_contact_forces_accumulator.clone()
        self.feet_contact_forces_accumulator.zero_()  # Reset for next step
        return feet_contact_forces

    #############################################################################################################
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_clip_penalty(self):
        # Penalize actions outside of clip_action range
        out_of_limits =  (self.preclip_actions - self.cfg.normalization.clip_actions) > 0.
        out_of_limits += (self.preclip_actions + self.cfg.normalization.clip_actions) < 0.
        return torch.sum(out_of_limits, dim=1)


    def _reward_gait_one_hot(self):
        # prev way 
        gait_outputs = self.actions[:,:len(self._cpg.gaits)]
        # Calculate how "one-hot" the output is
        # Perfect one-hot would have one value at 1.0 and others at 0.0
        max_val = torch.max(gait_outputs, dim=1)[0]
        sum_others = torch.sum(gait_outputs, dim=1) - max_val
        one_hot_quality = max_val - sum_others
        # Reward clear, distinct gait selections
        reward = torch.sigmoid(one_hot_quality * 5.0)  # Scale factor to make it more sensitive
        
        return reward


    def _get_pretrained_actions(self, obs_buf_pretrained, clip_actions):
        return self.pretrained_policy.act_inference(obs_buf_pretrained)


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
