# License: see [LICENSE]

import torch
import numpy as np 

MU_LOW = 1
MU_UPP = 2 
MAX_STEP_LEN = 0.23 

MAX_GROUND_PENETRATION = 0.015
MIN_GROUND_PENETRATION = 0 
MAX_STEP_HEIGHT = 0.12
MIN_STEP_HEIGHT = 0.02 
MAX_BODY_HEIGHT = 0.35 
MIN_BODY_HEIGHT = 0.18

MAX_FOOT_X_OFFSET =  0.08 
MIN_FOOT_X_OFFSET = -0.08

class CPG():
    def __init__(self,
          mu=1,
          gait="TROT",
          couple=True,
          coupling_strength=10,
          time_step=0.001,
          robot_height=0.3,        # in nominal case (standing)
          des_step_len=0.03,       # desired step length
          ground_clearance=0.05,   # foot swing height 
          ground_penetration=0.01, # foot stance penetration into ground
          x_offset=0,              # x offset 
          num_envs=1,
          num_oscillators=4,
          device=None,
          rl_task_string=None,
          test_time=False,
          add_cpg_actions=0,
        ):
        self.gaits = ["LSWALK", "DSWALK", "AMBLE", "TROT", "PACE", "PRONK", "HALFBOUND", "BOUND", "CANTER", "TGALLOP", "RGALLOP"]
        self._test_time = test_time
        # self.gaits = ["TEST1"]
        self.add_cpg_actions = add_cpg_actions
        self._rl_task_string = rl_task_string
        self.num_oscillators = num_oscillators 
        self._device = device 
        self.X = torch.zeros(num_envs,2,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self.X_dot = torch.zeros(num_envs,2,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self.d2X = torch.zeros(num_envs,1,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self.num_envs = num_envs
        self._mu = torch.ones(num_envs,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)

        # Y
        self.Y = torch.zeros(num_envs,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self.Y_dot = torch.zeros(num_envs,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self.d2Y = torch.zeros(num_envs,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self.y = torch.zeros(num_envs,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)

        # if "PSI" in rl_task_string:
        self.PSI = torch.zeros(num_envs,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self.PSI_dot = torch.zeros(num_envs,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)

        # PHI matrix 4x4 for each element 
        self.PHI_FULL = torch.zeros(num_envs,self.num_oscillators,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)

        self._robot_height = torch.zeros(num_envs,1,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self._ground_clearance = torch.zeros(num_envs,1,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self._ground_penetration = torch.zeros(num_envs,1,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)
        self._x_offset = torch.zeros(num_envs,1,self.num_oscillators,dtype=torch.float, device=device, requires_grad=False)

        self._robot_height[:,0,:]       = robot_height 
        self._ground_clearance[:,0,:]   = ground_clearance 
        self._ground_penetration[:,0,:] = ground_penetration
        self._x_offset[:,0,:]           = x_offset

        self._couple = couple
        self._coupling_strength = coupling_strength
        self._dt = time_step
        self._set_gait(gait)

        self._psi_max = 10
        self._a  = 150
        self._ay = 150
        self._dr_factor = 30
        self._dy_factor = 30
        self._omega_max = 45
        self._omega_min = 1 

        # set oscillator initial conditions  
        self.X[:,0,:] = 1 #torch.rand(num_envs,4,device=self._device) * .1
        self.X[:,1,:] = self.PHI[0,:] 

        self._des_step_len = des_step_len
        self.PHI_gaits = {}
        self.curr_gait_nums = torch.zeros(num_envs,dtype=torch.float, device=device, requires_grad=False)

    def resample_gaits(self, env_ids, gait_num=-1):
        if gait_num == -1:
            gait_num = np.random.randint(len(self.gaits))
            gait = self.gaits[gait_num]
            self.curr_gait_nums[env_ids] = gait_num
        else:
            gait = self.gaits[gait_num]
            self.curr_gait_nums[env_ids] = gait_num
        if gait == "LSWALK":
            self.PHI_FULL[env_ids,:,:] = self.PHI_lswalk
        elif gait == "DSWALK":
            self.PHI_FULL[env_ids,:,:] = self.PHI_dswalk
        elif gait == "AMBLE":
            self.PHI_FULL[env_ids,:,:] = self.PHI_amble
        elif gait == "TROT":
            self.PHI_FULL[env_ids,:,:] = self.PHI_trot
        elif gait == "PACE":
            self.PHI_FULL[env_ids,:,:] = self.PHI_pace
        elif gait == "PRONK":
            self.PHI_FULL[env_ids,:,:] = self.PHI_pronk
        elif gait == "HALFBOUND":
            self.PHI_FULL[env_ids,:,:] = self.PHI_halfbound
        elif gait == "BOUND":
            self.PHI_FULL[env_ids,:,:] = self.PHI_bound
        elif gait == "CANTER":
            self.PHI_FULL[env_ids,:,:] = self.PHI_canter
        elif gait == "TGALLOP":  
            self.PHI_FULL[env_ids,:,:] = self.PHI_tgallop
        elif gait == "RGALLOP":  
            self.PHI_FULL[env_ids,:,:] = self.PHI_rgallop
        elif gait == "TEST1":
            self.PHI_FULL[env_ids,:,:] = self.PHI_test1
        elif gait == "TEST2":
            self.PHI_FULL[env_ids,:,:] = self.PHI_test2
        else:
            raise ValueError( gait + 'not implemented.')

    def set_gait_params(self,env_ids,robot_height,ground_clearance,x_offset):
        self._robot_height[env_ids,0,:]       = robot_height 
        self._ground_clearance[env_ids,0,:]   = ground_clearance 
        self._x_offset[env_ids,0,:]           = x_offset
    
    def reset(self,env_ids):
        self._mu[env_ids,:] = 1
        self.X[env_ids,0,:] = 1 
        self.X[env_ids,1,:] = self.PHI_FULL[env_ids,0,:] 
        self.X_dot[env_ids,:,:] = 0.
        if "PSI" in self._rl_task_string:
            self.PSI[env_ids,:] = 0.
            self.PSI_dot[env_ids,:] = 0.
        if "Y" in self._rl_task_string:
            self.X[env_ids,0,:] = 1.5
            self.Y[env_ids,:] = 1.5 
            self.Y_dot[env_ids,:] = 0.

        if not self._test_time and self.add_cpg_actions == 0: 
            self._robot_height[env_ids,0,:] = MIN_BODY_HEIGHT + (MAX_BODY_HEIGHT - MIN_BODY_HEIGHT) * torch.rand(len(env_ids),1, device=self._device)
            self._ground_clearance[env_ids,0,:] = MIN_STEP_HEIGHT + (MAX_STEP_HEIGHT - MIN_STEP_HEIGHT) * torch.rand(len(env_ids),1, device=self._device)
            self._ground_penetration[env_ids,0,:] = MIN_GROUND_PENETRATION + (MAX_GROUND_PENETRATION - MIN_GROUND_PENETRATION) * torch.rand(len(env_ids),1, device=self._device)
            self._x_offset[env_ids,0,:] = MIN_FOOT_X_OFFSET + (MAX_FOOT_X_OFFSET - MIN_FOOT_X_OFFSET) * torch.rand(len(env_ids),1, device=self._device)
 
    
    def _set_gait(self,gait,printGait=True):
        # Assumes leg order: Front Right (FR), Front Left (FL), Hind Right (HR), Hind Left (HL)
        device = self._device 

        # lateral sequence walk
        lswalk = -torch.tensor([[     0,    np.pi, -np.pi/2, np.pi/2 ],
                              [   np.pi,       0,  np.pi/2,-np.pi/2 ],
                              [ np.pi/2,-np.pi/2,        0,   np.pi ],
                              [-np.pi/2, np.pi/2,    np.pi,       0 ]],dtype=torch.float, device=device, requires_grad=False) 
        self.PHI_lswalk = lswalk

        # diagonal sequence walk 
        dswalk = -torch.tensor([[     0,   np.pi,  np.pi/2,-np.pi/2 ],
                              [   np.pi,       0, -np.pi/2, np.pi/2 ],
                              [-np.pi/2, np.pi/2,        0,   np.pi ],
                              [ np.pi/2,-np.pi/2,    np.pi,       0 ]],dtype=torch.float, device=device, requires_grad=False) 
        self.PHI_dswalk = dswalk

        # amble (this is a diagonal amble )
        amble = torch.tensor([[   0,  -0.5, -0.2, -0.7 ],
                              [  0.5,    0,  0.3, -0.2 ],
                              [  0.2, -0.3,    0, -0.5 ],
                              [  0.7,  0.2,  0.5,    0 ]],dtype=torch.float, device=device, requires_grad=False) 
        self.PHI_amble = 2*np.pi * amble

        # trot 
        trot = torch.tensor([[ 0, 1, 1, 0 ],
                            [-1, 0, 0,-1 ],
                            [-1, 0, 0,-1 ],
                            [ 0, 1, 1, 0 ]],dtype=torch.float, device=device, requires_grad=False) 
        trot = np.pi * trot
        self.PHI_trot = trot 

        # pace 
        pace = torch.tensor([[ 0, 1, 0, 1 ],
                              [-1, 0,-1, 0 ],
                              [ 0, 1, 0, 1 ],
                              [-1, 0,-1, 0 ]],dtype=torch.float, device=device, requires_grad=False) 
        pace = np.pi * pace
        self.PHI_pace = pace

        # pronk 
        pronk = torch.tensor([[ 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0 ]],dtype=torch.float, device=device, requires_grad=False) 
        self.PHI_pronk = pronk

        # half-bound
        halfbound = torch.tensor([[   0, -0.2, -0.6, -0.6 ],
                                 [  0.2,    0, -0.4, -0.4 ],
                                 [  0.6,  0.4,    0,    0 ],
                                 [  0.6,  0.4,    0,    0 ]],dtype=torch.float, device=device, requires_grad=False) 
        self.PHI_halfbound = 2*np.pi * halfbound

        # bound 
        bound = torch.tensor([[ 0, 0, 1, 1 ],
                              [ 0, 0, 1, 1 ],
                              [-1,-1, 0, 0 ],
                              [-1,-1, 0, 0 ]],dtype=torch.float, device=device, requires_grad=False) 
        bound = -np.pi*bound
        self.PHI_bound = bound 

        # canter
        canter = torch.tensor([[   0,  0.7,  0.3,    0 ],
                              [ -0.7,    0, -0.4, -0.7 ],
                              [ -0.3,  0.4,    0, -0.3 ],
                              [    0,  0.7,  0.3,    0 ]],dtype=torch.float, device=device, requires_grad=False) 
        self.PHI_canter = 2*np.pi * canter

        # transverse gallop 
        tgallop = torch.tensor([[  0, -0.1, -0.5, -0.6 ],
                              [  0.1,    0, -0.4, -0.5 ],
                              [  0.5,  0.4,    0, -0.1 ],
                              [  0.6,  0.5,  0.1,    0 ]],dtype=torch.float, device=device, requires_grad=False) 
        self.PHI_tgallop = 2*np.pi * tgallop
        
        # rotary gallop
        rgallop = torch.tensor([[  0,  0.1, -0.4, -0.5 ],
                              [ -0.1,    0, -0.5, -0.6 ],
                              [  0.4,  0.5,    0, -0.1 ],
                              [  0.5,  0.6,  0.1,    0 ]],dtype=torch.float, device=device, requires_grad=False) 
        self.PHI_rgallop = 2*np.pi * rgallop

        # Test 1 (3 beat )
        test1 = torch.tensor([[  0, 1,  0,  0 ],
                              [ -1, 0, -1, -1 ],
                              [  0, 1,  0,  0 ],
                              [  0, 1,  0,  0 ]],dtype=torch.float, device=device, requires_grad=False) 
        test1 = -np.pi*test1
        self.PHI_test1 = test1

        test2 = torch.tensor([[  0, 1,  0,  0 ],
                              [  0, 1,  0,  0 ],
                              [  0, 1,  0,  0 ],
                              [ -1, 0, -1, -1 ]],dtype=torch.float, device=device, requires_grad=False) 
        test2 = -np.pi*test2
        self.PHI_test2 = test2

        if printGait:
            print(gait)
        if gait == "LSWALK":
            self.PHI = self.PHI_lswalk
        elif gait == "DSWALK":
            self.PHI = self.PHI_dswalk
        elif gait == "AMBLE":
            self.PHI = self.PHI_amble
        elif gait == "TROT":
            self.PHI = self.PHI_trot
        elif gait == "PACE":
            self.PHI = self.PHI_pace
        elif gait == "PRONK":
            self.PHI = self.PHI_pronk
        elif gait == "HALFBOUND":
            self.PHI = self.PHI_halfbound
        elif gait == "BOUND":
            self.PHI = self.PHI_bound
        elif gait == "CANTER":
            self.PHI = self.PHI_canter
        elif gait == "TGALLOP":
            self.PHI = self.PHI_tgallop
        elif gait == "RGALLOP":
            self.PHI = self.PHI_rgallop
        elif gait == "TEST1":
            self.PHI = self.PHI_test1
        else:
            raise ValueError( gait + 'not implemented.')

        self.PHI_GAITS = {
            "LSWALK": self.PHI_lswalk,
            "DSWALK": self.PHI_dswalk,
            "AMBLE": self.PHI_amble,
            "TROT": self.PHI_trot,
            "PACE": self.PHI_pace,
            "PRONK": self.PHI_pronk,
            "HALFBOUND": self.PHI_halfbound,
            "BOUND": self.PHI_bound, 
            "CANTER": self.PHI_canter,
            "TGALLOP":self.PHI_tgallop,
            "RGALLOP": self.PHI_rgallop,
        }
        self.gaits = ["LSWALK", "DSWALK", "AMBLE", "TROT", "PACE", "PRONK", "HALFBOUND", "BOUND", "CANTER", "TGALLOP", "RGALLOP"]
        self.gait_matrices = torch.stack([torch.tensor(self.PHI_GAITS[gait]) for gait in self.gaits])


    def _scale_helper(self, action, lower_lim, upper_lim):
        new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        new_a = torch.clip(new_a, lower_lim, upper_lim)
        return new_a

    def get_cpg_rl_commands(self,actions):
        """ Map actions to CPG signals, return desired joint angles. """
        a = torch.clip(actions, -1, 1)
        if self.add_cpg_actions == 3:
            self._robot_height[:,0,:] = self._scale_helper(a[:,-3],MIN_BODY_HEIGHT,MAX_BODY_HEIGHT).unsqueeze(1).expand(-1, 4)
            self._ground_clearance[:,0,:] = self._scale_helper(a[:,-2],MIN_STEP_HEIGHT,MAX_STEP_HEIGHT).unsqueeze(1).expand(-1, 4)
            self._x_offset[:,0,:] = self._scale_helper(a[:,-1],MIN_FOOT_X_OFFSET,MAX_FOOT_X_OFFSET).unsqueeze(1).expand(-1, 4)

        self._mu = self._scale_helper(a[:,:4],MU_LOW**2,MU_UPP**2) 
        self._omega_residuals = self._scale_helper(a[:,4:8],self._omega_min,self._omega_max) 
        if "PSI" in self._rl_task_string:
            self._psi_residuals = self._scale_helper(a[:,8:12],-self._psi_max,self._psi_max)

        if "Y" in self._rl_task_string:
            self._mu_y = self._scale_helper(a[:,8:12],MU_LOW**2,MU_UPP**2) 

        self._integrate_salamander_equations()

        # map CPG variables to Cartesian foot xyz positions
        r = torch.clip(self.X[:,0,:],MU_LOW,MU_UPP) 
        r = MAX_STEP_LEN * (r - MU_LOW) / (MU_UPP - MU_LOW)

        if "PSI" in self._rl_task_string:
            x = -r * torch.cos(self.X[:,1,:]) * torch.cos(self.PSI)
            y = -r * torch.cos(self.X[:,1,:]) * torch.sin(self.PSI)
        elif "Y" in self._rl_task_string:
            y = torch.clip(self.Y[:,:],MU_LOW,MU_UPP) 
            y = 2* ( (y - MU_LOW) / (MU_UPP - MU_LOW) ) - 1
            y = MAX_STEP_LEN * y * torch.cos(self.X[:,1,:])
            x = torch.clip(self.X[:,0,:],MU_LOW,MU_UPP) 
            x =  2* ( (x - MU_LOW) / (MU_UPP - MU_LOW) ) - 1
            x = - MAX_STEP_LEN * x * torch.cos(self.X[:,1,:])
        else:
            x = -r * torch.cos(self.X[:,1,:]) + self._x_offset[:,0,:]
            y = self.y
        # z  
        z = torch.where(torch.sin(self.X[:,1,:]) > 0, 
                        -self._robot_height[:,0,:] + self._ground_clearance[:,0,:]   * torch.sin(self.X[:,1,:]),# swing)
                        -self._robot_height[:,0,:] + self._ground_penetration[:,0,:] * torch.sin(self.X[:,1,:]))
        
        return x, y, z

    def _integrate_salamander_equations(self):
        # bookkeeping - save copies of current CPG states 
        device = self._device 
        X_dot = self.X_dot.clone()
        d2X = self.d2X.clone()
        Y_dot = self.Y_dot.clone()
        d2Y = self.d2Y.clone()
        _a  = self._a  
        _ay = self._ay 
        dt = 0.001
        for _ in range(int(self._dt/dt)):
            d2X_prev = self.d2X.clone()
            X_dot_prev = self.X_dot.clone()
            X = self.X.clone()
            d2X = (_a * ( _a/4 * (torch.sqrt(self._mu) - X[:,0,:]) - X_dot_prev[:,0,:] )).unsqueeze(1)
            X_dot[:,1,:] = self._omega_residuals

            if self._couple:
                if "Y" in self._rl_task_string:
                    Y = self.Y.clone()
                    for i in range(self.num_oscillators):
                        X_dot[:,1,i] += torch.sum(((X[:,0,:] + Y)/2) * self._coupling_strength * torch.sin(X[:,1,:] - X[:,1,i].unsqueeze(1) - self.PHI[i,:] ), 1)
                else:
                    for i in range(self.num_oscillators):
                        X_dot[:,1,i] += torch.sum(X[:,0,:] * self._coupling_strength * torch.sin(X[:,1,:] - X[:,1,i].unsqueeze(1) - self.PHI_FULL[:,i,:] ), 1)

            # integrate amplitude - get R_dot 
            X_dot[:,0,:] = X_dot_prev[:,0,:] + (d2X_prev[:,0,:] + d2X[:,0,:]) * dt / 2
            self.X = X + (X_dot_prev + X_dot) * dt / 2 
            self.X_dot = X_dot
            self.d2X = d2X 
            # mod phase variables to keep between 0 and 2pi
            self.X[:,1,:] = torch.remainder(self.X[:,1,:], (2*np.pi))

            if "PSI" in self._rl_task_string:
                PSI_dot = self.PSI_dot.clone()
                PSI_dot_prev = self.PSI_dot.clone()
                PSI_dot[:,:] = self._psi_residuals
                self.PSI = self.PSI + (PSI_dot_prev + PSI_dot) * dt / 2
                self.PSI_dot = PSI_dot 
                self.PSI = torch.remainder(self.PSI, (2*np.pi)) # mod to keep between 0 and 2pi

            if "Y" in self._rl_task_string:
                d2Y_prev = self.d2Y.clone()
                Y_dot_prev = self.Y_dot.clone()
                Y = self.Y.clone()
                d2Y = (_ay * ( _ay/4 * (torch.sqrt(self._mu_y) - Y) - Y_dot_prev ))
                Y_dot = Y_dot_prev + (d2Y_prev + d2Y) * dt / 2
                self.Y = Y + (Y_dot_prev + Y_dot) * dt / 2 
                self.Y_dot = Y_dot

 