import gym
from gym import spaces
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from solvers import transient
#from solvers.transient import *
from equations.flow import FlowConfig 
import equations.base as base
import equations.utils as utils 
from jax import *

import gym
from gym import spaces
import numpy as np
action_dim = 4
observation_dim = 4
episode_length = 10
flow = FlowConfig(grid_size=(256, 256))

# Physical parameters 
flow.Re = 250
flow.k = 4
dt = 0.001
save_time = 1
end_time = 10
n, m = 256, 256
total_seconds = 100
k1, k2, k3, k4 = 4, 5, 6, 7

class KolmogorovFlowEnv(gym.Env):
    
    def __init__(self):
        
        super(KolmogorovFlowEnv, self).__init__()
        # Observation space: energy of 4 modes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim*episode_length,), dtype=np.float32)
        
        # Action space: discrete actions, for increasing TKE the values can go from 0 to 1
        self.action_space = spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)

        self.episode_length = episode_length
        self.current_step = 0
        self.trajectory = None
        self.total_seconds = total_seconds
        self.state =  flow.initialize_state()
        
        self.total_steps = int(end_time // dt)
        self.step_to_save = int(save_time // dt) 

    def reset(self):
        
        equation = base.PseudoSpectralNavierStokes2D(flow)
        step_fn = transient.RK4_CN(equation, dt)
        
        _, self.trajectory = transient.iterative_func(step_fn, self.state, self.total_steps, self.step_to_save)
                
        return self._get_observation()

    def step(self, action):
        self._apply_action(action)
        
        observation = self._get_observation()
        
        reward = self._compute_reward(action)
        
        self.current_step += self.episode_length
        done = self.current_step >= self.total_seconds
        print("Current seconds: ", self.current_step)
        
        info = {}
        if done: 
            utils.create_animation(self.trajectory, 'testing_env', 0.06)
        return observation, reward, done, info

    def _apply_action(self, action): 
        x, y = flow.create_mesh()
        a1, a2, a3, a4 = action[0], action[1], action[2], action[3]
        def control_fn(y):
            return (a1*jnp.sin(k1*y) + a2*jnp.sin(k2*y) + a3*jnp.sin(k3*y) + a4*jnp.sin(k4*y), jnp.zeros_like(y))

        flow.control_function = control_fn(y)
        equation = base.PseudoSpectralNavierStokes2D(flow)
        step_fn = transient.RK4_CN(equation, dt)
        _, self.trajectory = transient.iterative_func(step_fn, self.state, self.total_steps, self.step_to_save)
                   
    
    def _get_observation(self):
        
        def calculate_energy(state):
            e1, e2, e3, e4 = self._calculate_mode_energy(state, k1), self._calculate_mode_energy(state, k2), self._calculate_mode_energy(state, k3), self._calculate_mode_energy(state, k4)
            return jnp.array([e1, e2, e3, e4])

        def scan_fn(carry, state):
            energies = calculate_energy(state)
            return carry, energies
        
        current_state = self.trajectory
        _, all_energy = lax.scan(scan_fn, None, current_state)
        
        return all_energy.flatten()

    def _calculate_mode_energy(self, state, k):
        # Calculate the energy of a mode
        kx, ky = flow.create_fft_mesh()
        uhat, vhat = utils.compute_velocity_fft(state, kx, ky)
        mode_energy = utils.compute_energy_mode(uhat, vhat, k, 0, n, m)
     
        # Implement the calculation here
        return mode_energy
    
    def _compute_avg_TKE(self):
        TKE = []
        current_state = self.trajectory
        # Calculate the energy of the first 4 modes from the state
        kx, ky = flow.create_fft_mesh()
        for timestep in current_state:
            tke_t = utils.compute_tke(timestep, kx, ky, n)
            TKE.append(tke_t)
        
        return jnp.mean(jnp.array(TKE))

    def _compute_reward(self, action):
        # Compute the reward based on the current state
        tke = self._compute_avg_TKE()
        a1, a2, a3, a4 = action[0], action[1], action[2], action[3]
        reward = -(tke + (a1 + a2 + a3 + a4))  
        return reward

    def render(self, mode='human'):
        # Render the environment (optional)
        pass

    def close(self):
        # Clean up resources (optional)
        pass