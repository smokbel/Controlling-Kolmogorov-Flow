import sys, os
import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from jax import *
from typing import Any, Dict, Optional, Tuple, Union
from gym import spaces
from flax import struct
from jax import lax
from gymnax.environments import environment
from gymnax.environments import spaces

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)
from equations.base import * 
from equations.utils import *
from solvers import transient
from equations.flow import FlowConfig 

#################################################
# Physical parameters of Kolmogorov environment #
################################################# 

n,m = 64,64 # Gridsize (square Cartesian grid)
flow = FlowConfig(grid_size=(n, m)) # Initializing flow env 
flow.Re = 350 # Reynolds number 
flow.k = 4 # Forcing wavenumber
dt = 0.001 # Timestep 
save_time = 1 # Save every 'save_time' seconds (for visualization)
end_time = 5 # Actor time length (action applied for 'end_time' seconds)

##########################################
# Gymnax environment for Kolmogorov flow #
##########################################
@struct.dataclass
class EnvState(environment.EnvState):
    trajectory: Any
    time: Any
    terminal: Any
    new_init: Any 

@struct.dataclass
class EnvParams(environment.EnvParams):
    min_action: float = 0.0 # This needs to be hardcoded in the gymnax ClipAction wrapper to actually work.
    max_action: float = 1.0
    min_obs: Any = -jnp.inf
    max_obs: Any = jnp.inf
    dt: float = 0.001
    k1: int = 1
    k2: int = 2
    k3: int = 3
    k4: int = 4
    k5: int = 5
    k6: int = 6
    k7: int = 7
    k8: int = 8
    action_dim: int = 8
    obs_dim: int = 8
    action_time: int = end_time # Action length in seconds, same as flow solver end time 
    max_steps_in_episode: int = 100 # Episode length in seconds 
    
    
class KolmogorovFlow(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible  version of Kolmogorov Flow gym environment."""

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()
    
    def _compute_avg_TKE(self, state: EnvState,):
        TKE = []
        current_state = state.trajectory
        # Calculate the energy of the first 4 modes from the state
        kx, ky = flow.create_fft_mesh()
        for timestep in current_state:
            tke_t = compute_tke(timestep, kx, ky, n)
            TKE.append(tke_t)
        
        return jnp.mean(jnp.array(TKE))

    def _compute_reward(self, 
                        action: jnp.ndarray, state: EnvState):
        # Compute the reward based on the current state
        tke = self._compute_avg_TKE(state)
        #jax.debug.print("Sampled action in REWARD: {}", action)
        a1, a2, a3, a4, a5, a6, a7, a8 = action[0], action[1], action[2], action[3], action[4], action[5], action[6], action[7]
        reward = -((-1*tke) + 37*(a1+a2+a3+a4+a5+a6+a7+a8))  # use abs value of the actions 
        return reward
    
    def _apply_action(self, 
                      action: jnp.ndarray, params: EnvParams, state: EnvState) -> jnp.ndarray: 
        x, y = flow.create_mesh()
        #jax.debug.print("Sampled action in APPLY ACTION: {}", action)
        a1, a2, a3, a4, a5, a6, a7, a8 = action[0], action[1], action[2], action[3], action[4], action[5], action[6], action[7]
        k1, k2, k3, k4, k5, k6, k7, k8 = params.k1, params.k2, params.k3, params.k4, params.k5, params.k6, params.k7, params.k8 
        def control_fn(y):
            return (a1*jnp.sin(k1*y) + a2*jnp.sin(k2*y) + a3*jnp.sin(k3*y) + a4*jnp.sin(k4*y) + a5*jnp.sin(k5*y) + a6*jnp.sin(k6*y) + a7*jnp.sin(k7*y) + a8*jnp.sin(k8*y), jnp.zeros_like(y))

        flow.control_function = control_fn(y)
        equation = PseudoSpectralNavierStokes2D(flow)
        step_fn = transient.RK4_CN(equation, params.dt)
        total_steps = int(end_time // dt)
        steps_to_save = int(save_time // dt)
        _, updated_trajectory = transient.iterative_func(step_fn, state.new_init, total_steps, steps_to_save)
        
        return updated_trajectory 
    
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: jnp.ndarray,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        
        updated_trajectory = self._apply_action(action, params, state)
        
        observation = self.get_obs(state, params, key)
        reward = self._compute_reward(action, state) 
        updated_time = state.time + params.action_time
        
        # Update state dict and evaluate termination conditions
        state = EnvState(
            trajectory= updated_trajectory,
            time = updated_time,
            terminal=False,
            new_init=updated_trajectory[-1],
        )
        
        done = self.is_terminal(state, params)
        
        return (
            lax.stop_gradient(observation),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)}
           
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        
        """Reset environment state by returning to initial position."""
        equation = PseudoSpectralNavierStokes2D(flow)
        step_fn = transient.RK4_CN(equation, dt)
        initial_state = flow.initialize_state()
        total_steps = int(5 // 0.001)
        steps_to_save = int(1 // 0.001)
        
        _, init_trajectory = transient.iterative_func(step_fn, initial_state, total_steps, steps_to_save)
        state = EnvState(time=0, trajectory=init_trajectory, terminal=False, new_init=init_trajectory[-1])
        return self.get_obs(state, params, key), state
    
    def _calculate_mode_energy(self, state, k):
        # Calculate the energy of a mode
        kx, ky = flow.create_fft_mesh()
        uhat, vhat = compute_velocity_fft(state, kx, ky)
        mode_energy = compute_energy_mode(uhat, vhat, k, 0, n, m)
     
        return mode_energy
    
    def _calculate_mode_velocity(self, state, k1, k2):
        kx, ky = flow.create_fft_mesh()
        uhat, vhat = compute_velocity_fft(state, kx, ky)
        mode_velocity = compute_velocity_mode(uhat, vhat, k1, k2, n, m)
        
        return mode_velocity 
    
    # def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        
    #     def calculate_energy(trajectory):
    #         e1, e2, e3, e4 = self._calculate_mode_energy(trajectory, params.k1), self._calculate_mode_energy(trajectory,  params.k2), self._calculate_mode_energy(trajectory,  params.k3), self._calculate_mode_energy(trajectory,  params.k4)
    #         return jnp.array([e1, e2, e3, e4])

    #     def scan_fn(carry, state):
    #         energies = calculate_energy(state)
    #         return carry, energies
        
    #     trajectory = state.trajectory
    #     _, all_energy = lax.scan(scan_fn, None, trajectory)
    #     jax.debug.print("all energy shape: {}", all_energy.shape)
                
    #     return jnp.mean(all_energy, axis=0)
    
    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        
        def calculate_velocity(trajectory):
            v1, v2, v3, v4, v5, v6, v7, v8 = self._calculate_mode_velocity(trajectory, params.k1, 0), self._calculate_mode_velocity(trajectory,  params.k2, 0), self._calculate_mode_velocity(trajectory,  params.k3, 0), self._calculate_mode_velocity(trajectory,  params.k4, 0), self._calculate_mode_velocity(trajectory, 0, params.k1), self._calculate_mode_velocity(trajectory, 0, params.k2), self._calculate_mode_velocity(trajectory, 0, params.k3), self._calculate_mode_velocity(trajectory, 0, params.k4)
            return jnp.array([v1, v2, v3, v4, v5, v6, v7, v8])

        def scan_fn(carry, state):
            energies = calculate_velocity(state)
            return carry, energies
        
        trajectory = state.trajectory
        _, all_energy = lax.scan(scan_fn, None, trajectory)
                        
        return jnp.mean(all_energy, axis=0)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done_steps, state.terminal)
        
        return done.squeeze()

    @property
    def name(self) -> str:
        """Environment name."""
        return "KolmogorovFlow"

    @property
    def num_actions(self, params: EnvParams) -> int:
        """Number of actions possible in environment."""
        return params.action_dim

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(params.action_dim,),
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(params.min_obs, params.max_obs, shape=(params.obs_dim,))

    # def state_space(self, params: EnvParams) -> spaces.Dict:
    #     """State space of the environment."""
    #     low = jnp.array(
    #         [params.min_position, -params.max_speed],
    #         dtype=jnp.float32,
    #     )
    #     high = jnp.array(
    #         [params.max_position, params.max_speed],
    #         dtype=jnp.float32,
    #     )
    #     return spaces.Dict(
    #         {
    #             "position": spaces.Box(low[0], high[0], (), dtype=jnp.float32),
    #             "time": spaces.Discrete(params.max_steps_in_episode),
    #         }
    #     )

