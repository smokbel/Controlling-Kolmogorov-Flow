import gym 
from gym import spaces 
import equations.flow as flow 
from solvers import transient 
import equations.base as base
import jax.numpy as jnp
import equations.utils as utils
import numpy as np
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env

flow = flow.FlowConfig()
flow.Re = 250
kx, ky = flow.create_fft_mesh()

class KolmogorovFlow(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, threshold, num_inputs, num_measurements):
        
        super(KolmogorovFlow, self).__init__()
        
        #self.action_space = spaces.MultiDiscrete([10, 10, 10, 10, 10])
                
        self.action_space = spaces.Box(low=-2.0, 
                                       high=2.0,
                                       shape=(num_inputs,))
        
        self.observation_space = spaces.Box(low=-1.0, 
                                       high=1.0,
                                       shape=(num_measurements,),
                                       dtype=np.float64)
        
        self.state = flow.initialize_state()
        
        self.threshold = threshold 
        
        self.y = flow.create_mesh()[1]
        
        self.k = [1,2,3,4,5,6]
        
        self.dt = 0.001
        
        self.step_count = 0
        
        self.max_steps = 10
        
    def reset(self):
        #super().reset(seed=seed)
        self.state = flow.initialize_state()
        observation = self.compute_energy(self.state)
        print(observation.shape)
        observation = np.array([observation]).flatten()
        return observation

    def control_fn(self, f, x, y):
        return (f, jnp.zeros_like(y))
    
    def timestep_fn(self, state, control_params, y, dt, flow):
        flow.control_function = self.control_fn(control_params, 0, y)
        equation = base.PseudoSpectralNavierStokes2D(flow)
        step_fn = transient.RK4_CN(equation, dt)(state)
        return step_fn
    
    def step(self, action):
        # Different amplitudes of modes up around the wavenumber  
        a1, a2, a3, a4, a5 = action 
        print("actions: ", a1,a2,a3,a4,a5)
        control_fn_arg = a1*jnp.sin(self.k[0]*self.y) + a2*jnp.sin(self.k[1]*self.y) + a3*jnp.sin(self.k[2]*self.y) + a4*jnp.sin(self.k[3]*self.y)  + a5*jnp.sin(self.k[4]*self.y)
        self.state = self.timestep_fn(self.state, control_fn_arg, self.y, self.dt, flow)
        
        measurement = self.compute_measurement(self.state)
        observation = self.compute_energy(self.state)
        
        reward = -abs(10*measurement)  # Example reward, target state is 50
        self.step_count += 1
        done = self.step_count >= self.max_steps
     
        return np.array([observation]), reward, done, {}
    
    def compute_measurement(self, state):
        return utils.compute_energy_dissipation(state, flow.nu)
    
    def compute_energy(self, state):
        uhat, vhat = utils.compute_velocity_fft(state, kx, ky)
        energy = utils.compute_energy_mode(uhat, vhat, 0, flow.k, 256, 256)
        return energy
    
    def render(self, mode='human'):
        pass

register(
    id='KolmogorovFlow-v0',
    entry_point='__main__:KolmogorovFlow',
)

# Create the environment with a specific threshold
threshold = 0.20
env = KolmogorovFlow(threshold=threshold, num_inputs=5, num_measurements=1)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.001 * np.ones(n_actions))

# Create the DDPG model
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

# Train the model
model.learn(total_timesteps=9000)

# Save the model
model.save("agent_1")