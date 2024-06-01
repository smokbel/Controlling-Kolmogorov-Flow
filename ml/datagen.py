import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
from sklearn.model_selection import train_test_split

from equations import utils
import jax.numpy as jnp
from equations import flow  

flow = flow.FlowConfig()
trajectory = jnp.load('kolmogorov_Re_250.npy')
kx, ky = flow.create_fft_mesh()
n, m = trajectory[0].shape 
kx = kx
ky = ky

# Generate the data 

def get_extreme_event_times(trajectory):
    """
    Given the fluid trajectory data, generate a 20x20 data matrix which stores the energy of the first ~10% of the modes.
    
    Args:
        trajectory: vorticity fft over time of full simulation.
    """
    
    # Compute energy dissipation for each timestep
    dissipation = []
    for omega_hat in trajectory:
        total_epsilon = utils.compute_energy_dissipation(omega_hat, kx, ky, flow.nu, n)    
        dissipation.append(total_epsilon)
    mean_diss = jnp.mean(np.array(dissipation))
    event_times = []
    for diss in dissipation:
        if diss >= 1.55*mean_diss:
            event_times.append(dissipation.index(diss))
    return

def data_gen(trajectory, event_times):
    """
    Given the trajectory data and event times, create input and target data related to the probability that an extreme event is occuring.
    
    Args:
        trajectory: vorticity fft over time of full simulation.
        event_times: times that extreme event is occuring.
    
    """
    energy_vals = np.empty(len(trajectory), 40)
    for omega_hat in trajectory:
        uhat, vhat = utils.compute_velocity_fft(omega_hat, kx, ky)
        energy_at_time = []
        for kx_idx in range(20):
            for ky_idx in range(20):
                energy = utils.compute_energy_mode(uhat, vhat, kx_idx, ky_idx, n, m)
                energy_at_time.append(energy)
        energy_vals.append(np.array(energy_at_time))
        
get_extreme_event_times(trajectory)
data = data_gen(trajectory)
print(data)

        
                
                
    
    
            
    
        
    