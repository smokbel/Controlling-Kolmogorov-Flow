import jax.numpy as jnp 
import jax 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
from equations.flow import *
from jax import lax

def compute_velocity_fft(omega_hat, kx, ky):
    """
    Computing the fourier velocity components (u_hat, v_hat) from the stream function (phi_hat)
    (Yin, Z. 2004)
        
    Args:
        omega_hat: the Fourier transform of the vorticity 
        grid: the jnp grid
        
    """
    double_derivative = (2 * jnp.pi * 1j) ** 2 * (abs(kx)**2 + abs(ky)**2)
    double_derivative = double_derivative.at[0, 0].set(1)  # avoiding division by 0.0 in the next step

    psi_hat = -1 * omega_hat / double_derivative 
    vxhat = (2 * jnp.pi * 1j) * ky * psi_hat # Get u,v from phi 
    vyhat = (-1 * 2 * jnp.pi * 1j) * kx * psi_hat
    return vxhat, vyhat

def dealiasing(advection_term):
    """ 
    
    Adds the 2/3 aliasing technique to the velocity field, which 
    sets the last 1/3 high frequency Fourier modes to 0. 
    Reference: https://notes.yeshiwei.com/pseudo_spectral_method/algorithm.html

    Args:
        vel_hat: velocity field in Fourier space
    """
    n, m = advection_term.shape[0], advection_term.shape[1]
    kn, km = int(n//2 * 2//3), int(m * 2//3)
    advection_term.at[kn:2*kn, :].set(0.0)
    advection_term.at[:, km:].set(0.0)
    
    return advection_term

def compute_energy_mode(uhat, vhat, kx, ky, n, m):
    """
    Compute the energy of a specific mode and wavenumber. 
    
    Args: 
        omega_hat: fft vorticity 
        kx: wavenumber x
        ky: wavenumber y 
        n, m: grid size
    """
    
    # Compute indices of wavenumber 
    #uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    kx_idx = kx % n
    ky_idx = ky % m 
    energy = 0.5 * (jnp.abs(uhat[kx_idx, ky_idx])**2 + jnp.abs(vhat[kx_idx, ky_idx])**2) / jnp.float32((n*m)**2)
    return energy 

def compute_velocity_mode(uhat, vhat, kx, ky, n, m):
    """
    Compute the velocity of a specific mode and wavenumber. 
    
    Args: 
        omega_hat: fft vorticity 
        kx: wavenumber x
        ky: wavenumber y 
        n, m: grid size
    """
    
    # Compute indices of wavenumber 
    #uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    kx_idx = kx % n
    ky_idx = ky % m 
    velocity_mag = jnp.sqrt((jnp.abs(uhat[kx_idx, ky_idx])**2 + jnp.abs(vhat[kx_idx, ky_idx])**2) / jnp.float32((n*m)**2))
    return velocity_mag 

def compute_energy_dissipation(omega_hat, kx, ky, nu, n):
    """
        Computes the energy dissipation of the system given the fft vorticity field.
        
    Args: 
        omega_hat: fft vorticity 
        kx: wavenumber x
        ky: wavenumber y 
        nu: kinematic viscosity 
        n: grid length
        
    """
    
    uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    ureal = jnp.fft.irfftn(uhat)
    vreal = jnp.fft.irfftn(vhat)
    du_dy = jnp.gradient(ureal, axis=1) 
    dv_dx = jnp.gradient(vreal, axis=0)
    avg_epsilon = 2 * nu * (du_dy+ dv_dx)**2 * (1/n)
    epsilon = jnp.sum(avg_epsilon)
    
    return epsilon 
    
def compute_tke(omega_hat, kx, ky, n):
    """
        Computes the TKE of the systen given the fft vorticity field.
        
    Args: 
        omega_hat: fft vorticity 
        kx: wavenumber x
        ky: wavenumber y 
        n: grid length
        
    """
    
    uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    ureal = jnp.fft.irfftn(uhat)
    vreal = jnp.fft.irfftn(vhat)
    avg_tke = 0.5 * (jnp.abs(ureal)**2 + jnp.abs(vreal)**2) 
    tke = jnp.sum(avg_tke) * (1/(n*n))
    
    return tke 

def compute_divergence(omega_hat, kx, ky):
    """
        Computes the divergence of the systen given the fft vorticity field.
        
    Args: 
        omega_hat: fft vorticity 
        kx: wavenumber x
        ky: wavenumber y 
    """
    
    uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    ureal = jnp.fft.irfftn(uhat)
    vreal = jnp.fft.irfftn(vhat)
    du_dx = jnp.gradient(ureal, axis=0) 
    dv_dy = jnp.gradient(vreal, axis=1)

    return du_dx + dv_dy 
    
def create_animation(trajectory, gif_name, frame_interval_factor):
    """
        Produces an animation of the trajectory.
        
    Args: 
        trajectory: numpy file of fft vorticity trajectory
        gif_name: file name of gif file that will be saved 
        interval: frame interval as related to the length of the trajectory. 
    """
    if type(trajectory) == str:
        trajectory = jnp.load(trajectory)
    simulation = jnp.fft.irfftn(trajectory, axes=(1,2))

    fig, ax = plt.subplots()
    cax = ax.imshow(simulation[0], cmap='icefire',interpolation='nearest')
    fig.colorbar(cax)
    
    num_frames = len(simulation)
    interval = int(num_frames * frame_interval_factor)

    timestamp = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')

    def update_frame(frame):
        cax.set_array(simulation[frame])
        timestamp.set_text(f'Time: {frame}')
        return cax, timestamp

    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=interval)

    # Save as a GIF
    ani.save('{}.gif'.format(gif_name), writer=PillowWriter(fps=interval))
    

def create_dynamic_visualization(simulation_no_control: jnp.array, simulation_control: jnp.array, 
                                 end_time: int, n: int, m: int, Re: int, colormap: str, fps: int,
                                 view_energy_dissipation: bool):
    """
    
    Function that visualizes the controlled vs uncontrolled case and their TKE in a pretty gif. 
    Optional to also visualize the energy dissipation.
    
    Args:
        simulation_no_control: FFT of the vorticity field without control 
        simulation_control: FFT of the vorticity field for the controlled case (after rollout)
        end_time: Length to visualize 
        n,m: Size of the simulation grid
        Re: Reynolds number of the case
        colormap: Seaborn colormap 
        fps: Frames per second   
    """    
    
    simulation_agent = simulation_control[:end_time]
    simulation = simulation_no_control[:end_time]
    
    # Initialize the flow configuration
    flow = flow.FlowConfig(grid_size=(n,m))
    kx, ky = flow.create_fft_mesh()
    
    # Compute TKE for both simulations
    tke_nocontrol = []
    tke_agent = []
    energy_dissipation = []
    
    for omega_hat_nc, omega_hat_agent in zip(simulation, simulation_agent):
        total_epsilon_nc = compute_tke(omega_hat_nc, kx, ky, n)    
        total_epsilon_agent = compute_tke(omega_hat_agent, kx, ky, n)    
        tke_nocontrol.append(total_epsilon_nc)
        tke_agent.append(total_epsilon_agent)
        
        dissipation = compute_energy_dissipation(omega_hat_agent, kx, ky, flow.nu, n)
        energy_dissipation.append(dissipation)

    # Initial plot setup
    pastel1 = plt.get_cmap('Set2')
    colors = [pastel1(i) for i in range(pastel1.N)]

    # Initialize plots
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # TKE plot in the top row, spanning both columns
    ax_tke = fig.add_subplot(gs[0, :])

    line_nocontrol, = ax_tke.plot([], [], label="No control", color=colors[0], linewidth=4)
    line_agent, = ax_tke.plot([], [], label='Controlled with agent', color=colors[1], linewidth=4)
    ax_tke.set_xlim(0, len(simulation_agent))
    ax_tke.set_ylim(min(min(tke_nocontrol), min(tke_agent)) * 0.9, max(max(tke_nocontrol), max(tke_agent)) * 1.1)
    ax_tke.legend()
    ax_tke.set_xlabel("Time (s)")
    ax_tke.set_ylabel("TKE")
    ax_tke.set_title("TKE Evolution, Re = {}".format(Re))

    img_simulation_agent = jnp.fft.irfftn(simulation_agent, axes=(1,2))
    img_simulation = jnp.fft.irfftn(simulation, axes=(1,2))

    ax_nocontrol = fig.add_subplot(gs[1, 0])
    
    # Display the grids side by side in the bottom row
    img_nocontrol = ax_nocontrol.imshow(img_simulation[0], cmap=colormap, aspect='auto', vmin=-8, vmax=8)
    ax_nocontrol.set_title("No Control")

    ax_agent = fig.add_subplot(gs[1, 1])
    img_agent = ax_agent.imshow(img_simulation_agent[0], cmap=colormap, aspect='auto', vmin=-8, vmax=8)
    ax_agent.set_title("Controlled with RL")

    # Function to update the frame
    def update(frame):
        # Update TKE plot
        line_nocontrol.set_data(range(frame + 1), tke_nocontrol[:frame + 1])
        line_agent.set_data(range(frame + 1), tke_agent[:frame + 1])

        # Update grid images
        img_nocontrol.set_array(img_simulation[frame])
        img_agent.set_array(img_simulation_agent[frame])
        
        return line_nocontrol, line_agent, img_nocontrol, img_agent

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(simulation_agent), blit=True)

    # Save the animation as a GIF
    ani.save('tke_evolution_{}.gif'.format(Re), writer='imagemagick', fps=fps)
    

def create_dynamic_visualization3(simulation_no_control: jnp.array, simulation_control: jnp.array, start_time: int,
                                 end_time: int, n: int, m: int, Re: int, colormap: str, fps: int,
                                 view_energy_dissipation: bool):
    
    simulation_agent = simulation_control[start_time:end_time]
    simulation = simulation_no_control[start_time:end_time]
    
    flow = FlowConfig(grid_size=(n, m))
    kx, ky = flow.create_fft_mesh()
    
    def compute_tke_and_dissipation(carry, omega_hats):
        omega_hat_nc, omega_hat_agent = omega_hats
        tke_nc = compute_tke(omega_hat_nc, kx, ky, n)
        tke_agent = compute_tke(omega_hat_agent, kx, ky, n)
        dissipation = compute_energy_dissipation(omega_hat_agent, kx, ky, flow.nu, n)
        
        carry['tke_nocontrol'].append(tke_nc)
        carry['tke_agent'].append(tke_agent)
        carry['energy_dissipation_agent'].append(dissipation)
        
        return carry, None

    # Initialize the carry
    carry = {
        'tke_nocontrol': [],
        'tke_agent': [],
        'energy_dissipation_agent': []
    }
    
    # Run lax.scan
    carry, _ = lax.scan(compute_tke_and_dissipation, carry, (simulation, simulation_agent))

    # Extract the results from the carry
    tke_nocontrol = carry['tke_nocontrol']
    tke_agent = carry['tke_agent']
    energy_dissipation_agent = carry['energy_dissipation_agent']

    # Initial plot setup
    pastel1 = plt.get_cmap('Set2')
    colors = [pastel1(i) for i in range(pastel1.N)]

    # Adjust figure and gridspec layout based on whether to display energy dissipation
    if view_energy_dissipation:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    else:
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        gs.update(wspace=0.3, hspace=0.3)
    
    # TKE plot on the top left
    ax_tke = fig.add_subplot(gs[0, 0])
    line_nocontrol, = ax_tke.plot([], [], label="No control", color=colors[0], linewidth=4)
    line_agent, = ax_tke.plot([], [], label='Controlled with agent', color=colors[1], linewidth=4)
    ax_tke.set_xlim(0, len(simulation_agent))
    ax_tke.set_ylim(min(min(tke_nocontrol), min(tke_agent)) * 0.9, max(max(tke_nocontrol), max(tke_agent)) * 1.1)
    ax_tke.legend()
    ax_tke.set_xlabel("Time (s)")
    ax_tke.set_ylabel("TKE")
    ax_tke.set_title("TKE Evolution, Re = {}".format(Re))
    
    if view_energy_dissipation:
        ax_dissipation = fig.add_subplot(gs[0, 1])
        line_dissipation, = ax_dissipation.plot([], [], label='Energy Dissipation', color='black', linestyle='--', linewidth=2)
        ax_dissipation.set_xlim(0, len(simulation_agent))
        ax_dissipation.set_ylim(min(energy_dissipation_agent) * 0.9, max(energy_dissipation_agent) * 1.1)
        ax_dissipation.set_xlabel("Time (s)")
        ax_dissipation.set_ylabel("Energy Dissipation")
        ax_dissipation.set_title("Energy Dissipation Evolution")
    
    img_simulation_agent = jnp.fft.irfftn(simulation_agent, axes=(1, 2))
    img_simulation = jnp.fft.irfftn(simulation, axes=(1, 2))

    ax_nocontrol = fig.add_subplot(gs[1, 0])
    img_nocontrol = ax_nocontrol.imshow(img_simulation[0], cmap=colormap, aspect='auto', vmin=-8, vmax=8)
    ax_nocontrol.set_title("No Control")

    ax_agent = fig.add_subplot(gs[1, 1])
    img_agent = ax_agent.imshow(img_simulation_agent[0], cmap=colormap, aspect='auto', vmin=-8, vmax=8)
    ax_agent.set_title("Controlled with RL")

    def update(frame):
        line_nocontrol.set_data(range(frame + 1), tke_nocontrol[:frame + 1])
        line_agent.set_data(range(frame + 1), tke_agent[:frame + 1])
        
        if view_energy_dissipation:
            line_dissipation.set_data(range(frame + 1), energy_dissipation_agent[:frame + 1])

        img_nocontrol.set_array(img_simulation[frame])
        img_agent.set_array(img_simulation_agent[frame])
        
        if view_energy_dissipation:
            return line_nocontrol, line_agent, line_dissipation, img_nocontrol, img_agent
        else:
            return line_nocontrol, line_agent, img_nocontrol, img_agent

    ani = animation.FuncAnimation(fig, update, frames=len(simulation_agent), blit=True)

    ani.save('tke_evolution_{}.gif'.format(Re), writer='pillow', fps=fps)
    
    
def create_dynamic_visualization_2(simulation_no_control: jnp.array, simulation_control: jnp.array, 
                                   start_time: int, end_time: int, n: int, m: int, Re: int, colormap: str, fps: int,
                                 view_energy_dissipation: bool):
    """
        
        Function that visualizes the controlled vs uncontrolled case and their TKE in a pretty gif. 
        Optional to also visualize the energy dissipation.
        
        Args:
            simulation_no_control: FFT of the vorticity field without control 
            simulation_control: FFT of the vorticity field for the controlled case (after rollout)
            end_time: Length to visualize 
            n,m: Size of the simulation grid
            Re: Reynolds number of the case
            colormap: Seaborn colormap 
            fps: Frames per second   
            view_energy_dissipation: Whether to also visualize energy dissipation
    """    

    flow = FlowConfig(grid_size=(n,m))
    kx, ky = flow.create_fft_mesh()
    
    simulation_agent = simulation_control[start_time:end_time]
    simulation = simulation_no_control[start_time:end_time]
    
    # Compute TKE for both simulations and energy dissipation for the controlled case
    tke_nocontrol = []
    tke_agent = []
    energy_dissipation_agent = []
    energy_dissipation = []
    
    for omega_hat_nc, omega_hat_agent in zip(simulation, simulation_agent):
        total_epsilon_nc = compute_tke(omega_hat_nc, kx, ky, n)    
        total_epsilon_agent = compute_tke(omega_hat_agent, kx, ky, n)    
        tke_nocontrol.append(total_epsilon_nc)
        tke_agent.append(total_epsilon_agent)
        
        dissipation_nc = compute_energy_dissipation(omega_hat_nc, kx, ky, flow.nu, n)
        dissipation_c = compute_energy_dissipation(omega_hat_agent, kx, ky, flow.nu, n)
        energy_dissipation_agent.append(dissipation_c)
        energy_dissipation.append(dissipation_nc)

    # Initial plot setup
    pastel1 = plt.get_cmap('Set2')
    colors = [pastel1(i) for i in range(pastel1.N)]

    # Adjust figure and gridspec layout based on whether to display energy dissipation
    if view_energy_dissipation:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    else:
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        gs.update(wspace=0.3, hspace=0.3)
    
    # TKE plot on the top left
    ax_tke = fig.add_subplot(gs[0, 0])
    line_nocontrol, = ax_tke.plot([], [], label="No control", color=colors[0], linewidth=4)
    line_agent, = ax_tke.plot([], [], label='Controlled with agent', color=colors[1], linewidth=4)
    ax_tke.set_xlim(0, len(simulation_agent))
    ax_tke.set_ylim(min(min(tke_nocontrol), min(tke_agent)) * 0.9, max(max(tke_nocontrol), max(tke_agent)) * 1.1)
    ax_tke.legend()
    ax_tke.set_xlabel("Time (s)")
    ax_tke.set_ylabel("TKE")
    ax_tke.set_title("TKE Evolution, Re = {}".format(Re))
    
    # Energy Dissipation plot on the top right if enabled
    if view_energy_dissipation:
        ax_dissipation = fig.add_subplot(gs[0, 1])
        line_dissipation, = ax_dissipation.plot([], [], label='Energy Dissipation', color='black', linestyle='--', linewidth=2)
        line_dissipation_c, = ax_dissipation.plot([], [], label='Energy Dissipation Controlled', color='blue', linestyle='--', linewidth=2)
        ax_dissipation.set_xlim(0, len(simulation_agent))
        ax_dissipation.set_ylim(min(energy_dissipation_agent) * 0.9, max(energy_dissipation_agent) * 1.1)
        ax_dissipation.set_xlabel("Time (s)")
        ax_dissipation.set_ylabel("Energy Dissipation")
        ax_dissipation.set_title("Energy Dissipation Evolution")
    
    # Convert the simulation data back to real space for visualization
    img_simulation_agent = jnp.fft.irfftn(simulation_agent, axes=(1, 2))
    img_simulation = jnp.fft.irfftn(simulation, axes=(1, 2))

    # Visualization of the simulation without control on the bottom left
    ax_nocontrol = fig.add_subplot(gs[1, 0])
    img_nocontrol = ax_nocontrol.imshow(img_simulation[0], cmap=colormap, aspect='auto', vmin=-8, vmax=8)
    ax_nocontrol.set_title("No Control")

    # Visualization of the agent-controlled simulation on the bottom right
    ax_agent = fig.add_subplot(gs[1, 1])
    img_agent = ax_agent.imshow(img_simulation_agent[0], cmap=colormap, aspect='auto', vmin=-8, vmax=8)
    ax_agent.set_title("Controlled with RL")

    # Function to update the frame
    def update(frame):
        # Update TKE plot
        line_nocontrol.set_data(range(frame + 1), tke_nocontrol[:frame + 1])
        line_agent.set_data(range(frame + 1), tke_agent[:frame + 1])
        
        # Update Energy Dissipation plot if enabled
        if view_energy_dissipation:
            line_dissipation.set_data(range(frame + 1), energy_dissipation[:frame + 1])
            line_dissipation_c.set_data(range(frame + 1), energy_dissipation_agent[:frame + 1])

        # Update grid images
        img_nocontrol.set_array(img_simulation[frame])
        img_agent.set_array(img_simulation_agent[frame])
        
        if view_energy_dissipation:
            return line_nocontrol, line_agent, line_dissipation, line_dissipation_c, img_nocontrol, img_agent
        else:
            return line_nocontrol, line_agent, img_nocontrol, img_agent

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(simulation_agent), blit=True)

    # Save the animation as a GIF
    ani.save('tke_evolution_{}.gif'.format(Re), writer='imagemagick', fps=fps)