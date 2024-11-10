# Controlling-Kolmogorov-Flow
Controlling extreme energy events in turbulent flow (2D Kolmogorov flow). 

## Usage notes

If you plan to use this code for research purposes that may result in publications, please contact me first at sajedam@uw.edu. 

## Requirements 

This project assumes you have the following Python libraries installed: 
1. Jax (https://jax.readthedocs.io/en/latest/installation.html)
2. Matplotlib and seaborn for visualization 

## Flow Configuration

The `PseudoSpectralNavierStokes2D` class expects a `FlowConfig` class argument. For simplicity, the `FlowConfig` contains default values. The user can, however, specify flow parameters such as: 
- Reynolds number (flow.Re)
- Grid size (flow.grid_size)
- Forcing wavenumber (flow.k)
- Timestep 

The vorticity field can be initialized with `flow.initialize_state()` for simplicity, as it creates a divergence-free field by default.  

## Solver details 

This project uses a fourth order implicit-explicit Runge-Kutta time stepping scheme from [2]. The user can specify the length of the simulation and the uniform interval desired for saving: 

```
dt = 0.001
end_time = 100
save_interval = 1
total_steps = int(end_time // dt)
step_to_save = int(save_interval // dt) 
vorticity_hat0 = flow.initialize_state()

step_fn = transient.RK4_CN(equation, dt)
end_state, full_trajectory = transient.iterative_func(step_fn, vorticity_hat0, total_steps, step_to_save)

```

In this case, the simulation trajectory will be stored in `full_trajectory`. 

## Further details

For computing and visualizing the extreme energy events in Kolmogorov flow, refer to `kolmogorov_demo.ipynb`. 

## References 

The references used when creating this code: 

[1] Z. Yin, H.J.H. Clercx, D.C. Montgomery,
An easily implemented task-based parallel scheme for the Fourier pseudospectral solver applied to 2D Navier–Stokes turbulence,
Computers & Fluids,Volume 33, Issue 4, 2004, Pages 509-520, ISSN 0045-7930,
https://doi.org/10.1016/j.compfluid.2003.06.003.

[2] G. Dresdner, D. Kochkov, P. Norgaard, L. Zepeda-N´u˜nez, J. A. Smith, M. P. Brenner, and
S. Hoyer, “Learning to correct spectral methods for simulating turbulent flows,” 2022.
