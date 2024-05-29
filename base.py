import dataclasses
import jax.numpy as jnp
from flow import FlowConfig
import utils

@dataclasses.dataclass
class PseudoSpectralNavierStokes2D():
  """ 
  Calculates the 2D Navier-Stokes equations using the pseudo-spectral solver. We transform the 2D Navier-Stokes equation to a vorticity equation:
    ∂/∂t ω + u·∇ω = v ∇²ω + ƒ ;
    ω = - ∇²φ ; 
  and solve in Fourier space
  
  """

  def __init__(self, flow: FlowConfig): 
      self.flow = flow 
      self.grid = flow.create_fft_mesh()
      self.real_grid = flow.create_mesh()
      self.kx, self.ky = self.grid
      self.x, self.y = self.real_grid
          
  def linear_terms(self, omega_hat):
    """Computes the linear (viscous) term of the vorticity equation
    """
    return self.flow.nu *  (2j * jnp.pi)**2 * (self.kx**2 + self.ky**2) * omega_hat
   
  def implicit_timestep(self,omega_hat, time_step):
    """
    Function that computes an implicit euler timestep,
      y_n+1 = y_n / (1-∇tλ). 
    
    """
    double_derivative = (2j * jnp.pi)**2 * (self.kx**2 + self.ky**2)
    return 1 / (1 - time_step * self.flow.nu * double_derivative) * omega_hat
    
  def nonlinear_terms(self, omega_hat):
    """Computes the explicit (nonlinear) terms in the vorticity equation. 
    Uses the stream function to compute velocity components in Fourier space.

    Args:
        omega_hat: fft of vorticity

    Returns:
        terms: Nonlinear terms of the equation.
    """
    
    kx, ky = self.kx, self.ky
    
    double_derivative = (2 * jnp.pi * 1j) ** 2 * (abs(self.kx)**2 + abs(ky)**2)
    double_derivative = double_derivative.at[0, 0].set(1)  # avoiding division by 0.0 in the next step

    psi_hat = -1 * omega_hat / double_derivative 
    uhat = (2 * jnp.pi * 1j) * ky * psi_hat # Get u,v from phi 
    vhat = (-1 * 2 * jnp.pi * 1j) * kx * psi_hat
  
    u, v = jnp.fft.irfftn(uhat), jnp.fft.irfftn(vhat)

    grad_x_hat = 2j * jnp.pi * self.kx * omega_hat
    grad_y_hat = 2j * jnp.pi * self.ky * omega_hat
    grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

    advection = -(grad_x * u + grad_y * v)
    advection_hat = jnp.fft.rfftn(advection)
    
    forcing_hat = self.forcing_term()
    advection_hat = utils.dealiasing(advection_hat) # 2/3 dealiasing rule

    terms = advection_hat + forcing_hat
    return terms
    
    
  def forcing_term(self):
      """Computes the user-specified forcing term of the vorticity equation 
      Args:
        omega_hat: Fourier transformed vorticity term
        forcing: Forcing function as specified by environment or user
      """
      forcing_func = self.flow.forcing_function
      if forcing_func is not None:
        kx, ky = self.grid
        x, y = self.real_grid
        fx, fy = forcing_func(k=self.flow.k, x=x, y=y)
        fx_hat, fy_hat = jnp.fft.rfft2(fx), jnp.fft.rfft2(fy)

        # Transform the velocity forcing into vorticity 
        derivative_term = (2j * jnp.pi)
        f_vorticity = derivative_term * (fy_hat*kx - fx_hat*ky)
        return f_vorticity
      else:
        return None 
