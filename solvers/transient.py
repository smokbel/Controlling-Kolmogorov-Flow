import jax.numpy as jnp 
from jax import lax
import tree_math

    
def RK4_CN(equation, dt):
  
  """ Crank-Nicolson RK4 implicit-explicit time stepping scheme
      Low storage scheme inspired by [1]. Method described in [2]. 
      
      Implicit-Explicit timestepping for an ODE of the form:
        ∂u/∂t = g(u,t) + l(u,t)
      where g(u,t) is the nonlinear advection term and l(u,t) is the linear diffusion term.
      
      [1] Kochkov, D., et. al. (2021) https://doi.org/10.1073/pnas.2101784118
      [2] PK Sweby, (1984). SIAM journal on numerical analysis 21, Appendix D.
  """
  
  a = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1] # Alphas
  B = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257] # Betas
  Y = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681] # Gammas
  g = tree_math.unwrap(equation.nonlinear_terms)
  l = tree_math.unwrap(equation.linear_terms)
  y = tree_math.unwrap(equation.implicit_timestep, vector_argnums=0)

  @tree_math.wrap
  def time_step_fn(u):
    h = 0
    for k in range(5):
      h = g(u) + B[k] * h 
      mu = 0.5 * dt * (a[k + 1] - a[k])
      yn = u + Y[k]*dt*h + mu*l(u)
      u = y(yn, mu)
    return u
  
  return time_step_fn

def iterative_func(func, initialization, steps, save_n):
  """Lax.scan to iteratively apply a function.

  Args:
      func (method): The time stepping function
      initialization(grid array): The initial fft vorticity field
      steps (int): number of timesteps
      save_n (int): Save every n steps
      
  """
  
  f = lambda init, inputs: (func(init), init)
  # Scan used to iteratively apply timestepping
  final_state, outputs = lax.scan(f, initialization, xs=None,length=steps)
  return final_state, outputs[save_n-1::save_n]
