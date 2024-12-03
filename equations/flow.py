import jax.numpy as jnp
from jax import grad

class FlowConfig:
    
    DEFAULT_REYNOLDS = 200
    DEFAULT_WAVENUMBER = 4
    DEFAULT_GRID_SIZE = (64, 64)
    DEFAULT_DOMAIN_X = (0, 2*jnp.pi)
    DEFAULT_DOMAIN_Y = (0, 2*jnp.pi)

    def __init__(self, **config):
        self.k = config.get("k", self.DEFAULT_WAVENUMBER)
        self.Re = config.get("Re", self.DEFAULT_REYNOLDS)
        self.grid_size = config.get("grid_size", self.DEFAULT_GRID_SIZE)
        self.domain_x = config.get("domain_x", self.DEFAULT_DOMAIN_X)
        self.domain_y = config.get("domain_y", self.DEFAULT_DOMAIN_Y)
        self.control_function = ( jnp.zeros_like(self.create_mesh()[0]), jnp.zeros_like(self.create_mesh()[1]) )
    
    def create_mesh(self):
        """Create jax grid given the desired dimensions and spacing in real space
        Returns:
            jax meshgrid 
        """
        x0, xn, nx = self.domain_x[0], self.domain_x[1], self.grid_size[1]
        y0, yn, ny = self.domain_y[0], self.domain_y[1], self.grid_size[0]
        x = jnp.linspace(x0, xn, nx)
        y = jnp.linspace(y0, yn, ny)
        return jnp.meshgrid(x, y, indexing='ij')
    
    def create_fft_mesh(self):
        """Create jax grid given desired dimensions and spacing in real Fourier space

        Returns:
            jax meshgrid 
        """
        N = self.grid_size[0]
        M = self.grid_size[1]
        dx = self.domain_x[1] / N
        dy = self.domain_y[1] / M
        kx = jnp.fft.fftfreq(N, dx)
        ky = jnp.fft.rfftfreq(M, dy) 
        return jnp.meshgrid(kx, ky, indexing='ij')
    
    def initialize_state(self):
        """Generate a divergence free velocity field to initialize the state
        Initializing with divergence free field specified with the following stream function:
        
        φ(x,y) = sin(x)cos(y)

        Returns:
            omega_0: fft vorticity field 
        """
        X, Y = self.create_mesh()
        # Gradients of φ(x,y) #
        def dstream_func_dx(x, y):
            return jnp.cos(x)

        def dstream_func_dy(x, y):
            return - jnp.sin(y)

        dudy = grad(dstream_func_dy, argnums=1)
        dvdx = grad(dstream_func_dx, argnums=0)
        du_dy = jnp.vectorize(dudy)(X, Y)
        dv_dx = jnp.vectorize(dvdx)(X, Y)
        omega  = dv_dx - du_dy
        omega_0 = jnp.fft.rfftn(omega)
        
        return omega_0
    
    def set_BCs(self):
        # Set the boundary conditions 
        pass

    def forcing_function(self, k, x, y):
        # Default kolmogorov forcing function 
        return ( jnp.sin(k*y), jnp.zeros_like(y) )
    
    @property
    def nu(self):
        return 1 / self.Re 
