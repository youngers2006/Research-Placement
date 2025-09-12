import sys
import jax
import jax.numpy as np
import os
from functools import partial
from jax import grad, hessian, random
# Import JAX-FEM specific modules
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from tqdm import tqdm

class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad)
    # to obtain the 'P_fn' function
    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy
        
        self.psi = psi

        P_fn = jax.grad(psi)
        self.P_fn = P_fn
        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        
        return first_PK_stress 
        
    def total_strain_energyTEMP(self, u):
        energy = 0.0
        u_grad_all = self.fes[0].sol_to_grad(u)  # shape: (num_cells, num_quads, dim, dim)
        weights = self.fes[0].JxW                # shape: (num_cells, num_quads)
    
        for cell_idx in range(u_grad_all.shape[0]):
            for q in range(u_grad_all.shape[1]):
                F = u_grad_all[cell_idx, q] + np.eye(self.dim)
                W = self.psi(F)
                energy += W * weights[cell_idx, q]
    
        return energy
    
    def total_strain_energy(self, u):
        # Get shape: (num_elements, num_quadrature_points, dim, dim)
        u_grad_all = self.fes[0].sol_to_grad(u)
        JxW = self.fes[0].JxW  # shape: (num_elements, num_quadrature_points)
    
        # Compute F = I + ∇u
        F = u_grad_all + np.eye(self.dim)
    
        # Vectorize psi over quadrature points
        psi_q = jax.vmap(jax.vmap(self.psi))(F)  # shape: (num_elements, num_quadrature_points)

        # Integrate energy
        energy = np.sum(psi_q * JxW)
        return energy
    
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

def top(point):
    return np.isclose(point[1], Ly, atol=1e-5)

def front(point):
    return np.isclose(point[2], 0., atol=1e-5)

def back(point):
    return np.isclose(point[2], Lz, atol=1e-5)

# Define Dirichlet boundary values
def zero_dirichlet_val(point):
    return 0.

def spatially_varying_displacement(point, key, scale):
    """Generates a unique random displacement for each spatial point."""
    weights = np.array([101, 757, 1553])
    point_hash = np.dot(point, weights).astype(np.int32)

    point_key = random.fold_in(key, point_hash)
    return random.normal(point_key) * scale

def Run_Sim():
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = 'data'

    # check directory exists
    os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)

    # define mesh
    Lx, Ly, Lz = 1., 1., 1.
    meshio_mesh = box_mesh_gmsh(
                Nx=9,
                Ny=9,
                Nz=9,
                domain_x=Lx,
                domain_y=Ly,
                domain_z=Lz,
                data_dir=data_dir,
                ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    num_simulations = 1000
    perturbation_scale = 0.0045 # Controls the magnitude of the random noise
    results = [] # List to store results from each simulation

    # Create a random key
    seed = 20
    key = random.PRNGKey(seed)

    for i in tqdm(range(num_simulations), leave=False):
        print(f"Running Simulation {i+1}/{num_simulations}")

        # Generate a key for each displacement component on each face 
        key, *subkeys = random.split(key, 19)
    
        # Create a displacement function for each component on each face
        face_fns = [partial(spatially_varying_displacement, key=k, scale=perturbation_scale) for k in subkeys]


