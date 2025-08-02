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

# Define constitutive relationship
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

# Specify mesh-related information (first-order hexahedron element)
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = 'data'

os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)

Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(
                Nx=10,
                Ny=10,
                Nz=10,
                domain_x=Lx,
                domain_y=Ly,
                domain_z=Lz,
                data_dir=data_dir,
                ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# Define boundary locations for all six faces of the cube
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


# This function applies a purely random displacement
def random_displacement(point, key, scale):
    """Generates a random displacement."""
    return random.normal(key) * scale

# Simulation Loop
num_simulations = 10000
perturbation_scale = 0.0045 # Controls the magnitude of the random noise
results = [] # List to store results from each simulation

# Create a random key
seed = 0
key = random.PRNGKey(seed)

for i in range(num_simulations):
    print(f"Running Simulation {i+1}/{num_simulations}")

    # Generate a key for each displacement component on each face 
    key, *subkeys = random.split(key, 19)
    
    # Create a displacement function for each component on each face
    face_fns = [partial(random_displacement, key=k, scale=perturbation_scale) for k in subkeys]

    # The 'dirichlet_bc_info' is defined with random displacements on all 6 faces
    dirichlet_bc_info = [
        # Location functions for each boundary condition
        [left, left, left,
         right, right, right,
         bottom, bottom, bottom,
         top, top, top,
         front, front, front,
         back, back, back],
        [0, 1, 2] * 6,
        # The 18 unique displacement functions
        face_fns
    ]

    # Create an instance of the problem for the current simulation step
    problem = HyperElasticity(mesh,
                              vec=3,
                              dim=3,
                              ele_type=ele_type,
                              dirichlet_bc_info=dirichlet_bc_info)

    # Solve the defined problem
    sol_list = solver(problem, use_petsc=True, petsc_options={
        'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
    })
    
    # Get the displacement field 
    u = sol_list[0]

    # Post-processing for this simulation step
    energy = problem.total_strain_energy(u)

    # Identify all boundary DOFs
    all_boundary_nodes = np.unique(np.hstack([
        np.where(left(mesh.points))[0], np.where(right(mesh.points))[0],
        np.where(bottom(mesh.points))[0], np.where(top(mesh.points))[0],
        np.where(front(mesh.points))[0], np.where(back(mesh.points))[0]
    ]))
    
    boundary_dofs_x = all_boundary_nodes * problem.vec + 0
    boundary_dofs_y = all_boundary_nodes * problem.vec + 1
    boundary_dofs_z = all_boundary_nodes * problem.vec + 2
    boundary_dofs = np.sort(np.hstack([boundary_dofs_x, boundary_dofs_y, boundary_dofs_z]))

    # Define a function that computes energy from ONLY the boundary displacements
    def energy_fn_wrt_boundary(boundary_disp, base_u, dofs_map):
        full_u = base_u.at[dofs_map].set(boundary_disp)
        return problem.total_strain_energy(full_u)

    # Compute the gradient of this new function
    boundary_u_from_sol = u[boundary_dofs]
    grad_fn = jax.grad(lambda b_u: energy_fn_wrt_boundary(b_u, u, boundary_dofs))
    boundary_energy_grad = grad_fn(boundary_u_from_sol)

    results.append({
        'simulation': i,
        'strain_energy': energy,
        'boundary_strain_energy_gradient': boundary_energy_grad,
        'applied_boundary_displacements': boundary_u_from_sol, # The input displacements
        'full_displacement_vector': u # The full output displacement vector
    })

    print(f"Strain Energy = {energy:.6f}, Boundary Gradient Norm = {np.linalg.norm(boundary_energy_grad):.6f}")

print("\n All simulations complete.")

import pickle
results_path = os.path.join(data_dir, 'simulation_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\n Full results saved to {results_path}")

print("\n Post-processing final simulation")

# Store the solution to local file
vtk_path = os.path.join(data_dir, f'vtk/u_final.vtu')
save_sol(problem.fes[0], u, vtk_path)
print(f"Saved final displacement field to {vtk_path}")

final_energy = results[-1]['strain_energy']
final_boundary_energy_grad = results[-1]['boundary_strain_energy_gradient']
vtk_path = os.path.join(data_dir, f'vtk/Jac_final_boundary.vtu')

# To visualize the gradient, create a zero vector of the full size and
# place the computed gradient values at the correct boundary DOF locations
full_grad_vector = np.zeros_like(u)
full_grad_vector = full_grad_vector.at[boundary_dofs].set(final_boundary_energy_grad)

# save the magnitude of this sparse gradient vector for visualization
grad_vec_mag = np.linalg.norm(full_grad_vector.reshape(-1, 3), axis=1)
save_sol(problem.fes[0], grad_vec_mag, vtk_path, is_nodal_sol=True)
print(f"Saved final boundary energy gradient magnitude to {vtk_path}")