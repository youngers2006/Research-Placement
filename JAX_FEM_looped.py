import sys
import jax
import jax.numpy as np
import os
from functools import partial
from jax import grad, hessian, random


# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh

# Define constitutive relationship.
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad)
    # to obtain the 'P_fn' function.
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

# Specify mesh-related information (first-order hexahedron element).
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = 'data'
# Ensure the VTK directory exists before starting.
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

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)


# Define Dirichlet boundary values
def zero_dirichlet_val(point):
    return 0.

def random_displacement(point, key, scale):
    """Generates a random displacement."""
    return random.normal(key) * scale

# Simulation Loop
num_simulations = 10000
perturbation_scale = 0.0045 # Controls the magnitude of the random noise
results = [] # List to store results from each simulation

# Create random key.
seed = 0
key = random.PRNGKey(seed)

for i in range(num_simulations):
    # new key for this simulation by splitting the master key
    key, subkey_y, subkey_z = random.split(key, 3)

    print(f"--- Running Simulation {i+1}/{num_simulations} ---")

    # Define the boundary condition values for the current random keys
    # The displacement is now purely random
    displace_y_fn = partial(random_displacement, key=subkey_y, scale=perturbation_scale)
    displace_z_fn = partial(random_displacement, key=subkey_z, scale=perturbation_scale)

    # The 'dirichlet_bc_info' is defined with the purely random displacement functions
    dirichlet_bc_info = [
        [left, left, left, right, right, right], # Boundary location functions
        [0, 1, 2, 0, 1, 2],                      # Components (0=x, 1=y, 2=z)
        [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val,
         zero_dirichlet_val, displace_y_fn, displace_z_fn] # Displacement value functions
    ]

    # Create an instance of the problem for the current simulation step.
    problem = HyperElasticity(mesh,
                              vec=3,
                              dim=3,
                              ele_type=ele_type,
                              dirichlet_bc_info=dirichlet_bc_info)

    # Solve the defined problem.
    sol_list = solver(problem, use_petsc=True, petsc_options={
        'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
    })
    
    # Get the solution 'u' (displacement field).
    u = sol_list[0]

    # Post-processing for this simulation step.
    energy = problem.total_strain_energy(u)

    # <--- CHANGE START: GRADIENT CALCULATION IS NOW MORE SPECIFIC --- >
    # 1. Identify the DOFs for the randomly displaced boundary (y and z components on the right face).
    right_node_inds = np.where(right(mesh.points))[0]
    dofs_y = right_node_inds * problem.vec + 1 # y-component DOFs
    dofs_z = right_node_inds * problem.vec + 2 # z-component DOFs
    boundary_dofs = np.sort(np.hstack([dofs_y, dofs_z]))

    # 2. Define a function that computes energy from ONLY the boundary displacements.
    # It takes a vector of boundary displacements, reconstructs the full solution
    # vector `u`, and returns the total energy. This focuses jax.grad.
    def energy_fn_wrt_boundary(boundary_disp, base_u, dofs_map):
        full_u = base_u.at[dofs_map].set(boundary_disp)
        return problem.total_strain_energy(full_u)

    # 3. Compute the gradient of this new function.
    # Get the actual displacement values at the boundary from the full solution vector.
    boundary_u_from_sol = u[boundary_dofs]
    
    # Create a lambda function to compute the gradient of. This tells JAX that
    # 'boundary_disp' is the variable and 'base_u' and 'dofs_map' are constants.
    grad_fn = jax.grad(
        lambda b_u: energy_fn_wrt_boundary(b_u, u, boundary_dofs)
    )
    boundary_energy_grad = grad_fn(boundary_u_from_sol)
    # <--- CHANGE END: GRADIENT CALCULATION --- >


    results.append({
        'simulation': i,
        'strain_energy': energy,
        'boundary_strain_energy_gradient': boundary_energy_grad # Store the new, smaller gradient
    })
    print(f"Strain Energy = {energy:.6f}, Boundary Gradient Norm = {np.linalg.norm(boundary_energy_grad):.6f}")

# --- End of Simulation Loop ---
print("\n--- All simulations complete. ---")

# Save the collected results to a file for later analysis.
import json
results_path = os.path.join(data_dir, 'simulation_results_random_displacements.json')
with open(results_path, 'w') as f:
    # Convert JAX arrays to standard Python lists of floats for JSON serialization
    serializable_results = []
    for r in results:
        serializable_results.append({
            'simulation': r['simulation'],
            'strain_energy': float(r['strain_energy']),
            # The gradient is now a different variable name and shape
            'boundary_strain_energy_gradient': r['boundary_strain_energy_gradient'].tolist()
        })
    json.dump(serializable_results, f) # Writing without indent for smaller file size
print(f"\nFull results saved to {results_path}")
# <--- CHANGE END --- >


# The following is your original post-processing code.
# It will now run on the results of the *last* simulation from the loop.
print("\n--- Post-processing final simulation ---")

# Store the solution to local file.
vtk_path = os.path.join(data_dir, f'vtk/u_final.vtu')
save_sol(problem.fes[0], u, vtk_path)
print(f"Saved final displacement field to {vtk_path}")

# The energy and gradient are already computed, but we can re-display the final ones.
final_energy = results[-1]['strain_energy']
final_boundary_energy_grad = results[-1]['boundary_strain_energy_gradient']
print(f"Final strain energy = {final_energy}")

# <--- CHANGE START: Visualize the boundary-specific gradient --- >
# Store the final gradient to a local file.
vtk_path = os.path.join(data_dir, f'vtk/Jac_final_boundary.vtu')

# To visualize the gradient, create a zero vector of the full size and
# place the computed gradient values at the correct boundary DOF locations.
full_grad_vector = np.zeros_like(u)
full_grad_vector = full_grad_vector.at[boundary_dofs].set(final_boundary_energy_grad)

# Now, save the magnitude of this sparse gradient vector for visualization.
grad_vec_mag = np.linalg.norm(full_grad_vector.reshape(-1, 3), axis=1)
save_sol(problem.fes[0], grad_vec_mag, vtk_path, is_nodal_sol=True)
print(f"Saved final boundary energy gradient magnitude to {vtk_path}")
# <--- CHANGE END --- >