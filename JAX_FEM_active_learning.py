import sys
import jax
import jax.numpy as np
import os
from functools import partial
from jax import grad, hessian, random
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem.core import DirichletBC
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
    
@partial(jax.jit, static_argnums=(0,))
def Run_sim(mesh, boundary_node_indices, boundary_displacement_values):
    boundary_nodes_set = frozenset(boundary_node_indices.tolist())
    def boundary_location_fn(point, node_id):
        return node_id in boundary_nodes_set
    
    predetermined_bc = DirichletBC(
        loc_func=boundary_location_fn,
        value=boundary_displacement_values
    )

    problem = HyperElasticity(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type='HEX8'
    )

    problem.set_dbcs([predetermined_bc])

    sol_list = solver(problem, solver_options={
        'ksp_type': 'preonly', 
        'pc_type': 'lu', 
        'pc_factor_mat_solver_type': 'mumps'
    })

    u = sol_list[0]
    energy = problem.total_strain_energy(u)

    def energy_fn_wrt_boundary(boundary_disp, base_u, boundary_idx):
        new_u = base_u.at[boundary_idx].set(boundary_disp)
        return problem.total_strain_energy(new_u)

    grad_fn = jax.grad(lambda b_u: energy_fn_wrt_boundary(b_u, u, boundary_node_indices))
    boundary_energy_grad = grad_fn(boundary_displacement_values)
    energy_grad_reshaped = boundary_energy_grad.reshape(-1, 3)

    return energy, energy_grad_reshaped