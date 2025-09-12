import os
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.nnx as nnx
from flax import struct
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Any
import jraph
import optax
from dataclasses import dataclass
from functools import partial
from JAX_FEM_active_learning import Run_Sim 
from ProjectUtils import restitch

class ActiveLearningModel:
    def __init__(self, seen_boundary_displacements, confidence_bound, Model, optimiser, learn_rate, epochs, alpha, gamma):
        self.Model = Model
        self.seen_bds = seen_boundary_displacements
        self.bound = confidence_bound
        self.LR = learn_rate
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.optimiser = optimiser
        
    @jax.jit
    def check_distances(self, applied_displacements):
        "Takes a batch of displacements and compares them to all displacements that have been seen by the model"
        seen_bds = self.seen_bds
        bound = self.bound

        def check_distance(applied_displacement, seen_displacements, bound):
            diff = seen_displacements - applied_displacement
            distances_sq = jnp.sum(jnp.square(diff), axis=(1,2))
            closest_vector_sq = jnp.min(distances_sq)
            should_query = (closest_vector_sq > bound)
            return should_query
        
        vmapped_check = jax.vmap(fun=check_distance, in_axes=(0, None, None))
        should_query_batch = vmapped_check(applied_displacements, seen_bds, bound)
        return should_query_batch
    
    def query_or_not(self, query_array):
        should_query = jnp.where(query_array)[0]
        not_query = jnp.where(~query_array)[0]
        return should_query, not_query
    
    def create_graphs(self, boundary_displacements: jax.Array) -> list:
        graphs = []
        boundary_nodes = self.Model.boundary_nodes
        base_nodes = self.Model.base_graph.nodes
        base_graph = self.Model.base_graph

        def create_graph(boundary_displacement, nodes, _graph, boundary_idx):
            new_nodes = nodes.at[boundary_idx].set(boundary_displacement)
            graph = _graph.replace(nodes=new_nodes)
            return graph
        
        create_graph_vmapped = jax.vmap(fun=create_graph, in_axes=(0, None, None, None))
        graphs = jraph.unbatch(create_graph_vmapped(boundary_displacements, base_nodes, base_graph, boundary_nodes))
        return graphs
    
    def loss_fn(self, target_e_batch, target_e_prime_batch, e_pred_batch, e_prime_pred_batch):
        loss_e = jnp.mean((e_pred_batch - target_e_batch)**2)
        loss_e_prime = jnp.mean((e_prime_pred_batch - target_e_prime_batch)**2)
        return (self.alpha * loss_e + self.gamma * loss_e_prime)
    
    @nnx.jit
    def train_step(self, target_e_batch, target_e_prime_batch, graphs_batch):
        def wrapped_loss(Model):
            e_pred_batch, e_prime_pred_batch = Model(graphs_batch)
            loss = self.loss_fn(
                target_e_batch,
                target_e_prime_batch,
                e_pred_batch,
                e_prime_pred_batch
            )
            return loss
    
        grads = nnx.grad(wrapped_loss, argnums=0)(self.Model)
        self.optimiser.update(self.Model, grads)
    
    def Learn(self, applied_displacement_graphs_list: jraph.GraphsTuple, target_e_from_sim, target_e_prime_from_sim):
        for _ in range(self.epochs):
            self.train_step(
                self.Model, 
                target_e_from_sim, 
                target_e_prime_from_sim, 
                applied_displacement_graphs_list
            )
    
    def query_fem(self, applied_displacements): # displacement batch (batch_size, num_nodes, 3)
        """calls a jax_fem sim function from another file"""
        batch_size = applied_displacements.shape[0]
        # need to vectorise this for the batch
        for disp_idx in range(batch_size):
            e, e_prime = Run_Sim(applied_displacements[disp_idx])
        return e, e_prime
        
    def __call__(self, applied_displacements: jax.Array) -> jax.Array:
        should_query = self.check_distance(applied_displacements)
        applied_displacement_graphs_list = self.create_graphs(applied_displacements) 
        query_idx, confident_idx = self.query_or_not(should_query)
        
        e_sim, e_prime_sim = self.query_fem(applied_displacements[query_idx]) 
        self.Learn(self.Model, applied_displacement_graphs_list[query_idx], e_sim, e_prime_sim)
        
        e_scaled, e_prime_scaled = self.Model.call_single(applied_displacement_graphs_list)
        e_predicted, e_prime_predicted = self.Model.unscale_predictions(e_scaled, e_prime_scaled)

        e_out = restitch(query_idx, confident_idx, e_sim, e_predicted)
        e_prime_out = restitch(query_idx, confident_idx, e_prime_sim, e_prime_predicted)
        return e_out, e_prime_out
           

