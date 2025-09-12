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
import JAX_FEM_active_learning.Run_Sim as Simulate

class ActiveLearning:
    def __init__(self, seen_boundary_displacements, confidence_bound, Model, learn_rate, epochs, alpha, gamma):
        self.Model = Model
        self.seen_bds = seen_boundary_displacements
        self.bound = confidence_bound
        self.LR = learn_rate
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.optimiser =  nnx.Optimizer(
                                self.Model,
                                optax.adam(
                                    learning_rate=learn_rate, 
                                    b1=0.999, 
                                    b2=0.9
                                ),
                            wrt=nnx.Param
                        )
    @jax.jit
    def check_distance(self, current_displacement) -> bool:
        diff = self.seen_bds - current_displacement
        distances_sq = jnp.sum(jnp.square(diff), axis=(1,2))
        closest_vector_sq = jnp.min(distances_sq)
        should_query = (closest_vector_sq > self.bound)
        return should_query
    
    def create_graph(self, boundary_displacements):
        boundary_nodes = self.Model.boundary_nodes
        nodes = self.Model.base_graph.nodes
        new_nodes = nodes.at[boundary_nodes].set(boundary_displacements)
        graph = self.Model.base_graph.replace(nodes=new_nodes)
        return graph
    
    def loss_fn(self, target_e, target_e_prime, e_pred, e_prime_pred):
        loss_e = jnp.mean((e_pred - target_e)**2)
        loss_e_prime = jnp.mean((e_prime_pred - target_e_prime)**2)
        return (self.alpha * loss_e + self.gamma * loss_e_prime)
    
    @nnx.jit
    def train_step(self, target_e, target_e_prime, graph):
        def wrapped_loss(Model):
            e_pred, e_prime_pred = Model.call_single(graph)
            loss = self.loss_fn(
                target_e,
                target_e_prime,
                e_pred,
                e_prime_pred
            )
            return loss
    
        grads = nnx.grad(wrapped_loss, argnums=0)(self.Model)
        self.optimiser.update(self.Model, grads)
    
    def Learn(self, current_displacement_graph: jraph.GraphsTuple, target_e_from_sim, target_e_prime_from_sim):
        for _ in range(self.epochs):
            loss = self.train_step(
                self.Model, 
                target_e_from_sim, 
                target_e_prime_from_sim, 
                current_displacement_graph
            )
    
    def query_fem(self, current_displacement):
        """calls a jax_fem sim function from another file"""
        e = 0
        e_prime = 0
        return e, e_prime
        
    def __call__(self, current_displacement: jax.Array) -> jax.Array:
        should_query = self.check_distance(current_displacement)
        current_displacement_graph = self.create_graph(current_displacement)
        if should_query:
            e_sim, e_prime_sim = self.query_fem(current_displacement)
            self.Learn(self.Model, current_displacement, e_sim, e_prime_sim)
            return e_sim, e_prime_sim
        else:
            e_scaled, e_prime_scaled = self.Model.call_single(current_displacement_graph)
            e, e_prime = self.Model.unscale_predictions(e_scaled, e_prime_scaled)
            return e, e_prime

