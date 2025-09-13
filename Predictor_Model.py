import os
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.nnx as nnx
from flax import struct
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Any
import jraph
from itertools import combinations
import meshio
import numpy as np
from dataclasses import dataclass

@dataclass
class GATEdges:
    score: jax.Array
    message: jax.Array

class GAT(nnx.Module):
    """
    Desc: Graph attention layer, does attention based message passing between nodes.
    Notes: jit friendly, uses summation pooling for message passing. Uses GATEdges dataclass so it is needed.
    Inputs: Graph: jraph.GraphsTuple
    Outputs: Processed graph: jraph.GraphsTuple
    """
    def __init__(self, in_features, out_features, *, rngs):
        self.in_features = in_features
        self.out_features = out_features

        self.GNN = jraph.GraphNetwork(
            update_edge_fn=self.update_edge_fn,
            update_node_fn=self.update_node_fn,
            aggregate_edges_for_nodes_fn=self.aggregate_edges_for_nodes_fn
        )

        initialiser = nnx.initializers.lecun_normal()
        self.Weight_mat = nnx.Param(initialiser(rngs.params(), (in_features, out_features)))
        self.Attention_mat = nnx.Param(initialiser(rngs.params(), (2 * out_features, 1)))

    def update_edge_fn(self, edges, senders, receivers, globals_):
        "computes edge features and outputs a dict containing the edge features"
        h_sender = senders @ self.Weight_mat
        h_reciever = receivers @ self.Weight_mat

        send_recieve_features = jnp.concatenate([h_sender, h_reciever], axis=-1)
        attention_scores = nnx.leaky_relu(send_recieve_features @ self.Attention_mat)

        return GATEdges(score=attention_scores, message=h_sender)
    
    def aggregate_edges_for_nodes_fn(self, edges: GATEdges, segment_ids, num_segments) -> jax.Array:
        "aggregates all edge messages for a node and outputs the aggregated messages"
        attention_coeffs = jraph.segment_softmax(
            logits=edges.score,
            segment_ids=segment_ids,
            num_segments=num_segments
        )

        weighted_messages = edges.message * attention_coeffs

        aggregated_messages = jraph.segment_sum(
            data=weighted_messages,
            segment_ids=segment_ids, 
            num_segments=num_segments
        )

        return aggregated_messages

    def update_node_fn(self, nodes, sent_edges, received_edges, globals_):
        "takes aggregated node messages and applies them to the graph"
        return received_edges

    def __call__(self, graph):
        return self.GNN(graph)

class BlockCoursening(nnx.Module): 
    """
    desc: Pooling operation, takes a mesh and coarsens it by combining blocks of nodes into supernodes.
    Notes: jit friendly.
    Inputs: 1.Graph to be pooled:jraph.GraphsTuple, 2. Geometric position of each node: jax.Array.
    Outputs: Coarsened graph: jraph.GraphsTuple
    """
    def __init__(self, block_size: tuple[int, int, int], max_coarsened_edges=10000, *, rngs: nnx.Rngs):
        self.block_dims = tuple(int(x) for x in block_size)
        self.num_blocks = int(self.block_dims[0] * self.block_dims[1] * self.block_dims[2])

    def partition(self, node_coords):
        dims_arr = jnp.array(self.block_dims)
        min_coords = jnp.min(node_coords, axis=0)
        max_coords = jnp.max(node_coords, axis=0)
        grid_cell_size = (max_coords - min_coords) / dims_arr
        relative_coords = node_coords - min_coords

        normalized_coords = relative_coords / grid_cell_size
        grid_indices = jnp.floor(normalized_coords).astype(jnp.int32)
        grid_indices = jnp.clip(grid_indices, 0, dims_arr - 1)
    
        block_ids = (
                grid_indices[:, 0] * (self.block_dims[1] * self.block_dims[2]) + 
                grid_indices[:, 1] * self.block_dims[2] + grid_indices[:, 2]
        )
        return block_ids
    
    def __call__(self, graph: jraph.GraphsTuple, node_coords: jax.Array):
        block_ids = self.partition(node_coords)
        num_blocks = self.num_blocks

        coarsened_nodes = jraph.segment_sum(
            data=graph.nodes,
            segment_ids=block_ids,
            num_segments=num_blocks
        )

        sizes = jraph.segment_sum(
            jnp.ones((graph.nodes.shape[0],1)), 
            block_ids, 
            num_blocks
        )          
        coarsened_nodes = coarsened_nodes / jnp.sqrt(sizes + 1e-10)

        block_senders = block_ids[graph.senders]
        block_receivers = block_ids[graph.receivers]

        not_self = block_senders != block_receivers

        keys = block_senders * num_blocks + block_receivers
        idx = jnp.arange(keys.shape[0])
        sorted_keys, perm = jax.lax.sort_key_val(keys, idx)  
    
        sentinel = jnp.array([-1], dtype=sorted_keys.dtype)
        prev = jnp.concatenate([sentinel, sorted_keys[:-1]])
        unique_sorted = sorted_keys != prev

        unique_mask = jnp.zeros_like(unique_sorted)
        unique_mask = unique_mask.at[perm].set(unique_sorted)

        final_mask = (unique_mask & not_self).astype(jnp.float32)  
        edge_weights = final_mask[:, None]

        pooled_graph = jraph.GraphsTuple(
            nodes=coarsened_nodes,
            edges=edge_weights,
            senders=block_senders,
            receivers=block_receivers,
            n_edge=jnp.array([edge_weights.shape[0]]),
            n_node=jnp.array([num_blocks]),
            globals=None
        )
        return pooled_graph

class GraphNorm(nnx.Module):
    """
    desc: Standardises the nodes of a graph passed through it with the mean and variance of all node features.
    Notes: Function is jit friendly 
    Inputs: Node features - jax.Array
    Outputs: Standardised Node features - jax.Array
    """
    def __init__(self, dim: int, *, eps: float = 1e-10, rngs: nnx.Rngs):
        self.eps = eps
        self.gamma = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))
        self.beta = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=0, keepdims=True)
        var = jnp.var(x, axis=0, keepdims=True)
        x_hat = (x - mean) / jnp.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class GNN(nnx.Module):
    """
    Desc: Main Graph Network class to predict strain energy and strain energy sensitivity wrt the boundary displacements.
    Notes: jit friendly
    Graph: jraph.GraphsTuple, Node Features: [node position, node displacement, known displacement flag].
    Input: Single graph (if called with .call_single()), list of graphs (if called with .__call__()), all graphs: jraph.GraphsTuple
    Output: 1. Strain energy: jax.Array, 2. Strain energy sensitivity.
    """
    def __init__(
            self, 
            node_feature_dim: int, 
            embedding_dim: int, 
            output_dim: int, 
            pooling_block_dims_1,  
            boundary_nodes, 
            base_graph,
            disp_mean,
            disp_std,
            e_mean,
            e_std,
            grad_mean, 
            grad_std,
            rngs: nnx.Rngs
        ):

        self.embedding_layer = nnx.Linear(node_feature_dim, embedding_dim, rngs=rngs)
        self.encoderL1 = GAT(embedding_dim, embedding_dim, rngs=rngs)
        self.graphNormL1 = GraphNorm(embedding_dim, eps=1e-5, rngs=rngs)
        self.encoderL2 = GAT(embedding_dim, embedding_dim, rngs=rngs)
        self.graphNormL2 = GraphNorm(embedding_dim, eps=1e-5, rngs=rngs)
        self.encoderL3 = GAT(embedding_dim, embedding_dim, rngs=rngs)
        self.decoding_layer = nnx.Linear(embedding_dim, output_dim, rngs=rngs)

        self.poolingLayer1 = BlockCoursening(pooling_block_dims_1, rngs=rngs)

        self.boundary_nodes = jnp.array(boundary_nodes, dtype=jnp.int32)
        self.base_graph = base_graph

        self.disp_mean = disp_mean ; self.disp_std = disp_std
        self.e_mean = e_mean ; self.e_std = e_std
        self.grad_mean = grad_mean ; self.grad_std = grad_std

    def embedder(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Maps current node features to higher dimensional node embeddings"""
        nodes = graph.nodes
        pos = nodes[:, 0:3]
        disp = nodes[:, 3:6]
        flag = nodes[:, 6:]

        disp_norm = disp
        b_disp = disp[self.boundary_nodes]
        b_disp_norm = (b_disp - self.disp_mean) / self.disp_std
        disp_norm = disp_norm.at[self.boundary_nodes].set(b_disp_norm)
        
        nodes_scaled = jnp.concatenate([pos, disp_norm, flag], axis=1)
        embeddings = self.embedding_layer(nodes_scaled)
        return graph._replace(nodes=embeddings)
    
    def apply_activation_and_res(self, graph: jraph.GraphsTuple, residual: jax.Array) -> jraph.GraphsTuple:
        """Applies activation function and residual to the graph"""
        activated_nodes = nnx.silu(graph.nodes) + residual
        return graph._replace(nodes=activated_nodes)
    
    def apply_res(self, graph: jraph.GraphsTuple, residual: jax.Array) -> jraph.GraphsTuple:
        """Applies activation_function to the graph"""
        new_nodes = graph.nodes + residual
        return graph._replace(nodes=new_nodes)
        
    def decoder(self, graph: jraph.GraphsTuple) -> jax.Array: 
        """Takes processed graph and aggregates nodes then passes them through the decoding layer to predict energy"""
        num_nodes = graph.nodes.shape[0]
        node_graph_indices = jnp.zeros(num_nodes, dtype=jnp.int32)
        aggregate_nodes = jraph.segment_sum(
            data=graph.nodes, 
            segment_ids=node_graph_indices,
            num_segments=1
        )
        out = self.decoding_layer(aggregate_nodes)
        return out.squeeze()
        
    def forward_pass(self, G: jraph.GraphsTuple) -> jax.Array:
        """Takes a graph and passes it through the GNN to predict energy"""
        node_coords = G.nodes[:, 0:3]
        G = self.embedder(G)
        res1 = G.nodes

        G = self.encoderL1(G)
        nodes_norm = self.graphNormL1(G.nodes)
        G = G._replace(nodes=nodes_norm)
        G = self.apply_activation_and_res(G, res1)
        G = self.poolingLayer1(G, node_coords)
        res2 = G.nodes

        G = self.encoderL2(G)
        nodes_norm = self.graphNormL2(G.nodes)
        G = G._replace(nodes=nodes_norm)
        G = self.apply_activation_and_res(G, res2)
        res3 = G.nodes

        G = self.encoderL3(G)
        G = self.apply_res(G, res3)

        e = self.decoder(G)
        return e
    
    def call_single(self, G: jraph.GraphsTuple):
        """Call the GNN for a single graph"""
        e = self.forward_pass(G)

        def energy_fn(nodes):
            G_temp = G._replace(nodes=nodes)
            result = self.forward_pass(G_temp)
            return result.squeeze()

        grad_fn = jax.grad(energy_fn)
        grads = grad_fn(G.nodes)
        e_prime_raw_full = grads[:, 3:6]
        e_prime_raw = e_prime_raw_full[self.boundary_nodes, :]

        e_prime_physical = e_prime_raw * (self.e_std / self.disp_std)
        e_prime = (e_prime_physical - self.grad_mean) / self.grad_std

        return e, e_prime
    
    def __call__(self, graphs_list: list):
        "Main call method for handling batches of identically structured graphs"
    
        base = graphs_list[0]
        stacked_nodes = jnp.stack([g.nodes for g in graphs_list])

        def per_sample(nodes_slice):
            g = base._replace(nodes=nodes_slice)
            return self.call_single(g)
        
        vmapped_call = jax.vmap(
            per_sample,
            in_axes=0
        )
        e_batch, e_prime_batch = vmapped_call(stacked_nodes)
        return e_batch, e_prime_batch
    
    def unscale_predictions(self, e_scaled, e_prime_scaled):
        """Unscales the predicted data to obtain physical predictions, should not be called during training"""
        e_physical = (e_scaled * self.e_std) + self.e_mean
        e_prime_physical = (e_prime_scaled * self.grad_std) + self.grad_mean
        return e_physical, e_prime_physical