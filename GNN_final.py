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

def Get_known(boundary_points, points):
    is_known = jnp.zeros(points.shape[0]) 
    is_known = is_known.at[boundary_points].set(1)
    return is_known

def build_send_receive(cell):
    sender_array = []
    receiver_array = []
    for edge in combinations(cell,2):
        sender_array.append(edge[0])
        receiver_array.append(edge[1])
    return sender_array, receiver_array

def build_graphs(senders, receivers, positions, boundary_points, U) -> jraph.GraphsTuple:
    is_known = Get_known(boundary_points, positions)
    U_applied = jnp.zeros_like(U).at[boundary_points].set(U[boundary_points])
        
    node_features = jnp.concatenate([positions, U_applied, jnp.expand_dims(is_known, axis=1)], axis=1)
    num_nodes = positions.shape[0]

    graph = jraph.GraphsTuple(
        nodes=node_features,
        senders=senders,
        receivers=receivers,
        edges=None,
        globals=None, 
        n_node=jnp.array([num_nodes]),
        n_edge=jnp.array([len(senders)])
    )
    return graph

def mean_and_std_dev(data,*,train_split):
    split_idx = int(data.shape[0] * train_split)
    train_data = data[:split_idx]
    mean = jnp.mean(train_data, axis=0)
    std_dev = jnp.std(train_data, axis=0)
    return {'mean':mean, 'std_dev':std_dev}

def scale_data(data,*, data_params):
    return (data - data_params['mean']) / data_params['std_dev']
    
def unscale_data(data,*,data_params):
    return (data * data_params['std_dev']) + data_params['mean']

def batch_and_split_dataset(dataset_dict, batch_size, train_split, CV_split, test_split, permutated_index_list):
    total_samples = permutated_index_list.shape[0]
    idx_train_samples = int(train_split * total_samples) 
    idx_test_samples = idx_train_samples + int(test_split * total_samples) 

    train_idx = list(permutated_index_list[:idx_train_samples])
    test_idx = list(permutated_index_list[idx_train_samples:idx_test_samples])
    CV_idx = list(permutated_index_list[idx_test_samples:])

    def batch_indices(idx):  
        if not idx:
            return
        num_samples = len(idx)
        num_batches = num_samples // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_idx = idx[start:end]
            
            graphs_in_batch = [dataset_dict['graphs_list'][i] for i in batch_idx]
            displacements_batch = [dataset_dict['displacements'][i] for i in batch_idx]
            e_batch = [dataset_dict['target_e'][i] for i in batch_idx]
            e_prime_batch = [dataset_dict['target_e_prime'][i] for i in batch_idx]

            batched_graphs = graphs_in_batch
            batched_displacements = jnp.array(displacements_batch)
            batched_e = jnp.array(e_batch)
            batched_e_prime = jnp.array(e_prime_batch)

            yield {
                'graphs': batched_graphs, 
                'displacements': batched_displacements, 
                'target_e': batched_e, 
                'target_e_prime': batched_e_prime
            }
    
    train_batches = list(batch_indices(train_idx))
    test_batches = list(batch_indices(test_idx))
    CV_batches = list(batch_indices(CV_idx))

    return train_batches, CV_batches, test_batches

@dataclass
class GATEdges:
    score: jax.Array
    message: jax.Array

class GAT(nnx.Module):
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

        aggregated_messages = jraph.segment_mean(
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
    def __init__(self, block_size, *, rngs: nnx.Rngs):
        self.block_dims = block_size

    def partition(self, node_coords):
        min_coords = jnp.min(node_coords, axis=0)
        max_coords = jnp.max(node_coords, axis=0)
        grid_cell_size = (max_coords - min_coords) / jnp.array(self.block_dims)
        relative_coords = node_coords - min_coords

        normalized_coords = relative_coords / grid_cell_size
        grid_indices = jnp.floor(normalized_coords).astype(jnp.int32)
    
        grid_indices = jnp.clip(grid_indices, 0, jnp.array(self.block_dims) - 1)
    
        block_ids = (grid_indices[:, 0] * self.block_dims[1] * self.block_dims[2] + 
                 grid_indices[:, 1] * self.block_dims[2] + 
                 grid_indices[:, 2]
            )
    
        return block_ids
    
    def __call__(self, graph: jraph.GraphsTuple, node_coords):
        block_ids = self.partition(node_coords)
        num_blocks = jnp.max(block_ids) + 1

        coarsened_nodes = jraph.segment_mean(
            data=graph.nodes,
            segment_ids=block_ids,
            num_segments=num_blocks
        )

        block_senders = block_ids[graph.senders]
        block_receivers = block_ids[graph.receivers]

        inter_block_mask = block_senders != block_receivers
        coarsened_senders_unfiltered = block_senders[inter_block_mask]
        coarsened_receivers_unfiltered = block_receivers[inter_block_mask]

        hashed_edge_ids = coarsened_senders_unfiltered * num_blocks + coarsened_receivers_unfiltered
        unique_edge_ids = jnp.unique(hashed_edge_ids)

        coarsened_senders = unique_edge_ids // num_blocks
        coarsened_receivers = unique_edge_ids % num_blocks
        coarsened_edges = jnp.ones(shape=(coarsened_senders.shape[0], 1))
        
        pooled_graph = jraph.GraphsTuple(
            nodes=coarsened_nodes,
            edges=coarsened_edges,
            senders=coarsened_senders,
            receivers=coarsened_receivers,
            n_edge=jnp.array([coarsened_edges.shape[0]]),
            n_node=jnp.array([num_blocks]),
            globals=None
        )
        return pooled_graph
    
class GNN(nnx.Module):
    def __init__(self, node_feature_dim: int, embedding_dim: int, output_dim: int, pooling_block_dims_1, rngs: nnx.Rngs):
        self.embedding_layer = nnx.Linear(node_feature_dim, embedding_dim, rngs=rngs)
        self.encoderL1 = GAT(embedding_dim, embedding_dim, rngs=rngs)
        self.BatchNormL1 = nnx.BatchNorm(num_features=embedding_dim, rngs=rngs)
        self.encoderL2 = GAT(embedding_dim, embedding_dim, rngs=rngs)
        self.BatchNormL2 = nnx.BatchNorm(num_features=embedding_dim, rngs=rngs)
        self.encoderL3 = GAT(embedding_dim, embedding_dim, rngs=rngs)
        self.decoding_layer = nnx.Linear(embedding_dim, output_dim, rngs=rngs)
        self.poolingLayer1 = BlockCoursening(pooling_block_dims_1, rngs=rngs)
    
    def embedder(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Maps current node features to higher dimensional node embeddings"""
        nodes = graph.nodes
        embeddings = self.embedding_layer(nodes)
        return graph._replace(nodes=embeddings)
    
    def apply_activation_and_res(self, graph: jraph.GraphsTuple, residual: jax.Array) -> jraph.GraphsTuple:
        """Applies activation function and residual to the graph"""
        nodes = graph.nodes
        activated_nodes = nnx.silu(nodes) + residual
        return graph._replace(nodes=activated_nodes)
    
    def apply_res(self, graph: jraph.GraphsTuple, residual: jax.Array) -> jraph.GraphsTuple:
        """Applies activation_function to the graph"""
        new_nodes = graph.nodes + residual
        return graph._replace(nodes=new_nodes)
        
    def decoder(self, graph: jraph.GraphsTuple) -> jax.Array: 
        """Takes processed graph and aggregates nodes then passes them through the decoding layer to predict energy"""
        num_nodes = graph.nodes.shape[0]
        node_graph_indices = jnp.zeros(num_nodes)
        aggregate_nodes = jraph.segment_sum(
            data=graph.nodes, 
            segment_ids=node_graph_indices,
            num_segments=1
        )
        return self.decoding_layer(aggregate_nodes)
        
    def forward_pass(self, G: jraph.GraphsTuple, use_running_average: bool) -> jax.Array:
        """Takes a graph and passes it through the GNN to predict energy"""
        node_coords = G.nodes[:,0:3]
        G = self.embedder(G)
        res1 = G.nodes

        G = self.encoderL1(G)
        nodes_norm = self.BatchNormL1(
            G.nodes, 
            use_running_average=use_running_average
        )
        G = G._replace(nodes=nodes_norm)
        G = self.apply_activation_and_res(G, res1)
        G = self.poolingLayer1(G, node_coords)
        res2 = G.nodes

        G = self.encoderL2(G)
        nodes_norm = self.BatchNormL2(
            G.nodes,
            use_running_average=use_running_average
        )
        G = G._replace(nodes=nodes_norm)
        G = self.apply_activation_and_res(G, res2)
        res3 = G.nodes

        G = self.encoderL3(G)
        G = self.apply_res(G, res3)

        e = self.decoder(G)
        return e
    
    def call_single(self, G: jraph.GraphsTuple, use_running_average: bool):
        """Call the GNN for a single graph"""
        e = self.forward_pass(G, use_running_average)

        def energy_fn(nodes):
            G_temp = G._replace(nodes=nodes)
            result = self.forward_pass(G_temp, use_running_average=True)
            return result.squeeze()

        grad_fn = jax.grad(energy_fn)
        grads = grad_fn(G.nodes)
        e_prime = grads[:, 3:6]
        return e, e_prime
    
    def __call__(self, graphs_list: list, use_running_average: bool):
        "Main call method for handling batches of identically structured graphs"
    
        stacked_nodes = jnp.stack([g.nodes for g in graphs_list])
        first_graph = graphs_list[0]
        
        batched_graph = jraph.GraphsTuple(
            nodes=stacked_nodes,  # Shape: (batch_size, num_nodes, num_features)
            senders=first_graph.senders, 
            receivers=first_graph.receivers,  
            edges=None,
            globals=None,
            n_node=first_graph.n_node,  
            n_edge=first_graph.n_edge   
        )
        
        vmapped_call = jax.vmap(
            self.call_single, 
            in_axes=(
                jraph.GraphsTuple(
                nodes=0,     
                senders=None,  
                receivers=None,
                edges=None,
                globals=None,
                n_node=None,   
                n_edge=None    
                ), 
            None 
            ),
        )
        e_batch, e_prime_batch = vmapped_call(batched_graph, use_running_average)
        return e_batch, e_prime_batch
        
def loss_fn(batch, batched_zero_graph,*, Model, use_running_average, alpha, gamma, lam): 
    """
    Calculates the loss of a model, works to minimise the mean square error of both 
    the strain energy prediction and the strain energy derivative prediction,
    whilst forcing the function through zero.
    """
    target_e_batch = batch['target_e']
    target_e_prime_batch = batch['target_e_prime']
    graph_batch = batch['graphs']
    
    prediction_e, prediction_e_prime = Model(graph_batch, use_running_average)
    loss_e = jnp.mean((prediction_e - target_e_batch)**2)
    loss_e_prime = jnp.mean((prediction_e_prime - target_e_prime_batch)**2)
    
    prediction_zero, _ = Model(batched_zero_graph, use_running_average=False)
    loss_zero = jnp.mean((prediction_zero - 0.0)**2)

    return (alpha * loss_e + gamma * loss_e_prime + lam * loss_zero)

def CV_loss_fn(CV_batches, batched_zero_graph, Model: GNN, alpha, gamma, lambda_):
    CV_loss = 0
    batch_count = 0

    Model.eval()
    for CV_batch in CV_batches:
        batch_count += 1

        loss = loss_fn(
            CV_batch,
            batched_zero_graph,
            Model=Model,
            use_running_average=True,
            alpha=alpha,
            gamma=gamma,
            lam=lambda_
        )

        CV_loss += loss

    if batch_count > 0:
        CV_loss = CV_loss / batch_count
        return CV_loss
    else:
        return 0

@nnx.jit
def train_step(Model, optimiser, GraphandTarget_batch, batched_zero_graph, *, alpha, gamma, lambda_):

    def wrapped_loss(Model):
        loss = loss_fn(
            GraphandTarget_batch,
            batched_zero_graph,
            Model=Model,
            use_running_average=False,
            alpha=alpha,
            gamma=gamma,
            lam=lambda_ 
        )
        return loss
    
    loss, grads = nnx.value_and_grad(wrapped_loss, argnums=0)(Model)
    optimiser.update(grads)

    return loss






def main():
    Epochs = 50
    alpha = 1.0 ; gamma = 1.0 ; lambda_ = 1.0
    beta_1 = 0.9 ; beta_2 = 0.999
    batch_size = 10 
    train_split = 0.9 ; CV_split = 0.05 ; test_split = 0.05
    Learn_Rate = 0.001

    seed = 42 # This can be changed but is here to make the results easy to reproduce
    base_key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(base_key)

    





if __name__ == "__main__":
    print()