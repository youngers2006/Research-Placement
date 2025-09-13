import os
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.nnx as nnx
from flax import struct
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from typing import Any
import jraph
import optax
from dataclasses import dataclass
from functools import partial

@jax.jit
def restitch(idx_1, idx_2, array1: jax.Array, array2: jax.Array) -> jax.Array:
    "Takes 2 arrays that have been separated from eachother and recombines them"
    length = idx_1.shape[0] + idx_2.shape[0]
    output_shape = (length,) + array1.shape[1:]
    stitched_array = jnp.zeros(shape=output_shape, dtype=jnp.float32)
    stitched_array = stitched_array.at[idx_1].set(array1).at[idx_2].set(array2)
    return stitched_array

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

def mean_and_std_dev(data,*, train_split, permutated_idxs):
    permuted = jnp.array(data)[permutated_idxs]
    split_idx = int(permutated_idxs.shape[0] * train_split)
    train_data = permuted[:split_idx]
    mean = jnp.mean(train_data, axis=0)
    std_dev = jnp.std(train_data, axis=0)
    return {'mean':mean, 'std_dev':std_dev}

def scale_data(data,*, data_params):
    return (data - data_params['mean']) / data_params['std_dev']
    
def unscale_data(data,*,data_params):
    return (data * data_params['std_dev']) + data_params['mean']

