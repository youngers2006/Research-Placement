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
from dataclasses import dataclass

class ActiveLearning:
    def __init__(self, seen_boundary_displacements):
        for i in range(len(seen_boundary_displacements)):
            = jnp.linalg.norm(seen_boundary_displacements)
            self.displacement_vectors_norm.append()

    def check_distance(self, current_displacement) -> bool:

        return should_query

