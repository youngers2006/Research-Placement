import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm import tqdm
from typing import Any
import sys
import types
import pickle

# ===============================================================
# 1. DATA LOADING & PREPARATION
# (Using your original data loading code)
# ===============================================================

# Unpickling fix for your custom DataStore object
fake_module = types.ModuleType("DataSetup")
class DataStore:
    def __init__(self):
        pass
fake_module.DataStore = DataStore
sys.modules["DataSetup"] = fake_module

# --- IMPORTANT: Make sure these file paths are correct ---
data_file_1 = r"C:\Users\samue\Downloads\Simulation.pickle"
data_file_2 = r"C:\Users\samue\Downloads\Simulation 2.pickle"

with open(data_file_1, "rb") as f:
    data_unpickled_1 = pickle.load(f)

with open(data_file_2, "rb") as f:
    data_unpickled_2 = pickle.load(f)

_, data_object_1 = data_unpickled_1
_, data_object_2 = data_unpickled_2

# Concatenate and reshape the datasets
input_dataset_1 = jnp.array(data_object_1.Indata)
e_dataset_1 = jnp.array(data_object_1.SE)
e_prime_dataset_1 = jnp.array(data_object_1.Jac)

input_dataset_2 = jnp.array(data_object_2.Indata)
e_dataset_2 = jnp.array(data_object_2.SE)
e_prime_dataset_2 = jnp.array(data_object_2.Jac)

input_dataset = jnp.concatenate([input_dataset_1, input_dataset_2], axis=0)
target_e_dataset = jnp.concatenate([e_dataset_1, e_dataset_2], axis=0)
target_e_prime_dataset = jnp.concatenate([e_prime_dataset_1, e_prime_dataset_2], axis=0)

# Reshape to final dimensions
input_dataset = input_dataset.reshape((20000, 456))
target_e_dataset = target_e_dataset.reshape((20000,))
target_e_prime_dataset = target_e_prime_dataset.reshape((20000, 456))

# Create the final dataset dictionary
Dataset = {
    'displacements': input_dataset,
    'target_e': target_e_dataset,
    'target_e_prime': target_e_prime_dataset
}

# ===============================================================
# 2. MODEL DEFINITIONS (Your correct code)
# ===============================================================

class Linear(nnx.Module):
    """Linear node for neural network"""
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.W = nnx.Param(jax.random.uniform(key=key, shape=(din, dout)))
        self.b = nnx.Param(jnp.zeros(shape=(dout,)))
        self.din, self.dout = din, dout

    def __call__(self, x: jax.Array):
        return x @ self.W + self.b

def SiLU(x: jax.Array):
    """Sigmoid Weighted Linear Unit activation function"""
    return x * jax.nn.sigmoid(x)

class energy_prediction(nnx.Module):
    """Model architecture"""
    def __init__(self, dim_in: int, dim_hidden1_in: int, dim_hidden2_in: int, dim_hidden3_in, dim_out: int, *, rngs: nnx.Rngs):
        self.layer1 = Linear(din=dim_in, dout=dim_hidden1_in, rngs=rngs)
        self.layer2 = Linear(din=dim_hidden1_in, dout=dim_hidden2_in, rngs=rngs)
        self.layer3 = Linear(din=dim_hidden2_in, dout=dim_hidden3_in, rngs=rngs)
        self.layer4 = Linear(din=dim_hidden3_in, dout=dim_out, rngs=rngs)
        self.silu = SiLU
        
    def __call__(self, x_in):
        def forwardPass(x):
            x = self.layer1(x)
            x = self.silu(x)
            x = self.layer2(x)
            x = self.silu(x)
            x = self.layer3(x)
            x = self.silu(x)
            x = self.layer4(x)
            return x.squeeze(-1)
        
        e = forwardPass(x_in)
        dedx = jax.vmap(jax.grad(forwardPass, argnums=(0)))
        e_prime = dedx(x_in)
        return e, e_prime

# ===============================================================
# 3. HELPER & LOSS FUNCTIONS
# ===============================================================

def split_and_batch_dataset(dataset, batch_size, test_split=0.2, shuffle=True):
    N = dataset['displacements'].shape[0]
    indices = jnp.arange(N)
    if shuffle:
        indices = jax.random.permutation(jax.random.PRNGKey(0), indices)
    
    split_idx = int(N * (1 - test_split))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    def batch_indices(idx):
        for start in range(0, len(idx), batch_size):
            end = start + batch_size
            yield {key: value[idx[start:end]] for key, value in dataset.items()}

    return list(batch_indices(train_idx)), list(batch_indices(test_idx))

# MODIFIED loss function for nnx.Optimizer

def loss_fn(model: energy_prediction, batch: dict):
    # Hyperparameters
    alpha, gamma, lam = 1.0, 1.0, 1.0
    
    # Unpack the data from the batch dictionary
    x = batch['displacements']
    target_e = batch['target_e']
    target_e_prime = batch['target_e_prime']
    
    # The rest of your loss logic is correct
    prediction_e, prediction_e_prime = model(x)
    loss_e = jnp.mean((prediction_e - target_e)**2)
    loss_e_prime = jnp.mean((prediction_e_prime - target_e_prime)**2)

    x_zero = jnp.zeros_like(x[:1])
    prediction_zero, _ = model(x_zero)
    loss_zero = jnp.mean((prediction_zero - 0)**2)
    
    return (alpha * loss_e + gamma * loss_e_prime + lam * loss_zero)

# ===============================================================
# 4. TRAINING EXECUTION
# ===============================================================

# -- Hyperparameters --
seed = 42
Epochs = 10
Learn_Rate = 0.001
Batch_size = 10

# -- Setup --
base_key = jax.random.PRNGKey(seed)
rngs = nnx.Rngs(base_key)

# Instantiate the model
Model = energy_prediction(
    dim_in=input_dataset.shape[1], 
    dim_hidden1_in=2024,
    dim_hidden2_in=1012,
    dim_hidden3_in=212, 
    dim_out=1,
    rngs=rngs
)

# Create the stateful nnx.Optimizer
optimizer = nnx.Optimizer(Model, optax.adam(learning_rate=Learn_Rate))

# Create the JIT-compiled training step
@jax.jit(static_argnums=0)
def update_step(optimizer: nnx.Optimizer, batch: dict):
    
    loss, grads = jax.value_and_grad(loss_fn)(optimizer.model, batch)
    
    optimizer.update(grads)
    
    return optimizer, loss

# -- Data Preparation --
train_batches, test_batches = split_and_batch_dataset(Dataset, Batch_size)

# -- Training Loop --
loss_record = []
print("Starting training with nnx.Optimizer...")

for epoch in range(Epochs):
    running_loss = 0.0
    for batch in tqdm(train_batches, desc=f"Epoch {epoch+1}/{Epochs}", leave=False):
        optimizer, loss_batch = update_step(optimizer, batch)
        running_loss += loss_batch
    
    avg_loss = running_loss / len(train_batches)
    loss_record.append(avg_loss)
    print(f"Epoch {epoch+1}/{Epochs} - Average Loss: {avg_loss:.6f}")

print("\nTraining finished successfully.")

# The final trained model is now stored inside the optimizer object
trained_model = optimizer.model