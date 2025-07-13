import jax
import jax.numpy as jnp
import jax.nn as jnn
from flax import nnx
from flax import struct
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Any

Epochs = 2000
alpha = 1.0
gamma = 0.4
lambda_ = 0.1
Learn_Rate = 0.001
beta_1 = 0.9
beta_2 = 0.999
Batch_size = 40
train_split = 0.9

import sys
import types
import pickle

fake_module = types.ModuleType("DataSetup")

class DataStore:
    def __init__(self):
        pass

fake_module.DataStore = DataStore

sys.modules["DataSetup"] = fake_module

data_file_1 = r"C:\Users\samue\Downloads\Simulation.pickle"
data_file_2 = r"C:\Users\samue\Downloads\Simulation 2.pickle"

with open(data_file_1,"rb") as f:
    data_unpickled_1 = pickle.load(f)

with open(data_file_2,"rb") as f:
    data_unpickled_2 = pickle.load(f)

_,data_object_1 = data_unpickled_1
_,data_object_2 = data_unpickled_2

input_dataset_1 = jnp.array(data_object_1.Indata)
#data_index_1 = data_object_1.i
e_dataset_1 = jnp.array(data_object_1.SE)
e_prime_dataset_1 = jnp.array(data_object_1.Jac)

input_dataset_2 = jnp.array(data_object_2.Indata)
#data_index_2 = data_object_2.i
e_dataset_2 = jnp.array(data_object_2.SE)
e_prime_dataset_2 = jnp.array(data_object_2.Jac)

input_dataset_2 = jnp.array(data_object_2.Indata)[0:2340]
e_dataset_2 = jnp.array(data_object_2.SE)[0:2340]
e_prime_dataset_2 = jnp.array(data_object_2.Jac)[0:2340]

print(input_dataset_2.shape)
print(input_dataset_1.shape)
print(e_dataset_1.shape)
print(e_prime_dataset_1.shape)

input_dataset = jax.numpy.concatenate([input_dataset_1,input_dataset_2],axis=0)
target_e_dataset = jax.numpy.concatenate([e_dataset_1, e_dataset_2],axis=0)
target_e_dataset = jax.numpy.expand_dims(target_e_dataset,axis=1)
target_e_prime_dataset = jax.numpy.concatenate([e_prime_dataset_1,e_prime_dataset_2],axis=0)

print(input_dataset.shape)
print(target_e_dataset.shape)
print(target_e_prime_dataset.shape)

seed = 42 # This can be changed but is here to make the results easy to reproduce
base_key = jax.random.PRNGKey(seed)
rngs = nnx.Rngs(base_key)

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

def add_square_feature(data,*,axis, feature_number):
    new_feature = jnp.square(data)
    new_data = jnp.concatenate([data,new_feature],axis=axis)
    feature_number += 1
    return new_data, feature_number

batch_num = input_dataset.shape[0] // Batch_size

input_dataset = input_dataset.reshape((input_dataset.shape[0],456))
displacement_dim = input_dataset.shape[1]

# add features
num_features = 0
input_dataset, num_features = add_square_feature(input_dataset,axis=1, feature_number=num_features)

target_e_dataset = target_e_dataset.reshape((target_e_dataset.shape[0],))
target_e_prime_dataset = target_e_prime_dataset.reshape((target_e_prime_dataset.shape[0],456))

params_dict_displacement = mean_and_std_dev(input_dataset,train_split=train_split)
params_dict_target_e = mean_and_std_dev(target_e_dataset,train_split=train_split)
params_dict_target_e_prime = mean_and_std_dev(target_e_prime_dataset,train_split=train_split)

input_dataset_scaled = scale_data(input_dataset,data_params=params_dict_displacement)
target_e_dataset_scaled = scale_data(target_e_dataset, data_params=params_dict_target_e)
target_e_prime_dataset_scaled = scale_data(target_e_prime_dataset, data_params=params_dict_target_e_prime)

Dataset_parameters = {
    'displacements':params_dict_displacement,
    'target_e':params_dict_target_e,
    'target_e_prime':params_dict_target_e_prime,
    'num_features':num_features,
    'standard_displacement_dim':displacement_dim
}

Dataset = {
    'displacements':input_dataset_scaled, 
    'target_e':target_e_dataset_scaled,
    'target_e_prime':target_e_prime_dataset_scaled
}

print("INSPECTING RAW DATASET")
for key, value in Dataset.items():
    print(f"Key: '{key}'")
    print(f"  - Type: {type(value)}")
    if hasattr(value, 'shape'):
        print(f"  - Shape: {value.shape}")
    else:
        print("  - No shape attribute.")
    if hasattr(value, 'dtype'):
        print(f"  - Dtype: {value.dtype}")
print("------------------------------")

class Linear(nnx.Module):
    """Linear node for neural network"""

    def __init__(self,din: int,dout: int,*,rngs: nnx.Rngs):
        key = rngs.params()
        self.W = nnx.Param(jax.random.uniform(key=key, shape=(din,dout)))
        self.b = nnx.Param(jnp.zeros(shape=(dout,)))
        self.din, self.dout = din, dout

    def __call__(self,x: jax.Array):
        return(x @ self.W + self.b)
    
def SiLU(x: jax.Array):
    """Sigmoid Weighted Linear Unit activation function"""
    return x * jax.nn.sigmoid(x)

class energy_prediction(nnx.Module):
    """
    Model architecture
    Inputs: standardised displacements and all engineered features and the parameters of the dataset
    Outputs: standardised energy value and standardised energy derivatives wrt each of the displacements
    """

    def __init__(self,dim_in: int, dim_hidden1_in: int, dim_hidden2_in: int, dim_out: int,*,rngs: nnx.Rngs):
        self.layer1 = Linear(din=dim_in,dout=dim_hidden1_in,rngs=rngs)
        self.layer2 = Linear(din=dim_hidden1_in,dout=dim_hidden2_in,rngs=rngs)
        self.output_layer = Linear(din=dim_hidden2_in,dout=dim_out,rngs=rngs)
        self.silu = SiLU

    def forwardPass(self,x):
            x = self.layer1(x)
            x = self.silu(x)
            x = self.layer2(x)
            x = self.silu(x)
            x = self.output_layer(x)
            return x.squeeze()
        
    def __call__(self,x_in,dataset_params):
        
        e = jax.vmap(self.forwardPass)(x_in)
        dedx = jax.vmap(jax.grad(self.forwardPass))
        e_prime_raw = dedx(x_in)
        e_prime_raw_lin_ft = e_prime_raw[:, :dataset_params['standard_displacement_dim']]

        sigma_e = dataset_params['target_e']['std_dev']
        sigma_x = dataset_params['displacements']['std_dev']
        mean_e_prime = dataset_params['target_e_prime']['mean']
        sigma_e_prime = dataset_params['target_e_prime']['std_dev']

        e_prime_physical = e_prime_raw_lin_ft * (sigma_e/sigma_x)
        e_prime = (e_prime_physical - mean_e_prime) / sigma_e_prime

        return e, e_prime
    
    optimiser = optax.adam(learning_rate=Learn_Rate, b1=beta_1, b2=beta_2)

def loss_fn(x: jax.Array, target_e, target_e_prime,*, Model, Dataset_parameters, alpha, gamma, lam): 
    """
    Calculates the loss of a model, works to minimise the mean square error of both 
    the strain energy prediction and the strain energy derivative prediction,
    whilst forcing the function through zero.
    """
    
    prediction_e, prediction_e_prime = Model(x, Dataset_parameters)
    loss_e = jnp.mean((prediction_e - target_e)**2)
    loss_e_prime = jnp.mean((prediction_e_prime - target_e_prime)**2)

    target_zero = (0 - Dataset_parameters['target_e']['std_dev']) / Dataset_parameters['target_e']['std_dev']
    x_zero = jnp.zeros(x[0].shape)
    x_zero = jnp.expand_dims(x_zero, axis=0)
    prediction_zero, _ = Model(x_zero, Dataset_parameters)
    loss_zero = jnp.mean((prediction_zero - target_zero)**2)

    return (alpha * loss_e + gamma * loss_e_prime + lam * loss_zero)

@nnx.dataclass
class TrainState(nnx.Object):
    params: Any
    graph_def: Any 
    state: Any
    alpha: float 
    gamma: float 
    lambda_: float 

@nnx.jit
def training_step(params,state,opt_state,batch,*,graph_def,Dataset_parameters,alpha,gamma,lambda_):

    disp_in = batch['displacements']
    e_target = batch['target_e']
    e_prime_target = batch['target_e_prime']

    def wrapped_loss_fn(params_,state_):
        Model = nnx.merge(graph_def,params_,state_)
        loss = loss_fn(
            disp_in,
            e_target,
            e_prime_target,
            Model=Model,
            Dataset_parameters=Dataset_parameters,
            alpha=alpha,
            gamma=gamma,
            lam=lambda_
        )
        return loss

    loss, grads = nnx.value_and_grad(wrapped_loss_fn, argnums=0)(params, state) 
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_state = state

    return new_params, new_state, new_opt_state, loss

def split_and_batch_dataset(dataset, batch_size, test_split=0.2, shuffle=True):
    """
    Splits the dataset into training and test sets, then yields batches for each.
    Returns: (train_batches, test_batches).
    """
    N = dataset['displacements'].shape[0]
    indices = jnp.arange(N)
    if shuffle:
        indices = jax.random.permutation(jax.random.PRNGKey(0), indices)
    split_idx = int(N * (1 - test_split))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    def batch_indices(idx):
        batch_num = len(idx) // batch_size
        for i in range(batch_num):
            start = i * batch_size
            end = start + batch_size
            batch_idx = idx[start:end]
            batch = {key: value[batch_idx] for key, value in dataset.items()}
            yield batch

    train_batches = list(batch_indices(train_idx))
    test_batches = list(batch_indices(test_idx))
    return train_batches, test_batches

train_batches, test_batches = split_and_batch_dataset(
    Dataset, 
    Batch_size, 
    test_split=(1 - train_split), 
    shuffle=True
)

# Instantiate energy prediction NN
Model = energy_prediction(
    dim_in=(input_dataset.shape[1] * 2), 
    dim_hidden1_in=128,
    dim_hidden2_in=64, 
    dim_out=1,
    rngs=rngs
)

graph_def,params,state = nnx.split(Model,nnx.Param,nnx.State)
opt_state = optimiser.init(params)

train_state = TrainState(
    graph_def=graph_def,
    params=params,
    state=state,
    alpha=alpha,
    gamma=gamma,
    lambda_=lambda_
    )

loss_record = []

for epoch in range(Epochs):
    running_loss = 0.0
    batch_count = 0

    for batch in tqdm(train_batches,desc=f"Epoch {epoch}/{Epochs}", leave=False):
        
        new_params, new_state, new_opt_state, loss_batch = training_step(
            train_state.params,
            train_state.state,
            opt_state,
            batch,
            graph_def=train_state.graph_def,
            Dataset_parameters=Dataset_parameters,
            alpha=train_state.alpha,
            gamma=train_state.gamma,
            lambda_=train_state.lambda_
        )

        opt_state = new_opt_state
        train_state.params = new_params
        train_state.state = new_state

        running_loss += loss_batch
        batch_count += 1
    
    avg_loss = avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
    loss_record.append(avg_loss)

@nnx.dataclass
class ModelData(nnx.Object):
    graph_def: Any
    params: Any
    state: Any
    Dataset_parameters: Any
    trained: bool

graph_def_trained = train_state.graph_def
params_trained = train_state.params
state_trained = train_state.state

model_data = ModelData(
    graph_def=graph_def_trained,
    params=params_trained,
    state=state_trained,
    Dataset_parameters=Dataset_parameters,
    trained=True
)

