import time

import catalyst
import jax
import optax
import pennylane as qml

from ..circuit_functions.setup_device import setup_device
from ..optimiser.setup_optimiser import setup_optimiser
from .find_best_loss_and_best_params import return_best_loss_and_parameters
from .create_data_dictionary import create_data_dictionary

def return_train_function(train_type):
    """Returns a training function depending on the optimiser architecture.

    Current options include:
    pytorch: Utilises PyTorch architecture

    Under development:
    jax: Utilises jax architecture
    pennylane: Utilises pennylane optimiser architecture

    Args:
        train_type (str):

    Returns:
        (function) The training function which is determined by the optimiser architecture."""
    
    if train_type == "torch":
        return train_torch
    
    elif train_type == "jax":
        return train_jax
    
    elif train_type == "pennylane":
        return train_pennylane
    
    elif train_type == "jit":
        return train_jit
    
    else:
        raise ValueError(
            f"Parameter train_type is invalid. Must either be torch, jax or pennylane; it is currently {train_type}"
        )

def train_torch(master_dictionary, loss_fn):
    """Train the variational quantum circuit using a PyTorch optimiser.
    
    Args:
        master_dictionary (dict): A dictionary which holds all yaml parameters, molecular data values and ansatz
            input parameters.
        loss_fn (function): The quantum circuit.
    Returns:
        (dict) A dictionary of the output data. See create_data_dictionary."""
    
    start_time = time.time()

    device_type = master_dictionary["device_type"]
    epochs = master_dictionary["epochs"]
    hamiltonian = master_dictionary["hamiltonian"]
    learning_rate = master_dictionary["learning_rate"]
    optimiser_type = master_dictionary["optimiser_type"]
    num_qubits = master_dictionary["num_qubits"]
    variational_circuit_params = master_dictionary["ansatz_config_params"]["variational_circuit_parameters"]
    device = setup_device(device_type, num_qubits)

    # Set up optimiser.
    optimiser = setup_optimiser(optimiser_type)
    opt = optimiser([variational_circuit_params], lr=learning_rate)

    results = {"params":[], 
               "energy_expectation_value":[]}

    # Training loop
    for i in range(epochs):
        # Ensure the gradients are set to zero.
        opt.zero_grad()

        # Run both the forward pass and calculate cost function.
        # Remember cost function is the expectation value.
        loss = loss_fn(master_dictionary, device)
        loss.backward()
        opt.step()
        parameter_value = variational_circuit_params.clone().detach().numpy()

        results["params"].append(parameter_value)
        results["energy_expectation_value"].append(loss.detach().numpy())

        print(f"Epoch {i + 1}: Loss = {loss.item():.8f} Ha")

    best_loss, best_parameters = return_best_loss_and_parameters(results)

    print(f"Best parameters: {best_parameters}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {loss:.4f}")

    duration = time.time() - start_time
    print(f"Training time: {duration:.4f}")
    output_data = create_data_dictionary(
        best_parameters,
        best_loss,
        results["energy_expectation_value"],
        results["params"],
        hamiltonian,
        master_dictionary["num_qubits"],
        loss_fn,
        duration
    )

    return output_data

def train_jax(master_dictionary, loss_fn):
    """Train the variational quantum circuit using a Jax optimiser.
    
    Args:
        master_dictionary (dict): A dictionary which holds all yaml parameters, molecular data values and ansatz
            input parameters.
        loss_fn (function): The quantum circuit.
    Returns:
        (dict) A dictionary of the output data. See create_data_dictionary."""
    pass

def train_pennylane():
    """"""
    pass

def train_jit(master_dictionary, loss_fn):
    """Train the variational quantum circuit using a Jax optimiser with JIT compilation.
    
    Args:
        master_dictionary (dict): A dictionary which holds all yaml parameters, molecular data values and ansatz
            input parameters.
        loss_fn (function): The quantum circuit.
    Returns:
        (dict) A dictionary of the output data. See create_data_dictionary."""

    start_time = time.time()

    device_type = master_dictionary["device_type"]
    epochs = master_dictionary["epochs"]
    hamiltonian = master_dictionary["hamiltonian"]
    learning_rate = master_dictionary["learning_rate"]
    optimiser_type = master_dictionary["optimiser_type"]
    num_qubits = master_dictionary["num_qubits"]
    variational_circuit_params = master_dictionary["ansatz_config_params"]["variational_circuit_parameters"]
    device = setup_device(device_type, num_qubits)

    # Set up optimiser.
    optimiser = setup_optimiser(optimiser_type)
    opt = optimiser(learning_rate=learning_rate)

    results = {"params":[], 
               "energy_expectation_value":[]}
    
    @qml.qjit
    def update_step(params, opt_state):
        grads = catalyst.grad(loss_fn)(params, device)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state)
    
    state = opt.init(variational_circuit_params)
    
    # Optimization loop
    for i in range(epochs):
        variational_circuit_params, state = update_step(variational_circuit_params, state)
        
        # Compute and print the current cost
        loss = loss_fn(variational_circuit_params, device)
        print(f"Iteration {i+1}, Cost: {loss}")