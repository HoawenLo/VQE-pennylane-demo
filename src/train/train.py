from catalyst import qjit
import jax
import jaxopt
import numpy as np
import torch


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
    
    else:
        raise ValueError(
            f"Parameter train_type is invalid. Must either be torch, jax or pennylane; it is currently {train_type}"
        )

def train_torch(ansatz_type, ansatz_config_params, variational_circuit_params, epochs, device, hamiltonian, loss_fn, learning_rate, optimiser_type):
    """Train the variational quantum circuit using a PyTorch optimiser.
    
    Args:
        ansatz_type (str): The circuit ansatz type to be created.
        ansatz_config_params (dict): The parameters for each ansatz type contained
            within a dictionary.
        variational_circuit_params (torch.Tensor): The variational circuit parameters that
            are optimised to find the best quantum circuit.
        device (pennylane.devices): The quantum hardware to be used.
        hamiltonian (pennylane.Hamiltonian): The hamiltonian to minimise the
            energy expectation value for.
        loss_fn (function): The quantum circuit.
        optimiser_learning_rate (float): The learning rate of the optimiser.
        optimiser (torch.optim): An object representing the PyTorch optimiser.
    
    Returns:
        (tuple) """
    device = setup_device(device, ansatz_config_params["num_qubits"])

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
        loss = loss_fn(ansatz_type, ansatz_config_params, variational_circuit_params, device, hamiltonian)
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

    output_data = create_data_dictionary(
        best_parameters,
        best_loss,
        results["energy_expectation_value"],
        results["params"],
        hamiltonian,
        ansatz_config_params["num_qubits"],
        loss_fn
    )

    return output_data

# @qjit
def train_jax(ansatz_type, ansatz_config_params, variational_circuit_params, epochs, device, hamiltonian, loss_fn):
    """Train the variational quantum circuit using a Jax optimiser.
    
    Args:
        ansatz_type (str): The circuit ansatz type to be created.
        ansatz_config_params (dict): The parameters for each ansatz type contained
            within a dictionary.
        variational_circuit_params (torch.Tensor): The variational circuit parameters that
            are optimised to find the best quantum circuit.
        device (pennylane.devices): The quantum hardware to be used.
        hamiltonian (pennylane.Hamiltonian): The hamiltonian to minimise the
            energy expectation value for.
        loss_fn (function): The quantum circuit.
    
    Returns:
        (tuple) Returns a tuple of the final parameters and the final loss function value."""
    device = setup_device(device, ansatz_config_params["num_qubits"])

    # Set up optimiser.
    opt = jaxopt.GradientDescent(loss_fn, stepsize=0.4, value_and_grad=True)

    results = {"theta_param":[], 
               "energy_expectation_value":[]}

    # Training loop
    for i in range(epochs):
        # Run both the forward pass and calculate cost function.
        # Remember cost function is the expectation value.
        loss = loss_fn(ansatz_type, ansatz_config_params, variational_circuit_params, device, hamiltonian)
        update = lambda i, args: tuple(opt.update(*args))


        results["theta_param"].append(parameter_value)
        results["energy_expectation_value"].append(loss.detach().numpy())

        print(f"Epoch {i + 1}: Loss = {loss.item():.8f} Ha, Theta = {variational_circuit_params.detach().numpy()}")

    print(f"Final parameters: {variational_circuit_params}")
    print(f"Final loss: {loss:.4f}")

    return results, variational_circuit_params, loss

def train_pennylane():
    """"""
    pass