import pennylane as qml
import torch

from ..circuit_functions.setup_device import setup_device

@qml.qjit
def train(ansatz_type, ansatz_config_params, variational_circuit_params, epochs, device, hamiltonian, loss_fn):
    """Train the variational quantum circuit.
    
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
    opt = torch.optim.Adam([variational_circuit_params], lr=0.1)

    results = {"theta_param":[], 
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

        results["theta_param"].append(parameter_value)
        results["energy_expectation_value"].append(loss.detach().numpy())

        print(f"Epoch {i + 1}: Loss = {loss.item():.8f} Ha, Theta = {variational_circuit_params.detach().numpy()}")

    print(f"Final parameters: {variational_circuit_params}")
    print(f"Final loss: {loss:.4f}")

    return results, variational_circuit_params, loss

