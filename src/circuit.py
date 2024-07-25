import matplotlib.pyplot as plt
import pennylane as qml
import torch

from pennylane import numpy as np

def create_circuit(n_qubits, params):
    """Create a quantum circuit. Initial state of the quantum circuit will be the circuit
    ansatz, determined by the shape and structure of the quantum circuit and the input
    parameters.

    Args:
        n_qubits: The number of qubits for the quantum circuit.
        params (torch.Tensor): A torch.Tensor as parameters for the quantum circuit.
    
    Returns:
        (torch.Tensor, scalar) A single float value representing the expectation value 
        of the hamiltonain
    """