import pennylane as qml
import matplotlib.pyplot as plt

from ..hea_ansatz.hea_ansatz import hea_circuit_base
from ..circuit_functions.setup_device import setup_device

def display_circuit(ansatz_type, ansatz_config_params, variational_circuit_params, device, hamiltonian):
    """Display a circuit ansatz, either hea or uccsd.
    
    Args:
        ansatz_type (str): The circuit ansatz type to be created.
        ansatz_config_params (dict): The parameters for each ansatz type contained
            within a dictionary.
        variational_circuit_params (torch.Tensor): The variational circuit parameters that
            are optimised to find the best quantum circuit.
        device (pennylane.devices): The quantum hardware to be used.
        hamiltonian (pennylane.Hamiltonian): The hamiltonian to fo minimise 
            the energy expectation value for.

    Returns:
        None"""
    device = setup_device(device, ansatz_config_params["num_qubits"])

    if ansatz_type == "hea":
        num_qubits = ansatz_config_params["num_qubits"]
        num_layers = ansatz_config_params["num_layers"]
        hea_circuit = hea_circuit_base(device)
        qml.draw_mpl(hea_circuit)(num_qubits, num_layers, variational_circuit_params, hamiltonian)
        plt.tight_layout()
        plt.show()