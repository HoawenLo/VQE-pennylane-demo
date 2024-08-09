from ..ansatz.hea_ansatz.hea_ansatz import hea_circuit_base
from ..ansatz.uccsd_ansatz.preset_pennylane_uccsd_ansatz import preset_pennylane_uccsd_circuit_base


def run_circuit(master_dictionary, device):
    """Create a circuit ansatz, either hea or uccsd.
    
    Args:
        master_dictionary (dict): A dictionary which holds all yaml parameters, molecular data values and ansatz
            input parameters.
        device (pennylane.devices): The quantum hardware to be used.

    Returns:
        (torch.Tensor) A torch tensor scalar representing the expectation value of the
            Hamiltonian."""
    
    ansatz_type = master_dictionary["ansatz_type"]

    if ansatz_type == "hea":
        hea_circuit = hea_circuit_base(device)
        circuit_result = hea_circuit(master_dictionary)
    elif ansatz_type == "preset_pennylane_uccsd":
        preset_pennylane_uccsd_circuit = preset_pennylane_uccsd_circuit_base(device)


    return circuit_result

def loss_fn(master_dictionary, device):
    """Calculate the loss value.
    
    Args:
        master_dictionary (dict): A dictionary which holds all yaml parameters, molecular data values and ansatz
            input parameters.
        device (pennylane.devices): The quantum hardware to be used.

    Returns:
        (torch.Tensor) A torch tensor scalar representing the expectation value of the
            Hamiltonian."""
    return run_circuit(master_dictionary, device)