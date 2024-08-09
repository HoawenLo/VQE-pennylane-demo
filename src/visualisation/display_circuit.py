import pennylane as qml
import matplotlib.pyplot as plt

from ..ansatz.hea_ansatz.hea_ansatz import hea_circuit_base
from ..ansatz.uccsd_ansatz.preset_pennylane_uccsd_ansatz import preset_pennylane_uccsd_circuit_base
from ..circuit_functions.setup_device import setup_device

def display_circuit(master_dictionary):
    """Display a circuit ansatz, either hea or uccsd.
    
    Args:
        master_dictionary (dict): A dictionary which holds all inputs values.
            This includes molecular dataset, ansatz inputs and yaml parameters.

    Returns:
        None"""
    ansatz_type = master_dictionary["ansatz_type"]
    device_type = master_dictionary["device_type"]
    num_qubits = master_dictionary["num_qubits"]

    device = setup_device(device_type, num_qubits)

    if ansatz_type == "hea":
        circuit = hea_circuit_base(device)
    elif ansatz_type == "preset_pennylane_uccsd":
        circuit = preset_pennylane_uccsd_circuit_base(device)

    qml.draw_mpl(circuit)(master_dictionary)
    plt.tight_layout()
    plt.show()