import pennylane as qml

from ...logging.log import get_logger

def setup_preset_pennylane_uccsd(num_electrons, qubits):
    """Set up the hartree fock state, calculate combinations of single and double excitations
    and the wires to apply those single and double excitations to.

    Args:
        num_electrons (int): The number of electrons applied.
        qubits (int): The number of qubits

    Returns:
        (dict) A dictionary with the hartree fock state, all single excitations, all double excitations
        all single and double excitations applied to relevant wires."""

    hf_state = qml.qchem.hf_state(num_electrons, qubits)
    singles, doubles = qml.qchem.excitations(num_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    output_data = {
        "hf_state":hf_state,
        "singles":singles,
        "doubles":doubles,
        "s_wires":s_wires,
        "d_wires":d_wires
    }
    return output_data

def preset_pennylane_uccsd_circuit_base(device):
    """Create a unitary coupled cluster ansatz with the preset UCCSD pennylane class.
    
    This circuit ansatz is a Qnode Pennylane object. Returns a function to match the 
    pennylane functionality with the draw function.
    
    Args:
        device (pennylane.devices): The simulator or device to connect the high level 
            circuit representation to.
            
    Returns:
        (Qnode) A qnode function which is a Pennylane circuit."""
    @qml.qnode(device=device, interface="torch")
    def preset_pennylane_uccsd_circuit(master_dictionary):
        """Create a quantum circuit with a unitary coupled cluster ansatz.
        
        Args:
            master_dictionary (dict): A dictionary which holds all inputs values.
                This includes molecular dataset, ansatz inputs and yaml parameters.
        Returns:
            (torch.Tensor.float) The expectation value of the Hamiltonian."""
        d_wires = master_dictionary["d_wires"]
        hamiltonian = master_dictionary["hamiltonian"]
        hf_state = master_dictionary["hf_state"]
        num_qubits = master_dictionary["num_qubits"]
        s_wires = master_dictionary["s_wires"]
        variational_circuit_params = master_dictionary["ansatz_config_params"]["variational_circuit_params"]

        qml.UCCSD(variational_circuit_params, num_qubits, s_wires, d_wires, hf_state)
        return qml.expval(hamiltonian)
    
    return preset_pennylane_uccsd_circuit