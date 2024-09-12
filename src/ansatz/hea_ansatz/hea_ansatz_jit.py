import pennylane as qml
import jax

from ...logging.log import get_logger
from .hea_ansatz import calculate_params_one_line_hea, create_layer

def create_hea_params_jit(num_qubits, num_layers):
    """Create the parameters for the hardware efficient ansatz quantum circuit.

    Args:
        num_qubits (int): The number of qubits used in the quantum circuit.
        num_layers (int): The number of layers of the HEA circuit ansatz.
    
    Returns:
        (torch.Tensor) A set of parameters of shape (num_qubits, num_layers * 3)"""

    logger = get_logger("Gate parameters")

    logger.info("Create gate parameters.")
    key = jax.random.PRNGKey(0)
    num_params = calculate_params_one_line_hea(num_layers)
    shape = (num_qubits, num_params)
    circuit_weights = jax.random.normal(key, shape) * -0.1
    logger.info("Gate parameters:")
    print(circuit_weights)
    return circuit_weights
    
def hea_circuit_base_jit(device):
    """Create a hardware efficient ansatz. This circuit ansatz is a Qnode Pennylane
    object. Returns a function to match the pennylane functionality with the 
    draw function.
    
    Args:
        device (pennylane.devices): The simulator or device to connect the high level 
            circuit representation to.
            
    Returns:
        (Qnode) A qnode function which is a Pennylane circuit."""
    @qml.qjit(autograph=True)
    @qml.qnode(device=device, interface="torch")
    def hea_circuit(master_dictionary):
        """Create a quantum circuit with a hardware efficient ansatz.
        
        Args:
            master_dictionary (dict): A dictionary which holds all inputs values.
                This includes molecular dataset, ansatz inputs and yaml parameters.
                
        Returns:
            (torch.Tensor.float) The expectation value of the Hamiltonian."""
        
        num_qubits = master_dictionary["num_qubits"]
        num_layers = master_dictionary["num_layers"]
        variational_circuit_params = master_dictionary["ansatz_config_params"]["variational_circuit_parameters"]
        hamiltonian = master_dictionary["hamiltonian"]

        for i in range(num_layers):
            create_layer(num_qubits, variational_circuit_params, i)
        return qml.expval(hamiltonian)
    
    return hea_circuit