import pennylane as qml
import torch

from ...logging.log import get_logger

def calculate_params_one_line_hea(num_layers):
    """Calculate the number of parameters in the hardware efficient ansatz.
    Each layer consist of one of each of the rotation gates followed by two 
    alternating cnot gates.
    
    Args:
        num_layers: The number of layers in the quantum circuit. Minimum amount
            is one.
        
    Returns:
        (int) The number of parameters in the ansatz."""
    
    num_params_per_rotation_block = 3
    num_params = num_params_per_rotation_block * num_layers
    return num_params

def create_hea_params(num_qubits, num_layers):
    """Create the parameters for the hardware efficient ansatz quantum circuit.

    Args:
        num_qubits (int): The number of qubits used in the quantum circuit.
        num_layers (int): The number of layers of the HEA circuit ansatz.
    
    Returns:
        (torch.Tensor) A set of parameters of shape (num_qubits, num_layers * 3)"""

    logger = get_logger("Gate parameters")

    logger.info("Create gate parameters.")
    param_number = calculate_params_one_line_hea(num_layers)
    torch_random_values = torch.randn([num_qubits, param_number], requires_grad=True) * 0.1
    torch_params = torch.nn.Parameter(torch_random_values)
    logger.info("Gate parameters:")
    print(torch_params)
    return torch_params
    
def create_layer(num_qubits, params, layer_number):
    """Create a layer which consists of rotation gates with alternating
    cnot gates. Number of qubits must be minimum 4.
    
    Args:
        num_qubits (int): The number of qubits.
        params (torch.Tensor): The parameters of the rotation gates.
        layer_number (int): The current layer number of this section.
        
    Returns:
        () A layer which is just a quantum circuit; a smaller section of a larger quantum circuit."""
    
    rx_param_value = (layer_number * 3) - 3
    ry_param_value = (layer_number * 3) - 2
    rz_param_value = (layer_number * 3) - 1

    for i in range(num_qubits):
        qml.RX(params[i, rx_param_value], wires=i)
        qml.RY(params[i, ry_param_value], wires=i)
        qml.RZ(params[i, rz_param_value], wires=i)

    # First alternating CNOT gate layer.
    for i in range(0, num_qubits - 1, 2):
        qml.CNOT(wires=[i, i+1])
    
    # Add identity if space so rotation gates all aligned.
    if i + 1 < num_qubits:
        qml.Identity(num_qubits - 1)

    qml.Identity(0)

    # Second alternating CNOT gate layer.
    for i in range(1, num_qubits - 1, 2):
        qml.CNOT(wires=[i, i+1])

def hea_circuit_base(device):
    """Create a hardware efficient ansatz. This circuit ansatz is a Qnode Pennylane
    object. Returns a function to match the pennylane functionality with the 
    draw function.
    
    Args:
        device (pennylane.devices): The simulator or device to connect the high level 
            circuit representation to.
            
    Returns:
        (Qnode) A qnode function which is a Pennylane circuit."""
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