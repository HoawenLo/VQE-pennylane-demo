import pennylane as qml
from pennylane import numpy as np
import torch

def calculate_excitations(num_electrons, num_qubits):
    """Calculate both singles and doubles excitations.

    Args:
        num_electrons (int): Number of electrons
        num_qubits (int): Number of qubits.

    Returns:
        (tuple) A tuple with two elements, each element a list of all combinations of
        single excitations and double excitations for the number of qubits."""
    
    singles, doubles = qml.qchem.excitations(num_electrons, num_qubits)
    return singles, doubles

def create_parameters(excitations):
    """Create parameters depending on the number of excitations.

    Args:
        exictations (list): A list of excitations for qubits.

    Returns:
        (numpy.ndarray) A numpy array of variational circuit parameters of a size of the
        number of excitations."""
    return [0.0] * len(excitations)

def base_circuit(variational_circuit_params, excitations, hf_state, qubits, hamiltonian):
    """Setup an initial circuit with all doubles excitations. This circuit will be used to
    calculate gradients to determine which gates are redundant.

    Args:
        excitations (list): The excitations of electrons in terms of qubits. 
        hf_state (numpy.ndarray): The initial state which is a Hartree Fock state.
        qubits (int): The total number of qubits.
        variational_circuit_params (numpy.ndarray): A numpy array which contains parameters of the
            gates of the variational circuits.
        hamiltonian (pennylane.ops): The hamiltonian.

    Returns:
        (float) The expectation value of the Hamiltonian."""
    qml.BasisState(hf_state, wires=range(qubits))
    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(variational_circuit_params[i], wires=excitation)
        else:
            qml.SingleExcitation(variational_circuit_params[i], wires=excitation)

    return qml.expval(hamiltonian)

def singles_excitations_circuit(variational_circuit_params, single_excitations, gates_select, params_select, num_qubits, hf_state, hamiltonian):
    """The circuit that will calculate the gradients for single excitations.

    Args:
        variational_circuit_params (numpy.ndarray): The single excitation gate parameters.
        single_excitations (list): A list containing all singles excitations.
        gates_select (list): The selected doubles excitation gates.
        params_select (numpy.ndarray): A numpy array containing the partially trained doubles excitation
            gate parameters.
        num_qubits (int): Total number of qubits.
        hf_state (numpy.ndarray): The initial Hartree Fock state.
        hamiltonian (pennylane.ops): The Hamiltonian of the quantum system.

    Returns:
        (list) A list of selected singles excitation gates."""
    
    qml.BasisState(hf_state, wires=range(num_qubits))

    for i, gate in enumerate(gates_select):
        if len(gate) == 4:
            qml.DoubleExcitation(params_select[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(params_select[i], wires=gate)

    for i, gate in enumerate(single_excitations):
        if len(gate) == 4:
            qml.DoubleExcitation(variational_circuit_params[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(variational_circuit_params[i], wires=gate)
    return qml.expval(hamiltonian)

def setup_hf_state(num_electrons, num_qubits):
    """Setup the initial Hartree Fock state.

    Args:
        num_electrons (int): The number of electrons.
        num_qubits (int): The number of qubits.

    Returns:
        (numpy.ndarray) The hartree fock state."""
    return qml.qchem.hf_state(num_electrons, num_qubits)

def setup_device(device_type, num_qubits):
    """Setup the device.

    Args:
        device_type (str): The name of the device type to use, either a simulator or real quantum
            hardware.
        num_qubits (int): The total number of qubits.

    Returns:
        (pennylane.devices) The device."""
    return qml.device(device_type, wires=num_qubits)

def select_parameters(excitations, gradients, gradient_threshold):
    """After calculating the gradient contributions for each gate which corresponds to an excitation.
    Select the gates that have gradients greater than the gradient threshold. These gates are the 
    which will contribute significantly to the optimisation process to find the target quantum state.

    Args:
        excitations (list): A list of excitations for each qubit.
        gradients (list): A list of gradient values for each quantum gate.
        gradient_threshold (float): The gradient threshold value. Any gates with gradient values below
            it are discarded.
    
    Returns:
        (list) The selected excitations that were not filtered out."""
    selected_excitations = [excitations[i] for i in range(len(excitations)) if abs(gradients[i]) > gradient_threshold]
    return selected_excitations

def create_cost_function(device, circuit_template):
    """Create a cost function.

    Args:
        device (pennylane.devices): The device the quantum circuit will be ran on.
        circuit_template (function): The quantum circuit ansatz structure.

    Returns:
        (pennylane.qnode) A pennylane qnode function."""
    cost_fn = qml.QNode(circuit_template, device, interface="autograd")
    return cost_fn

def truncate_doubles_circuit(cost_fn, doubles, gradient_threshold, hf_state, num_qubits, hamiltonian):
    """Truncate the doubles circuit by calculating the gradient contribution for each gate.

    Args:
        cost_fn (pennylane.qnode): The cost function.
        doubles (list): A list of doubles excitations.
        gradient_threshold (float): Gradient values of gates below this threshold will be discarded.
        hf_state (numpy.ndarray): The initial hartree fock state.
        num_qubits (int): The number of qubits in the quantum circuit.
        hamiltonian (pennylane.ops): The Hamiltonian of the circuit in terms of bit strings.

    Returns:
        (list) The selected doubles excitations that had gradient values above the gradient threshold value."""
    
    variational_circuit_parameters = create_parameters(doubles)
    circuit_gradient = qml.grad(cost_fn, argnum=0)
    grads = circuit_gradient(variational_circuit_parameters, excitations=doubles, hf_state=hf_state, qubits=num_qubits, hamiltonian=hamiltonian)
    selected_excitations = select_parameters(doubles, grads, gradient_threshold)
    return selected_excitations

def train_truncated_doubles_circuit(step_size, selected_excitations, cost_fn, epochs, hf_state, qubits, hamiltonian):
    """Train the truncated doubles circuit to pass onto the circuit that will truncate the singles excitations.
    With the doubles gate trained to a degree, this should allow the focus to be on the singles gates as their
    parameters will be from zero.

    Args:
        step_size (float): The step size of the optimiser, equivalent to learning rate.
        selected_excitations (list): A list of the selected doubles excitations.
        cost_fn (pennylane.workflow.qnode): The cost function which simply the circuit.
        epochs (int): The total number of epochs.
    
    Returns:
        (numpy.ndarray) A numpy array of the partially trained parameters for the doubles
        exicitation gates."""
    
    opt = qml.GradientDescentOptimizer(step_size)
    params_doubles = np.zeros(len(selected_excitations), requires_grad=True)

    for _ in range(epochs):
        params_doubles = opt.step(cost_fn, params_doubles, excitations=selected_excitations, hf_state=hf_state, qubits=qubits, hamiltonian=hamiltonian)

    return params_doubles

def truncate_singles_circuit(device, singles, doubles_select, params_doubles, gradient_threshold, num_qubits, hf_state, hamiltonian):
    """Truncate the singles circuit by calculating the gradient contribution for each gate.
    Args:
        device (pennylane.devices): The device that the circuit will run; simulator or real quantum hardware.
        singles (list): A list of singles excitations.
        doubles_select (list): A list of the filtered doubles excitations.
        params_doubles (numpy.ndarray): The partially trained parameters of the doubles excitations gates.
        gradient_threshold (float): Gradient values of gates below this threshold will be discarded.
        num_qubits (int): The number of qubits in the quantum circuit.
        hf_state (numpy.ndarray): The initial hartree fock state.
        hamiltonian (pennylane.ops): The Hamiltonian of the circuit in terms of bit strings.

    Returns:
        (list) The selected doubles excitations that had gradient values above the gradient threshold value."""
    cost_fn = qml.QNode(singles_excitations_circuit, device, interface="autograd")
    circuit_gradient = qml.grad(cost_fn, argnum=0)
    variational_circuit_params = [0.0] * len(singles)

    grads = circuit_gradient(
        variational_circuit_params,
        singles,
        doubles_select,
        params_doubles,
        num_qubits,
        hf_state,
        hamiltonian
    )

    selected_excitations = select_parameters(singles, grads, gradient_threshold)
    return selected_excitations


def filter_excitation_gates(device_type, epochs, gradient_threshold, hamiltonian, train_type, num_electrons, num_qubits, partial_doubles_circuit_learning_rate):
    """Filter both single and double excitation gates by calculating their gradient contributions and 
    if they are below a gradient threshold value discard them.

    Args:
        device_type (pennylane.devices): The device the quantum circuit will run on, either a simulator or real quantum hardware.
        epochs (int): Number of training epochs for the partial doubles excitation circuit.
        gradient_threshold (float): The gradient threshold value to filter excitation gates.
        hamiltonian (pennylane.ops): The Hamiltonian of the system.
        train_type (str): The optimiser architecture to decide the datatype of the variational circuit parameters.
        num_electrons (int): The number of electrons of the quantum system.
        num_qubits (int): The number of qubits of the quantum circuit.
        partial_doubles_circuit_learning_rate (float): The learning rate of the partial doubles circuit.

    Returns:
        (list) A list of the gates which were above the gradient threshold."""

    device = setup_device(device_type, num_qubits)
    hf_state = setup_hf_state(num_electrons, num_qubits)
    singles, doubles = calculate_excitations(num_electrons, num_qubits)
    partial_doubles_circuit_cost_fn = create_cost_function(device, base_circuit)

    selected_doubles = truncate_doubles_circuit(partial_doubles_circuit_cost_fn, doubles, gradient_threshold, hf_state, num_qubits, hamiltonian)
    params_doubles = train_truncated_doubles_circuit(
        partial_doubles_circuit_learning_rate, 
        selected_doubles, 
        partial_doubles_circuit_cost_fn, 
        epochs,
        hf_state,
        num_qubits,
        hamiltonian
        )

    selected_singles = truncate_singles_circuit(
        device,
        singles,
        selected_doubles,
        params_doubles,
        gradient_threshold,
        num_qubits,
        hf_state,
        hamiltonian
    )

    selected_excitation_gates = selected_doubles + selected_singles

    if train_type == "torch":
        variational_circuit_parameters = torch.randn(len(selected_excitation_gates), requires_grad=True) * 0.1
        variational_circuit_parameters = torch.nn.Parameter(variational_circuit_parameters)
    else:
        raise ValueError(
            f"Invalid parameter train_type. Should be torch, jax or pennylane but instead is {train_type}."
        )

    output_data = {
        "selected_excitation_gates":selected_excitation_gates,
        "variational_circuit_parameters":variational_circuit_parameters
    }
    return output_data

def adaptive_uccsd_circuit_base(device):
    """Create a unitary coupled cluster ansatz with that is adaptive.
    
    This circuit ansatz is a Qnode Pennylane object. Returns a function to match the 
    pennylane functionality with the draw function.
    
    Args:
        device (pennylane.devices): The simulator or device to connect the high level 
            circuit representation to.
            
    Returns:
        (Qnode) A qnode function which is a Pennylane circuit."""
    @qml.qnode(device=device, interface="torch")
    def adaptive_uccsd_circuit(master_dictionary):
        """Create a quantum circuit with a unitary coupled cluster ansatz.
        
        Args:
            master_dictionary (dict): A dictionary which holds all inputs values.
                This includes molecular dataset, ansatz inputs and yaml parameters.
        Returns:
            (torch.Tensor.float) The expectation value of the Hamiltonian."""
        hamiltonian = master_dictionary["hamiltonian"]
        num_qubits = master_dictionary["num_qubits"]
        num_electrons = master_dictionary["num_electrons"]
        variational_circuit_params = master_dictionary["ansatz_config_params"]["variational_circuit_parameters"]
        selected_excitation_gates = master_dictionary["ansatz_config_params"]["selected_excitation_gates"]

        hf_state = qml.qchem.hf_state(num_electrons, num_qubits)

        qml.BasisState(hf_state, wires=range(num_qubits))
        for i, excitation_gate in enumerate(selected_excitation_gates):
            if len(excitation_gate) == 4:
                qml.DoubleExcitation(variational_circuit_params[i], wires=excitation_gate)
            else:
                qml.SingleExcitation(variational_circuit_params[i], wires=excitation_gate)
        return qml.expval(hamiltonian)
    return adaptive_uccsd_circuit