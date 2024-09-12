from .hea_ansatz.hea_ansatz import create_hea_params
from .hea_ansatz.hea_ansatz_jit import create_hea_params_jit
from .uccsd_ansatz.preset_pennylane_uccsd_ansatz import setup_preset_pennylane_uccsd
from .uccsd_ansatz.adaptive_uccsd import filter_excitation_gates

def package_all_inputs(molecular_dataset, yaml_parameters, ansatz_inputs):
    """Combine yaml parameters and molecular dataset parameters and combines it into one dictionary
    to allow for efficiency.

    Args:
        molecular_dataset (dict): A dictionary containing data values loaded from molecular dataset.
        yaml_parameters (dict): A dictionary containing the parameter values from the params.yaml file.
        ansatz_inptus (dict): A dictionary containing all ansatz inputs.

    Returns:
        (dict) A master dictionary containing both yaml parameters and values from molecular dataset."""
    yaml_parameters.update(molecular_dataset)
    yaml_parameters["ansatz_config_params"] = ansatz_inputs
    return yaml_parameters

def setup_hea_ansatz_parameters(num_layers, num_qubits, jit_enabled):
    """Setup hardware efficient ansatz parameters. Provides option to setup jit compiled version and
    non jit compiled version.

    Args:
        num_layers (int): The number of layers for the hardware efficient ansatz.
        num_qubits (int): The number of qubits used in the circuit.
        jit_enabled (bool): A bool indicating whether jit compilation is enabled or not.

    Returns:
        (dict) Containing the ansatz parameters, which in this case for the hardware efficient ansatz
        is only the gate parameters that will be optimised."""

    if jit_enabled:
        ansatz_parameters = create_hea_params_jit(num_qubits, num_layers)
        return {"variational_circuit_parameters":ansatz_parameters}
    else:
        ansatz_parameters = create_hea_params(num_qubits, num_layers)
        return {"variational_circuit_parameters":ansatz_parameters}


def setup_ansatz_parameters(yaml_parameters, molecular_dataset):
    """Call the relevant functions to setup parameters for a circuit ansatz and return
    those relevant values.

    Args:
        yaml_parameters (dict): A dictionary which contains parameters from params.yaml file.
        molecular_dataset (dict): A dictionary which contains values from the molecular dataset.

    Returns:
        (varies) The relevant output for the ansatz type."""
    
    ansatz_type = yaml_parameters["ansatz_type"]
    
    # Set up bool if jit compilation is enabled.
    if yaml_parameters["train_type"] == "jit":
        jit_enabled = True
    else:
        jit_enabled = False

    # Data parameters
    if yaml_parameters["data_type"] == "preset":
        num_qubits = molecular_dataset["num_qubits"]
        num_electrons = molecular_dataset["num_electrons"]
    elif yaml_parameters["data_type"] == "manual_inputs":
        num_qubits = yaml_parameters["num_qubits"]
        num_electrons = yaml_parameters["num_electrons"]

    # Specified ansatz parameters - vary with different ansatz
    if ansatz_type == "hea":
        num_layers = yaml_parameters["num_layers"]
        ansatz_parameters = setup_hea_ansatz_parameters(num_layers, num_qubits, jit_enabled)
        return ansatz_parameters
    elif ansatz_type == "preset_pennylane_uccsd":
        basis_type = yaml_parameters["basis_type"]
        ansatz_parameters = setup_preset_pennylane_uccsd(num_electrons, num_qubits, basis_type)
        return ansatz_parameters
    elif ansatz_type == "adaptive_uccsd":
        basis_type = yaml_parameters["basis_state"]
        device_type = yaml_parameters["device_type"]
        epochs = yaml_parameters["partial_doubles_circuit_epochs"]
        gradient_threshold = yaml_parameters["gradient_threshold"]
        partial_doubles_circuit_learning_rate = yaml_parameters["partial_doubles_circuit_learning_rate"]
        train_type = yaml_parameters["train_type"]
        hamiltonian = molecular_dataset["hamiltonian"]
        
        ansatz_parameters = filter_excitation_gates(
            device_type,
            epochs,
            gradient_threshold,
            hamiltonian,
            train_type,
            num_electrons,
            num_qubits,
            partial_doubles_circuit_learning_rate,
            basis_type
        )
        return ansatz_parameters