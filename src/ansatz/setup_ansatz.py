from .hea_ansatz.hea_ansatz import create_hea_params
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

def setup_ansatz_parameters(yaml_parameters, molecular_dataset):
    """Call the relevant functions to setup parameters for a circuit ansatz and return
    those relevant values.

    Args:
        yaml_parameters (dict): A dictionary which contains parameters from params.yaml file.
        molecular_dataset (dict): A dictionary which contains values from the molecular dataset.

    Returns:
        (varies) The relevant output for the ansatz type."""
    
    ansatz_type = yaml_parameters["ansatz_type"]

    if yaml_parameters["data_type"] == "preset":
        num_qubits = molecular_dataset["num_qubits"]
        num_electrons = molecular_dataset["num_electrons"]
    elif yaml_parameters["data_type"] == "manual_inputs":
        num_qubits = yaml_parameters["num_qubits"]
        num_electrons = yaml_parameters["num_electrons"]

    if ansatz_type == "hea":
        num_layers = yaml_parameters["num_layers"]
        ansatz_parameters = create_hea_params(num_qubits, num_layers)
        return {"variational_circuit_parameters":ansatz_parameters}
    elif ansatz_type == "preset_pennylane_uccsd":
        ansatz_parameters = setup_preset_pennylane_uccsd(num_electrons, num_qubits)
        return ansatz_parameters
    elif ansatz_type == "adaptive_uccsd":
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
            partial_doubles_circuit_learning_rate
        )
        return ansatz_parameters