from .hea_ansatz.hea_ansatz import create_hea_params

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

    if ansatz_type == "hea":
        num_qubits = yaml_parameters["num_qubits"]
        num_layers = yaml_parameters["num_layers"]
        ansatz_parameters = create_hea_params(num_qubits, num_layers)
        return {"variational_circuit_parameters":ansatz_parameters}
    elif ansatz_type == "preset_pennylane_uccsd":
        print("")
