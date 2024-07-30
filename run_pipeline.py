from src.circuit_functions.circuit import loss_fn
from src.data_loading.extract_params_from_yaml import load_and_extract_parameters_from_config
from src.data_loading.loading_pennylane_datasets import run_pennylane_molecular_dataset_pipeline
from src.hea_ansatz.hea_ansatz import create_hea_params
from src.train.train import train
from src.visualisation.display_circuit import display_circuit
from src.visualisation.visualisation import create_loss_graph, output_graph

def run_pipeline(yaml_filepath="params.yaml"):
    """Master train function."""
    parameters = load_and_extract_parameters_from_config(yaml_filepath)
    
    ansatz_type = parameters["ansatz_type"]
    num_layers = parameters["num_layers"]
    epochs = parameters["epochs"]
    device = parameters["device_type"]
    molecule_name = parameters["molecule_name"]
    bond_length = parameters["bond_length"]
    show_circuit = parameters["show_circuit"]
    show_loss_graph = parameters["show_loss_graph"]

    molecular_dataset = run_pennylane_molecular_dataset_pipeline(molecule_name, bond_length)

    hamiltonian = molecular_dataset["hamiltonian"]
    num_qubits = molecular_dataset["num_qubits"]
    fci_energy = molecular_dataset["fci_energy"]
    
    torch_params = create_hea_params(num_qubits, num_layers)
    ansatz_params = {"num_qubits":num_qubits, "num_layers":num_layers}

    if show_circuit:
        display_circuit(ansatz_type, ansatz_params, torch_params, device, hamiltonian)
    
    loss_function = loss_fn
    results, variational_circuit_params, loss = train(ansatz_type, ansatz_params, torch_params, epochs, device, hamiltonian, loss_function)

    create_loss_graph(results, fci_energy, molecule_name, num_layers, epochs)
    output_graph(show_loss_graph)

    return results, variational_circuit_params, loss