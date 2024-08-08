import argparse

import matplotlib.pyplot as plt

from src.circuit_functions.circuit import loss_fn
from src.data_loading.extract_params_from_yaml import load_and_extract_parameters_from_config
from src.data_loading.loading_pennylane_datasets import run_pennylane_molecular_dataset_pipeline
from src.hea_ansatz.hea_ansatz import create_hea_params
from src.logging.log import get_logger
from src.output_data.output_data import run_export_pipeline
from src.train.train import return_train_function
from src.visualisation.display_circuit import display_circuit
from src.visualisation.visualisation import create_loss_graph, show_graph

def run_pipeline(config_path):
    """Master pipeline function. Load data. Create parameters and train quantum circuit.
    
    Args:
        config_path (str): The filepath to params.yaml file which is a yaml file holding 
            parameters.
            
    Returns:
        None"""
    logger = get_logger("Master pipeline")

    logger.info(f"Loading parameters from parameter config yaml file.")
    parameters = load_and_extract_parameters_from_config(config_path)
    
    logger.info(f"Extracting parameters from parameter dictionary.")
    data_type = parameters["data_type"]
    ansatz_type = parameters["ansatz_type"]
    num_layers = parameters["num_layers"]
    epochs = parameters["epochs"]
    device = parameters["device_type"]
    molecule_name = parameters["molecule_name"]
    bond_length = parameters["bond_length"]
    
    show_circuit = parameters["show_circuit"]
    show_loss_graph = parameters["show_loss_graph"]
    
    export_graph = parameters["export_graph"]
    export_parameters = parameters["export_parameters"]
    
    input_symbols = parameters["input_symbols"]
    input_coordinates = parameters["input_coordinates"]
    input_fci_energy = parameters["fci_energy"]
    
    learning_rate = parameters["learning_rate"]
    optimiser_type = parameters["optimiser_type"]
    train_type = parameters["train_type"]

    if data_type == "preset":
        logger.info(f"Loading preset Pennylane molecular dataset.")
    elif data_type == "manual_inputs":
        logger.info(f"Building molecular dataset with manual inputs.")
    else:
        raise ValueError(
            f"The parameter input data_type is invalid. It must either be preset or manual_inputs. The parameter data_type is instead {data_type}."
        )
    molecular_dataset = run_pennylane_molecular_dataset_pipeline(
        data_type, molecule_name, bond_length, input_symbols, input_coordinates, input_fci_energy
    )

    hamiltonian = molecular_dataset["hamiltonian"]
    num_qubits = molecular_dataset["num_qubits"]
    fci_energy = molecular_dataset["fci_energy"]
    
    torch_params = create_hea_params(num_qubits, num_layers)
    ansatz_params = {"num_qubits":num_qubits, "num_layers":num_layers}

    if show_circuit:
        display_circuit(ansatz_type, ansatz_params, torch_params, device, hamiltonian)

    
    logger.info(f"Setup training architecture, selected architecture: {train_type}")
    train = return_train_function(train_type)
    
    logger.info(f"Training quantum circuit.")
    loss_function = loss_fn
    output_data = train(
        ansatz_type, 
        ansatz_params, 
        torch_params, 
        epochs, 
        device, 
        hamiltonian, 
        loss_function, 
        learning_rate,
        optimiser_type
    )

    logger.info(f"Creating graph of results.")
    create_loss_graph(output_data["loss_data"], fci_energy, molecule_name, num_layers, epochs)
    show_graph(show_loss_graph)

    run_export_pipeline(export_parameters, export_graph, output_data)

    return output_data

if __name__ == "__main__":
    # Create an parser
    parser = argparse.ArgumentParser("Train quantum circuit.")

    # Add config argument
    parser.add_argument("--config", dest="config", help="Params.yaml filepath", required=True)

    # Parse arguments
    args = parser.parse_args()
    
    run_pipeline(config_path=args.config)