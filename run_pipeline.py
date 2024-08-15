import argparse

import matplotlib.pyplot as plt

from src.ansatz.setup_ansatz import package_all_inputs, setup_ansatz_parameters
from src.circuit_functions.circuit import loss_fn
from src.data_loading.batch_mode import extract_batch_values
from src.data_loading.extract_params_from_yaml import load_and_extract_parameters_from_config
from src.data_loading.loading_pennylane_datasets import run_pennylane_molecular_dataset_pipeline
from src.logging.log import get_logger
from src.output_data.output_data import run_export_pipeline
from src.train.train import return_train_function
from src.visualisation.display_circuit import display_circuit
from src.visualisation.visualisation import create_loss_graph, show_graph

def reorganise_train_export_sub_pipeline(batch_components, logger, yaml_parameters):
    """A sub-component of the master pipeline which runs multiple times if batch mode is active, or
    just once if batch mode is inactive.

    Args:
        batch_components (tuple): The element of the batch.

    Returns:
        None"""
    
    if batch_components != None and yaml_parameters["data_type"] == "preset":
        yaml_parameters["molecule_name"] = batch_components[0]
        yaml_parameters["bond_length"] = batch_components[1]
    elif batch_components != None and yaml_parameters["data_type"] == "manual_inputs":
        yaml_parameters["input_symbols"] = batch_components[0]
        yaml_parameters["input_coordinates"] = batch_components[1]

    molecular_dataset = run_pennylane_molecular_dataset_pipeline(yaml_parameters)
    ansatz_inputs = setup_ansatz_parameters(yaml_parameters, molecular_dataset)
    master_dictionary = package_all_inputs(molecular_dataset, yaml_parameters, ansatz_inputs)

    if master_dictionary["show_circuit"]:
        display_circuit(master_dictionary)

    logger.info(f"Setup training architecture, selected architecture: {master_dictionary['train_type']}")
    train = return_train_function(master_dictionary["train_type"])
    
    logger.info(f"Training quantum circuit.")
    loss_function = loss_fn
    
    output_data = train(
        master_dictionary,
        loss_function
    )

    logger.info(f"Creating graph of results.")
    create_loss_graph(output_data, master_dictionary)
    run_export_pipeline(master_dictionary["export_parameters"], master_dictionary["export_graph"], output_data)
    show_graph(master_dictionary["show_loss_graph"])

def run_pipeline(config_path):
    """Master pipeline function. Load data. Create parameters and train quantum circuit.
    
    Args:
        config_path (str): The filepath to params.yaml file which is a yaml file holding 
            parameters.
            
    Returns:
        None"""
    logger = get_logger("Master pipeline")

    logger.info(f"Loading parameters from parameter config yaml file.")
    yaml_parameters = load_and_extract_parameters_from_config(config_path)
    
    logger.info(f"Extracting parameters from parameter dictionary.")

    if yaml_parameters["data_type"] == "preset":
        logger.info(f"Loading preset Pennylane molecular dataset.")
    elif yaml_parameters["data_type"] == "manual_inputs":
        logger.info(f"Building molecular dataset with manual inputs.")
    else:
        raise ValueError(
            f"The parameter input data_type is invalid. It must either be preset or manual_inputs. The parameter data_type is instead {yaml_parameters['data_type']}."
        )

    if yaml_parameters["run_batch"]:
        batch_values = extract_batch_values(yaml_parameters)

        for batch in batch_values:
            reorganise_train_export_sub_pipeline(
                batch,
                logger,
                yaml_parameters
            )
    else:
        reorganise_train_export_sub_pipeline(
            None,
            logger,
            yaml_parameters
        )




if __name__ == "__main__":
    # Create an parser
    parser = argparse.ArgumentParser("Train quantum circuit.")

    # Add config argument
    parser.add_argument("--config", dest="config", help="Params.yaml filepath", required=True)

    # Parse arguments
    args = parser.parse_args()
    
    run_pipeline(config_path=args.config)