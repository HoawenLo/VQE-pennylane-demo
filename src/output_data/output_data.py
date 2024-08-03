import os
import pickle

import matplotlib.pyplot as plt

def return_new_directory_number():
    """Check to see if the directory exists in the results directory.

    Args:
        None
    
    Returns:
        (int) The directory number."""
    all_files = os.listdir("results")
    directory_number = 0

    for file in all_files:
        first_split_file = file.split("results_")
        if not first_split_file[0]:
            directory_number = max(int(first_split_file[1]), directory_number)

    directory_number += 1

    return directory_number

def create_directory_name(directory_number):
    """Create directory name which has the template of results followed by 
    an integer number.
    
    Args:
        directory_number (int): The iteration of the results directory.
        
    Returns:
        (str) A string representing the name of the directory."""
    
    directory_name = "results_" + str(directory_number)
    return directory_name

def setup_output_directory(directory_name):
    """Create a directory where the results graph and a pickle file holding all data will be outputted.
    The pickle file will hold:
    - The best variational circuit parameters.
    - The lowest expectation (best loss) value.
    - All the expectation values (loss value) for each epoch.
    - All the parameters for each epoch.
    - The Hamiltonian in Pauli string format with the Pennylane package.
    - The number of qubits of the quantum circuit.
    - The quantum circuit object; quantum circuit utilised.

    
    Args:
        directory_name (str): The name of the directory.
        
    Returns:
        None"""
    
    target_directory = "results"
    directory_filepath = os.path.join(target_directory, directory_name)
    os.mkdir(directory_filepath)

def create_dataset_filepath(dataset_number):
    """Create the filepath of the pickled dataset.

    Args:
        dataset_number (int): The dataset iteration.

    Returns:
        (str) Return a string which is the filepath to the pickled dataset."""
    
    pickle_filename = f"results/results_{dataset_number}/dataset_{dataset_number}.pkl"
    return pickle_filename

def pickle_data(dataset_filepath, data_dictionary):
    """Pickle the data and export it.

    Args:
        dataset_filepath (str): The filepath to pickled dataset.
        data_dictionary (dict):  A dictionary containing all the important information from the 
            training of the VQC.
    
    Returns:
        None"""
    with open(dataset_filepath, "wb") as file:
        pickle.dump(data_dictionary, file)


def export_graph(graph_number, directory_name):
    """Output the graph showing how the loss changes over time.

    Args:
        graph_number (int): A graph number related to the directory number.
        dictory_name (str): The name of directory in which the results graph will be
            outputted to.

    Returns:
        None"""
    plt.savefig(f"results/{directory_name}/energy_expectation_value_{graph_number}.png")
        
def run_export_pipeline(export_data_param, export_graph_param, data_dictionary):
    """Run the export pipeline which consists of exporting the results graph and a pickle file which holds:
    - The variational circuit parameters.
    - The Hamiltonian in Pauli string format with the Pennylane package.
    - The number of qubits of the quantum circuit.
    - All the expectation values (loss value) for each epoch.
    - All the parameters for each epoch.

    Args:
        export_data_param (bool): A bool to indicate whether to export data or not.
        export_graph_param (bool): A bool to indicate whether to export the training pickle data file.
        data_dictionary (dict): A dictionary holding the data for the pickle file.

    Returns:
        None"""
    
    if export_data_param == True or export_graph_param == True:
        number = return_new_directory_number()
        directory_name = create_directory_name(number)
        setup_output_directory(directory_name)

    if export_graph_param:
        export_graph(number, directory_name)

    if export_data_param:
        dataset_filepath = create_dataset_filepath(number)
        pickle_data(dataset_filepath, data_dictionary)