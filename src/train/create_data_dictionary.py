def create_data_dictionary(best_variational_circuit_parameters, best_loss_value, loss_data, variational_circuit_parameters_over_epochs, hamiltonian, number_qubits, quantum_circuit):
    """Create a dictionary which will be pickled. Contains data from training the VQC. Dictionary will contain:


    Args:
        best_variational_circuit_parameters (varies): The best parameters of the variational circuit. Data type will
            vary depending on the optimiser used.
        best_loss_value (float): The lowest loss value found.
        loss_data (list): A list holding the loss (expectation value of the Hamiltonian) data.
        variational_circuit_parameters_over_epochs (list): A list holding all the variational circuit parameters
            over time.        
        hamiltonian (varies): The Hamiltonian used format will also vary depending on the optimiser used.
        number_qubits (int): The total number of qubits used in the quantum circuit.
        quantum_circuit (varies): The quantum circuit used in the training.
        
    Returns:
        (dict) A dictionary containing all the important information from the training of the VQC."""
    
    data = {
        "best_variational_circuit_params":best_variational_circuit_parameters,
        "best_loss_value":best_loss_value,
        "hamiltonian":hamiltonian,
        "number_qubits":number_qubits,
        "loss_data":loss_data,
        "variational_circuit_parameters_over_epochs":variational_circuit_parameters_over_epochs,
        "quantum_circuit":quantum_circuit
    }

    return data