from pennylane import numpy as np

def find_position_of_best_loss(loss_data):
    """Find the position of the best loss; the lowest loss value.

    Args:
        loss_data (dict): The dictionary which holds all loss values and training parameters over 
            each epoch

    Returns:
        (int) The position of the smallest loss in the container which holds loss values."""
    
    position_of_smallest_loss = np.argmin(loss_data["energy_expectation_value"])
    return position_of_smallest_loss


def return_best_loss_and_parameters(loss_data):
    """Find the position of the smallest loss and return that value and its corresponding parameters.
    
    Args:
        loss_data (dict): The dictionary which holds all loss values and training parameters over 
            each epoch
        
    Returns:
        (tuple) A tuple holding the smallest loss and the corresponding parameters that lead to that
        loss."""
    
    position = find_position_of_best_loss(loss_data)
    best_loss_value = loss_data["energy_expectation_value"][position]
    best_parameters = loss_data["params"][position]

    return best_loss_value, best_parameters