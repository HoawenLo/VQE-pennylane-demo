import torch

def setup_optimiser(optimiser_type):
    """Returns an optimiser.

    Args:
        optimiser_type (str): The name of the optimiser.

    Returns:
        (varies) Returns an optimiser object."""
    
    if optimiser_type == "torch_adam":
        return torch.optim.Adam