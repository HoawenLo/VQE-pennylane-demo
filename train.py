import yaml

from src.train.train import train
from src.hea_ansatz.hea_ansatz import create_hea_params 


def train_circuit(hamiltonian, yaml_filepath="params.yaml"):
    """Master train function."""
    with open(yaml_filepath, 'r') as file:
        config = yaml.safe_load(file)

    ansatz_type = config["circuit_params"]["ansatz_type"]
    num_qubits = config["circuit_params"]["num_qubits"]
    num_layers = config["circuit_params"]["num_layers"]
    epochs = config["training_params"]["epochs"]
    device = config["device_type"]

    torch_params = create_hea_params(num_qubits, num_layers)
    ansatz_params = {"num_qubits":num_qubits, "num_layers":num_layers}

    results, variational_circuit_params, loss = train(ansatz_type, ansatz_params, torch_params, epochs, device, hamiltonian)

    return results, variational_circuit_params, loss
    