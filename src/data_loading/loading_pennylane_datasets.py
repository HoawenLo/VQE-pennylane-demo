import pennylane as qml
from pennylane import numpy as np

from ..logging.log import get_logger

def load_pennylane_molecule_dataset(molecule_name, bond_length):
    """Load a Pennylane molecule dataset. Data provided for the molecules includes
    the Hamiltonian, number of qubits required based off the number of electrons in the
    molecule, and full configuration energy, and exact solution to the Schrodinger equations
    for a particular system.

    Args:
        molecule_name (str): The chemical formula of a molecule.
        bond_length (float): The bond length of the molecule.

    Returns:
        (pennylane.data.base.dataset) We only have one particular input configuration of the
        molecule hence the dataset corresponding to that input configuration is the returned
        object."""
    logger = get_logger("Data")

    logger.info(f"Loading pennylane dataset, molecule_name: {molecule_name}, bond_length{bond_length}.")
    dataset = qml.data.load("qchem", molname=molecule_name, bondlength=bond_length, basis="STO-3G", attributes=["fci_energy", "hamiltonian"])[0]
    return dataset

def extract_dataset_information(molecular_dataset):
    """Extracts the Hamiltonian, number of qubits and fci energy from a pennylane molecular
    dataset.

    Args:
        molecular_dataset (pennylane.data.base.dataset): The molecular dataset from a load dataset function.

    Returns:
        (dict) Returns a dictionary containing the Hamiltonian, number of qubits and fci energy."""
    logger = get_logger("Data")

    logger.info(f"Extracting data from molecular dataset.")
    H, qubits, fci_energy = molecular_dataset.hamiltonian, len(molecular_dataset.hamiltonian.wires), molecular_dataset.fci_energy
    data = {"hamiltonian":H, "num_qubits":qubits, "fci_energy":fci_energy}
    logger.info(f"Extracted Hamiltonian, number of qubits and full configuration energy: {data}")
    return data

def create_manual_hamiltonian(input_symbols, input_coordinates):
    """Create a manual molecular Hamiltonian from input symbols and coordinates.

    Args:
        input_symbols (list): A list of strings which have the chemical symbol of the Hamiltonian.
        input_coordinates (np.array): A numpy array of coordinates of each atom.

    Returns:
        (dict) A dictionary with the hamiltonian and number of qubits."""
    logger = get_logger("Data")

    logger.info("Creating Hamiltonian from manual inputs.")
    print(f"Paramters:")
    print(f"input_coordinates:")
    print(input_coordinates)
    print(f"input_symbols:")
    print(input_symbols)
    input_coordinates = np.array(input_coordinates)
    molecule = qml.qchem.Molecule(input_symbols, input_coordinates)
    H, qubits = qml.qchem.molecular_hamiltonian(molecule)
    logger.info("Output Hamiltonian and number of qubits.")
    print("Hamiltonian")
    print(H)
    print(f"Number of qubits: {qubits}")
    return {"hamiltonian":H, "num_qubits":qubits}

def run_pennylane_molecular_dataset_pipeline(data_type, molecule_name, bond_length, input_symbols, input_coordinates, fci_energy):
    """Run the entire data pipeline for extracting the data from a pennylane molecular
    dataset.
    
    Args:
        data_type (str): The data loading method. Input preset will extract the molecular data from
            qml.data.load(), Pennylane's presets, whilst the input manual_inputs will generate a 
            molecular hamiltonian and number of qubits from input coordinates.
        molecule_name (str): The chemical formula of a molecule.
        bond_length (float): The bond length of the molecule.
        input_symbols (list): A list of strings which have the chemical symbol of the Hamiltonian.
        input_coordinates (np.array): A numpy array of coordinates of each atom.
        fci_energy (float): The full configuration energy, an exact solution to the Schrodinger equation for
            a particular quantum system's configuration.

    Returns:
        (dict) Returns a dictionary containing the Hamiltonian, number of qubits and fci energy."""

    if data_type == "preset":
        molecular_dataset = load_pennylane_molecule_dataset(molecule_name, bond_length)
        data = extract_dataset_information(molecular_dataset)
    elif data_type == "manual_inputs":
        data = create_manual_hamiltonian(input_symbols, input_coordinates)
        data["fci_energy"] = fci_energy
    else:
        raise ValueError(
            f"The parameter input data_type is invalid. It must either be preset or manual_inputs. The parameter data_type is instead {data_type}."
        )
    return data

