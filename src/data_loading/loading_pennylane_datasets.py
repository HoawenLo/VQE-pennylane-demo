import pennylane as qml

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

    dataset = qml.data.load("qchem", molname=molecule_name, bondlength=bond_length, basis="STO-3G", attributes=["fci_energy", "hamiltonian"])[0]
    return dataset

def extract_dataset_information(molecular_dataset):
    """Extracts the Hamiltonian, number of qubits and fci energy from a pennylane molecular
    dataset.

    Args:
        molecular_dataset (pennylane.data.base.dataset): The molecular dataset from a load dataset function.

    Returns:
        (dict) Returns a dictionary containing the Hamiltonian, number of qubits and fci energy."""
    
    H, qubits, fci_energy = molecular_dataset.hamiltonian, len(molecular_dataset.hamiltonian.wires), molecular_dataset.fci_energy
    data = {"hamiltonian":H, "num_qubits":qubits, "fci_energy":fci_energy}
    return data

def run_pennylane_molecular_dataset_pipeline(molecule_name, bond_length):
    """Run the entire data pipeline for extracting the data from a pennylane molecular
    dataset.
    
    Args:
        molecule_name (str): The chemical formula of a molecule.
        bond_length (float): The bond length of the molecule.

    Returns:
        (dict) Returns a dictionary containing the Hamiltonian, number of qubits and fci energy."""

    molecular_dataset = load_pennylane_molecule_dataset(molecule_name, bond_length)
    data = extract_dataset_information(molecular_dataset)
    return data

