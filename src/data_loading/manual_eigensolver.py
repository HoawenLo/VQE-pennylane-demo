import pennylane as qml
from pennylane import numpy as np
import scipy

def convert_hamiltonian_to_array(hamiltonian, num_qubits):
    """Convert the Hamiltonian to an array.

    Args:
        hamiltonian (pennylane.ops): The hamiltonian object from pennylane.
        num_qubits (int): The number of qubits.

    Returns:
        (numpy.ndarray) Returns an array which is a matrix of the Hamiltonian."""
    
    h_matrix = hamiltonian.sparse_matrix()
    h_sparse = qml.SparseHamiltonian(h_matrix, range(num_qubits))
    return h_sparse

def solve_eigenvalues(hamiltonian, num_qubits, eigensolver_type):
    """Solve the eigenvalues manually for a Hamiltonian.

    Args:
        hamiltonian (pennylane.ops): The hamiltonian object from pennylane.
        num_qubits (int): The number of qubits.
        eigensolver_type (str): The name of the eigensolver, either pennylane, numpy
            or scipy.

    Returns:
        (dict) Returns a dictionary containing the smallest eigenvalue corresponding
        to the ground state energy, an array containing all solved eigenvalues and the
        matrix as a numpy array."""
    
    hamiltonian_matrix = convert_hamiltonian_to_array(hamiltonian, num_qubits)
    hamiltonian_array = hamiltonian_matrix.matrix()


    if eigensolver_type == "pennylane":
        eigenvalues = hamiltonian_matrix.eigvals()
    elif eigensolver_type == "scipy":
        eigenvalues, eigenvectors = scipy.linalg.eig(hamiltonian_array)
    elif eigensolver_type == "numpy":
        eigenvalues, eigenvectors = np.linalg.eig(hamiltonian_array)
    else:
        raise ValueError(
            f"Invalid eigenvalue_type provided: {eigensolver_type}. Must be pennylane, numpy or scipy."
        )

    ground_state_energy = np.min(eigenvalues).real
    ground_state_energy = np.round(ground_state_energy, decimals=6)

    return {
        "ground_state_energy":ground_state_energy,
        "eigenvalues":eigenvalues,
        "hamiltonian_matrix":hamiltonian_matrix
    }