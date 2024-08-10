import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def create_loss_graph(output_data, master_dictionary):
    """Creates a graph which visualises the change of loss over time.
    Where the loss is the expectation value for hamiltonian.

    Args:
        output_data (dict): A dictionary which holds the loss data and the change of parameters
            over time.
        fci_energy: Full configuration energy, and exact solution to the Schrodinger equations
            for a particular system.
        molecule_chemical_symbol (str): The molecule chemical symbol to be added in the title.
        number_layers (str): The number of layers of the hardware efficient ansatz.
        number_epochs (int): The number of epochs.


    Returns:
        None"""

    ansatz_type = master_dictionary["ansatz_type"]
    fci_energy = master_dictionary["fci_energy"]
    number_epochs= master_dictionary["epochs"]
    results = output_data["loss_data"]

    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(12)
    
    num_data_points = len(output_data["loss_data"])
    final_data_energy_expectation_value = np.round(results[-1], 4)
    fci_energy = np.round(fci_energy, 4)
    # Add energy plot on column 1
    ax1 = fig.add_subplot(111)
    ax1.plot(range(0, num_data_points), results, "go", ls="dashed")
    ax1.plot(range(0, num_data_points), np.full(num_data_points, fci_energy), color="red")
    ax1.set_xlabel("Optimization step", fontsize=13)
    ax1.set_ylabel("Energy (Hartree)", fontsize=13)
    ax1.text(0, results[0], r"$E_\mathrm{HF}$", fontsize=15)
    ax1.text(num_data_points - 1, final_data_energy_expectation_value, final_data_energy_expectation_value, fontsize=10)
    ax1.text(0, fci_energy, r"$E_\mathrm{FCI}$", fontsize=15)
    ax1.text(num_data_points - 1, fci_energy, fci_energy, fontsize=10)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    molecule_chemical_symbol = master_dictionary["molecule_name"]

    if ansatz_type == "hea":
        number_layers = master_dictionary["num_layers"]
        plt.title(f"Energy expectation value over {number_epochs} training epochs for {molecule_chemical_symbol} with {number_layers} layers of a hardware efficient ansatz", fontsize=16)
    elif ansatz_type == "preset_pennylane_uccsd":
        plt.title(f"Energy expectation value over {number_epochs} training epochs for {molecule_chemical_symbol} with a the preset unitary coupled cluster ansatz with pennylane", fontsize=16)
    elif ansatz_type == "adaptive_uccsd":
        plt.title(f"Energy expectation value over {number_epochs} training epochs for {molecule_chemical_symbol} with an adaptive unitary coupled cluster ansatz.", fontsize=16)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    

def return_new_graph_number():
    """Check to see if the graph exists in the results directory.

    Args:
        None
    
    Returns:
        (int) The graph number on the title of the exported graph."""
    all_files = os.listdir("results")
    graph_number = 0

    for file in all_files:
        first_split_file = file.split("energy_expectation_value_")
        if not first_split_file[0]:
            second_split_file = first_split_file[1].split(".png")
            graph_number = max(int(second_split_file[0]), graph_number)

    graph_number += 1

    return graph_number

def show_graph(show_graph):
    """Show the results graph if set to True.

    Args:
        show_graph (bool): A bool that indicates whether to show graph or not. If used in jupyter
            notebook can show graph.

    Returns:
        None"""
    
    if show_graph:
        plt.show()

    