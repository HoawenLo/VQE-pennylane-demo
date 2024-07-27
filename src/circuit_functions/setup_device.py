import pennylane as qml

def setup_device(device_name, num_qubits):
    """Setup the device; the quantum simulator.
    
    Args:
        device_name (str): The device name.
        num_qubits (int): The number of qubits.
        
    Returns:
        (pennylane.devices) A device object."""
    dev = qml.device(device_name, wires=num_qubits)
    return dev