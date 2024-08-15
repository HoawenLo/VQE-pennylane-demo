import yaml

def extract_batch_values(yaml_parameters):
    """Extract the batch values from the yaml batch file.

    Args:
        yaml_parameters (dict): A dictionary holding all yaml parameters.

    Returns:
        (list) A list of yaml parameters. If the data loading mode is set to preset each
        element of the list consists of tuples of the following structure:

        (molecule_name, bond _length)

        Whilst if the data loading mode is set to manual inputs, each element of the list
        will consist of tuples of the following structure:

        (input_symbols, input_coordinates)"""
    
    batch_yaml_filepath ="batch.yaml"

    with open(batch_yaml_filepath, 'r') as file:
        config = yaml.safe_load(file)

    if yaml_parameters["data_type"] == "preset":
        molecule_name_batch = config["preset"]["molecule_name"]
        bond_length_batch = config["preset"]["bond_length"]
        batch_elements = [(molecule_name_batch[i], bond_length_batch[i]) for i in range(len(molecule_name_batch))]
        return batch_elements
    elif yaml_parameters["data_type"] == "manual_inputs":
        input_symbols_batch = config["manual_inputs"]["input_symbols"]
        input_coordinates_batch = config["manual_inputs"]["input_coordinates"]
        batch_elements = [(input_symbols_batch[i], input_coordinates_batch[i]) for i in range(len(input_symbols_batch))]
        return batch_elements
    else:
        raise ValueError(
            f"Input data_type is {yaml_parameters['data_type']} which is invalid. Valid inputs are manual_inputs or preset."
        )


