import yaml

def extract_parameters(yaml_nested_dict):
    """Extract the parameters into a dictionary from the nested dictionary which holds the parameters
    from the params.yaml file.

    Args:
        yaml_nest_dict (dict): The nested dictionary which contains parameters from yaml file.
        
    Returns:
        (dict) A flattened dictionary holding just the parameter name with their corresponding value."""
    result = {}
    stack = list(yaml_nested_dict.items())
    while stack:
        key, value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.items())
        else:
            result[key] = value
    
    return result

def load_and_extract_parameters_from_config(yaml_filepath):
    """Extract parameters from the config file to be passed onto master functions.
    
    Args:
        yaml_filepath (str): The name of yaml filepath.
        
    Returns:
        (dict) A dictionary of all parameters."""
    
    with open(yaml_filepath, 'r') as file:
        config = yaml.safe_load(file)

    output_parameters = extract_parameters(config)

    return output_parameters