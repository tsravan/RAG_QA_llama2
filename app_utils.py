import yaml

def read_yaml_file(yaml_file_path):
    with open(yaml_file_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg