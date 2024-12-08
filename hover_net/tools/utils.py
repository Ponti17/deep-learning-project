import yaml

def read_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def dump_yaml(save_path, dictionary):
    with open(save_path, "w") as f:
        yaml.dump(
            dictionary,
            f,
            default_flow_style=False,
            sort_keys=False
        )
